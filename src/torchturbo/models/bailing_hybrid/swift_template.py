"""ms-swift chat template for Bailing V3.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple, Union

from swift.agent_template.base import BaseAgentTemplate
from swift.infer_engine import Function
from swift.template import Prompt, Template
from swift.template.template_meta import TemplateMeta

from ...logging import get_logger

logger = get_logger(__name__)

# Thinking markers (literal special tokens of the bailing tokenizer; verified
# byte-for-byte against ms-swift's ring2 template and antllm's chat_format).
THINK_START = '<think>'
THINK_END = '</think>'

# Role wrappers.
ROLE_END = '<|role_end|>'
SYS_OPEN = '<role>SYSTEM</role>'
HUMAN_OPEN = '<role>HUMAN</role>'
ASSISTANT_OPEN = '<role>ASSISTANT</role>'
OBSERVATION_OPEN = '<role>OBSERVATION</role>'
EOS = '<|endoftext|>'

# ``detailed thinking`` directive options injected into the system prompt.
DETAILED_THINKING_ON = 'detailed thinking on'
DETAILED_THINKING_OFF = 'detailed thinking off'

_FORMAT_EXAMPLE = ('<tool_call>{function-name}\n'
                   '<arg_key>{arg-key-1}</arg_key>\n'
                   '<arg_value>{arg-value-1}</arg_value>\n'
                   '<arg_key>{arg-key-2}</arg_key>\n'
                   '<arg_value>{arg-value-2}</arg_value>\n'
                   '...\n'
                   '</tool_call>\n')

# Module-level classes let dill serialize Swift preprocessing tasks by reference.


class LingV3AgentTemplate(BaseAgentTemplate):
    """Agent template for Ling-3.0 models (Bailing V3 format).

    Tool calling uses tool_call/arg_key/arg_value XML tags,
    matching the model's chat_template.jinja.
    """

    @staticmethod
    def _find_function_call(single_content: str) -> Optional[Function]:
        single_content = single_content.strip()
        func_name_match = re.match(r'^([^\n<]+)', single_content)
        if not func_name_match:
            return None
        func_name = func_name_match.group(1).strip()
        keys = re.findall(r'<arg_key>(.*?)</arg_key>', single_content, re.DOTALL)
        values = re.findall(r'<arg_value>(.*?)</arg_value>', single_content, re.DOTALL)
        if len(keys) != len(values):
            return None
        args = {k.strip(): v.strip() for k, v in zip(keys, values)}
        return Function(name=func_name, arguments=json.dumps(args, ensure_ascii=False))

    def get_toolcall(self, response: str) -> List[Function]:
        toolcall_list = re.findall(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
        functions = []
        for toolcall in toolcall_list:
            function = self._find_function_call(toolcall)
            if function:
                functions.append(function)
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = [
            '# Tools\n\n'
            'You may call one or more functions to assist with the user query.\n\n'
            'You are provided with function signatures within '
            '<tools></tools> XML tags:\n<tools>'
        ]
        for tool in tools:
            tool = self.wrap_tool(tool)
            tool_descs.append(json.dumps(tool, ensure_ascii=False))
        tool_descs.append('</tools>\n\n'
                        'If none of the functions can be used, point it out. '
                        'If the given question lacks the parameters required by the function, '
                        'also point it out.\n'
                        'If you need to use a function, for each function call, '
                        'output the function name and arguments within the following '
                        'XML format:\n' + _FORMAT_EXAMPLE)
        tool_section = '\n'.join(tool_descs)
        if system is not None and system.strip():
            return system.strip() + '\n' + tool_section
        return tool_section

    def _format_tool_calls(self, tool_call_messages) -> str:
        # Jinja: \n before each tool_call when (loop.first and content) or not loop.first.
        # Since _format_tool_calls returns ONLY tool calls (content is separate),
        # the first call gets no prefix, subsequent calls get \n.
        parts = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            tc = '<tool_call>' + tool_call['name']
            for arg_key, arg_value in tool_call['arguments'].items():
                tc += '<arg_key>' + arg_key + '</arg_key>'
                tc += '\n<arg_value>'
                if isinstance(arg_value, str):
                    tc += arg_value
                else:
                    tc += json.dumps(arg_value, ensure_ascii=False)
                tc += '</arg_value>'
            tc += '\n</tool_call>'
            parts.append(tc)
        return '\n'.join(parts)

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = (self.keyword.action in assistant_content and self.keyword.action_input in assistant_content)
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        # Close assistant turn, open OBSERVATION, then re-open ASSISTANT.
        # chat_sep is NOT added when next query is tool, so we must include <|role_end|>.
        res = ['<|role_end|><role>OBSERVATION</role>']
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res.append('\n<tool_response>\n' + tool_content + '\n</tool_response>')
        res.append('<|role_end|><role>ASSISTANT</role>\n')
        return assistant_content, res

    def _format_standalone_tool_responses(self, tool_messages) -> 'Prompt':
        # Standalone tool messages follow a user message and are folded into the query.
        # Close HUMAN, open OBSERVATION; template prompt will re-open ASSISTANT.
        res = ['<|role_end|><role>OBSERVATION</role>']
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res.append('\n<tool_response>\n' + tool_content + '\n</tool_response>')
        return res


class LingV3Template(Template):
    """Ling v3 chat template with hybrid thinking + detailed-thinking directive."""
    def init_env_args(self) -> None:
        from swift.utils import get_env_args

        super().init_env_args()
        self.enable_thinking = get_env_args('enable_thinking', bool, True)

    def _get_system(self, inputs):
        """Inject the ``detailed thinking {on|off}`` directive into the system
        (default ``on`` unless ``enable_thinking`` is False), preserved when already
        present. Placement follows the jinja ordering:

            no tools: system + '\n' + directive   (or directive if no system)
            tools:    system + '\n' + tools_block + directive
                      ^ directive AFTER tools (jinja ``{%- if tools %}`` branch)
        """
        template_meta = self.template_meta
        system = inputs.system
        tools = inputs.tools
        template_meta.check_system(system)
        if system is None:
            system = template_meta.default_system

        enable = self._get_enable_thinking(inputs)
        thinking_on = enable is not False
        directive = DETAILED_THINKING_ON if thinking_on else DETAILED_THINKING_OFF
        has_directive = isinstance(system, str) and (
            DETAILED_THINKING_ON in system or DETAILED_THINKING_OFF in system)

        if tools:
            inner = self.agent_template._format_tools(
                tools, system if isinstance(system, str) else None, inputs.messages[0])
            if not has_directive:
                inner = inner + directive
            return inner

        if isinstance(system, str) and system:
            return system if has_directive else system.rstrip('\n') + '\n' + directive
        return directive


def _make_ling_v3_template_meta(
    template_type: str,
    template_cls,
    *,
    suffix: str | list[str] = ROLE_END,
) -> 'TemplateMeta':
    """Build the shared ``TemplateMeta`` for the ling v3 template family.
    """
    return TemplateMeta(
        template_type=template_type,
        prefix=[],
        system_prefix=[SYS_OPEN + '{{SYSTEM}}' + ROLE_END],
        prompt=[HUMAN_OPEN + '{{QUERY}}' + ROLE_END + ASSISTANT_OPEN + '\n'],
        chat_sep=[ROLE_END],
        suffix=[suffix] if isinstance(suffix, str) else suffix,
        template_cls=template_cls,
        default_system=None,
        is_thinking=True,
        thinking_prefix=THINK_START,
        non_thinking_prefix=THINK_START + THINK_END,
        history_thinking_prefix=THINK_START + THINK_END,
        agent_template='ling_v3',
    )


def _register_ling_v3_agent_template() -> None:
    from swift.agent_template import agent_template_map
    if 'ling_v3' in agent_template_map:
        return
    agent_template_map['ling_v3'] = LingV3AgentTemplate
    logger.debug_rank0("Registered ms-swift agent_template 'ling_v3'")


def register() -> None:
    """Register the ``ling_v3`` / ``ling_v3_eos`` templates + agent template
    and point ``turbo_bailing_hybrid`` at ``ling_v3``."""
    from swift.model.model_meta import MODEL_MAPPING
    from swift.template.register import TEMPLATE_MAPPING, register_template

    _register_ling_v3_agent_template()

    if 'ling_v3' not in TEMPLATE_MAPPING:
        meta = _make_ling_v3_template_meta('ling_v3', LingV3Template)
        register_template(meta, exist_ok=True)
        logger.debug_rank0("Registered ms-swift template 'ling_v3'")

    if 'ling_v3_eos' not in TEMPLATE_MAPPING:
        meta = _make_ling_v3_template_meta(
            'ling_v3_eos', LingV3Template, suffix=[ROLE_END, EOS])
        register_template(meta, exist_ok=True)
        logger.debug_rank0("Registered ms-swift template 'ling_v3_eos'")

    # ``turbo_bailing_hybrid`` would otherwise inherit the V2.5 template: the
    # turbo name ``bailing_hybrid`` collides with swift's native V2.5
    # ``LLMModelType.bailing_hybrid``. architectures/requires/tags need no patch
    # (model_registry keeps architectures empty; V2.5 sets no requires/tags).
    turbo_meta = MODEL_MAPPING.get('turbo_bailing_hybrid')
    if turbo_meta is not None:
        turbo_meta.template = 'ling_v3'
        logger.debug_rank0("Patched turbo_bailing_hybrid ModelMeta -> template='ling_v3'")

__all__ = ['register']
