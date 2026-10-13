#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLaDA2 block-diffusion inference entry point.

LLaDA2 is a discrete diffusion model, not autogregressive. Generation starts
from a full ``<|mask|>`` template and iteratively denoises it block-by-block,
so the standard HF ``model.generate(**model_inputs, do_sample=...)`` path does
not apply — call the model's own diffusion ``generate`` (see
``modeling_llada2_moe.LLaDA2MoeModelLM.generate``).

Two hard constraints: (1) ``prompt_length`` must divide ``block_length``
(we right-pad the prompt with ``mask_id`` — never ``pad_id``, since
``eos == pad == 156892`` here, padding with it biases the output toward eos);
(2) this inference path builds its own attention mask as additive ``0`` / ``-inf``
(see ``modeling_llada2_moe.LLaDA2MoeModelLM.generate``: ``torch.tril`` + ``where``).
That mask is block-causal and small (``[num_blocks, num_blocks]`` expanded), so it
stays additive for clarity — independent of the *training* path, where
``bd.block_diffusion_mask_4d`` returns a **bool** mask (``True=attend``) to keep the
much larger ``[beam_len, beam_len]`` tensor (~4x smaller) within memory at long
cutoff. Both bool and additive masks are accepted by ``sdpa``; the two paths do not
share mask construction.

Usage::

    python [path/to/your/torchturbo]/src/torchturbo/models/llada2_moe/generate.py \\
        --model /path/to/LLaDA2.1-mini/

The model class is loaded from this directory (the torchturbo source tree), not
from the trust-remote-code copy shipped in the weights dir — see
``_load_torchturbo_modeling`` for why.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import sys
import time
import types

import torch

# Fallback ids for when neither config nor tokenizer exposes them
# (LLaDA2.1-mini values; production reads config.mask_token_id / tokenizer.*).
_DEFAULT_MASK_ID = 156895   # <|mask|>
_DEFAULT_EOS_ID = 156892    # same as pad_token_id in LLaDA2.

_DEFAULT_PROMPT = "Provide a comprehensive yet accessible introduction to Large Language Models (LLMs)."


def _load_torchturbo_runtime():
    """Load the ``torchturbo.runtime`` package without importing torchturbo.

    This keeps the standalone inference entry point independent from the full
    engine/registry/peft import chain.
    Returns ``(is_npu_available, get_device_name)``.

    Walks up from this file (``.../models/llada2_moe/generate.py``) looking for
    a sibling ``runtime`` package -- robust to wherever the script lives under
    ``torchturbo/src/torchturbo``.
    """
    cur = os.path.dirname(os.path.abspath(__file__))
    runtime_dir = None
    while cur and cur != os.path.dirname(cur):  # stop at filesystem root
        cand = os.path.join(cur, "runtime")
        if os.path.isfile(os.path.join(cand, "runtime.py")):
            runtime_dir = cand
            break
        cur = os.path.dirname(cur)
    if runtime_dir is None:
        raise SystemExit(
            "torchturbo runtime package not found. Walked up from "
            f"{os.path.dirname(os.path.abspath(__file__))} without finding runtime/."
        )

    package_name = "tt_standalone_runtime"
    package = types.ModuleType(package_name)
    package.__path__ = [runtime_dir]
    package.__package__ = package_name
    sys.modules[package_name] = package

    for submodule in ("base", "runtime"):
        full_name = f"{package_name}.{submodule}"
        spec = importlib.util.spec_from_file_location(
            full_name,
            os.path.join(runtime_dir, f"{submodule}.py"),
        )
        if spec is None or spec.loader is None:
            raise SystemExit(f"Unable to load torchturbo runtime module: {submodule}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)

    runtime_module = sys.modules[f"{package_name}.runtime"]
    return runtime_module.is_npu_available, runtime_module.get_device_name


def _load_torchturbo_modeling():
    """Load the torchturbo modeling + configuration as a nominal package.

    ``modeling_llada2_moe.py`` uses ``from .configuration_llada2_moe import ...``
    (relative import), so it must exec inside a package context. We register a
    synthetic parent package ``llada2_std`` pointing at this directory and exec
    ``configuration`` then ``modeling`` under it. This skips ``__init__.py``
    (which pulls in the training-only engine / registry / peft chain) and the
    stale trust-remote-code copy in the weights dir (which imports the removed
    ``transformers.masking_utils.create_bidirectional_mask``).
    """
    base = os.path.dirname(os.path.abspath(__file__))  # script lives next to modeling
    if not os.path.isfile(os.path.join(base, "modeling_llada2_moe.py")):
        raise SystemExit("modeling_llada2_moe.py not found beside this script.")

    pkg_name = "llada2_std"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [base]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg
    # configuration must exec first — modeling does `from .configuration import ...`
    for sub in ("configuration_llada2_moe", "modeling_llada2_moe"):
        full = f"{pkg_name}.{sub}"
        spec = importlib.util.spec_from_file_location(full, os.path.join(base, sub + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{pkg_name}.modeling_llada2_moe"]


def _pad_prompt_to_block(input_ids: torch.Tensor, block_length: int,
                         pad_id: int) -> torch.Tensor:
    """Right-pad the prompt to a multiple of ``block_length``. Returns [1, L]."""
    length = input_ids.shape[-1]
    rem = length % block_length
    if rem == 0:
        return input_ids
    pad_n = block_length - rem
    pad = torch.full((1, pad_n), pad_id, dtype=input_ids.dtype,
                     device=input_ids.device)
    return torch.cat([input_ids, pad], dim=1)


def _resolve_special_ids(config, tok) -> tuple[int, int]:
    """Resolve ``(mask_id, eos_id)`` from config then tokenizer, with fallbacks."""
    mask_id = getattr(config, "mask_token_id", None)
    if mask_id is None:
        mask_id = getattr(tok, "mask_token_id", None) or _DEFAULT_MASK_ID
    eos_id = getattr(config, "eos_token_id", None)
    if isinstance(eos_id, (list, tuple)):  # some configs store a list
        eos_id = eos_id[0]
    if eos_id is None:
        eos_id = getattr(tok, "eos_token_id", None) or _DEFAULT_EOS_ID
    return mask_id, eos_id


def _apply_attn_implementation(config, impl: str) -> None:
    """Set the attention backend on the config (covers public + internal attrs)."""
    config._attn_implementation = impl
    if hasattr(config, "_attn_implementation_internal"):
        config._attn_implementation_internal = impl


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLaDA2 block-diffusion inference")
    parser.add_argument(
        "--model", default="/path/to/LLaDA2.1-mini",
        help="HuggingFace weights directory",
    )
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--gen_length", type=int, default=512,
                        help="Max generated tokens (excluding prompt)")
    parser.add_argument("--steps", type=int, default=512,
                        help="Denoising steps per block; default = gen_length for full decoding")
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy (recommended)")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="mask->token lock confidence; 0.5 recommended (0.95 collapses to \\n)")
    parser.add_argument("--editing_threshold", type=float, default=0.0,
                        help="editing threshold for settled tokens; 0 = disable (recommended)")
    parser.add_argument("--num_to_transfer", type=int, default=1,
                        help="tokens locked per step; 1 = conservative")
    parser.add_argument("--max_post_steps", type=int, default=16,
                        help="global refinement steps after all masks resolved")
    parser.add_argument("--eos_early_stop", action="store_true", default=False,
                        help="Stop early on eos. Off by default (eos==pad, prone to false stops)")
    parser.add_argument("--device", default="npu",
                        help="npu (default; multi-card triggers NPU TBE compile bug). "
                             "Auto-falls back to cpu/cuda if npu absent.")
    parser.add_argument(
        "--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"],
        help="Attention backend. sdpa (default) on NPU goes via FlashAttentionScore "
             "(fast, but is_causal=False may be unstable). eager = pure torch (safest)."
    )
    parser.add_argument("--dtype", default="auto",
                        help="auto / bfloat16 / float16 / float32")
    return parser


@contextlib.contextmanager
def _counted_forward(model):
    """Wrap ``model.forward`` to count calls (one per diffusion step) and
    expose time-per-forward telemetry. Restores the original forward on exit."""
    original = model.forward
    counter = {"n": 0}

    def _forward(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    model.forward = _forward
    try:
        yield counter
    finally:
        model.forward = original


def _synchronize():
    """No-op if torch has no npu (local dev); otherwise block until NPU is idle."""
    if hasattr(torch, "npu"):
        torch.npu.synchronize()  # type: ignore[attr-defined]


def main() -> None:
    args = _build_arg_parser().parse_args()

    is_npu_available, get_device_name = _load_torchturbo_runtime()
    # If --device npu is requested but unavailable, fall back to the host's best device.
    device = (
        args.device
        if (args.device != "npu" or is_npu_available())
        else get_device_name()
    )
    print(f"[generate] device={device} dtype={args.dtype}", flush=True)

    # 1. Tokenizer — plain AutoTokenizer, no trust_remote_code pitfalls here.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 2. Modeling — load our torchturbo class (not the stale trc copy in the
    #    weights dir) and set the chosen attention backend on its config.
    modeling_mod = _load_torchturbo_modeling()
    LLaDA2MoeConfig = modeling_mod.LLaDA2MoeConfig
    LLaDA2MoeModelLM = modeling_mod.LLaDA2MoeModelLM

    config = LLaDA2MoeConfig.from_pretrained(args.model)
    _apply_attn_implementation(config, args.attn)

    model = LLaDA2MoeModelLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=args.dtype,        # "auto" -> bfloat16
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn,
    )
    model.eval()
    print(f"[generate] attn_implementation={config._attn_implementation}", flush=True)

    mask_id, eos_id = _resolve_special_ids(config, tok)

    # 3. Prompt — chat template (matches the official inference regime).
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if not torch.is_tensor(prompt_ids):  # some trc tokenizers ignore return_tensors
        prompt_ids = torch.tensor([list(prompt_ids)], dtype=torch.long)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_ids = prompt_ids.to(model.device)

    # 4. Right-pad the prompt to a block_length multiple (generate requireit).
    prompt_ids = _pad_prompt_to_block(prompt_ids, args.block_length, mask_id)
    prompt_len = prompt_ids.shape[1]
    print(f"[generate] prompt_len={prompt_len} block_length={args.block_length} "
          f"mask_id={mask_id} eos_id={eos_id}", flush=True)

    # 5. Diffusion generation (model's own generate, not HF's).
    _synchronize()
    t0 = time.time()
    with _counted_forward(model) as fwd_counter:
        with torch.inference_mode():
            out_ids = model.generate(
                inputs=prompt_ids,
                temperature=args.temperature,
                block_length=args.block_length,
                steps=args.steps,
                gen_length=args.gen_length,
                top_p=args.top_p,
                top_k=args.top_k,
                threshold=args.threshold,
                editing_threshold=args.editing_threshold,
                num_to_transfer=args.num_to_transfer,
                max_post_steps=args.max_post_steps,
                eos_early_stop=args.eos_early_stop,
                mask_id=mask_id,
                eos_id=eos_id,
            )
    _synchronize()
    n_tok = out_ids.shape[1]
    dt = time.time() - t0
    n_fwd = fwd_counter["n"]
    print(f"[generate] {n_tok} tok | {dt:.2f}s | {n_tok / dt:.1f} tok/s | "
          f"fwd={n_fwd} | TPF={n_tok / max(n_fwd, 1):.2f}", flush=True)

    # 6. Decode (out_ids is already prompt-stripped inside generate).
    response = tok.batch_decode(out_ids, skip_special_tokens=True)[0]
    print("*" * 60)
    print(response)
    print("*" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
