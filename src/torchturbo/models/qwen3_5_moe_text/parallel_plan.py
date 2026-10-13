from ...parallel import FSDPParallelPlan, FSDPPlanSpec
from ..modules.moe.config_converter import get_base_moe_config
from ..modules.moe.experts import GroupedMLP
from ..modules.moe.moe_layer import MoEConfig, MoELayer, MoESubmodules
from ..modules.moe.module import ModuleSpec
from ..qwen3_5_moe.parallel_plan import _replace_qwen3_5_moe_only


def get_parallel_plan() -> FSDPParallelPlan:
    from torch.distributed._tensor import Shard

    from ...parallel import FSDPPlacementSpec

    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))

    return FSDPParallelPlan(
        plan={
            "model.layers.*": FSDPPlanSpec(activation_checkpoint=True),
            "model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "model.layers.*.mlp": FSDPPlanSpec(
                replace_module=_replace_qwen3_5_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            "model.embed_tokens": FSDPPlanSpec(standalone_world_shard=True),
            "lm_head": FSDPPlanSpec(standalone_world_shard=True),
            "mtp.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "mtp.layers.*": FSDPPlanSpec(activation_checkpoint=True),
            "mtp.layers.*.mlp": FSDPPlanSpec(
                replace_module=_replace_qwen3_5_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
        },
        sp_size=1,
        ep_size=1,
    )
