"""Make flash-linear-attention importable on ROCm-Windows torch (built without distributed).

Root cause: this torch build has no `torch._C._distributed_c10d`, so `torch.distributed.is_available()`
is False and any `from torch.distributed.* import X` (DeviceMesh, DTensor, ParallelStyle, ...) raises.
FLA's high-level modules import those unguarded. We never use tensor-parallel on a single GPU, so we
stub the torch.distributed.* submodules with catch-all dummies. Same idea as transformers PR #46205,
applied generically. Call `install()` BEFORE importing anything from `fla`.
"""
import sys
import types
import torch


def install():
    if torch.distributed.is_available():
        return
    import torch.distributed as dist
    if not hasattr(dist, "DeviceMesh"):
        dist.DeviceMesh = object
    if not hasattr(dist, "init_device_mesh"):
        dist.init_device_mesh = lambda *a, **k: None

    class _AnyModule(types.ModuleType):
        def __getattr__(self, name):
            return object                       # any symbol -> a harmless dummy class

    for sub in ["torch.distributed.device_mesh",
                "torch.distributed.tensor",
                "torch.distributed.tensor.parallel",
                "torch.distributed.tensor.placement_types",
                "torch.distributed._tensor",
                "torch.distributed.algorithms",
                "torch.distributed.algorithms._checkpoint.checkpoint_wrapper",
                "torch.distributed.fsdp"]:
        if sub not in sys.modules:
            sys.modules[sub] = _AnyModule(sub)
