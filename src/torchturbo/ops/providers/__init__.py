from __future__ import annotations

from importlib import import_module
from pkgutil import walk_packages

from torchturbo.runtime import TurboDevice


def _iter_package_modules(package_name: str) -> tuple[str, ...]:
    package = import_module(package_name)
    modules: list[str] = []
    for module_info in walk_packages(package.__path__, f"{package.__name__}."):
        relative_name = module_info.name.removeprefix(f"{package.__name__}.")
        if module_info.ispkg:
            continue
        if any(part.startswith("_") for part in relative_name.split(".")):
            continue
        modules.append(module_info.name)
    return tuple(sorted(modules))


def iter_provider_modules(device: TurboDevice) -> tuple[str, ...]:
    modules = list(_iter_package_modules("torchturbo.ops.providers.native"))
    if device is TurboDevice.CUDA:
        modules.extend(_iter_package_modules("torchturbo.ops.providers.cuda"))
    elif device is TurboDevice.NPU:
        modules.extend(_iter_package_modules("torchturbo.ops.providers.npu"))
    elif device is TurboDevice.CPU:
        pass
    else:
        raise ValueError(f"Unsupported op provider device: {device!r}.")
    return tuple(modules)


__all__ = ["iter_provider_modules"]
