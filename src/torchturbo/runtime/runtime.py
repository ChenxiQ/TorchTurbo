from __future__ import annotations

from functools import lru_cache

from .base import DeviceRuntime, TurboDevice


_SELECTED_RUNTIME: DeviceRuntime | None = None
_AUTO_DETECT_ORDER: tuple[TurboDevice, ...] = (
    TurboDevice.CUDA,
    TurboDevice.NPU,
    TurboDevice.CPU,
)

def _build_runtime(device: TurboDevice) -> DeviceRuntime:
    match device:
        case TurboDevice.CUDA:
            from .cuda import CudaRuntime

            return CudaRuntime()
        case TurboDevice.NPU:
            from .npu import NpuRuntime

            return NpuRuntime()
        case TurboDevice.CPU:
            from .cpu import CpuRuntime

            return CpuRuntime()

    raise ValueError(f"Unsupported runtime device: {device!r}.")


@lru_cache(maxsize=None)
def is_device_available(device: TurboDevice) -> bool:
    return _build_runtime(device).available()


def is_cuda_available() -> bool:
    return is_device_available(TurboDevice.CUDA)


def is_npu_available() -> bool:
    return is_device_available(TurboDevice.NPU)


def _auto_detect_runtime() -> DeviceRuntime:
    for device in _AUTO_DETECT_ORDER:
        runtime = _build_runtime(device)
        if runtime.available():
            runtime.initialize()
            return runtime

    raise RuntimeError("No available device runtime was detected.")


def get_device_runtime() -> DeviceRuntime:
    global _SELECTED_RUNTIME

    if _SELECTED_RUNTIME is None:
        _SELECTED_RUNTIME = _auto_detect_runtime()
    return _SELECTED_RUNTIME


def get_device_name() -> str:
    return get_device_runtime().device.type


__all__ = [
    "DeviceRuntime",
    "TurboDevice",
    "get_device_name",
    "get_device_runtime",
    "is_cuda_available",
    "is_device_available",
    "is_npu_available",
]
