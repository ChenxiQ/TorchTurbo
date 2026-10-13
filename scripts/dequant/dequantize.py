"""Tensor dequantization shared by offline and online loading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorSource:
    logical_key: str
    physical_keys: tuple[str, ...]
    quantized: bool = False

    @property
    def primary_key(self) -> str:
        return self.physical_keys[0]


class DequantizationFormat(ABC):
    """Interface implemented by model-specific quantization formats."""

    @abstractmethod
    def resolve_sources(
        self,
        weight_map: Mapping[str, str],
    ) -> dict[str, TensorSource]:
        raise NotImplementedError

    @abstractmethod
    def dequantize(
        self,
        source: TensorSource,
        tensors: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError


class Dequantizer:
    """Apply a model-specific format to quantized tensors."""

    def __init__(self, tensor_format: DequantizationFormat) -> None:
        self._format = tensor_format

    def resolve_sources(
        self,
        weight_map: Mapping[str, str],
    ) -> dict[str, TensorSource]:
        return self._format.resolve_sources(weight_map)

    def dequantize(
        self,
        source: TensorSource,
        tensors: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return self._format.dequantize(source, tensors)

    @staticmethod
    def apply_scales_to_bf16(
        weight: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """Apply expanded scales and return a contiguous BF16 tensor."""

        if weight.shape != scales.shape:
            raise ValueError(
                f"Weight shape {tuple(weight.shape)} does not match "
                f"scale shape {tuple(scales.shape)}"
            )
        return (
            weight.to(torch.float32)
            .mul(scales.to(torch.float32))
            .to(torch.bfloat16)
            .contiguous()
        )
