# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Protocol

import torch

from megatron.core.jit import jit_fuser
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_torch_min_version


class LayerNormInterface(Protocol):
    """Interface that all LayerNorm implementations should follow."""

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        """Forward method for a LayerNorm implementation."""
        ...


class LayerNormBuilder(Protocol):
    """A protocol showing how Modules are expected to construct LayerNorms."""

    def __call__(
        self, *, config: TransformerConfig, hidden_size: int, eps: float
    ) -> LayerNormInterface: ...


class AccuracyCompatibleRMSNorm(torch.nn.Module, LayerNormInterface):
    """RMSNorm with explicit fp32 reduction and one output cast."""

    def __init__(
        self,
        normalized_shape: int | None = None,
        eps: float = 1e-5,
        *,
        hidden_size: int | None = None,
        config: TransformerConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        normalized_shape = hidden_size if normalized_shape is None else normalized_shape
        if normalized_shape is None:
            raise ValueError("normalized_shape or hidden_size is required")
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        dtype = config.params_dtype if config is not None else None
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        output = x_float * torch.rsqrt(variance + self.eps)
        return (output * self.weight.float()).to(x.dtype)


class WrappedTorchNorm:
    """
    A conditional wrapper to initialize an instance of PyTorch's
    `LayerNorm` or `RMSNorm` based on input
    """

    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        # TODO: unused arguments.
        # See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/223
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",
    ) -> LayerNormInterface:
        assert (
            not config.layernorm_zero_centered_gamma
        ), f"zero_centered_gamma not supported by torch LayerNorm"

        assert not config.persist_layer_norm, f"persist_layer_norm not supported by torch LayerNorm"

        assert not config.sequence_parallel, f"sequence parallel not supported by torch LayerNorm"

        assert (
            not config.memory_efficient_layer_norm
        ), f"memory_efficient_layer_norm not supported by torch LayerNorm"

        if config.normalization == "LayerNorm":
            norm_cls = torch.nn.LayerNorm
        elif config.normalization == "RMSNorm":
            if config.norm_accuracy_compatible:
                return AccuracyCompatibleRMSNorm(normalized_shape=hidden_size, eps=eps)
            assert is_torch_min_version(
                "2.4.0a0"
            ), 'Torch RMSNorm requires PyTorch version >= 2.4.0'

            norm_cls = torch.nn.RMSNorm
        elif config.normalization == "L2Norm":
            norm_cls = torch.nn.L2Norm
        else:
            raise Exception("Only LayerNorm, RMSNorm and L2Norm are currently supported")

        return norm_cls(normalized_shape=hidden_size, eps=eps)


class L2Norm(torch.nn.Module, LayerNormInterface):
    """
    Applies L2 normalization to the input tensor along the last dimension.

    This module normalizes the input tensor such that the mean of the squared values
    along the last dimension is 1 (within a small epsilon for numerical stability).

    Args:
        hidden_size (int): Expected input shape for normalization (not used internally).
        eps (float, optional): A small value added to the denominator for numerical stability.
            Default: 1e-6.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    @jit_fuser
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the actual L2 normalization.

        Args:
            x (torch.Tensor): The input tensor to normalize.

        Returns:
            torch.Tensor: The L2-normalized tensor.
        """
        x_float = x.float()
        return (x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the L2Norm module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: L2-normalized tensor with the same dtype as input.
        """
        return self._norm(x)
