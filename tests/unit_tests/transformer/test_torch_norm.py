# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.torch_norm import TorchRMSNorm, WrappedTorchNorm
from megatron.core.transformer.transformer_config import TransformerConfig


def _config(normalization="RMSNorm", zero_centered_gamma=True):
    return TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=1,
        normalization=normalization,
        layernorm_zero_centered_gamma=zero_centered_gamma,
        persist_layer_norm=False,
        sequence_parallel=False,
    )


@pytest.mark.parametrize("shape", [(2, 3, 8), (1, 5, 16)])
def test_zero_centered_rmsnorm_matches_explicit_qwen_formula(shape):
    torch.manual_seed(1234)
    norm = WrappedTorchNorm(_config(), hidden_size=shape[-1], eps=1e-6)
    assert isinstance(norm, TorchRMSNorm)
    assert list(norm.state_dict()) == ["weight"]
    assert torch.count_nonzero(norm.weight) == 0

    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.25, 0.25, shape[-1]))

    x = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    weight = norm.weight.detach().clone().requires_grad_(True)

    actual = norm(x)
    reference_fp32 = x.float()
    reference_fp32 = reference_fp32 * torch.rsqrt(
        reference_fp32.pow(2).mean(dim=-1, keepdim=True) + 1e-6
    )
    expected = (reference_fp32 * (1.0 + weight.float())).to(dtype=x.dtype)
    assert torch.equal(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad, retain_graph=True)
    actual_x_grad = x.grad.detach().clone()
    actual_weight_grad = norm.weight.grad.detach().clone()

    x.grad = None
    expected.backward(grad)
    assert torch.equal(actual_x_grad, x.grad)
    assert torch.equal(actual_weight_grad, weight.grad)


def test_non_zero_centered_rmsnorm_keeps_native_torch_module():
    norm = WrappedTorchNorm(_config(zero_centered_gamma=False), hidden_size=16, eps=1e-6)
    assert isinstance(norm, torch.nn.RMSNorm)


def test_zero_centered_layernorm_remains_unsupported():
    with pytest.raises(AssertionError, match="zero_centered_gamma not supported by torch LayerNorm"):
        WrappedTorchNorm(_config(normalization="LayerNorm"), hidden_size=16, eps=1e-6)
