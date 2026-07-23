# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.transformer.torch_norm import (
    AccuracyCompatibleRMSNorm,
    WrappedTorchNorm,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def _config(**overrides):
    values = {
        "num_layers": 1,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "normalization": "RMSNorm",
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_accuracy_compatible_rmsnorm_matches_explicit_formula():
    config = _config(norm_accuracy_compatible=True)
    norm = WrappedTorchNorm(config=config, hidden_size=64, eps=1e-5).cuda().bfloat16()
    x = torch.randn(2, 3, 64, device="cuda", dtype=torch.bfloat16)

    output = norm(x)
    x_float = x.float()
    expected = (
        x_float
        * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        * norm.weight.float()
    ).to(torch.bfloat16)

    assert isinstance(norm, AccuracyCompatibleRMSNorm)
    assert torch.equal(output, expected)


def test_accuracy_compatible_rmsnorm_canonicalizes_zero_input_gradients():
    config = _config(norm_accuracy_compatible=True)
    norm = WrappedTorchNorm(config=config, hidden_size=4, eps=1e-5).cuda().bfloat16()
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([-1.0, 1.0, -2.0, 2.0], device="cuda"))
    x = torch.tensor(
        [[[1.0, -1.0, 2.0, -2.0]]], device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    norm(x).backward(torch.zeros_like(x))

    assert torch.equal(x.grad, torch.zeros_like(x.grad))
    assert torch.equal(x.grad.view(torch.uint16), torch.zeros_like(x.grad.view(torch.uint16)))


def test_default_rmsnorm_stays_native():
    config = _config(norm_accuracy_compatible=False)
    norm = WrappedTorchNorm(config=config, hidden_size=64, eps=1e-5)

    assert isinstance(norm, torch.nn.RMSNorm)
