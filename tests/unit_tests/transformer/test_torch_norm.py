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


def test_default_rmsnorm_stays_native():
    config = _config(norm_accuracy_compatible=False)
    norm = WrappedTorchNorm(config=config, hidden_size=64, eps=1e-5)

    assert isinstance(norm, torch.nn.RMSNorm)
