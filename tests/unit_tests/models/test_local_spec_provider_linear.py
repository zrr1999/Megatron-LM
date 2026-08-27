# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""LocalSpecProvider must expose backend.linear() for DSA/MLA down-projections."""

from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.tensor_parallel.layers import ColumnParallelLinear


def test_local_spec_provider_linear_is_replicated_te_linear():
    backend = LocalSpecProvider()
    assert backend.linear() is TELinear
    assert backend.column_parallel_linear() is ColumnParallelLinear
    assert backend.linear() is not backend.column_parallel_linear()
