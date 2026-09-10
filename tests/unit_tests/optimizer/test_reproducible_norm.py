# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer.reproducible_norm import ReproducibleL2Norm


@pytest.mark.parametrize(
    'values,expected',
    [
        ([0.0], 0.0),
        ([3.0, 4.0], 25.0),
        ([4096.0, 1.0], 16777216.0),
        ([4096.0, 1.0, 1.0, 1.0], 16777220.0),
        ([2.0**-70], 2.0**-140),
        ([float('inf')], float('inf')),
        ([3e38], float('inf')),
    ],
)
def test_sum_squares_rounding(values, expected):
    norm = ReproducibleL2Norm()
    _, squared = norm.finish(norm.accumulate(norm.zeros(), norm.tensor(values)))
    assert squared.item() == expected


def test_nan_and_invalid_input():
    norm = ReproducibleL2Norm()
    actual, _ = norm.finish(
        norm.accumulate(norm.zeros(), norm.tensor([float('inf'), float('nan')]))
    )
    assert torch.isnan(actual).item()
    with pytest.raises(TypeError, match='FP32'):
        norm.accumulate(norm.zeros(), norm.tensor([1.0]).bfloat16())
    bins = norm.zeros()
    bins[22] = 2**40 + 1
    with pytest.raises(OverflowError):
        norm.finish(bins)


def test_layout_and_chunk_invariance():
    norm = ReproducibleL2Norm()
    gradient = torch.arange(1, 12289, device='cuda', dtype=torch.float32).reshape(96, 128) / 16384
    whole = norm.accumulate(norm.zeros(), gradient)
    transposed = norm.accumulate(norm.zeros(), gradient.T.contiguous(), chunk_size=123)
    split = norm.accumulate(norm.zeros(), gradient.flatten()[:1234])
    split += norm.accumulate(norm.zeros(), gradient.flatten()[1234:])
    assert torch.equal(whole, transposed)
    assert torch.equal(whole, split)


def test_chained_owner_groups_and_clipping(monkeypatch):
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    singleton = None
    for member in range(world):
        group = torch.distributed.new_group([member])
        if member == rank:
            singleton = group
    config = OptimizerConfig(use_accuracy_compatible=True, clip_grad=1.0)
    # Dense 3 and 4 have distinct owners; the expert 12 is replicated between
    # singleton stats groups. Finishing each child separately loses this contract.
    dense = [torch.tensor([3.0 if rank == 0 else 4.0], device='cuda')] if rank < 2 else []
    if world == 1:
        dense = [torch.tensor([3.0, 4.0], device='cuda')]
    expert = [torch.tensor([12.0], device='cuda')]
    children = [
        SimpleNamespace(
            config=config,
            get_grads_for_grad_norm=lambda _group=None: dense,
            get_grad_stats_parallel_group=lambda: torch.distributed.group.WORLD,
        ),
        SimpleNamespace(
            config=config,
            get_grads_for_grad_norm=lambda _group=None: expert,
            get_grad_stats_parallel_group=lambda: singleton,
        ),
    ]
    actual = ChainedOptimizer(children).get_grad_norm()
    assert actual.item() == 13.0
    parameter = torch.nn.Parameter(torch.zeros(2, device='cuda'))
    parameter.grad = torch.tensor([3.0, 4.0], device='cuda')
    monkeypatch.setattr('megatron.core.optimizer.clip_grads.multi_tensor_scale_tensor_impl', None)
    clip_grad_by_total_norm_fp32([parameter], 1.0, actual)
    expected = torch.tensor([3.0, 4.0], device='cuda') * (1.0 / (actual + 1e-6))
    assert torch.equal(parameter.grad, expected)


@pytest.fixture(scope='module', autouse=True)
def distributed_norm_device():
    import os

    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    owns_group = not torch.distributed.is_initialized()
    if owns_group:
        torch.distributed.init_process_group(backend='nccl')
    yield
    if owns_group:
        torch.distributed.destroy_process_group()


@pytest.fixture(scope='session')
def ensure_test_data():
    """The norm tests are self-contained and do not consume external datasets."""
