# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch
from __future__ import annotations

import os
import warnings
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing_extensions import override

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.utils import (
    divide,
    get_pg_rank,
    get_pg_size,
    get_tensor_model_parallel_group_if_none,
    is_torch_min_version,
    make_tp_sharded_tensor_for_checkpoint,
    prepare_input_tensors_for_wgrad_compute,
)

from ..dist_checkpointing.mapping import ShardedStateDict
from ..transformer.utils import make_sharded_tensors_for_checkpoint
from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from .random import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from .utils import VocabUtility

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

try:
    import transformer_engine  # pylint: disable=unused-import
    from transformer_engine.pytorch.module.base import get_dummy_wgrad

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from megatron.core.transformer.module import _use_accuracy_compatible

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "expert_tp": False,
    "is_qkv": False,
    "qkv_split_shapes": None,
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}

try:
    if is_torch_min_version("2.4.0a0"):
        custom_fwd = partial(torch.amp.custom_fwd, device_type="cuda")
        custom_bwd = partial(torch.amp.custom_bwd, device_type="cuda")
    else:
        custom_fwd = torch.cuda.amp.custom_fwd
        custom_bwd = torch.cuda.amp.custom_bwd
except:
    custom_fwd = torch.cuda.amp.custom_fwd
    custom_bwd = torch.cuda.amp.custom_bwd

try:
    if is_torch_min_version("1.13.0"):
        dist_all_gather_func = torch.distributed.all_gather_into_tensor
        dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
    else:
        dist_all_gather_func = torch.distributed._all_gather_base
        dist_reduce_scatter_func = torch.distributed._reduce_scatter_base
except:
    dist_all_gather_func = torch.distributed._all_gather_base
    dist_reduce_scatter_func = torch.distributed._reduce_scatter_base


def param_is_not_tensor_parallel_duplicate(param, tp_group=None):
    """Returns true if the passed-in parameter is not a duplicate parameter
    on another TP rank."""
    if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
        return True
    # Prefer provided tp_group when available (new explicit path).
    if tp_group is not None:
        return tp_group.rank() == 0
    # Fallback to legacy global state (back-compat).
    return get_tensor_model_parallel_rank() == 0


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    """Sets tp attributes to tensor"""
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    """Set default model parallel attributes if not set explicitly already."""

    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    """Copy model parallel attributes from one tensor to another."""

    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1, is_expert=False):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    if not is_expert:
        with get_cuda_rng_tracker().fork():
            init_method(weight)
    else:
        with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
            init_method(weight)


def _initialize_affine_weight_cpu(
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
    *,
    params_dtype=torch.float32,
    rank=None,
    world_size=None,
    skip_set_tensor_parallel_attributes=False,
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    if not skip_set_tensor_parallel_attributes:
        set_tensor_model_parallel_attributes(
            tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
        )

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size, dtype=torch.float, requires_grad=False)
    init_method(master_weight)
    master_weight = master_weight.to(dtype=params_dtype)
    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    if rank is None:
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        # all tensors must live on the same device
        cpu_weight = torch.cat(my_weight_list, dim=partition_dim).to_dense()
        weight.data.copy_(cpu_weight)
    if return_master_weight:
        return master_weight
    return None


_EMBED_IDS_CALL = 0
_EMBED_DY_HITS = 0
_EMBED_Y_HITS = 0
_EMBED_W_AT_FUSED_HITS = 0


def _maybe_register_embed_dy_dump(output_parallel, masked_input):
    """Dump-only live dY of VocabParallelEmbedding (E-530). Observation hook.

    MODEL_REPRO_EMBED_DY_DIR: bin dump (dump-on). MODEL_REPRO_EMBED_DY_HASH_DIR:
    jsonl sha only (dump-off; no bin). Hook returns g unchanged.
    """
    dump = os.environ.get("MODEL_REPRO_EMBED_DY_DIR")
    hashdir = os.environ.get("MODEL_REPRO_EMBED_DY_HASH_DIR")
    if not dump and not hashdir:
        return output_parallel

    def _hook(grad_output):
        import hashlib
        import json

        global _EMBED_DY_HITS
        _EMBED_DY_HITS += 1
        hit = _EMBED_DY_HITS
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        idn = masked_input.detach().cpu().numpy()
        dy_f32 = grad_output.detach().float().cpu().numpy()
        rec = {
            "kind": "bwd",
            "tag": "embed_dy",
            "framework": "torch",
            "rank": int(rank),
            "hit": int(hit),
            "ids_shape": list(idn.shape),
            "dy_shape": list(dy_f32.shape),
            "ids_n": int(idn.size),
            "dy_sha256": hashlib.sha256(dy_f32.tobytes()).hexdigest(),
            "ids_sha256": hashlib.sha256(idn.tobytes()).hexdigest(),
        }
        if hashdir:
            os.makedirs(hashdir, exist_ok=True)
            with open(
                os.path.join(hashdir, f"rank{rank}.jsonl"), "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if hit == 1:
                print(
                    f"[E546-EMBED-DY-HASH] dir={hashdir} rank={rank} hit={hit} "
                    f"ids_n={rec['ids_n']} sha={rec['dy_sha256'][:16]}",
                    flush=True,
                )
        if dump and int(idn.size) in (36, 37):
            os.makedirs(dump, exist_ok=True)
            stem = f"torch_embed_dy_r{rank}_h{hit}_L{int(idn.size)}"
            dy_f32.tofile(os.path.join(dump, f"{stem}.f32.bin"))
            idn.tofile(os.path.join(dump, f"{stem}_ids.i64.bin"))
            with open(os.path.join(dump, f"{stem}.json"), "w", encoding="utf-8") as handle:
                json.dump(rec, handle, sort_keys=True)
                handle.write("\n")
            print(
                f"[E612-EMBED-DY-DUMP] r{rank} hit={hit} "
                f"ids={tuple(idn.shape)} dy={tuple(dy_f32.shape)} "
                f"sha={rec['dy_sha256'][:16]}",
                flush=True,
            )
        return grad_output

    output_parallel.register_hook(_hook)
    return output_parallel


def _dump_embed_lookup_ids(input_, masked_input, input_mask, vocab_start, vocab_end):
    """Dump-only embedding lookup ids. Observation, not a wrap."""
    dump = os.environ.get("MODEL_REPRO_EMBED_IDS_DIR")
    hashdir = os.environ.get("MODEL_REPRO_EMBED_IDS_HASH_DIR")
    if not dump and not hashdir:
        return
    import hashlib
    import json

    global _EMBED_IDS_CALL
    _EMBED_IDS_CALL += 1
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    step_env = os.environ.get("MODEL_REPRO_STEP", "")
    ids = masked_input.detach().cpu().numpy()
    raw = input_.detach().cpu().numpy()
    mask = None
    if input_mask is not None:
        mask = input_mask.detach().cpu().numpy().astype("uint8")
    n = int(ids.size)
    prefix = ids.reshape(-1)[:-1] if n > 1 else ids.reshape(-1)
    meta = {
        "kind": "fwd",
        "tag": "embed_ids",
        "framework": "torch",
        "rank": rank,
        "call": _EMBED_IDS_CALL,
        "step_env": step_env,
        "shape": list(ids.shape),
        "n": n,
        "vocab_start": int(vocab_start),
        "vocab_end": int(vocab_end),
        "masked_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "input_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "prefix_n": int(prefix.size),
        "prefix_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
        "n_oov": int(mask.sum()) if mask is not None else 0,
        "n_unique_masked": int(len(set(ids.reshape(-1).tolist()))),
    }
    if hashdir:
        os.makedirs(hashdir, exist_ok=True)
        with open(os.path.join(hashdir, f"rank{rank}.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        if _EMBED_IDS_CALL == 1:
            print(
                f"[E611-EMBED-IDS-HASH] dir={hashdir} rank={rank} "
                f"n={n} prefix_n={meta['prefix_n']}",
                flush=True,
            )
    if dump:
        os.makedirs(dump, exist_ok=True)
        stem = f"torch_embed_r{rank}_c{_EMBED_IDS_CALL}_L{int(ids.size)}"
        ids.tofile(os.path.join(dump, f"{stem}_masked.i64.bin"))
        raw.tofile(os.path.join(dump, f"{stem}_input.i64.bin"))
        if mask is not None:
            mask.tofile(os.path.join(dump, f"{stem}_mask.u8.bin"))
        with open(os.path.join(dump, f"{stem}.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, sort_keys=True)
            handle.write("\n")
        print(
            f"[EMBED-IDS-DUMP] r{rank} c{_EMBED_IDS_CALL} n={ids.size} "
            f"step_env={step_env} oov={meta['n_oov']} sha={meta['masked_sha256'][:16]}",
            flush=True,
        )


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        reduce_scatter_embeddings: Decides whether to perform ReduceScatter after embedding lookup

    Keyword Args:
        config: A megatron.core.ModelParallelConfig object
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable,
        reduce_scatter_embeddings: bool = False,
        config: ModelParallelConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reduce_scatter_embeddings = reduce_scatter_embeddings
        self.tp_group = tp_group

        self.tp_group = get_tensor_model_parallel_group_if_none(self.tp_group)

        (self.vocab_start_index, self.vocab_end_index) = (
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_pg_rank(self.tp_group), get_pg_size(self.tp_group)
            )
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        self.deterministic_mode = config.deterministic_mode or _use_accuracy_compatible()
        self.config = config

        self.use_inference_optimized_reduce_scatter = (
            getattr(config, 'transformer_impl', None) == 'inference_optimized'
        )

        # Allocate weights and initialize.
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition, self.embedding_dim, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.num_embeddings,
                    self.embedding_dim,
                    self.num_embeddings_per_partition,
                    0,
                    init_method,
                    params_dtype=config.params_dtype,
                    rank=get_pg_rank(self.tp_group),
                    world_size=get_pg_size(self.tp_group),
                )
            else:
                set_tensor_model_parallel_attributes(
                    tensor=self.weight, is_parallel=True, dim=0, stride=1
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)
            else:
                set_tensor_model_parallel_attributes(
                    tensor=self.weight, is_parallel=True, dim=0, stride=1
                )

    def forward(self, input_):
        """Forward.

        Args:
            input_ (torch.Tensor): Input tensor.
        """
        if self.tp_group.size() > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            input_mask = None
        _dump_embed_lookup_ids(
            input_,
            masked_input,
            input_mask,
            self.vocab_start_index,
            self.vocab_end_index,
        )
        # E-576 dump-off: lookup W hash once per rank. Observation only.
        dump_w = os.environ.get("MODEL_REPRO_EMBED_W_HASH_DIR")
        if dump_w and not getattr(self, "_e576_w_hashed", False):
            import hashlib
            import json

            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            os.makedirs(dump_w, exist_ok=True)
            w = self.weight.detach().cpu()
            rec = {
                "kind": "param",
                "tag": "embed_w",
                "framework": "torch",
                "rank": int(rank),
                "shape": list(w.shape),
                "dtype": str(w.dtype),
                "sha_w": hashlib.sha256(
                    w.contiguous().view(torch.uint16).numpy().tobytes()
                    if w.dtype == torch.bfloat16
                    else w.contiguous().numpy().tobytes()
                ).hexdigest(),
                "vocab_start": int(self.vocab_start_index),
                "vocab_end": int(self.vocab_end_index),
            }
            with open(os.path.join(dump_w, f"rank{rank}.jsonl"), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"[E576-EMBED-W-HASH] dir={dump_w} rank={rank} shape={rec['shape']} "
                f"sha={rec['sha_w'][:16]}",
                flush=True,
            )
            self._e576_w_hashed = True
        # E-578 dump-off: W at fused-equivalent n=168 calls, not first lookup.
        dump_w_fused = os.environ.get("MODEL_REPRO_EMBED_W_AT_FUSED_HASH_DIR")
        if dump_w_fused:
            import hashlib
            import json

            idn = int(masked_input.numel())
            if idn in (36, 37, 168, 169):
                global _EMBED_W_AT_FUSED_HITS
                _EMBED_W_AT_FUSED_HITS += 1
                hit = _EMBED_W_AT_FUSED_HITS
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                os.makedirs(dump_w_fused, exist_ok=True)
                w = self.weight.detach().cpu()
                rec = {
                    "kind": "param",
                    "tag": "embed_w_at_fused",
                    "framework": "torch",
                    "rank": int(rank),
                    "hit": int(hit),
                    "ids_n": int(idn),
                    "shape": list(w.shape),
                    "dtype": str(w.dtype),
                    "sha_w": hashlib.sha256(
                        w.contiguous().view(torch.uint16).numpy().tobytes()
                        if w.dtype == torch.bfloat16
                        else w.contiguous().numpy().tobytes()
                    ).hexdigest(),
                    "vocab_start": int(self.vocab_start_index),
                    "vocab_end": int(self.vocab_end_index),
                }
                with open(
                    os.path.join(dump_w_fused, f"rank{rank}.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(
                    f"[E611-EMBED-W-AT-FUSED-HASH] dir={dump_w_fused} rank={rank} "
                    f"hit={hit} ids_n={idn} sha={rec['sha_w'][:16]}",
                    flush=True,
                )
        # E-579 dump-off: unique W rows at first n=168, not the 905MiB table.
        dump_rows = os.environ.get("MODEL_REPRO_EMBED_W_ROWS_DIR")
        if dump_rows and int(masked_input.numel()) == 168:
            import hashlib
            import json

            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if not getattr(self, "_e579_rows_dumped", False):
                os.makedirs(dump_rows, exist_ok=True)
                ids_full = masked_input.detach().reshape(-1)
                uniq = torch.unique(ids_full)
                rows = self.weight.detach()[uniq].contiguous().cpu()
                uniq_np = uniq.cpu().numpy()
                rows_u16 = (
                    rows.view(torch.uint16).numpy()
                    if rows.dtype == torch.bfloat16
                    else rows.numpy()
                )
                stem = f"torch_embed_w_rows_r{rank}_n168"
                uniq_np.astype("int64").tofile(os.path.join(dump_rows, f"{stem}_ids.i64.bin"))
                rows_u16.tofile(os.path.join(dump_rows, f"{stem}.u16.bin"))
                rec = {
                    "kind": "param",
                    "tag": "embed_w_rows",
                    "framework": "torch",
                    "rank": int(rank),
                    "ids_n": 168,
                    "n_unique": int(uniq_np.size),
                    "shape_rows": list(rows.shape),
                    "sha_rows": hashlib.sha256(rows_u16.tobytes()).hexdigest(),
                    "sha_uniq_ids": hashlib.sha256(uniq_np.tobytes()).hexdigest(),
                    "vocab_start": int(self.vocab_start_index),
                    "vocab_end": int(self.vocab_end_index),
                    "sha_w_full": hashlib.sha256(
                        self.weight.detach()
                        .contiguous()
                        .cpu()
                        .view(torch.uint16)
                        .numpy()
                        .tobytes()
                    ).hexdigest()
                    if self.weight.dtype == torch.bfloat16
                    else None,
                }
                with open(
                    os.path.join(dump_rows, f"{stem}.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(rec, handle, sort_keys=True)
                with open(
                    os.path.join(dump_rows, f"rank{rank}.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(
                    f"[E579-EMBED-W-ROWS] dir={dump_rows} rank={rank} "
                    f"n_unique={rec['n_unique']} sha_w={rec['sha_w_full'][:16] if rec['sha_w_full'] else None}",
                    flush=True,
                )
                self._e579_rows_dumped = True
        # Get the embeddings.
        if self.deterministic_mode:
            output_parallel = self.weight[masked_input]
        else:
            # F.embedding currently has a non-deterministic backward function
            output_parallel = F.embedding(masked_input, self.weight)
        output_parallel = _maybe_register_embed_dy_dump(output_parallel, masked_input)
        # E-574 dump-off: lookup Y (pre-OOV-mask) + dY. Torch first has no extra
        # token; prefix is the full tensor. Separate env from EMBED_DY / SLICE.
        dump_y = os.environ.get("MODEL_REPRO_EMBED_Y_HASH_DIR")
        if dump_y:
            import hashlib
            import json

            global _EMBED_Y_HITS
            _EMBED_Y_HITS += 1
            hit = _EMBED_Y_HITS
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            os.makedirs(dump_y, exist_ok=True)
            y = output_parallel.detach().cpu()
            rec = {
                "kind": "fwd",
                "tag": "embed_y",
                "framework": "torch",
                "rank": int(rank),
                "hit": int(hit),
                "ids_n": int(masked_input.numel()),
                "shape_y": list(y.shape),
                "sha_y": hashlib.sha256(
                    y.contiguous().view(torch.uint16).numpy().tobytes()
                    if y.dtype == torch.bfloat16
                    else y.contiguous().numpy().tobytes()
                ).hexdigest(),
                "shape_y_prefix": list(y.shape),
                "sha_y_prefix": hashlib.sha256(
                    y.contiguous().view(torch.uint16).numpy().tobytes()
                    if y.dtype == torch.bfloat16
                    else y.contiguous().numpy().tobytes()
                ).hexdigest(),
            }
            with open(os.path.join(dump_y, f"rank{rank}.jsonl"), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if hit == 1:
                print(
                    f"[E610-EMBED-Y-HASH] dir={dump_y} rank={rank} hit={hit} "
                    f"ids_n={rec['ids_n']}",
                    flush=True,
                )

            def _on_embed_y_dy(g, *, _dump=dump_y, _rank=rank, _hit=hit, _ids_n=int(masked_input.numel())):
                if g is None:
                    return g
                seq = int(g.shape[1]) if g.ndim >= 2 else int(g.shape[0])
                bwd = {
                    "kind": "bwd",
                    "tag": "embed_y",
                    "framework": "torch",
                    "rank": int(_rank),
                    "hit": int(_hit),
                    "ids_n": int(_ids_n),
                    "shape_dy": list(g.shape),
                    "sha_dy": hashlib.sha256(
                        g.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes()
                        if g.dtype == torch.bfloat16
                        else g.detach().contiguous().cpu().numpy().tobytes()
                    ).hexdigest(),
                    "shape_dy_prefix": list(g.shape),
                    "sha_dy_prefix": hashlib.sha256(
                        g.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes()
                        if g.dtype == torch.bfloat16
                        else g.detach().contiguous().cpu().numpy().tobytes()
                    ).hexdigest(),
                }
                with open(os.path.join(_dump, f"rank{_rank}.jsonl"), "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(bwd, ensure_ascii=False) + "\n")
                return g

            if output_parallel.requires_grad:
                output_parallel.register_hook(_on_embed_y_dy)
        # Mask the output embedding.
        if self.tp_group.size() > 1:
            output_parallel[input_mask, :] = 0.0

        if self.reduce_scatter_embeddings:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            output_parallel = output_parallel.transpose(0, 1).contiguous()
            if self.use_inference_optimized_reduce_scatter and not self.training:
                # Deferred to avoid circular import: inference_layers → TE → layers.
                from .inference_layers import inference_reduce_scatter_to_sequence_parallel_region

                output = inference_reduce_scatter_to_sequence_parallel_region(
                    output_parallel, self.tp_group, self.config
                )
            else:
                output = reduce_scatter_to_sequence_parallel_region(
                    output_parallel, group=self.tp_group
                )
        elif self.tp_group.size() > 1:
            # Reduce across all the model parallel GPUs.
            output = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
        else:
            output = output_parallel
        return output

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Non-default implementation for embeddings due to `allow_shape_mismatch` param"""
        state_dict = self.state_dict(prefix="", keep_vars=True)

        weight_prefix = f"{prefix}weight"
        return {
            weight_prefix: make_tp_sharded_tensor_for_checkpoint(
                tensor=state_dict["weight"],
                key=weight_prefix,
                allow_shape_mismatch=True,
                prepend_offsets=sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        }


class LinearWithFrozenWeight(torch.autograd.Function):
    """Linear operator that does not calculate gradient for weight.
    This op and LinearWithGradAccumulationAndAsyncCommunication performs
    mathematically-identical forward and DGRAD.

    Conceptually this op is the same as torch.nn.functional.linear with
    weight.requires_grad==False, but in experiments they are not identical
    mathematically."""

    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias, allreduce_dgrad, tp_group):
        """Forward with frozen weight."""
        ctx.save_for_backward(weight)
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """Backward with frozen weight."""
        (weight,) = ctx.saved_tensors
        if grad_output.dim() > 2:
            # Work around PyTorch matmul not folding some size-1 leading dims to mm.
            # Remove this once https://github.com/pytorch/pytorch/issues/186148 is fixed.
            grad_output_2d = grad_output.reshape(-1, grad_output.size(-1))
            grad_input = grad_output_2d.matmul(weight)
            grad_input = grad_input.reshape(*grad_output.shape[:-1], weight.size(1))
        else:
            grad_input = grad_output.matmul(weight)

        if ctx.allreduce_dgrad:
            # All-reduce. Note: here async and sync are effectively the same.
            torch.distributed.all_reduce(grad_input, group=ctx.tp_group)

        return grad_input, None, None, None, None


def linear_with_frozen_weight(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    tp_group: Optional[torch.distributed.ProcessGroup],
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: None = None,
) -> torch.Tensor:
    """Linear layer execution with weight.requires_grad == False.

    This function handles linear layers with weight frozen (untrainable).
    In the forward, it only saves weight and does not save input activations.
    In the backward, it does not perform weight gradient calculation, or
    weight gradient allreduce.

    Args:

    input (torch.Tensor required): input like torch.nn.functional.linear

    weight (torch.Tensor required): weight like torch.nn.functional.linear

    bias (torch.Tensor optional): bias like torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): dummy argument, used to
    keep the API unified between all forward implementation functions.

    allreduce_dgrad (bool, required): Do the allreduce of input gradients.
        Here, async and sync allreduce are the same. If sequence_parallel is
        True, this must be False, as no all reduce is performed.

    sequence_parallel (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.

    tp_group (torch.distributed.ProcessGroup): The process group to use for tensor
                                                       parallel operations.

    grad_output_buffer (List[torch.Tensor] optional): dummy argument, used to
    keep the API unified between all forward implementation functions.

    wgrad_deferral_limit (int optional): dummy argument, used to
    keep the API unified between all forward implementation functions.
    """

    assert grad_output_buffer is None, (
        "grad_output_buffer kwarg is only supported with "
        "linear_with_grad_accumulation_and_async_allreduce"
    )

    assert wgrad_deferral_limit is None, (
        "This arg is only supported with " "linear_with_grad_accumulation_and_async_allreduce"
    )

    tp_group = get_tensor_model_parallel_group_if_none(tp_group)

    if sequence_parallel:
        input = gather_from_sequence_parallel_region(
            input, tensor_parallel_output_grad=True, group=tp_group
        )
    else:
        input = input

    args = [input, weight, bias, allreduce_dgrad, tp_group]

    return LinearWithFrozenWeight.apply(*args)


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
    ):
        """Forward."""
        if gradient_accumulation_fusion and hasattr(weight, "main_grad"):
            main_grad = weight.main_grad
        else:
            main_grad = None
        ctx.save_for_backward(input, weight)
        # We can't save main_grad in save_for_backward as this module would be
        # reused across layers like MTP logits. So, to prevent in-place modification
        # checks we save the tensor in ctx.
        ctx.main_grad = main_grad
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.grad_output_buffer = grad_output_buffer
        ctx.tp_group = tp_group

        if sequence_parallel:
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * tp_group.size()

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            dist_all_gather_func(all_gather_buffer, input, group=tp_group)
            total_input = all_gather_buffer
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """Backward."""
        input, weight = ctx.saved_tensors
        main_grad = ctx.main_grad
        use_bias = ctx.use_bias
        grad_output_buffer = ctx.grad_output_buffer
        wgrad_deferral_limit = ctx.wgrad_deferral_limit
        handle = None
        tp_group = ctx.tp_group

        if ctx.gradient_accumulation_fusion:
            weight.main_grad = main_grad

        wgrad_compute = True
        if grad_output_buffer is not None:
            if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

        if wgrad_compute:
            if ctx.sequence_parallel:
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * tp_group.size()

                all_gather_buffer = get_global_memory_buffer().get_tensor(
                    dim_size, input.dtype, "mpu"
                )
                handle = dist_all_gather_func(
                    all_gather_buffer, input, group=tp_group, async_op=True
                )

                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation
                total_input = all_gather_buffer
            else:
                total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel and wgrad_compute:
            # pylint: disable=possibly-used-before-assignment
            handle.wait()

        if wgrad_compute:
            grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
                grad_output, total_input
            )

        if ctx.allreduce_dgrad:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(grad_input, group=tp_group, async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.allreduce_dgrad
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = dist_reduce_scatter_func(
                sub_grad_input, grad_input, group=tp_group, async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        if ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                # In case of Megatron-FSDP, need to create main grad buffers in-place
                if hasattr(weight, "__fsdp_param__"):
                    weight.main_grad = weight.get_main_grad()
                    # Import here to avoid circular import
                    from megatron.core.extensions.transformer_engine import te_general_gemm

                    if te_general_gemm is not None:
                        # Use TE general_gemm to support mixed-precision output
                        # (e.g. bf16 input -> fp32 main_grad) which torch.matmul
                        # does not support via the out= parameter.
                        te_general_gemm(
                            total_input,
                            grad_output,
                            out_dtype=weight.main_grad.dtype,
                            layout="NT",
                            out=weight.main_grad,
                            grad=True,
                        )
                    else:
                        torch.matmul(grad_output.t(), total_input, out=weight.main_grad)
                else:
                    if weight.main_grad.dtype == torch.float32:
                        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                            total_input, grad_output, weight.main_grad
                        )
                    elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                            total_input, grad_output, weight.main_grad
                        )
                    else:
                        raise RuntimeError(
                            "Unsupported gradient type for gradient accumulation fusion"
                        )

            if hasattr(weight, "grad_added_to_main_grad"):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, "zero_out_wgrad", False):
                    if HAVE_TE:
                        # get_dummy_wgrad function in TE enables reuse of single dummy wgrad buffer
                        # across different layers/microbatches. The function accepts shape as list.
                        grad_weight = get_dummy_wgrad(
                            list(weight.main_grad.shape), input.dtype, zero=True
                        )
                    else:
                        grad_weight = torch.zeros(
                            weight.main_grad.shape,
                            dtype=input.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                else:
                    if HAVE_TE:
                        grad_weight = get_dummy_wgrad(list(weight.main_grad.shape), input.dtype)
                    else:
                        grad_weight = torch.empty(
                            weight.main_grad.shape,
                            dtype=input.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            # Need to return None's as gradient has to flow for all the input arguments
            # provided during forward
            return (sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None)

        if ctx.allreduce_dgrad:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: Optional[int] = 0,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calculation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Args:
        input (torch.Tensor required): input like torch.nn.functional.linear

        weight (torch.Tensor required): weight like torch.nn.functional.linear

        bias (torch.Tensor optional): bias like torch.nn.functional.linear

        gradient_accumulation_fusion (bool required): Perform the gradient
            accumulation fusion, requires the custom CUDA extension
            fused_weight_gradient_mlp_cuda module. To use
            gradient_accumulation_fusion you must install APEX with
            --cpp_ext and --cuda_ext. For example: "pip install
            --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
            " Note that the extension requires CUDA>=11. Otherwise, you
            must turn off gradient accumulation fusion."

        allreduce_dgrad (bool required): Do the allreduce of input gradients.
            The allreduce is done asynchronously with the computation of weight
            gradients. If sequence_parallel is True, this must be
            False, as no all reduce is performed.

        sequence_parallel (bool required): Indicates that sequence
            parallelism is used and thus in the forward pass the input is
            all gathered, and the backward pass the input gradients are
            reduce scattered.

        tp_group (torch.distributed.ProcessGroup required): The process group to use for tensor
                                                   parallel operations.

        grad_output_buffer (List[torch.Tensor] optional): Buffer used to save
            output gradients when embedding table wgrad compute is deferred.
            Defaults to None.

        wgrad_deferral_limit (int optional): Limit on the number of
            micro-batches for which embedding weight gradient GEMM should be
            deferred. Disable by setting this to 0. Defaults to 0.
    """

    tp_group = get_tensor_model_parallel_group_if_none(tp_group)

    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:
        if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1":
            if sequence_parallel:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

            if allreduce_dgrad:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)


linear_with_grad_accumulation_and_async_allreduce.warned = False


def _expert_grads_need_own_dp_domain(config) -> bool:
    """Whether expert parameters must be reduced over the expert-data-parallel group.

    mcore normally decides this from ``expert_model_parallel_size > 1`` alone, which
    is correct only when the expert tensor-parallel size equals the dense one: the
    expert parameter is then sharded exactly like a dense parameter and its
    data-parallel domain coincides with ``dp_cp``.

    With ``expert_tensor_parallel_size < tensor_model_parallel_size`` (the accuracy
    -compatible topology uses ETP=1 with TP=2) every rank in the tensor-parallel
    group holds a FULL copy of the expert weight while consuming only its own
    sequence-parallel shard of the tokens. Those partial weight gradients live in a
    larger data-parallel domain (``expt_dp`` = TP/ETP times ``dp_cp``) and must be
    summed. Leaving ``allreduce=True`` puts them in the dense bucket, which is
    reduced over ``dp_cp`` — size 1 in this topology — so the reduction silently
    never happens and every expert gradient stays a per-rank partial sum.
    """
    if config.expert_model_parallel_size > 1:
        return True
    etp = getattr(config, 'expert_tensor_parallel_size', None)
    if etp is None:
        return False
    return etp != config.tensor_model_parallel_size


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size:
            first dimension of matrix A.
        output_size:
            second dimension of matrix A.
        bias:
            If true, add bias
        gather_output:
            If true, call all-gather on output and make Y available to all GPUs,
            otherwise, every GPU will have its output which is Y_i = XA_i
        init_method:
            method to initialize weights. Note that bias is always set to zero.
        stride:
            For the strided linear layers.
        keep_master_weight_for_test:
            This was added for testing and should be set to False. It
            returns the master weights used for initialization.
        skip_bias_add:
            If True, do not add the bias term, instead return it to be added by the
            caller. This enables performance optimizations where bias can be fused with other
            elementwise operations.
        skip_weight_param_allocation:
            If True, weight parameter is not allocated and must be passed
            as a keyword argument `weight` during the forward pass. Note that this does not
            affect bias, which will be allocated if bias is True. Defaults to False.
        embedding_activation_buffer:
            This buffer holds the input activations of the final embedding
            linear layer on the last pipeline stage when defer_embedding_wgrad_compute is enabled.
        grad_output_buffer:
            This buffer holds the gradient outputs of the final embedding linear
            layer on the last pipeline stage when defer_embedding_wgrad_compute is enabled.
        is_expert:
            If True, the layer is treated as an MoE expert layer.
        config:
            ModelParallelConfig object
        tp_comm_buffer_name:
            Communication buffer name is not used in non-Transformer-Engine modules.
        disable_grad_reduce:
            If True, reduction of output gradients across tensor-parallel ranks
            will be disabled. Defaults to False. This feature is used by Lora Adapter in Nemo to
            delay and fuse reduction along with other gradients for performance optimization.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,  # Not used
        disable_grad_reduce: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        name: str | None = None,
    ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.embedding_activation_buffer = embedding_activation_buffer
        self.grad_output_buffer = grad_output_buffer
        self.config = config
        self.disable_grad_reduce = disable_grad_reduce
        self.tp_group = tp_group

        self.tp_group = get_tensor_model_parallel_group_if_none(
            self.tp_group, is_expert=self.is_expert
        )
        world_size = get_pg_size(self.tp_group)
        rank = get_pg_rank(self.tp_group)
        self.explicit_expert_comm = self.is_expert and (world_size > 1 or self.expert_parallel)
        self.output_size_per_partition = divide(output_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if not skip_weight_param_allocation:
            if config.use_cpu_initialization:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition, self.input_size, dtype=config.params_dtype
                    )
                )
                if config.perform_initialization:
                    self.master_weight = _initialize_affine_weight_cpu(
                        self.weight,
                        self.output_size,
                        self.input_size,
                        self.output_size_per_partition,
                        0,
                        init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                        rank=rank,
                        world_size=world_size,
                    )
                else:
                    set_tensor_model_parallel_attributes(
                        tensor=self.weight, is_parallel=True, dim=0, stride=stride
                    )
            else:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        self.input_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
                if config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        is_expert=self.is_expert,
                    )
                else:
                    set_tensor_model_parallel_attributes(
                        tensor=self.weight, is_parallel=True, dim=0, stride=stride
                    )

            setattr(
                self.weight,
                "allreduce",
                not (self.is_expert and _expert_grads_need_own_dp_domain(config)),
            )
        else:
            self.weight = None

        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            if config.perform_initialization:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
            setattr(
                self.bias,
                "allreduce",
                not (self.is_expert and _expert_grads_need_own_dp_domain(config)),
            )
        else:
            self.register_parameter("bias", None)

        self.use_inference_optimized_all_gather = (
            getattr(config, 'transformer_impl', None) == 'inference_optimized'
        )

        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and world_size <= 1:
            warnings.warn(
                "`sequence_parallel` is set to `True`, but tensor model parallel size "
                f"is {world_size}. Disabling sequence parallel."
            )
            self.sequence_parallel = False

        self.allreduce_dgrad = (
            world_size > 1 and not self.sequence_parallel and not self.disable_grad_reduce
        )

        if config.gradient_accumulation_fusion and not _grad_accum_fusion_available:
            raise RuntimeError(
                "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                "module is not found. To use gradient_accumulation_fusion you must "
                "install APEX with --cpp_ext and --cuda_ext. For example: "
                'pip install --global-option="--cpp_ext" --global-option="--cuda_ext ." '
                "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                "gradient accumulation fusion."
            )
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion

        if self.allreduce_dgrad and self.sequence_parallel:
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f"{prefix}_extra_state"
            )
        )

    def _forward_impl(self, input, weight, *args, **kwargs):
        if not weight.requires_grad:
            return linear_with_frozen_weight(input, weight, *args, **kwargs)
        else:
            return linear_with_grad_accumulation_and_async_allreduce(input, weight, *args, **kwargs)

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of ColumnParallelLinear

        Args:
            input_:
                3D tensor whose order of dimension is [sequence, batch, hidden]
            weight (optional):
                weight tensor to use, compulsory when skip_weight_param_allocation is True.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `gather_output` arg in the constructor will be used.

        Returns:
            - output
            - bias

        """
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        bias = self.bias if not self.skip_bias_add else None

        if (
            self.allreduce_dgrad
            or self.sequence_parallel
            or self.explicit_expert_comm
            or self.disable_grad_reduce
        ):
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)

        if self.config.defer_embedding_wgrad_compute:
            if (
                self.config.wgrad_deferral_limit == 0
                or len(self.embedding_activation_buffer) < self.config.wgrad_deferral_limit
            ):
                self.embedding_activation_buffer.append(input_parallel)

        # Matrix multiply.
        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context is True:
                if not HAVE_TE:
                    assert (
                        self.config.cpu_offloading is False
                    ), "CPU Offloading cannot be enabled while TE is not present"
                else:
                    input_parallel.activation_offloading = self.config.cpu_offloading_activations

        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=weight,
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            allreduce_dgrad=allreduce_dgrad,
            sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
            grad_output_buffer=(
                self.grad_output_buffer if self.config.defer_embedding_wgrad_compute else None
            ),
            wgrad_deferral_limit=(
                self.config.wgrad_deferral_limit
                if self.config.defer_embedding_wgrad_compute
                else None
            ),
            tp_group=self.tp_group,
        )

        gather_output = self.gather_output
        # Use the runtime gather output if it's set explicitly.
        if runtime_gather_output is not None:
            gather_output = runtime_gather_output

        if gather_output:
            # All-gather across the partitions.
            if self.use_inference_optimized_all_gather and not self.training:
                # Deferred to avoid circular import: inference_layers → TE → layers.
                from .inference_layers import inference_all_gather_from_tensor_model_parallel_region

                output = inference_all_gather_from_tensor_model_parallel_region(
                    output_parallel, self.tp_group, self.config
                )
            else:
                output = gather_from_tensor_model_parallel_region(
                    output_parallel, group=self.tp_group
                )
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def backward_dw(self) -> None:
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled.

        Not supported - does nothing.
        """
        pass

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 0, "bias": 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata['dp_cp_group'],
        )

    def set_extra_state(self, state: Any):
        """Extra state is ignored"""

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None

    @override
    def extra_repr(self) -> str:
        """Extra context to add to the module's string representation."""
        tp = self.output_size // self.output_size_per_partition
        use_bias = self.bias is not None
        return (
            f"in_features={self.input_size}, "
            f"out_features={self.output_size}, "
            f"bias={use_bias}, "
            f"TP={tp}"
        )


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X
    along its second dimension. A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]

    Args:
        input_size:
            first dimension of matrix A.
        output_size:
            second dimension of matrix A.
        bias:
            If true, add bias. Note that bias is not parallelized.
        input_is_parallel:
            If true, we assume that the input is already split across the GPUs
            and we do not split again.
        init_method:
            method to initialize weights. Note that bias is always set to zero.
        stride:
            For the strided linear layers.
        keep_master_weight_for_test:
            This was added for testing and should be set to False. It returns the master weights
            used for initialization.
        skip_bias_add:
            If True, do not add the bias term, instead return it to be added by the
            caller. This enables performance optimizations where bias can be fused with other
            elementwise operations.
        is_expert:
            If True, the layer is treated as an MoE expert layer
        tp_comm_buffer_name:
            Communication buffer name. Not used in non-Transformer-Engine modules.
        config:
            ModelParallelConfig object

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,  # Not used
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        name: str | None = None,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.config = config
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.sequence_parallel = config.sequence_parallel
        self.tp_group = tp_group

        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

        # Divide the weight matrix along the last dimension.
        self.tp_group = get_tensor_model_parallel_group_if_none(
            self.tp_group, is_expert=self.is_expert
        )

        world_size = get_pg_size(self.tp_group)
        rank = get_pg_rank(self.tp_group)
        self.explicit_expert_comm = self.is_expert and (world_size > 1 or self.expert_parallel)

        self.input_size_per_partition = divide(input_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size, self.input_size_per_partition, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.input_size_per_partition,
                    1,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    params_dtype=config.params_dtype,
                    rank=rank,
                    world_size=world_size,
                )
            else:
                set_tensor_model_parallel_attributes(
                    tensor=self.weight, is_parallel=True, dim=1, stride=stride
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=1,
                    stride=stride,
                    is_expert=self.is_expert,
                )
            else:
                set_tensor_model_parallel_attributes(
                    tensor=self.weight, is_parallel=True, dim=1, stride=stride
                )
        setattr(
                self.weight,
                "allreduce",
                not (self.is_expert and _expert_grads_need_own_dp_domain(config)),
            )

        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )

            if config.perform_initialization:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
            setattr(
                self.bias,
                "allreduce",
                not (self.is_expert and _expert_grads_need_own_dp_domain(config)),
            )
            setattr(self.bias, "sequence_parallel", self.sequence_parallel)
        else:
            self.register_parameter("bias", None)

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f"{prefix}_extra_state"
            )
        )

    def _forward_impl(self, input, weight, *args, **kwargs):
        if not weight.requires_grad:
            return linear_with_frozen_weight(input, weight, *args, **kwargs)
        else:
            return linear_with_grad_accumulation_and_async_allreduce(input, weight, *args, **kwargs)

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """

        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_, group=self.tp_group)
        # Matrix multiply.
        allreduce_dgrad = False

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context is True:
                if not HAVE_TE:
                    assert (
                        self.config.cpu_offloading is False
                    ), "CPU Offloading cannot be enabled while TE is not present"
                else:
                    input_parallel.activation_offloading = self.config.cpu_offloading_activations

        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            allreduce_dgrad=allreduce_dgrad,
            sequence_parallel=False,
            tp_group=None,
            grad_output_buffer=None,
        )

        # All-reduce across all the partitions.
        if self.explicit_expert_comm:
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(
                output_parallel, group=self.tp_group
            )
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

    def backward_dw(self) -> None:
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled.

        Not supported - does nothing.
        """
        pass

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 1, bias not sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 1},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata['dp_cp_group'],
        )

    def set_extra_state(self, state: Any):
        """Extra state is ignored"""

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None

    @override
    def extra_repr(self) -> str:
        """Extra context to add to the module's string representation."""
        tp = self.input_size // self.input_size_per_partition
        use_bias = self.bias is not None
        return (
            f"in_features={self.input_size}, "
            f"out_features={self.output_size}, "
            f"bias={use_bias}, "
            f"TP={tp}"
        )
