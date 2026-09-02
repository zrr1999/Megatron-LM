"""Redirect megatron.core editable finder from repos/ to this Explore worktree."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_WT = Path(__file__).resolve().parent
if str(_WT) not in sys.path:
    sys.path.insert(0, str(_WT))

_CORE = _WT / "megatron" / "core"
_TRAINING = _WT / "megatron" / "training"

_dump = os.environ.get("MODEL_REPRO_LIVE_XY_DUMP_DIR")
if _dump:
    try:
        os.makedirs(_dump, exist_ok=True)
        open(os.path.join(_dump, "torch_sitecustomize.txt"), "a").write(
            f"pid={os.getpid()} wt={_WT}\n"
        )
    except OSError:
        pass

try:
    import __editable___megatron_core_0_19_0_07a79ee2b_finder as _finder
except Exception:
    _finder = None

if _finder is not None and _CORE.is_dir():
    mapping = getattr(_finder, "MAPPING", None)
    if isinstance(mapping, dict):
        mapping["megatron.core"] = str(_CORE)
        if _TRAINING.is_dir():
            mapping["megatron.training"] = str(_TRAINING)
    namespaces = getattr(_finder, "NAMESPACES", None)
    if isinstance(namespaces, dict):
        for key, paths in list(namespaces.items()):
            if not isinstance(paths, list):
                continue
            rewritten = []
            for path in paths:
                if isinstance(path, str) and "/repos/Megatron-LM/" in path:
                    rewritten.append(path.replace("/repos/Megatron-LM/", f"{_WT}/"))
                else:
                    rewritten.append(path)
            namespaces[key] = rewritten

# Live GLM-5.2 DSA uses mcore_bridge AbsorbedMLA, not megatron.core's class.
# Nested tofile on the megatron class never runs. Wrap torch.einsum here
# (sitecustomize loads for every worker; PYTHONSTARTUP does not).
if _dump:
    import builtins as _builtins

    def _repro_patch_torch_einsum() -> None:
        import torch

        orig = torch.einsum
        if getattr(orig, "_repro_kabsorb_wrapped", False):
            return

        def _wrapped(equation, *operands, **kwargs):
            out = orig(equation, *operands, **kwargs)
            dump_dir = os.environ.get("MODEL_REPRO_LIVE_XY_DUMP_DIR")
            if not dump_dir or not isinstance(equation, str):
                return out
            if equation.replace(" ", "") != "...nd,ndk->...nk":
                return out
            ops = operands
            if len(ops) == 1 and isinstance(ops[0], (tuple, list)):
                ops = ops[0]
            if len(ops) != 2 or getattr(ops[1], "ndim", 0) != 3:
                return out
            q_no_pe, k_up_weight = ops
            try:
                import torch.distributed as _td

                rank = _td.get_rank() if _td.is_initialized() else 0
                os.makedirs(dump_dir, exist_ok=True)
                lay = os.environ.get("MODEL_REPRO_DSA_LAYER", "-1")
                mtp = os.environ.get("MODEL_REPRO_DSA_MTP", "0")
                q_no_pe.detach().float().cpu().numpy().tofile(
                    os.path.join(dump_dir, f"torch_qnope_l{lay}_mtp{mtp}_r{rank}.f32.bin")
                )
                k_up_weight.detach().float().cpu().numpy().tofile(
                    os.path.join(dump_dir, f"torch_kup_l{lay}_mtp{mtp}_r{rank}.f32.bin")
                )
                with open(os.path.join(dump_dir, "torch_kabsorb_dump.txt"), "a") as fh:
                    fh.write(
                        f"pid={os.getpid()} lay={lay} mtp={mtp} r={rank} "
                        f"q={tuple(q_no_pe.shape)} w={tuple(k_up_weight.shape)}\n"
                    )

                def _dump_dqabs(g, lay=lay, mtp=mtp, r=rank, d=dump_dir):
                    if g is None:
                        return g
                    g.detach().float().cpu().numpy().tofile(
                        os.path.join(d, f"torch_dqabs_l{lay}_mtp{mtp}_r{r}.f32.bin")
                    )
                    return g

                if getattr(out, "requires_grad", False):
                    out.register_hook(_dump_dqabs)
            except OSError:
                pass
            return out

        _wrapped._repro_kabsorb_wrapped = True
        torch.einsum = _wrapped

    def _repro_patch_mcore_bridge_absorbed() -> None:
        try:
            from mcore_bridge.model.modules import absorbed_mla as _mcb
        except Exception:
            return
        cls = getattr(_mcb, "AbsorbedMLASelfAttention", None)
        if cls is None or getattr(cls, "_repro_layer_env_patched", False):
            return
        _fwd = cls.forward

        def _fwd_wrapped(self, *args, **kwargs):
            os.environ["MODEL_REPRO_DSA_LAYER"] = str(int(self.layer_number))
            os.environ["MODEL_REPRO_DSA_MTP"] = (
                "1" if int(self.layer_number) == 1 else "0"
            )
            kvln = getattr(self, "kv_layernorm", None)
            dump_dir = os.environ.get("MODEL_REPRO_LIVE_XY_DUMP_DIR")
            if dump_dir and kvln is not None and not getattr(kvln, "_repro_kvln_hooked", False):
                _kvln_fwd = kvln.forward

                def _kvln_wrapped(x, *a, _mod=kvln, _orig=_kvln_fwd, **k):
                    out = _orig(x, *a, **k)
                    d = os.environ.get("MODEL_REPRO_LIVE_XY_DUMP_DIR")
                    if not d:
                        return out
                    try:
                        import torch.distributed as _td

                        rank = _td.get_rank() if _td.is_initialized() else 0
                        os.makedirs(d, exist_ok=True)
                        lay = os.environ.get("MODEL_REPRO_DSA_LAYER", "-1")
                        mtp = os.environ.get("MODEL_REPRO_DSA_MTP", "0")
                        x.detach().float().cpu().numpy().tofile(
                            os.path.join(d, f"torch_kvlnx_l{lay}_mtp{mtp}_r{rank}.f32.bin")
                        )
                        _mod.weight.detach().float().cpu().numpy().tofile(
                            os.path.join(d, f"torch_kvlnw_l{lay}_mtp{mtp}_r{rank}.f32.bin")
                        )
                        out.detach().float().cpu().numpy().tofile(
                            os.path.join(d, f"torch_kvlny_l{lay}_mtp{mtp}_r{rank}.f32.bin")
                        )

                        def _dump_dy(g, lay=lay, mtp=mtp, r=rank, dd=d):
                            if g is None:
                                return g
                            g.detach().float().cpu().numpy().tofile(
                                os.path.join(dd, f"torch_kvlndy_l{lay}_mtp{mtp}_r{r}.f32.bin")
                            )
                            return g

                        if getattr(out, "requires_grad", False):
                            out.register_hook(_dump_dy)
                    except OSError:
                        pass
                    return out

                kvln.forward = _kvln_wrapped
                kvln._repro_kvln_hooked = True
            return _fwd(self, *args, **kwargs)

        cls.forward = _fwd_wrapped
        cls._repro_layer_env_patched = True

        # Wrap the gather used inside get_query_key_value_tensors at import
        # time so the first MTP call is already hooked.
        orig_gath = getattr(_mcb, "gather_from_sequence_parallel_region", None)
        if orig_gath is not None and not getattr(orig_gath, "_repro_gath_wrapped", False):
            def _gath_wrapped(t, *a, **k):
                out = orig_gath(t, *a, **k)
                d = os.environ.get("MODEL_REPRO_LIVE_XY_DUMP_DIR")
                if not d or getattr(t, "shape", None) is None:
                    return out
                try:
                    last = int(t.shape[-1])
                except Exception:
                    return out
                if last != 512:
                    return out
                try:
                    import torch.distributed as _td

                    rank = _td.get_rank() if _td.is_initialized() else 0
                    os.makedirs(d, exist_ok=True)
                    ntok = int(out.numel() // 512)
                    if ntok != 60:
                        return out
                    # PP stage-1 (ranks 2/3) first 60-token gather is MTP;
                    # later gathers are decoder. Count per process.
                    ncall = int(getattr(_gath_wrapped, "_repro_full_calls", 0))
                    _gath_wrapped._repro_full_calls = ncall + 1
                    tag = "mtp" if ncall == 0 else f"dec{ncall}"
                    out.detach().float().cpu().numpy().tofile(
                        os.path.join(
                            d, f"torch_kvgath_full_{tag}_r{rank}.f32.bin"
                        )
                    )

                    def _dump_gdy(g, tag=tag, r=rank, dd=d):
                        if g is None:
                            return g
                        g.detach().float().cpu().numpy().tofile(
                            os.path.join(
                                dd, f"torch_kvgathdy_full_{tag}_r{r}.f32.bin"
                            )
                        )
                        return g

                    if getattr(out, "requires_grad", False):
                        out.register_hook(_dump_gdy)
                except OSError:
                    pass
                return out

            _gath_wrapped._repro_gath_wrapped = True
            _mcb.gather_from_sequence_parallel_region = _gath_wrapped

    try:
        _repro_patch_torch_einsum()
    except Exception:
        pass

    _real_import = _builtins.__import__

    def _repro_import(name, globals=None, locals=None, fromlist=(), level=0):
        mod = _real_import(name, globals, locals, fromlist, level)
        try:
            if name == "torch" or (isinstance(name, str) and name.startswith("torch")):
                _repro_patch_torch_einsum()
            if isinstance(name, str) and name.startswith("mcore_bridge"):
                _repro_patch_mcore_bridge_absorbed()
        except Exception:
            pass
        return mod

    _builtins.__import__ = _repro_import
