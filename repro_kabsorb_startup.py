"""PYTHONSTARTUP: dump live K-absorb einsum operands. Worktree-only.

Live GLM-5.2 DSA uses mcore_bridge AbsorbedMLA; nested tofile on megatron.core
never runs. Wrap torch.einsum after import and stamp DSA layer/mtp on
mcore_bridge.AbsorbedMLASelfAttention.forward.
"""
from __future__ import annotations

import os

_DUMP = os.environ.get("MODEL_REPRO_LIVE_XY_DUMP_DIR")
if not _DUMP:
    raise SystemExit  # noqa: not used as CLI

try:
    os.makedirs(_DUMP, exist_ok=True)
    open(os.path.join(_DUMP, "torch_startup.txt"), "a").write(f"pid={os.getpid()}\n")
except OSError:
    pass


def _patch_einsum() -> None:
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
        if len(ops) != 2:
            return out
        q_no_pe, k_up_weight = ops
        if getattr(k_up_weight, "ndim", 0) != 3:
            return out
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


def _patch_mcore_bridge() -> None:
    try:
        from mcore_bridge.model.modules.absorbed_mla import AbsorbedMLASelfAttention
    except Exception:
        return
    if getattr(AbsorbedMLASelfAttention, "_repro_layer_env_patched", False):
        return
    _fwd = AbsorbedMLASelfAttention.forward

    def _fwd_wrapped(self, *args, **kwargs):
        os.environ["MODEL_REPRO_DSA_LAYER"] = str(int(self.layer_number))
        os.environ["MODEL_REPRO_DSA_MTP"] = "1" if int(self.layer_number) == 1 else "0"
        return _fwd(self, *args, **kwargs)

    AbsorbedMLASelfAttention.forward = _fwd_wrapped
    AbsorbedMLASelfAttention._repro_layer_env_patched = True


try:
    _patch_einsum()
except Exception:
    pass

import builtins as _builtins

_real_import = _builtins.__import__


def _repro_import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _real_import(name, globals, locals, fromlist, level)
    try:
        if name == "torch" or (isinstance(name, str) and name.startswith("torch.")):
            _patch_einsum()
        if isinstance(name, str) and name.startswith("mcore_bridge"):
            _patch_mcore_bridge()
    except Exception:
        pass
    return mod


_builtins.__import__ = _repro_import
