"""Model compatibility patches for FakeTensorMode tracing.

Contains the minimal set of patches needed to run arbitrary HuggingFace
causal LMs through FakeTensorMode without allocating real memory.

Patch inventory
---------------
apply_compat_patches()
    Adds deprecated transformers attributes expected by older model code
    (is_torch_fx_available, is_torch_greater_or_equal_than_1_13).

patch_moe_for_fake(model)
    Replaces MoE module forwards with a simplified version that avoids
    .cpu().numpy() and torch.bincount() calls on routing indices, which
    crash on fake tensors.  Only applied when the standard heuristic
    identifies a module as MoE (has nn.ModuleList experts, not already
    patched).

patch_indexer_for_fake(model)
    Patches DeepSeek-V3.2 Indexer modules whose original forward contains
    a 3-D tensor .transpose(2,3) that is invalid under FakeTensorMode.
    The original modeling files from HF are kept untouched; this patch
    supplies a corrected forward at runtime only.

What is intentionally NOT patched
----------------------------------
* Autocast / dtype casting — FakeTensorMode handles these transparently.
* Meta-device specific hacks — superseded by FakeTensorMode.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Transformers compatibility ────────────────────────────────────────────────

def apply_compat_patches() -> None:
    """Apply all compatibility patches needed before model loading.

    Call order matters: version shims must be injected *before* any model file
    is imported, so that ``from transformers.xxx import yyy`` at module level
    finds the stub symbols even when the installed transformers version removed them.
    """
    # 1. Version shims (missing symbols injected into transformers sub-modules)
    from python.zrt.graph.compat import apply_version_shims
    apply_version_shims()

    # 2. Legacy attrs still expected by some older model files
    try:
        import transformers.utils.import_utils as _iu
        if not hasattr(_iu, "is_torch_fx_available"):
            _iu.is_torch_fx_available = lambda: True
    except ImportError:
        pass
    try:
        import transformers.pytorch_utils as _pu
        if not hasattr(_pu, "is_torch_greater_or_equal_than_1_13"):
            _pu.is_torch_greater_or_equal_than_1_13 = True
    except ImportError:
        pass


# ── MoE patch ─────────────────────────────────────────────────────────────────
# Many MoE implementations call .cpu().numpy() or torch.bincount() on routing
# indices, which crash on fake tensors.  This simplified forward exercises the
# gate + one expert + shared experts — enough to capture the full op pattern.

def is_moe_module(module: nn.Module) -> bool:
    """True if module looks like a MoE layer that needs patching."""
    experts = getattr(module, "experts", None)
    return (
        isinstance(experts, nn.ModuleList)
        and any(e is not None for e in experts)
        and not getattr(module, "_fake_patched", False)
    )


def _returns_router_tuple(mod: nn.Module) -> bool:
    """True if this MoE forward returns (hidden, router_logits) tuple."""
    try:
        src = inspect.getsource(type(mod).forward)
        if any(pat in src for pat in ("router_logits", "aux_loss",
                                      "return hidden_states,")):
            return True
    except Exception:
        pass
    return hasattr(mod, "router") and not hasattr(mod, "gate")


def _make_fake_moe_forward(mod: nn.Module):
    _tuple_return = _returns_router_tuple(mod)

    def _forward(hidden_states: torch.Tensor, *args: Any, **kwargs: Any):
        try:
            result = _impl(hidden_states)
            return (result, None) if _tuple_return else result
        except Exception as exc:
            logger.debug("Fake MoE forward error (%s) — returning identity.", exc)
            return (hidden_states, None) if _tuple_return else hidden_states

    def _impl(hidden_states: torch.Tensor) -> torch.Tensor:
        orig = hidden_states
        bs, seq, h = orig.shape
        flat = orig.reshape(bs * seq, h)

        gate_weight: Optional[torch.Tensor] = None
        gate = getattr(mod, "gate", None)
        if gate is not None and callable(gate):
            try:
                gate_out = gate(orig)
                if isinstance(gate_out, (tuple, list)):
                    gate_weight = gate_out[1]
                else:
                    gate_weight = torch.softmax(gate_out.float(), dim=-1)[:, :1]
            except Exception:
                try:
                    gate_out = gate(flat)
                    if isinstance(gate_out, (tuple, list)):
                        gate_weight = gate_out[1]
                    else:
                        gate_weight = torch.softmax(gate_out.float(), dim=-1)[:, :1]
                except Exception as exc:
                    logger.debug("Gate forward failed (%s).", exc)

        first_expert = next((e for e in mod.experts if e is not None), None)
        if first_expert is None:
            return orig

        try:
            expert_out = first_expert(flat)
        except Exception:
            expert_out = first_expert(orig).reshape(bs * seq, -1)

        y = (expert_out * gate_weight[:, :1]
             if gate_weight is not None else expert_out)
        try:
            y = y.reshape(bs, seq, -1)
        except Exception:
            y = orig

        for attr in ("shared_experts", "shared_expert"):
            shared = getattr(mod, attr, None)
            if shared is not None and callable(shared):
                try:
                    y = y + shared(orig)
                except Exception as exc:
                    logger.debug("Shared expert failed (%s).", exc)
                break

        return y

    return _forward


def patch_moe_for_fake(model: nn.Module) -> None:
    """Replace MoE forwards with a fake-tensor-safe simplified version."""
    patched = 0
    for _, module in model.named_modules():
        if not is_moe_module(module):
            continue
        module._fake_patched = True
        module.forward = _make_fake_moe_forward(module)
        patched += 1
    if patched:
        logger.info("Applied fake-tensor MoE patch to %d module(s).", patched)


# ── DeepSeek-V3.2 Indexer patch ──────────────────────────────────────────────
# The Indexer forward in the original HF modeling file (kept unmodified) uses
# k_nope.transpose(1, 2).transpose(2, 3) on a 3-D tensor, which is invalid.
# We supply a corrected forward at runtime without touching the model files.

def _make_indexer_forward_fake(IndexerClass: type) -> None:
    """Replace Indexer.forward with a FakeTensorMode-compatible version."""

    def _forward(self, x, qr, position_ids=None, attention_mask=None):
        import torch.nn.functional as F
        bsz, seqlen, _ = x.size()
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.index_n_heads, self.index_head_dim)
        # Split into RoPE and NoPE parts (mirrors the original design intent)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.index_head_dim - self.rope_head_dim], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.index_head_dim - self.rope_head_dim], dim=-1)
        # q_nope: (bsz, seqlen, n_heads, nope_dim) -> (bsz, n_heads, seqlen, nope_dim)
        q_nope = q_nope.transpose(1, 2)
        # k_nope: (bsz, seqlen, nope_dim) -> broadcast across heads
        k_nope = k_nope.unsqueeze(1).expand(-1, self.index_n_heads, -1, -1)
        scores = torch.matmul(q_nope, k_nope.transpose(2, 3))
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        topk_indices = scores.topk(min(self.index_topk, seqlen), dim=-1)[1]
        return topk_indices

    IndexerClass.forward = _forward


def patch_indexer_for_fake(model: nn.Module) -> None:
    """Patch DeepSeek-V3.2 Indexer modules for FakeTensorMode compatibility.

    The original HF modeling files are not modified.  The fix is applied
    in-memory at runtime: any module whose class name contains 'Indexer'
    and which has the expected MLA attributes (wq_b, index_n_heads) gets
    a corrected forward method.
    """
    patched_classes: set = set()
    for _, module in model.named_modules():
        cls = type(module)
        if cls in patched_classes:
            continue
        if "Indexer" in cls.__name__ and hasattr(module, "wq_b") and hasattr(module, "index_n_heads"):
            _make_indexer_forward_fake(cls)
            patched_classes.add(cls)
            logger.debug("Patched Indexer class: %s", cls.__name__)
    if patched_classes:
        logger.info("Applied Indexer patch to: %s",
                    ", ".join(c.__name__ for c in patched_classes))


# Backward-compatible aliases
patch_moe_for_meta = patch_moe_for_fake
_is_moe_module = is_moe_module
