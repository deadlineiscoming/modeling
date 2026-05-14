# CP Kind CLI Parameter and Communication Mask Correction

**Date**: 2025-05-14
**Status**: Approved

## Summary

Add `--cp-kind` CLI parameter to explicitly specify context parallel strategy, with model-specific constraints. Correct communication node `mask` annotations for all four CP strategies based on theoretical overlap characteristics.

## Background

### Current State

- CLI has `--cp` for CP degree but no `--cp-kind` parameter
- `cp_kind` is auto-selected via `TrainingConfig.resolve_cp_kind()`:
  - DSV4/DV3.2 → `"compressed"`
  - Other models → `"ulysses"`
- Communication nodes have `mask` annotations, but 5 out of 8 are incorrect

### Problem

1. Users cannot override auto-selected `cp_kind` from CLI
2. `mask` annotations don't reflect actual overlap characteristics:
   - A2A nodes marked `mask=True` but should be `False` (blocking)
   - Compressed Stage1 P2P marked `mask=False` but should be `True` (overlapable)

## Design

### Part 1: CLI Parameter

#### Parameter Definition

```python
parser.add_argument(
    "--cp-kind",
    default="none",
    choices=["none", "ulysses", "ring", "hybrid", "compressed"],
    metavar="KIND",
    help="Context parallel strategy (default: auto-select by model type). "
         "DSV4/DV3.2: 'compressed' only. "
         "Other models: 'ulysses' (default), 'ring', or 'hybrid'.",
)
```

#### Constraint Logic

In `cli.py` main() and `_run_training_modelling()`:

```python
def _validate_cp_kind(model_id: str, cp_kind: str, cp: int) -> str:
    """Validate and resolve cp_kind based on model type and CLI input.
    
    Returns resolved cp_kind string.
    Raises ValueError on constraint violation.
    """
    if cp <= 1:
        return "none"
    
    model_slug = _make_model_slug(model_id).lower()
    is_dsv4 = any(x in model_slug for x in ["deepseek_v4", "deepseek_v3_2", "dsv4", "dv32"])
    
    if is_dsv4:
        if cp_kind not in ("none", "compressed"):
            raise ValueError(
                f"Model '{model_id}' only supports cp_kind='compressed'. "
                f"Got '{cp_kind}'. Use --cp-kind compressed or remove flag."
            )
        return "compressed" if cp_kind == "none" else cp_kind
    
    if cp_kind == "compressed":
        raise ValueError(
            f"cp_kind='compressed' is reserved for DeepSeek-V4/V3.2. "
            f"Model '{model_id}' should use 'ulysses', 'ring', or 'hybrid'."
        )
    
    return "ulysses" if cp_kind == "none" else cp_kind
```

#### Integration Points

1. `cli.py:main()` — validate after model_id resolution
2. `TransformContext` — pass resolved `cp_kind` via `ParallelConfig` or `TrainingConfig`
3. `config_loader.py:load_specs()` — already supports YAML `cp_kind` field

### Part 2: Communication Mask Correction

#### Theory: Which Operations Can Overlap?

| Operation Type | Blocking? | Maskable? | Reason |
|---------------|-----------|-----------|--------|
| **P2P send_recv** | No | **Yes** | Asynchronous, overlaps with compute tiles |
| **A2A all_to_all** | Yes | **No** | Synchronous barrier, all ranks must participate |
| **AllGather** | Yes | **No** | Collective operation, blocking |
| **ReduceScatter** | Yes | **No** | Collective operation, blocking |

#### Correction Table

| File:Line | Strategy | Role | Current | Correct | Reason |
|-----------|----------|------|---------|---------|--------|
| `comm_inserter.py:341` | Ulysses | pre-A2A | `True` | `False` | A2A blocking |
| `comm_inserter.py:364` | Ulysses | post-A2A | `True` | `False` | A2A blocking |
| `comm_inserter.py:407` | Ring | P2P | `True` | `True` ✓ | Correct |
| `comm_inserter.py:446` | Hybrid | A2A-pre | `True` | `False` | A2A blocking |
| `comm_inserter.py:472` | Hybrid | P2P | `True` | `True` ✓ | Correct |
| `comm_inserter.py:495` | Hybrid | A2A-post | `True` | `False` | A2A blocking |
| `comm_inserter.py:542` | Compressed | Stage1 P2P | `False` | `True` | P2P overlapable |
| `comm_inserter.py:570` | Compressed | Stage2 AG | `True` | `False` | AG blocking |

#### Code Changes

Replace `mask=True` → `mask=False` for A2A nodes, `mask=False` → `mask=True` for Compressed Stage1 P2P.

### Part 3: Verification

Run user command and inspect communication node annotations:

```bash
python -m python.zrt --model-id hf_models/deepseek_v4 --layers 3 --train \
    --hw nvidia_h100_sxm --seq-len 20000 --cp 2 --cp-kind compressed
```

Expected output:
- Communication nodes inserted with correct `mask` values
- Compressed CP: Stage1 P2P `mask=True`, Stage2 AG `mask=False`

## Implementation Plan

1. Add `--cp-kind` argument in `cli.py`
2. Add `_validate_cp_kind()` helper in `cli.py`
3. Integrate resolved cp_kind into `TrainingConfig` flow
4. Correct 5 mask annotations in `comm_inserter.py`
5. Update test cases for mask validation
6. Run user command to verify

## Testing

- Unit test for `_validate_cp_kind()` constraints
- Test mask annotations in `test_context_parallel.py`
- Integration test with `--cp-kind compressed` on DSV4
- Error test for invalid combinations (e.g., `--cp-kind ring` on DSV4)

## Files Changed

- `python/zrt/cli.py` — add parameter + validation
- `python/zrt/transform/parallel/comm_inserter.py` — correct mask annotations
- `tests/test_cp_shape_split.py` or `tests/training/test_context_parallel.py` — add mask tests

## Success Criteria

1. CLI accepts `--cp-kind` and validates model constraints
2. All communication nodes have correct `mask` annotations
3. User command runs without error with `--cp-kind compressed`
4. Error message shown for invalid combinations