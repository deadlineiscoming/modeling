# Training modelling reuses hardware registry

## Context

The training modelling has **two CLI surfaces** that redeclare hardware numbers
instead of loading from the shared registry at `python/zrt/hardware/`:

1. **`model-training` subcommand** — `python/zrt/training/cli.py:79-93` builds a
   `HardwareSpec` ad-hoc from 4 raw flags (`--gpu-name`, `--gpu-bf16-tflops`,
   `--gpu-hbm-gbps`, `--gpu-hbm-gb`) and hardcodes interconnect values
   (NVLink 900 GB/s, IB 400 GB/s, 8 intra-devs, 64 inter-devs). Interconnect
   cannot be changed without editing source.

2. **`estimate` subcommand** — `python/zrt/training/io/config_loader.py:56-81`
   parses a `system:` YAML block into a *parallel* dataclass hierarchy
   (`SystemSpec` / `GPU` / `NetTier` in `python/zrt/training/spec/system.py`),
   duplicating data shapes that already exist in the registry.

Downstream analysis passes (`python/zrt/transform/analysis/{passes,memory_train,
comm_latency,modeller,training}.py`) already consume `HardwareSpec` correctly —
the problem is only at the two ingestion boundaries. The unified inference CLI
at `python/zrt/cli.py:196-204` already uses `hw_registry.load(args.hw)` and is
the reference pattern to match.

**Intended outcome:** a single source of truth — the YAML configs under
`python/zrt/hardware/configs/` — for every per-device number (compute, HBM,
interconnect). CLI and YAML callers reference entries by name; cluster-shape
fields (`nodes`, `gpus_per_node`, `host_mem_gb`) remain in the training config
since they are cluster-level, not device-level.

## Critical files to modify

| File | Change |
|------|--------|
| `python/zrt/training/cli.py` | Drop 4 `--gpu-*` flags; add `--hw <name>` (default `nvidia_h100_sxm`); load via `hw_registry.load()`. |
| `python/zrt/training/io/config_loader.py` | Rewrite `_parse_system` to resolve `system.hw: <name>` via registry; translate `HardwareSpec` → `SystemSpec` (`GPU` + `NetTier`s). |
| `configs/training/h100_8x8.yaml` | Replace inline `gpu:`/`nets:` blocks with `hw: nvidia_h100_sxm`. Keep `nodes`, `gpus_per_node`, `host_mem_gb`. |

Other YAMLs under `configs/training/` (`deepseek_v3.yaml`, `llama3_70b*.yaml`,
`strategy_3d.yaml`) are model/strategy configs, not system configs — no change.

## Critical files to reuse (no modification)

| File | Role |
|------|------|
| `python/zrt/hardware/registry.py:34-73` | `load(name)` by stem or display name; `list_available()`. |
| `python/zrt/hardware/spec.py:77-115` | `HardwareSpec` with `compute`, `memory`, `interconnect`; `peak_flops()`, `hbm_bandwidth()`. |
| `python/zrt/hardware/configs/*.yaml` | 5 existing configs: `nvidia_h100_sxm`, `nvidia_a100_80g`, `nvidia_h800`, `ascend_910b`, `ascend_910c`. |
| `python/zrt/cli.py:196-204` | Reference pattern — `--hw` → `hw_registry.load()`. |
| `python/zrt/training/spec/system.py` | `SystemSpec`/`GPU`/`NetTier` kept as internal consumer shape. Wide downstream surface in `training/{search,models,compose}/` stays untouched. |

## Implementation

### 1. `model-training` subcommand (`python/zrt/training/cli.py`)

**Remove** lines 37-41 (`--gpu-name`, `--gpu-bf16-tflops`, `--gpu-hbm-gbps`,
`--gpu-hbm-gb`) from the argparse block.

**Add** one flag:
```python
mt.add_argument("--hw", default="nvidia_h100_sxm",
                help="Hardware registry name (see zrt.hardware.registry.list_available())")
```

**Replace** `_cmd_model_training` body (lines 79-93) — drop the manual
`HardwareSpec(...)` construction and use the registry:
```python
from zrt.hardware import registry as hw_registry
hw = hw_registry.load(args.hw)
```
`hw_spec=hw` pass-through at line 104 is unchanged.

Default `nvidia_h100_sxm` preserves the current default **compute/memory** numbers
exactly (the registry YAML has `bf16_tflops=989`, `capacity_gb=80`,
`hbm_bandwidth_gbps=3350`). Interconnect numbers change from the hardcoded
`(900, 400)` to registry `(900, 400)` — identical — but topology and
`num_devices` become properly driven by YAML.

### 2. `estimate` subcommand YAML schema

**New schema** for `system:` blocks — every system YAML must specify `hw`:
```yaml
system:
  hw: nvidia_h100_sxm     # registry stem or display name
  nodes: 8
  gpus_per_node: 8
  host_mem_gb: 2048       # optional, defaults to 256
```
Cluster shape (`nodes`, `gpus_per_node`, `host_mem_gb`) stays in training YAML;
everything else comes from the registry.

**Rewrite** `_parse_system` in `python/zrt/training/io/config_loader.py:56-81`:
```python
def _parse_system(d: dict) -> SystemSpec:
    from zrt.hardware import registry as hw_registry
    hw = hw_registry.load(d["hw"])                    # KeyError if missing/unknown

    gpu = GPU(
        name=hw.name,
        flops_bf16=hw.compute.bf16_tflops,
        flops_fp8=hw.compute.fp8_tops or hw.compute.bf16_tflops * 2,
        hbm_gb=hw.memory.capacity_gb,
        hbm_bw_gbps=hw.memory.hbm_bandwidth_gbps,
    )
    nets = [
        NetTier(scope="intra_node",
                bw_gbps=hw.interconnect.intra_node.bandwidth_gbps,
                latency_us=hw.interconnect.intra_node.latency_us,
                topology=hw.interconnect.intra_node.topology),
        NetTier(scope="inter_node",
                bw_gbps=hw.interconnect.inter_node.bandwidth_gbps,
                latency_us=hw.interconnect.inter_node.latency_us,
                topology=hw.interconnect.inter_node.topology),
    ]
    return SystemSpec(
        gpu=gpu,
        host_mem_gb=d.get("host_mem_gb", 256),
        nets=nets,
        nodes=d["nodes"],
        gpus_per_node=d["gpus_per_node"],
    )
```

Keep the `GPU`/`NetTier` imports — they are still constructed inside
`_parse_system`.

### 3. Migrate the lone system YAML (`configs/training/h100_8x8.yaml`)

Replace the 19-line `system:` block (lines 4-22) with:
```yaml
system:
  hw: nvidia_h100_sxm
  nodes: 8
  gpus_per_node: 8
  host_mem_gb: 2048
```

### Data-change note to surface to user on execution

The registry's `nvidia_h100_sxm.yaml` has `fp8_tops: 3958` (sparse, doubled).
The current `h100_8x8.yaml` has `flops_fp8: 1979` (dense). After migration, any
FP8 calculation in the `estimate` path will **double**. If `h100_8x8.yaml`'s
`1979` is deliberate (dense-only), either add a dense variant to the registry
YAML or override at the consumer. Flag this to the user before editing and let
them decide.

## Verification

1. **Registry contract test** — `python -c "from zrt.hardware import registry;
   print(registry.list_available())"` returns the 5 configs. Confirm
   `registry.load('nvidia_h100_sxm').compute.bf16_tflops == 989`.

2. **`model-training` smoke** — from repo root:
   ```
   python -m zrt.training model-training \
     --model-id hf_models/llama3_8b --num-layers 2 \
     --hw nvidia_h100_sxm --tp 2 --dp 4
   ```
   Expect a report with non-zero step time and 0 ≤ MFU ≤ 1. Rerun with
   `--hw ascend_910b` and confirm output differs (different peak FLOPS/BW).

3. **`estimate` smoke** — `python -m zrt.training estimate --config
   configs/training/h100_8x8.yaml` succeeds and prints a summary. Diff report
   numbers against pre-change baseline: BF16 compute ceilings and intra/inter
   BW should be unchanged (registry values match current YAML values); only
   FP8-dependent numbers may shift per the data-change note above.

4. **Unit tests** — `pytest tests/training/test_captured_graph_modelling.py -v
   2>&1 | tail -n 50`. These build `HardwareSpec` directly via `_hw()` in the
   test (line 18-28) and don't touch the CLI layer, so they must continue to
   pass unchanged. Also run `pytest tests/ -v -k hardware 2>&1 | tail -n 50`
   for any registry regression tests.

5. **Unified CLI regression** — `python -m python.zrt hf_models/llama3_8b
   --layers 2 --hw nvidia_h100_sxm --train --tp 2` (the inference CLI's
   `--train` path from `python/zrt/cli.py:196-204`) must still run; this path
   was already correct but we verify nothing broke downstream.

6. **Help-text check** — `python -m zrt.training model-training --help` shows
   `--hw` and no longer advertises the 4 `--gpu-*` flags.

## Out of scope

- Adding training-specific fields to `HardwareSpec` (activation write BW,
  optimizer state scale, backward efficiency). Current `HardwareSpec` suffices
  for consumers today.
- Unifying `SystemSpec`/`GPU`/`NetTier` with `HardwareSpec` at the dataclass
  level. The bridge inside `_parse_system` is sufficient; a full merge would
  require rewriting `python/zrt/training/{search,models,compose}/*.py`
  downstream consumers.
- Adding more hardware YAMLs (e.g. B200, MI300X) — additive change unrelated
  to this plumbing work.
