# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ZRT-Sim** — an LLM performance modeling and simulation system. Captures the operator sequence of any HuggingFace causal LM using `TorchDispatchMode` inside `FakeTensorMode` (no weights or real memory needed), then applies parallelization transforms and simulates performance across hardware configurations.

## Commands

```bash
# Install
pip install -r requirements.txt

# Run all tests
pytest tests/ -v 2>&1 | tail -n 50

# Run a single test function (preferred)
pytest tests/test_transform.py::test_tp_shape_modification -v
pytest tests/test_executor.py -v -k "overlap"
pytest tests/test_train_trace.py -v

# Run specific test files
pytest tests/test_transform.py tests/test_executor.py tests/test_simulator.py -v 2>&1 | tail -n 50

# CLI: trace a model and export Excel
python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 4
python -m python.zrt deepseek-ai/DeepSeek-V3-0324 --layers 4 --hw nvidia_h100_sxm --tp 8

# Training modeling
python -m python.zrt hf_models/deepseek_v3 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --pp 4 --dp 2

# Training CLI (alternative entrypoint)
PYTHONPATH=python python -m zrt.training model-training hf_models/llama3_8b --num-layers 2 --hw nvidia_h100_sxm --tp 2 --dp 4

# End-to-end validation
python e2e_check.py
```

## Architecture

Four-stage pipeline, all centered on the `OpGraph` IR:

```
Graph Capture → Transform Pipeline → DAGScheduler → Report Generator
                                          ↑
                              MemoryModel (feasibility)
                              SimulatorHub (latency)
```

**Stage 1 — Graph Capture** (`python/zrt/graph/`)
- `model_loader.py`: loads HF model via `FakeTensorMode` — no real weights or memory allocated
- `dispatch.py` + `tracker.py`: intercept aten ops during forward pass via `TorchDispatchMode` + `ModuleTracker`
- `fusion.py`: two-stage fusion — group by leaf module, then merge up to parent if ≤30 child ops
- `graph_builder.py`: produces raw `OpGraph` and fused `OpGraph`
- `patches.py`: MoE meta patch (replaces `.cpu().numpy()` on meta tensors); Indexer patch for DeepSeek V3.2
- `transform_runner.py`: `run_transform()` — applies the transform pipeline and exports results

**Stage 2 — Transform Pipeline** (`python/zrt/transform/`)
- `context.py`: `TransformContext`, `ParallelConfig` (TP/EP/PP/DP/SP), `StreamConfig`
- `pipeline.py`: `build_default_pipeline()` — pluggable pass system
- Pass order: Parallel split (TP → EP → SP → PP) → Comm insertion → Fusion → Optimization (quant, recomp) → Analysis (FLOPs, Roofline, StreamAssign)
- `training/`: additional passes for training — `zero_fsdp.py`, `recompute.py`, `optimizer.py`, `offload.py`
- `analysis/modeller.py`: `estimate_training_from_graphs()` — FLOPs, MFU, memory, 1F1B pipeline scheduling
- Transforms always **clone before mutating** — functional style throughout

**Stage 3 — Executor** (`python/zrt/executor/`)
- `scheduler.py`: `DAGScheduler` — topological sort + greedy multi-stream assignment → `Timeline`
- `Timeline.overlap_us`: quantifies compute/comm masking benefit
- `overlap.py`: stream overlap analysis helpers

**Stage 4 — Simulator** (`python/zrt/simulator/`)
- `hub.py`: `SimulatorHub` — fallback chain: Roofline → Regression → ProfileDB → Tiling
- `backends/roofline.py`: primary backend, uses `HardwareSpec` arithmetic/memory intensity

**Supporting modules**
- `python/zrt/ir/`: `OpGraph`, `OpNode`, `OpEdge`, `DType` (`types.py`), `GraphHierarchy` (`hierarchy.py`), `serde.py`
- `python/zrt/hardware/`: `HardwareSpec`, `hw_registry.load("nvidia_h100_sxm")` — YAML configs in `hardware/configs/` (`ascend_910b`, `ascend_910c`, `nvidia_a100_80g`, `nvidia_h100_sxm`, `nvidia_h800`)
- `python/zrt/memory/`: `model.py` (feasibility), `activation.py` (peak estimation), `budget.py`
- `python/zrt/report/summary.py`: `E2ESummary` + `build_summary()` — TTFT, TPOT, MFU, memory
- `python/zrt/layers/`: typed operator layer classes (`op_mm`, `op_attention`, `op_communication`, `op_fused`, etc.) with `build_dynamic_input()` and `__call__`
- `python/zrt/policy_model/`: policy-based optimization models (`micro_architecture_model`, `open_box_model`, `priority_model`)
- `python/zrt/training/`: dedicated training performance modeling with its own IR (`training/ir/`), spec (`training/spec/`), search (`training/search/`), and compose (`training/compose/pipeline.py` for 1F1B)

## Key Rules (from .clauderules)

### Token Compression
- **Never dump full test logs.** Always pipe through `2>&1 | tail -n 50` or `grep` for errors.
- Run single test functions, not whole files.
- Before running prefill/decode simulations, check NumPy version and Python path.

### Session State
- After each bug fix or feature implementation (tests passing), update `SESSION_PROGRESS.md` with: current file state, resolved issues, next steps.
- After updating, archive to `SESSION_HISTORY.md` (append with timestamp, no need to read first).
- Remind user: **"进度已同步至存档，请执行 /clear 开启新会话或者执行 /compact 压缩成摘要以节省 Token。"**

### Fixed Task Workflow
1. Read `SESSION_PROGRESS.md` and continue from last state
2. Execute task
3. Append `SESSION_PROGRESS.md` content (timestamped) to `SESSION_HISTORY.md`
4. Update `SESSION_PROGRESS.md` with new state

### Error Handling
- Before retrying a fix, check `SESSION_PROGRESS.md` for previously attempted (and failed) solutions.

## Model-Specific Notes

- **Dense models** (Llama, Qwen2, Mistral): 2 layers captures all operator patterns
- **DeepSeek-V3 / V3.2**: 4 layers needed (first 3 dense, layer 4 is MoE); requires `trust_remote_code`
- **MoE models** (Mixtral, Qwen3-MoE): 2 layers usually sufficient (first layer is already MoE)

Local model configs (no weights) live in `hf_models/` (deepseek_v3, deepseek_v3_2, llama3_8b, llama3_70b, qwen2_7b, qwen2_72b, mistral_7b, mixtral_8x7b).
