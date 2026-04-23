"""Training modeller: main entry point for training performance estimation.

Two usage modes:

1. **From an existing OpGraph** (lower-level)::

       from python.zrt.transform.analysis import estimate_training
       report = estimate_training(graph, ctx)

2. **From a model directory** (end-to-end, captures graph then estimates)::

       from python.zrt.transform.analysis import model_training
       report = model_training(
           model_id="hf_models/deepseek_v3_2",
           num_layers=4,
           seq_len=128,
           hw_spec=my_hardware_spec,
           total_params=671e9,
           tp=8, pp=4, dp=2,
           micro_batch=1, global_batch=8192,
       )
       print(report.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


@dataclass
class TrainingReport:
    """Training performance estimation report."""

    # Config summary
    config_summary: str = ""

    # Timing metrics
    step_time_ms: float = 0.0
    per_stage_ms: float = 0.0

    # Efficiency metrics
    mfu: float = 0.0  # Model FLOPs Utilization

    # FLOPs breakdown
    training_flops: float = 0.0
    forward_flops: float = 0.0
    backward_flops: float = 0.0

    # Memory breakdown (per GPU)
    memory_breakdown: dict[str, float] = field(default_factory=dict)

    # Pipeline metrics
    warmup_steps: int = 0
    cooldown_steps: int = 0
    steady_steps: int = 0
    bubble_fraction: float = 0.0

    # Model info
    total_params: int = 0

    def to_dict(self) -> dict:
        """Convert report to JSON-serializable dict."""
        return {
            "config_summary": self.config_summary,
            "step_time_ms": self.step_time_ms,
            "per_stage_ms": self.per_stage_ms,
            "mfu": self.mfu,
            "training_flops": self.training_flops,
            "forward_flops": self.forward_flops,
            "backward_flops": self.backward_flops,
            "memory_breakdown_gb": {
                k: v / 1e9 for k, v in self.memory_breakdown.items()
            },
            "warmup_steps": self.warmup_steps,
            "cooldown_steps": self.cooldown_steps,
            "steady_steps": self.steady_steps,
            "bubble_fraction": self.bubble_fraction,
            "total_params": self.total_params,
        }

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "Training Estimation Report",
            "=" * 40,
            f"Config: {self.config_summary}",
            "",
            "Timing:",
            f"  Step time: {self.step_time_ms:.2f} ms",
            f"  Per-stage: {self.per_stage_ms:.2f} ms",
            "",
            "Efficiency:",
            f"  MFU: {self.mfu:.1%}",
            "",
            "FLOPs:",
            f"  Training: {self.training_flops/1e12:.2f} TFLOPs",
            f"  Forward: {self.forward_flops/1e12:.2f} TFLOPs",
            f"  Backward: {self.backward_flops/1e12:.2f} TFLOPs",
            "",
            "Memory (per GPU):",
        ]
        for k, v in self.memory_breakdown.items():
            lines.append(f"  {k}: {v/1e9:.2f} GB")
        lines.extend([
            "",
            "Pipeline:",
            f"  Warmup steps: {self.warmup_steps}",
            f"  Steady steps: {self.steady_steps}",
            f"  Cooldown steps: {self.cooldown_steps}",
            f"  Bubble fraction: {self.bubble_fraction:.1%}",
            "",
            f"Total params: {self.total_params/1e9:.2f}B",
        ])
        return "\n".join(lines)


def estimate_training(
    graph: "OpGraph",
    ctx: "TransformContext",
) -> TrainingReport:
    """Estimate training performance metrics.

    This function runs training-specific analysis passes on the graph
    and returns a comprehensive training performance report.

    Parameters
    ----------
    graph : OpGraph
        The computation graph (typically a forward pass graph)
    ctx : TransformContext
        Transform context with training configuration (ctx.training must be set)

    Returns
    -------
    TrainingReport
        Training performance estimation report

    Examples
    --------
    >>> from python.zrt.transform.context import TransformContext, TrainingConfig
    >>> ctx = TransformContext(
    ...     hw_spec=my_hw,
    ...     training=TrainingConfig(optimizer="adam", zero_stage=1, ...),
    ... )
    >>> report = estimate_training(graph, ctx)
    >>> print(report.summary())
    """
    from .training import TrainingFlopsPass, TrainingMemoryPass, TrainingPipelinePass  # noqa: F401

    # Run training analysis passes
    flops_pass = TrainingFlopsPass()
    memory_pass = TrainingMemoryPass()
    pipeline_pass = TrainingPipelinePass()

    g = flops_pass.run(graph, ctx)
    g = memory_pass.run(g, ctx)

    # Annotate per-node latency before pipeline timing (requires hw_spec)
    if ctx.hw_spec is not None:
        from .passes import RooflinePass
        g = RooflinePass().run(g, ctx)

    g = pipeline_pass.run(g, ctx)

    # Extract metrics from graph metadata
    pipeline_metrics = g.metadata.get("pipeline_metrics")
    memory_breakdown = g.metadata.get("memory_breakdown")

    # Build config summary
    parallel = ctx.parallel
    training = ctx.training
    config_parts = []
    if parallel.tp > 1:
        config_parts.append(f"TP{parallel.tp}")
    if parallel.pp > 1:
        config_parts.append(f"PP{parallel.pp}")
    if parallel.ep > 1:
        config_parts.append(f"EP{parallel.ep}")
    if parallel.dp > 1:
        config_parts.append(f"DP{parallel.dp}")
    if training:
        config_parts.append(f"ZeRO-{training.zero_stage}")
        config_parts.append(f"{training.optimizer}")
        config_parts.append(f"micro{training.micro_batch}")

    config_summary = "-".join(config_parts) if config_parts else "default"

    # Build report
    report = TrainingReport(
        config_summary=config_summary,
        step_time_ms=pipeline_metrics.step_time_ms if pipeline_metrics else 0.0,
        per_stage_ms=pipeline_metrics.per_stage_ms if pipeline_metrics else 0.0,
        mfu=pipeline_metrics.mfu if pipeline_metrics else 0.0,
        training_flops=g.metadata.get("training_flops", 0.0),
        forward_flops=g.metadata.get("forward_flops", 0.0),
        backward_flops=g.metadata.get("backward_flops", 0.0),
        memory_breakdown=memory_breakdown.to_dict() if memory_breakdown else {},
        warmup_steps=pipeline_metrics.warmup_steps if pipeline_metrics else 0,
        cooldown_steps=pipeline_metrics.cooldown_steps if pipeline_metrics else 0,
        steady_steps=pipeline_metrics.steady_steps if pipeline_metrics else 0,
        bubble_fraction=pipeline_metrics.bubble_fraction if pipeline_metrics else 0.0,
        total_params=g.metadata.get("total_params", 0),
    )

    return report


def model_training(
    model_id: str,
    num_layers: int = 4,
    batch_size: int = 1,
    seq_len: int = 128,
    total_params: int | None = None,
    hidden: int = 7168,
    num_layers_full: int | None = None,
    hw_spec: "HardwareSpec | None" = None,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    dp: int = 1,
    zero_stage: int = 1,
    optimizer: str = "adam",
    micro_batch: int = 1,
    global_batch: int = 32,
    output_dir: str | None = None,
) -> TrainingReport:
    """Capture a computation graph from a model and estimate training performance.

    This is the main end-to-end entry point. It chains graph capture
    (``run_trace_phases``) → IR conversion → ``estimate_training`` in a
    single call.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID or local path (e.g. ``"hf_models/deepseek_v3_2"``).
    num_layers : int
        Number of layers to trace (keep small for speed; results are scaled).
    batch_size, seq_len : int
        Trace input dimensions.
    total_params : int or None
        Full model parameter count.  If provided, stored in graph metadata
        so the FLOPs pass uses the authoritative count.
    hidden : int
        Hidden dimension (for memory estimation).
    num_layers_full : int or None
        Total layers in the full model (for memory scaling).  Defaults to
        *num_layers* if not provided.
    hw_spec : HardwareSpec or None
        Target hardware.  Required for realistic latency / MFU estimates.
    tp, pp, ep, dp : int
        Parallelism dimensions.
    zero_stage : int
        ZeRO stage (0–3).
    optimizer : str
        Optimizer name (``"adam"``, ``"adamw"``, ``"muon"``).
    micro_batch, global_batch : int
        Batch configuration.
    output_dir : str or None
        Where to write trace outputs (Excel, JSON, ONNX).  None = temp dir.

    Returns
    -------
    TrainingReport
    """
    from python.zrt.graph import run_trace_phases
    from python.zrt.transform.context import ParallelConfig, TrainingConfig, TransformContext

    # 1. Capture
    _, phase_records = run_trace_phases(
        model_id=model_id,
        num_layers=num_layers,
        batch_size=batch_size,
        seq_len=seq_len,
        phases=("train_forward",),
        output_dir=output_dir,
    )
    records = phase_records["train_forward"]

    # 2. Build OpGraph with full-model metadata
    from python.zrt.ir.adapter import records_to_opgraph

    metadata: dict = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_layers": num_layers_full or num_layers,
        "hidden": hidden,
    }
    if total_params is not None:
        metadata["total_params"] = int(total_params)

    graph = records_to_opgraph(
        records=records,
        name=model_id.replace("/", "_"),
        phase="train_forward",
        metadata=metadata,
    )

    # 3. Estimate
    ctx = TransformContext(
        hw_spec=hw_spec,
        parallel=ParallelConfig(tp=tp, pp=pp, ep=ep, dp=dp),
        training=TrainingConfig(
            optimizer=optimizer,
            zero_stage=zero_stage,
            micro_batch=micro_batch,
            global_batch=global_batch,
        ),
    )
    return estimate_training(graph, ctx)
