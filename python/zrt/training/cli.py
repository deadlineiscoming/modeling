"""CLI: python -m zrt.training estimate --config config.yaml

Also: python -m zrt.training model-training --model-id hf_models/deepseek_v3_2 ...
"""

from __future__ import annotations

import argparse
import json
import sys

from zrt.training.io.config_loader import load_specs
from zrt.training.search.estimator import estimate
from zrt.training.search.report import report_summary, report_to_json


def main():
    parser = argparse.ArgumentParser(
        description="AI Training Infra Modeller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # estimate subcommand (from YAML config)
    est = sub.add_parser("estimate", help="Estimate training performance for a config")
    est.add_argument("--config", required=True, help="Path to YAML config file")
    est.add_argument("--output", "-o", default=None, help="Output JSON path (default: stdout summary)")

    # model-training subcommand (capture graph from model, then estimate)
    mt = sub.add_parser("model-training", help="Capture graph from model and estimate training performance")
    mt.add_argument("--model-id", required=True, help="HuggingFace model ID or local path")
    mt.add_argument("--num-layers", type=int, default=4, help="Layers to trace (default: 4)")
    mt.add_argument("--seq-len", type=int, default=128, help="Sequence length for tracing (default: 128)")
    mt.add_argument("--total-params", type=float, default=None, help="Full model param count (e.g. 671e9)")
    mt.add_argument("--hidden", type=int, default=7168, help="Hidden dimension (default: 7168)")
    mt.add_argument("--num-layers-full", type=int, default=None, help="Total layers in full model")
    # Hardware
    mt.add_argument("--gpu-name", default="h100", help="GPU name (default: h100)")
    mt.add_argument("--gpu-bf16-tflops", type=float, default=989, help="GPU BF16 TFLOP/s (default: 989)")
    mt.add_argument("--gpu-hbm-gbps", type=float, default=3350, help="GPU HBM bandwidth GB/s (default: 3350)")
    mt.add_argument("--gpu-hbm-gb", type=float, default=80, help="GPU HBM capacity GB (default: 80)")
    # Parallelism
    mt.add_argument("--tp", type=int, default=1)
    mt.add_argument("--pp", type=int, default=1)
    mt.add_argument("--ep", type=int, default=1)
    mt.add_argument("--dp", type=int, default=1)
    # Training config
    mt.add_argument("--zero-stage", type=int, default=1)
    mt.add_argument("--optimizer", default="adam")
    mt.add_argument("--micro-batch", type=int, default=1)
    mt.add_argument("--global-batch", type=int, default=32)
    # Output
    mt.add_argument("--output", "-o", default=None, help="Output JSON path (default: stdout summary)")
    mt.add_argument("--trace-dir", default=None, help="Directory for trace outputs (Excel/ONNX/JSON)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "estimate":
        _cmd_estimate(args.config, args.output)
    elif args.command == "model-training":
        _cmd_model_training(args)


def _cmd_estimate(config_path: str, output_path: str | None) -> None:
    model, system, strategy = load_specs(config_path)
    report = estimate(model, system, strategy)

    if output_path:
        report_to_json(report, output_path)
        print(f"Report written to {output_path}")
    else:
        print(report_summary(report))


def _cmd_model_training(args) -> None:
    from zrt.hardware.spec import HardwareSpec, ComputeSpec, MemorySpec, InterconnectSpec, LinkSpec
    from zrt.transform.analysis import model_training

    hw = HardwareSpec(
        name=args.gpu_name,
        vendor="nvidia",
        device_type="gpu",
        compute=ComputeSpec(bf16_tflops=args.gpu_bf16_tflops),
        memory=MemorySpec(capacity_gb=args.gpu_hbm_gb, hbm_bandwidth_gbps=args.gpu_hbm_gbps),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="nvlink", num_devices=8, bandwidth_gbps=900, latency_us=1.0),
            inter_node=LinkSpec(type="ib", num_devices=64, bandwidth_gbps=400, latency_us=5.0),
        ),
    )

    total_params = int(args.total_params) if args.total_params is not None else None

    report = model_training(
        model_id=args.model_id,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        total_params=total_params,
        hidden=args.hidden,
        num_layers_full=args.num_layers_full,
        hw_spec=hw,
        tp=args.tp, pp=args.pp, ep=args.ep, dp=args.dp,
        zero_stage=args.zero_stage,
        optimizer=args.optimizer,
        micro_batch=args.micro_batch,
        global_batch=args.global_batch,
        output_dir=args.trace_dir,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report written to {args.output}")
    else:
        print(report.summary())


if __name__ == "__main__":
    main()
