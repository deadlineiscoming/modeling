"""Integration test: verify how DataParallel (DP) affects TrainingReport fields.

This test runs the training modelling CLI twice (dp=1 and dp=4), loads the
generated `deepseek_v4_training_report.json` files and asserts expected
relationships:

- optimizer state (opt_state) per-GPU ≈ 1/dp
- total per-GPU memory drops when dp increases
- step_time decreases and tokens/sec increases when dp increases
- dp_hidden_ms and dp_exposed_ms are zero for dp=1, non-zero for dp>1

This is a long-running integration test that captures real reports. To avoid
running it by default in fast CI, it is skipped unless the environment
variable ``RUN_DP_TEST`` is set to ``1``.

Run locally (PowerShell):

```powershell
$env:RUN_DP_TEST='1'; pytest tests/integration/test_dp_effects.py -q
```
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli_and_load_report(repo_root: Path, outdir: Path, dp: int, timeout: int = 900) -> dict:
    """Run `python -m python.zrt` with given dp and return parsed report JSON.

    Raises subprocess.CalledProcessError on failure.
    """
    env = os.environ.copy()
    # Ensure the training subpackage imports as expected (PYTHONPATH=python)
    env["PYTHONPATH"] = str(repo_root / "python")

    cmd = [
        sys.executable,
        "-m",
        "python.zrt",
        "--model-id",
        "hf_models/deepseek_v4",
        "--train",
        "--hw",
        "nvidia_h100_sxm",
        "--dp",
        str(dp),
        "--layers",
        "4",
        "--batch-size",
        "1",
        "--seq-len",
        "128",
        "--output-dir",
        str(outdir),
    ]

    proc = subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    # Write debug output to the outdir for inspection
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cli_output.log").write_text(proc.stdout)

    report_path = outdir / "reports" / "deepseek_v4_training_report.json"
    assert report_path.exists(), f"Report not found at {report_path}"
    return json.loads(report_path.read_text())


def test_dp_changes_affect_expected_report_fields(tmp_path: Path):
    # Skip by default because this is an expensive integration test.
    if os.environ.get("RUN_DP_TEST") != "1":
        pytest.skip("Set RUN_DP_TEST=1 to run this long integration test")

    repo_root = Path(__file__).resolve().parents[3]

    out_dp1 = tmp_path / "out_dp1"
    out_dp4 = tmp_path / "out_dp4"

    rep1 = _run_cli_and_load_report(repo_root, out_dp1, dp=1)
    rep4 = _run_cli_and_load_report(repo_root, out_dp4, dp=4)

    # Basic sanity
    assert rep1.get("memory_breakdown_gb") and rep4.get("memory_breakdown_gb")

    opt1 = rep1["memory_breakdown_gb"].get("opt_state")
    opt4 = rep4["memory_breakdown_gb"].get("opt_state")
    assert opt1 is not None and opt4 is not None

    # optimizer state per-GPU should roughly scale ~1/dp (ZeRO-1 behaviour)
    expected_opt4 = opt1 / 4.0
    assert opt4 == pytest.approx(expected_opt4, rel=0.25), f"opt_state did not scale near 1/dp: {opt1} -> {opt4}"

    # Total per-GPU memory should decrease when dp increases
    total1 = rep1["memory_breakdown_gb"].get("total")
    total4 = rep4["memory_breakdown_gb"].get("total")
    assert total1 is not None and total4 is not None
    assert total4 < total1, f"expected total memory per-GPU to decrease with DP: {total1} -> {total4}"

    # Step time should decrease and tokens/sec should increase (throughput improvement)
    st1 = rep1.get("step_time_ms", None)
    st4 = rep4.get("step_time_ms", None)
    tps1 = rep1.get("tokens_per_sec", None)
    tps4 = rep4.get("tokens_per_sec", None)
    assert st1 is not None and st4 is not None
    assert tps1 is not None and tps4 is not None
    assert st4 < st1, f"expected step_time to decrease when DP increases: {st1} -> {st4}"
    assert tps4 > tps1, f"expected tokens/sec to increase when DP increases: {tps1} -> {tps4}"

    # DP communication: hidden and exposed portions
    dp_hidden1 = rep1.get("dp_hidden_ms", None)
    dp_hidden4 = rep4.get("dp_hidden_ms", None)
    dp_exposed1 = rep1.get("dp_exposed_ms", None)
    dp_exposed4 = rep4.get("dp_exposed_ms", None)
    assert dp_hidden1 is not None and dp_hidden4 is not None
    assert dp_exposed1 is not None and dp_exposed4 is not None

    # dp=1: no DP communication at all
    assert dp_hidden1 == 0.0, f"dp_hidden should be 0 for dp=1, got {dp_hidden1}"
    assert dp_exposed1 == 0.0, f"dp_exposed should be 0 for dp=1, got {dp_exposed1}"

    # dp=4: DP AR/RS communication exists, split into hidden + exposed
    assert dp_hidden4 > 0, f"dp_hidden should be >0 for dp=4, got {dp_hidden4}"
    assert dp_exposed4 > 0, f"dp_exposed should be >0 for dp=4, got {dp_exposed4}"