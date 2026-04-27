"""Hardware registry and specification dataclasses.

Quickstart::

    from python.zrt.hardware import load, list_available
    from python.zrt.hardware import HardwareSpec, ComputeSpec, MemorySpec
    from python.zrt.hardware import InterconnectSpec, LinkSpec, MemoryTier

    hw = load("ascend_910b")
    print(hw)
    # → HardwareSpec('Ascend 910B', npu, bf16=320T, hbm=64GB@1600GB/s)

    print(hw.peak_flops(DType.BF16))   # 3.2e14 (FLOPs/s)
    print(hw.hbm_bandwidth())           # 1.6e12 (bytes/s)

    print(list_available())
    # → ['ascend_910b', 'ascend_910c', 'nvidia_a100_80g', ...]
"""
from python.zrt.hardware.spec import (
    ComputeSpec,
    HardwareSpec,
    InterconnectSpec,
    LinkSpec,
    MemorySpec,
    MemoryTier,
)
from python.zrt.hardware.registry import load, list_available

__all__ = [
    # spec types
    "HardwareSpec",
    "ComputeSpec",
    "MemorySpec",
    "MemoryTier",
    "InterconnectSpec",
    "LinkSpec",
    # registry
    "load",
    "list_available",
]
