from __future__ import annotations

from zrt.hardware.spec import InterconnectSpec, LinkSpec, TopologyTier


def test_equal_interconnect_specs_have_equal_hashes():
    a = InterconnectSpec(tiers=[
        TopologyTier(
            name="tray",
            link=LinkSpec(
                type="NVLink",
                bandwidth_gbps=900,
                latency_us=1.0,
                topology="all_to_all",
                num_devices=8,
            ),
        ),
    ])
    b = InterconnectSpec(tiers=[
        TopologyTier(
            name="tray",
            link=LinkSpec(
                type="NVLink",
                bandwidth_gbps=900,
                latency_us=1.0,
                topology="all_to_all",
                num_devices=8,
            ),
        ),
    ])

    assert a == b
    assert hash(a) == hash(b)
    assert {a: "spec"}[b] == "spec"
