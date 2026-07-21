from examples.simBenchmarks.CoT.geometry_probe.probe_utils import build_latency_fields


def test_build_latency_fields_keeps_raw_stage_times_and_transport_overhead():
    fields = build_latency_fields(
        client_roundtrip_ms=120.0,
        server_timing={
            "server_total_ms": 100.0,
            "preprocess_ms": 4.0,
            "qwen_backbone_ms": 70.0,
            "action_expert_ms": 20.0,
        },
    )

    assert fields["client_roundtrip_ms"] == 120.0
    assert fields["server_total_ms"] == 100.0
    assert fields["preprocess_ms"] == 4.0
    assert fields["qwen_backbone_ms"] == 70.0
    assert fields["action_expert_ms"] == 20.0
    assert fields["transport_overhead_ms"] == 20.0
