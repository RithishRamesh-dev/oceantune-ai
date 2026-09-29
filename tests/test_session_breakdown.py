"""Tests for session_breakdown CLOSE artifact."""

from core.session_breakdown import build_session_breakdown, write_session_breakdown


def test_build_session_breakdown_fitness_gain():
    doc = build_session_breakdown(
        session_id="abc123",
        model_id="Qwen/Qwen2.5-7B",
        gpu_type="H200",
        stage1_fitness=1.0,
        stage2_fitness=1.1,
        stage3_fitness=1.2,
        winner_flags={"attention_backend": "FLASH_ATTN"},
        peak_throughput=9000.0,
        best_concurrency=128,
        recipe_id="rid1",
    )
    assert doc["schema_version"] == "1.0.0"
    assert doc["fitness"]["final"] == 1.2
    assert abs(doc["fitness"]["gain_vs_stage1_pct"] - 20.0) < 1e-6
    assert doc["config"]["attention_backend"] == "FLASH_ATTN"
    assert doc["recipe_id"] == "rid1"
    assert doc["stage4"]["integrated_into_serving"] is False


def test_write_session_breakdown(tmp_path):
    doc = build_session_breakdown(
        session_id="sess-xyz",
        model_id="m",
        gpu_type="MI300X",
        stage1_fitness=0.5,
    )
    path = write_session_breakdown(doc, output_dir=tmp_path)
    assert path.exists()
    assert "session_breakdown_" in path.name
    text = path.read_text()
    assert "sess-xyz" in text
    assert "schema_version" in text
