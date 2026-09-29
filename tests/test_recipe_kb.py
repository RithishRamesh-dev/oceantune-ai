"""Unit tests for OceanTune Recipe Knowledge Base."""

from __future__ import annotations

from core.recipe_kb import (
    Lesson,
    Pitfall,
    Recipe,
    RecipeKnowledgeBase,
    canonical_recipe_id,
    lessons_from_analyst,
    pitfalls_from_failures,
)


def test_canonical_recipe_id_stable():
    a = canonical_recipe_id(model_id="Qwen/Qwen2.5-7B", gpu_type="H200")
    b = canonical_recipe_id(model_id="Qwen/Qwen2.5-7B", gpu_type="H200")
    assert a == b
    assert "H200" in a


def test_canonical_recipe_id_differs_by_gpu():
    a = canonical_recipe_id(model_id="m", gpu_type="H100")
    b = canonical_recipe_id(model_id="m", gpu_type="H200")
    assert a != b


def test_recipe_planner_context_includes_lessons():
    r = Recipe(
        recipe_id="x",
        model_id="Qwen/Qwen2.5-7B",
        gpu_type="H200",
        best_fitness=0.7,
        best_flags={"attention_backend": "FLASHINFER", "kv_cache_dtype": "fp8"},
        lessons=[Lesson(statement="Try FLASHINFER on GQA", measured_impact="+3%")],
        pitfalls=[Pitfall(description="OOM at gpu_memory_utilization=0.95", severity="high")],
    )
    ctx = r.planner_context()
    assert "FLASHINFER" in ctx
    assert "Try FLASHINFER" in ctx
    assert "OOM" in ctx


def test_recipe_roundtrip_mongo_doc():
    r = Recipe(
        recipe_id="id1",
        model_id="m",
        gpu_type="H200",
        best_flags={"a": 1},
        best_fitness=0.5,
        lessons=[Lesson(statement="L1")],
    )
    doc = r.to_mongo_doc()
    r2 = Recipe.from_mongo_doc(doc)
    assert r2.recipe_id == "id1"
    assert r2.best_flags == {"a": 1}
    assert r2.lessons[0].statement == "L1"


def test_lessons_from_analyst():
    lessons = lessons_from_analyst(
        explanation="The primary bottleneck is memory bandwidth at high concurrency.",
        recommendation="Enable fp8 KV cache when VRAM is saturated.",
        session_id="s1",
    )
    assert len(lessons) >= 1
    assert any("fp8" in l.statement.lower() or "bottleneck" in l.statement.lower() for l in lessons)


def test_pitfalls_from_failures():
    pits = pitfalls_from_failures(
        [
            {"error": "CUDA OOM", "flags": {"gpu_memory_utilization": 0.95}},
            {"error": "startup timeout"},
        ],
        session_id="s1",
    )
    assert len(pits) == 2
    assert pits[0].severity == "high"
