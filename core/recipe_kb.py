"""
core/recipe_kb.py
-----------------
OceanTune Recipe Knowledge Base.

Inspired by Hyperloom's recipe-snapshot KB (cascade warm-start, lessons,
pitfalls, session sedimentation) but implemented as an OceanTune-native
MongoDB-backed store — no Hyperloom source is vendored.

A Recipe is the durable memory for one (model_id, gpu_type, framework,
precision) identity: winning flags, measured fitness, lessons, pitfalls,
and session history. New runs warm-start from the closest match.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("core.recipe_kb")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_recipe_id(
    *,
    model_id: str,
    gpu_type: str,
    framework: str = "vllm",
    precision: str = "auto",
) -> str:
    """Stable identity string for a recipe row."""
    parts = [
        (model_id or "").strip().lower(),
        (gpu_type or "").strip().upper(),
        (framework or "vllm").strip().lower(),
        (precision or "auto").strip().lower(),
    ]
    raw = "|".join(parts)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{parts[1]}::{parts[0].replace('/', '_')}::{parts[2]}::{parts[3]}::{digest}"


@dataclass
class Lesson:
    statement: str
    measured_impact: str = ""
    source_session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Lesson":
        return cls(
            statement=str(d.get("statement") or ""),
            measured_impact=str(d.get("measured_impact") or ""),
            source_session_id=str(d.get("source_session_id") or ""),
        )


@dataclass
class Pitfall:
    description: str
    severity: str = "medium"  # low | medium | high
    source_session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Pitfall":
        return cls(
            description=str(d.get("description") or ""),
            severity=str(d.get("severity") or "medium"),
            source_session_id=str(d.get("source_session_id") or ""),
        )


@dataclass
class SessionRecord:
    session_id: str
    fitness_before: float = 0.0
    fitness_after: float = 0.0
    peak_throughput: float = 0.0
    gain_pct: float = 0.0
    closed_at: str = ""
    stage2_strategy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionRecord":
        return cls(
            session_id=str(d.get("session_id") or ""),
            fitness_before=float(d.get("fitness_before") or 0.0),
            fitness_after=float(d.get("fitness_after") or 0.0),
            peak_throughput=float(d.get("peak_throughput") or 0.0),
            gain_pct=float(d.get("gain_pct") or 0.0),
            closed_at=str(d.get("closed_at") or ""),
            stage2_strategy=dict(d.get("stage2_strategy") or {}),
        )


@dataclass
class Recipe:
    """One durable optimization memory row."""

    recipe_id: str
    model_id: str
    gpu_type: str
    framework: str = "vllm"
    precision: str = "auto"
    version: int = 1
    created_at: str = ""
    updated_at: str = ""

    best_flags: Dict[str, Any] = field(default_factory=dict)
    best_fitness: float = 0.0
    best_fingerprint: str = ""
    peak_throughput_tokens_per_sec: float = 0.0

    lessons: List[Lesson] = field(default_factory=list)
    pitfalls: List[Pitfall] = field(default_factory=list)
    what_worked: List[Dict[str, Any]] = field(default_factory=list)
    what_failed: List[Dict[str, Any]] = field(default_factory=list)
    sessions: List[SessionRecord] = field(default_factory=list)

    confidence: float = 0.5

    def to_mongo_doc(self) -> Dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "model_id": self.model_id,
            "gpu_type": self.gpu_type,
            "framework": self.framework,
            "precision": self.precision,
            "version": self.version,
            "created_at": self.created_at or _utc_now(),
            "updated_at": self.updated_at or _utc_now(),
            "best_flags": self.best_flags,
            "best_fitness": self.best_fitness,
            "best_fingerprint": self.best_fingerprint,
            "peak_throughput_tokens_per_sec": self.peak_throughput_tokens_per_sec,
            "lessons": [x.to_dict() for x in self.lessons],
            "pitfalls": [x.to_dict() for x in self.pitfalls],
            "what_worked": list(self.what_worked),
            "what_failed": list(self.what_failed),
            "sessions": [x.to_dict() for x in self.sessions],
            "confidence": self.confidence,
        }

    @classmethod
    def from_mongo_doc(cls, doc: Dict[str, Any]) -> "Recipe":
        return cls(
            recipe_id=str(doc.get("recipe_id") or ""),
            model_id=str(doc.get("model_id") or ""),
            gpu_type=str(doc.get("gpu_type") or ""),
            framework=str(doc.get("framework") or "vllm"),
            precision=str(doc.get("precision") or "auto"),
            version=int(doc.get("version") or 1),
            created_at=str(doc.get("created_at") or ""),
            updated_at=str(doc.get("updated_at") or ""),
            best_flags=dict(doc.get("best_flags") or {}),
            best_fitness=float(doc.get("best_fitness") or 0.0),
            best_fingerprint=str(doc.get("best_fingerprint") or ""),
            peak_throughput_tokens_per_sec=float(
                doc.get("peak_throughput_tokens_per_sec") or 0.0
            ),
            lessons=[Lesson.from_dict(x) for x in (doc.get("lessons") or [])],
            pitfalls=[Pitfall.from_dict(x) for x in (doc.get("pitfalls") or [])],
            what_worked=list(doc.get("what_worked") or []),
            what_failed=list(doc.get("what_failed") or []),
            sessions=[
                SessionRecord.from_dict(x) for x in (doc.get("sessions") or [])
            ],
            confidence=float(doc.get("confidence") or 0.5),
        )

    def planner_context(self, *, max_lessons: int = 8, max_pitfalls: int = 8) -> str:
        """Compact text block for PlannerAgent / StrategyOptimizer prompts."""
        lines = [
            f"=== Recipe warm-start ({self.recipe_id}) ===",
            f"model={self.model_id} gpu={self.gpu_type} framework={self.framework}",
            f"best_fitness={self.best_fitness:.4f} peak_tok_s="
            f"{self.peak_throughput_tokens_per_sec:.1f} confidence={self.confidence:.2f}",
        ]
        if self.best_flags:
            key_flags = {
                k: v
                for k, v in self.best_flags.items()
                if k
                in (
                    "attention_backend",
                    "kv_cache_dtype",
                    "gpu_memory_utilization",
                    "max_num_batched_tokens",
                    "max_num_seqs",
                    "enable_chunked_prefill",
                    "enable_prefix_caching",
                    "tensor_parallel_size",
                )
            }
            lines.append(f"key_flags={key_flags}")
        if self.lessons:
            lines.append("lessons:")
            for les in self.lessons[-max_lessons:]:
                impact = f" ({les.measured_impact})" if les.measured_impact else ""
                lines.append(f"  - {les.statement}{impact}")
        if self.pitfalls:
            lines.append("pitfalls:")
            for pit in self.pitfalls[-max_pitfalls:]:
                lines.append(f"  - [{pit.severity}] {pit.description}")
        return "\n".join(lines)


# Cascade: relax identity one field at a time (Hyperloom-inspired order).
_CASCADE_STEPS: Tuple[Dict[str, Optional[str]], ...] = (
    {},  # exact
    {"precision": None},  # any precision
    {"framework": None, "precision": None},  # any framework+precision
)


class RecipeKnowledgeBase:
    """
    Lookup / sediment recipes via Database methods.

    Cascade search: exact (model, gpu, framework, precision) → drop precision
    → drop framework → same model+gpu any recipe → empty.
    """

    def __init__(self, db: Any) -> None:
        self._db = db

    async def lookup(
        self,
        *,
        model_id: str,
        gpu_type: str,
        framework: str = "vllm",
        precision: str = "auto",
    ) -> Optional[Recipe]:
        """Cascade warm-start lookup. Returns best matching Recipe or None."""
        # Exact
        doc = await self._db.get_recipe(
            model_id=model_id,
            gpu_type=gpu_type,
            framework=framework,
            precision=precision,
        )
        if doc:
            return Recipe.from_mongo_doc(doc)

        # Same model+gpu, prefer highest fitness
        docs = await self._db.search_recipes(
            model_id=model_id,
            gpu_type=gpu_type,
            limit=5,
        )
        if not docs:
            # Relax gpu: same model any GPU (low confidence)
            docs = await self._db.search_recipes(model_id=model_id, limit=5)
        if not docs:
            return None

        # Prefer matching framework, then fitness
        def _rank(d: Dict[str, Any]) -> Tuple[int, float]:
            fw_match = 1 if d.get("framework") == framework else 0
            return (fw_match, float(d.get("best_fitness") or 0.0))

        docs_sorted = sorted(docs, key=_rank, reverse=True)
        recipe = Recipe.from_mongo_doc(docs_sorted[0])
        # Downgrade confidence for non-exact matches
        if (
            recipe.gpu_type != gpu_type
            or recipe.framework != framework
            or recipe.precision != precision
        ):
            recipe.confidence = min(recipe.confidence, 0.4)
        return recipe

    async def sediment(
        self,
        *,
        model_id: str,
        gpu_type: str,
        session_id: str,
        best_flags: Dict[str, Any],
        best_fitness: float,
        best_fingerprint: str = "",
        peak_throughput: float = 0.0,
        fitness_before: float = 0.0,
        stage2_strategy: Optional[Dict[str, Any]] = None,
        lessons: Optional[List[Lesson]] = None,
        pitfalls: Optional[List[Pitfall]] = None,
        what_worked: Optional[List[Dict[str, Any]]] = None,
        what_failed: Optional[List[Dict[str, Any]]] = None,
        framework: str = "vllm",
        precision: str = "auto",
    ) -> Recipe:
        """
        Write/update recipe after a successful session CLOSE.

        Only upgrades best_flags when new fitness beats stored best.
        Always appends session record and merges lessons/pitfalls.
        """
        recipe_id = canonical_recipe_id(
            model_id=model_id,
            gpu_type=gpu_type,
            framework=framework,
            precision=precision,
        )
        existing = await self._db.get_recipe_by_id(recipe_id)
        if existing:
            recipe = Recipe.from_mongo_doc(existing)
            recipe.version = int(recipe.version) + 1
        else:
            recipe = Recipe(
                recipe_id=recipe_id,
                model_id=model_id,
                gpu_type=gpu_type,
                framework=framework,
                precision=precision,
                created_at=_utc_now(),
            )

        gain = 0.0
        if fitness_before > 0:
            gain = (best_fitness - fitness_before) / fitness_before * 100.0

        recipe.sessions.append(
            SessionRecord(
                session_id=session_id,
                fitness_before=fitness_before,
                fitness_after=best_fitness,
                peak_throughput=peak_throughput,
                gain_pct=gain,
                closed_at=_utc_now(),
                stage2_strategy=dict(stage2_strategy or {}),
            )
        )
        # Cap history
        if len(recipe.sessions) > 50:
            recipe.sessions = recipe.sessions[-50:]

        if best_fitness >= recipe.best_fitness and best_flags:
            recipe.best_flags = dict(best_flags)
            recipe.best_fitness = best_fitness
            recipe.best_fingerprint = best_fingerprint
            recipe.peak_throughput_tokens_per_sec = peak_throughput
            recipe.confidence = min(0.95, 0.5 + 0.05 * len(recipe.sessions))

        for les in lessons or []:
            if les.statement and not any(
                x.statement == les.statement for x in recipe.lessons
            ):
                recipe.lessons.append(les)
        for pit in pitfalls or []:
            if pit.description and not any(
                x.description == pit.description for x in recipe.pitfalls
            ):
                recipe.pitfalls.append(pit)
        if len(recipe.lessons) > 40:
            recipe.lessons = recipe.lessons[-40:]
        if len(recipe.pitfalls) > 40:
            recipe.pitfalls = recipe.pitfalls[-40:]

        for row in what_worked or []:
            recipe.what_worked.append(row)
        for row in what_failed or []:
            recipe.what_failed.append(row)
        if len(recipe.what_worked) > 30:
            recipe.what_worked = recipe.what_worked[-30:]
        if len(recipe.what_failed) > 30:
            recipe.what_failed = recipe.what_failed[-30:]

        recipe.updated_at = _utc_now()
        await self._db.upsert_recipe(recipe.to_mongo_doc())
        log.info(
            "Recipe sedimented: id=%s fitness=%.4f sessions=%d lessons=%d pitfalls=%d",
            recipe.recipe_id,
            recipe.best_fitness,
            len(recipe.sessions),
            len(recipe.lessons),
            len(recipe.pitfalls),
        )
        return recipe


def lessons_from_analyst(
    explanation: str = "",
    recommendation: str = "",
    session_id: str = "",
) -> List[Lesson]:
    """Extract coarse lessons from Analyst text (heuristic, no LLM)."""
    out: List[Lesson] = []
    if recommendation and len(recommendation.strip()) > 20:
        out.append(
            Lesson(
                statement=recommendation.strip()[:500],
                measured_impact="analyst_recommendation",
                source_session_id=session_id,
            )
        )
    if explanation and "bottleneck" in explanation.lower():
        # Keep first sentence mentioning bottleneck
        for sentence in explanation.replace("\n", " ").split("."):
            if "bottleneck" in sentence.lower() and len(sentence.strip()) > 15:
                out.append(
                    Lesson(
                        statement=sentence.strip()[:400],
                        measured_impact="bottleneck_note",
                        source_session_id=session_id,
                    )
                )
                break
    return out


def pitfalls_from_failures(
    failed_configs: List[Dict[str, Any]],
    session_id: str = "",
) -> List[Pitfall]:
    """
    Build pitfalls from failed / OOM configs.

    failed_configs entries: {flags?, error?, reason?}
    """
    out: List[Pitfall] = []
    for item in failed_configs[:10]:
        err = str(item.get("error") or item.get("reason") or "")
        if not err:
            continue
        severity = "high" if "oom" in err.lower() or "out of memory" in err.lower() else "medium"
        flags = item.get("flags") or {}
        hint = ""
        if isinstance(flags, dict) and flags.get("gpu_memory_utilization"):
            hint = f" (gpu_memory_utilization={flags.get('gpu_memory_utilization')})"
        out.append(
            Pitfall(
                description=f"{err[:300]}{hint}",
                severity=severity,
                source_session_id=session_id,
            )
        )
    return out
