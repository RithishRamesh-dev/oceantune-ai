"""
core/db.py
----------
MongoDB connection and all document-level operations for OceanTune AI.

Collections
-----------
  sessions       — one document per optimisation run (model, gpu, strategy, timestamps)
  nodes          — heartbeat / capacity records for each GPU droplet
  configs        — candidate VLLMFlags configs with status tracking
  benchmark_runs — raw + enriched benchmark results per config per context
  kernel_runs    — Stage-2 kernel-level results

Usage
-----
    from core.db import Database
    db = Database(uri="mongodb://localhost:27017", db_name="oceantune")
    await db.connect()
    session_id = await db.create_session(model_id="...", gpu_type="H100", strategy="evolutionary")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.errors import DuplicateKeyError

log = logging.getLogger("core.db")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _oid_str(doc: Dict) -> str:
    """Return the string representation of a document's _id."""
    return str(doc["_id"])


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class Database:
    """
    Async MongoDB client wrapper.

    All public methods are coroutines and must be awaited.
    Call ``await db.connect()`` once before using any other method.
    Call ``await db.close()`` on shutdown.
    """

    def __init__(self, uri: str, db_name: str = "oceantune"):
        self._uri = uri
        self._db_name = db_name
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the MongoDB connection and ensure indexes exist."""
        self._client = AsyncIOMotorClient(self._uri, serverSelectionTimeoutMS=5000)
        self._db = self._client[self._db_name]
        await self._ensure_indexes()
        log.info("Connected to MongoDB: %s / %s", self._uri, self._db_name)

    async def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            log.info("MongoDB connection closed")

    @property
    def db(self) -> AsyncIOMotorDatabase:
        if self._db is None:
            raise RuntimeError("Database.connect() has not been called")
        return self._db

    # ------------------------------------------------------------------
    # Index creation
    # ------------------------------------------------------------------

    async def _ensure_indexes(self) -> None:
        await self.db["sessions"].create_indexes([
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("model_id", ASCENDING), ("gpu_type", ASCENDING)]),
        ])
        await self.db["nodes"].create_indexes([
            IndexModel([("host", ASCENDING), ("node_port", ASCENDING)], unique=True),
        ])
        # Drop the old global fingerprint unique index if it exists
        # (it was incorrectly global; the correct scope is per-session)
        try:
            await self.db["configs"].drop_index("fingerprint_1")
        except Exception:
            pass
        await self.db["configs"].create_indexes([
            IndexModel([("session_id", ASCENDING), ("status", ASCENDING)]),
            IndexModel(
                [("session_id", ASCENDING), ("fingerprint", ASCENDING)],
                unique=True,
                name="session_fingerprint_unique",
            ),
        ])
        await self.db["benchmark_runs"].create_indexes([
            IndexModel([("session_id", ASCENDING), ("config_id", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("fitness_score", DESCENDING)]),
        ])
        await self.db["kernel_runs"].create_indexes([
            IndexModel([("session_id", ASCENDING), ("iteration", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
        ])
        log.debug("MongoDB indexes ensured")

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def create_session(
        self,
        *,
        model_id: str,
        gpu_type: str,
        strategy: str,
        context_configs: Optional[List] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Insert a new session document. Returns the session_id (str)."""
        doc: Dict[str, Any] = {
            "model_id": model_id,
            "gpu_type": gpu_type,
            "strategy": strategy,
            "context_configs": context_configs or [],
            "status": "running",
            "created_at": _now(),
            "updated_at": _now(),
            "metadata": metadata or {},
        }
        result = await self.db["sessions"].insert_one(doc)
        session_id = str(result.inserted_id)
        log.info("Session created: %s  model=%s gpu=%s", session_id, model_id, gpu_type)
        return session_id

    async def update_session_status(self, session_id: str, status: str) -> None:
        """Set session.status and bump updated_at."""
        from bson import ObjectId
        await self.db["sessions"].update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"status": status, "updated_at": _now()}},
        )

    async def get_session(self, session_id: str) -> Optional[Dict]:
        from bson import ObjectId
        return await self.db["sessions"].find_one({"_id": ObjectId(session_id)})

    async def list_sessions(self, limit: int = 20) -> List[Dict]:
        cursor = self.db["sessions"].find().sort("created_at", DESCENDING).limit(limit)
        return await cursor.to_list(length=limit)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    async def upsert_node(
        self,
        *,
        host: str,
        node_port: int,
        gpu_type: str,
        gpu_count: int,
        status: str = "online",
        metadata: Optional[Dict] = None,
    ) -> str:
        """Upsert a node heartbeat record. Returns the node document _id."""
        doc: Dict[str, Any] = {
            "host": host,
            "node_port": node_port,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "status": status,
            "last_seen": _now(),
            "metadata": metadata or {},
        }
        result = await self.db["nodes"].find_one_and_update(
            {"host": host, "node_port": node_port},
            {"$set": doc},
            upsert=True,
            return_document=True,
        )
        return str(result["_id"])

    async def list_nodes(self, status: Optional[str] = None) -> List[Dict]:
        filt: Dict = {}
        if status:
            filt["status"] = status
        cursor = self.db["nodes"].find(filt)
        return await cursor.to_list(length=100)

    # ------------------------------------------------------------------
    # Configs (candidate VLLMFlags)
    # ------------------------------------------------------------------

    async def insert_config(
        self,
        *,
        session_id: str,
        fingerprint: str,
        flags: Dict[str, Any],
        generation: int = 0,
        priority: int = 0,
    ) -> Optional[str]:
        """
        Insert a candidate config document.
        Returns the config_id (str), or None if fingerprint already exists.
        """
        doc: Dict[str, Any] = {
            "session_id": session_id,
            "fingerprint": fingerprint,
            "flags": flags,
            "generation": generation,
            "priority": priority,
            "status": "pending",      # pending | running | done | failed
            "retry_count": 0,
            "created_at": _now(),
            "updated_at": _now(),
        }
        try:
            result = await self.db["configs"].insert_one(doc)
            return str(result.inserted_id)
        except DuplicateKeyError:
            log.debug("Config fingerprint already exists: %s", fingerprint)
            return None

    async def claim_pending_config(self, session_id: str) -> Optional[Dict]:
        """
        Atomically find-and-update the highest-priority pending config
        for a session, setting status=running. Returns the document or None.
        """
        return await self.db["configs"].find_one_and_update(
            {"session_id": session_id, "status": "pending"},
            {"$set": {"status": "running", "updated_at": _now()}},
            sort=[("priority", DESCENDING), ("created_at", ASCENDING)],
            return_document=True,
        )

    async def get_config_by_id(self, config_id: str) -> Optional[Dict]:
        """Return a config document by its string ID, or None if not found."""
        from bson import ObjectId
        return await self.db["configs"].find_one({"_id": ObjectId(config_id)})

    async def mark_config_done(self, config_id: str, fitness_score: float) -> None:
        from bson import ObjectId
        await self.db["configs"].update_one(
            {"_id": ObjectId(config_id)},
            {"$set": {"status": "done", "fitness_score": fitness_score, "updated_at": _now()}},
        )

    async def mark_config_failed(self, config_id: str, error: str = "") -> None:
        from bson import ObjectId
        await self.db["configs"].update_one(
            {"_id": ObjectId(config_id)},
            {
                "$set": {"status": "failed", "error": error, "updated_at": _now()},
                "$inc": {"retry_count": 1},
            },
        )

    async def requeue_config(self, config_id: str) -> None:
        """Reset a failed config to pending (for retry)."""
        from bson import ObjectId
        await self.db["configs"].update_one(
            {"_id": ObjectId(config_id)},
            {"$set": {"status": "pending", "updated_at": _now()}},
        )

    async def count_pending_configs(self, session_id: str) -> int:
        return await self.db["configs"].count_documents(
            {"session_id": session_id, "status": "pending"}
        )

    async def get_failed_fingerprints(self, session_id: str) -> List[str]:
        """Return fingerprints of all failed configs for a session."""
        cursor = self.db["configs"].find(
            {"session_id": session_id, "status": "failed"},
            {"fingerprint": 1},
        )
        docs = await cursor.to_list(length=10000)
        return [d["fingerprint"] for d in docs]

    # ------------------------------------------------------------------
    # Benchmark runs
    # ------------------------------------------------------------------

    async def insert_benchmark_run(
        self,
        *,
        session_id: str,
        config_id: str,
        fingerprint: str,
        flags: Optional[Dict[str, Any]] = None,
        context: Dict[str, int],
        raw_metrics: Dict[str, Any],
        levels: Optional[List[Dict[str, Any]]] = None,
        enriched_metrics: Optional[Dict[str, Any]] = None,
        fitness_score: float = 0.0,
        error: Optional[str] = None,
        node_host: str = "localhost",
    ) -> str:
        """Insert a benchmark result document. Returns the run_id (str)."""
        doc: Dict[str, Any] = {
            "session_id": session_id,
            "config_id": config_id,
            "fingerprint": fingerprint,
            "flags": flags or {},
            "context": context,
            "raw_metrics": raw_metrics,
            "levels": levels or [],          # per-concurrency breakdown
            "enriched_metrics": enriched_metrics or {},
            "fitness_score": fitness_score,
            "error": error,
            "node_host": node_host,
            "created_at": _now(),
        }
        result = await self.db["benchmark_runs"].insert_one(doc)
        return str(result.inserted_id)

    async def get_top_configs(
        self,
        session_id: str,
        n: int = 5,
        metric: str = "fitness_score",
    ) -> List[Dict]:
        """Return the top-N benchmark runs by a given metric for a session."""
        cursor = (
            self.db["benchmark_runs"]
            .find({"session_id": session_id, "error": None})
            .sort(metric, DESCENDING)
            .limit(n)
        )
        return await cursor.to_list(length=n)

    async def get_best_run(self, session_id: str) -> Optional[Dict]:
        """Return the single best benchmark run for a session."""
        runs = await self.get_top_configs(session_id, n=1)
        return runs[0] if runs else None

    async def list_benchmark_runs(
        self,
        session_id: str,
        limit: int = 200,
    ) -> List[Dict]:
        cursor = (
            self.db["benchmark_runs"]
            .find({"session_id": session_id})
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    # ------------------------------------------------------------------
    # Kernel runs (Stage 2)
    # ------------------------------------------------------------------

    async def insert_kernel_run(
        self,
        *,
        session_id: str,
        iteration: int,
        kernel_config: Dict[str, Any],
        raw_metrics: Dict[str, Any],
        fitness_score: float = 0.0,
        llm_reasoning: str = "",
        error: Optional[str] = None,
    ) -> str:
        """Insert a Stage-2 kernel optimisation result. Returns the run_id."""
        doc: Dict[str, Any] = {
            "session_id": session_id,
            "iteration": iteration,
            "kernel_config": kernel_config,
            "raw_metrics": raw_metrics,
            "fitness_score": fitness_score,
            "llm_reasoning": llm_reasoning,
            "error": error,
            "created_at": _now(),
        }
        result = await self.db["kernel_runs"].insert_one(doc)
        return str(result.inserted_id)

    async def get_best_kernel_run(self, session_id: str) -> Optional[Dict]:
        cursor = (
            self.db["kernel_runs"]
            .find({"session_id": session_id, "error": None})
            .sort("fitness_score", DESCENDING)
            .limit(1)
        )
        docs = await cursor.to_list(length=1)
        return docs[0] if docs else None

    async def list_kernel_runs(self, session_id: str, limit: int = 100) -> List[Dict]:
        cursor = (
            self.db["kernel_runs"]
            .find({"session_id": session_id})
            .sort("iteration", ASCENDING)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    # ------------------------------------------------------------------
    # Analytics aggregation pipelines
    # ------------------------------------------------------------------

    async def top_configs_by_throughput(
        self, session_id: str, n: int = 10
    ) -> List[Dict]:
        """Aggregate top-N configs ranked by max throughput across all contexts."""
        pipeline = [
            {"$match": {"session_id": session_id, "error": None}},
            {"$group": {
                "_id": "$fingerprint",
                "max_throughput": {"$max": "$enriched_metrics.peak_throughput_tokens_per_sec"},
                "avg_fitness": {"$avg": "$fitness_score"},
                "config_id": {"$first": "$config_id"},
                "run_count": {"$sum": 1},
            }},
            {"$sort": {"max_throughput": -1}},
            {"$limit": n},
        ]
        cursor = self.db["benchmark_runs"].aggregate(pipeline)
        return await cursor.to_list(length=n)

    async def kernel_impact_analysis(self, session_id: str) -> List[Dict]:
        """
        Aggregate Stage-2 kernel runs: show average fitness delta per
        kernel_config key to rank which flags matter most.
        """
        pipeline = [
            {"$match": {"session_id": session_id, "error": None}},
            {"$sort": {"fitness_score": -1}},
            {"$limit": 50},
            {"$project": {
                "fitness_score": 1,
                "kernel_config": {"$objectToArray": "$kernel_config"},
            }},
            {"$unwind": "$kernel_config"},
            {"$group": {
                "_id": {"key": "$kernel_config.k", "value": "$kernel_config.v"},
                "avg_fitness": {"$avg": "$fitness_score"},
                "count": {"$sum": 1},
            }},
            {"$sort": {"avg_fitness": -1}},
        ]
        cursor = self.db["kernel_runs"].aggregate(pipeline)
        return await cursor.to_list(length=200)

    async def oom_patterns(self, session_id: str) -> List[Dict]:
        """Return configs that triggered OOM errors, grouped by flag pattern."""
        pipeline = [
            {"$match": {"session_id": session_id, "error": {"$regex": "OOM|out of memory", "$options": "i"}}},
            {"$lookup": {
                "from": "configs",
                "localField": "config_id",
                "foreignField": "_id",
                "as": "config",
            }},
            {"$unwind": "$config"},
            {"$project": {
                "fingerprint": 1,
                "error": 1,
                "gpu_memory_utilization": "$config.flags.gpu_memory_utilization",
                "max_num_seqs": "$config.flags.max_num_seqs",
                "created_at": 1,
            }},
            {"$sort": {"created_at": -1}},
        ]
        cursor = self.db["benchmark_runs"].aggregate(pipeline)
        return await cursor.to_list(length=500)

    async def performance_over_time(self, session_id: str) -> List[Dict]:
        """Return fitness_score time-series for a session (for progress charts)."""
        pipeline = [
            {"$match": {"session_id": session_id}},
            {"$sort": {"created_at": ASCENDING}},
            {"$project": {
                "created_at": 1,
                "fitness_score": 1,
                "fingerprint": 1,
                "error": 1,
            }},
        ]
        cursor = self.db["benchmark_runs"].aggregate(pipeline)
        return await cursor.to_list(length=10000)

    async def cross_session_seen_fingerprints(
        self, model_id: str, gpu_type: str, limit: int = 5000
    ) -> List[str]:
        """
        Return fingerprints of configs that were benchmarked in any prior
        session for the same model+GPU combination.  Used by PopulationManager
        to skip already-evaluated configs across restarts.
        """
        pipeline = [
            {"$lookup": {
                "from": "sessions",
                "localField": "session_id",
                "foreignField": "_id",
                "as": "session",
            }},
            {"$unwind": "$session"},
            {"$match": {
                "session.model_id": model_id,
                "session.gpu_type": gpu_type,
            }},
            {"$group": {"_id": "$fingerprint"}},
            {"$limit": limit},
        ]
        cursor = self.db["benchmark_runs"].aggregate(pipeline)
        docs = await cursor.to_list(length=limit)
        return [d["_id"] for d in docs]
