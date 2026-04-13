"""
node/node_server.py
-------------------
FastAPI server that runs on each GPU Droplet.

The Coordinator (core/coordinator.py) sends HTTP requests to this server
to dispatch benchmark jobs, query capacity, and receive job results.

Start command (on the GPU droplet):
    python3 -m node.node_server --port 9000 --gpu-type H100 --gpu-indices 0,1,2,3,4,5,6,7

Environment variables:
    MONGO_URI          — MongoDB connection string
    DO_INFERENCE_KEY   — DO Serverless Inference API key (optional)
    NODE_HOST          — hostname / IP reported to the Coordinator

API
---
    GET  /health                  → {"status": "ok", "free_gpus": N, "free_ports": N}
    POST /jobs                    → submit a job (async; responds immediately with job_id)
    GET  /jobs/{job_id}           → poll job status
    GET  /capacity                → {"free_gpus": N, "free_ports": N, "in_use_ports": [...]}
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agents.do_client import DOClient
from core.config import load_config
from core.db import Database
from core.gpu_allocator import GPUSlotAllocator
from core.port_allocator import PortAllocator
from node.node_worker import NodeWorker, JobRequest, JobResult

log = logging.getLogger("node.server")


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class JobSubmitRequest(BaseModel):
    session_id: str
    config_id: str
    fingerprint: str
    flags: Dict[str, Any]
    context_configs: List[List[int]]   # [[input, output], ...]
    model_id: str
    gpu_type: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str                        # "pending" | "running" | "done" | "failed"
    config_id: str
    fingerprint: str
    best_fitness: Optional[float] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(
    gpu_type: str = "H100",
    gpu_indices: Optional[List[int]] = None,
    port_pool_start: int = 8000,
    port_pool_end: int = 8099,
    startup_timeout_sec: int = 1200,
    node_host: str = "localhost",
) -> FastAPI:
    """
    Build the FastAPI application with all dependencies wired up.
    Call this from the ``if __name__ == "__main__"`` block or from tests.
    """
    app = FastAPI(title="OceanTune Node Server", version="1.0.0")

    # Shared state attached to app
    cfg = load_config()
    db = Database(uri=cfg.database.uri, db_name=cfg.database.name)
    do_client = DOClient.from_env(
        max_tokens=cfg.agent.max_tokens,
        temperature=cfg.agent.temperature,
        timeout_sec=cfg.agent.timeout_sec,
    )
    gpu_alloc = GPUSlotAllocator(
        gpu_indices=gpu_indices or [0],
        gpu_type=gpu_type,
    )
    port_alloc = PortAllocator(start=port_pool_start, end=port_pool_end)
    worker = NodeWorker(
        db=db,
        do_client=do_client,
        gpu_alloc=gpu_alloc,
        port_alloc=port_alloc,
        node_host=node_host,
        startup_timeout_sec=startup_timeout_sec,
        primary_metric=cfg.optimiser.primary_metric,
    )

    # In-memory job registry  job_id → {"status": ..., "result": JobResult|None}
    jobs: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @app.on_event("startup")
    async def _startup() -> None:
        await db.connect()
        log.info(
            "Node server ready: gpu_type=%s indices=%s ports=%d-%d",
            gpu_type, gpu_indices, port_pool_start, port_pool_end,
        )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await db.close()
        await do_client.close()

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "gpu_type": gpu_type,
            "free_gpus": worker.free_gpus,
            "free_ports": worker.free_ports,
        })

    @app.get("/capacity")
    async def capacity() -> JSONResponse:
        return JSONResponse({
            "gpu_type": gpu_type,
            "total_gpus": gpu_alloc.total,
            "free_gpus": worker.free_gpus,
            "free_ports": worker.free_ports,
            "in_use_ports": list(port_alloc.in_use),
        })

    @app.post("/jobs", status_code=202)
    async def submit_job(req: JobSubmitRequest) -> JSONResponse:
        job_id = str(uuid.uuid4())
        context_tuples: List[Tuple[int, int]] = [
            (pair[0], pair[1]) for pair in req.context_configs
        ]
        job_req = JobRequest(
            session_id=req.session_id,
            config_id=req.config_id,
            fingerprint=req.fingerprint,
            flags=req.flags,
            context_configs=context_tuples,
            model_id=req.model_id,
            gpu_type=req.gpu_type,
        )

        jobs[job_id] = {"status": "pending", "config_id": req.config_id,
                        "fingerprint": req.fingerprint, "result": None}

        # Fire-and-forget: run the job in a background task
        async def _run() -> None:
            jobs[job_id]["status"] = "running"
            result: JobResult = await worker.submit(job_req)
            jobs[job_id]["status"] = result.status
            jobs[job_id]["result"] = result

        asyncio.create_task(_run())
        log.info("Job accepted: job_id=%s config_id=%s", job_id, req.config_id)
        return JSONResponse({"job_id": job_id, "status": "pending"}, status_code=202)

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job(job_id: str) -> JobStatusResponse:
        entry = jobs.get(job_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        result: Optional[JobResult] = entry.get("result")
        return JobStatusResponse(
            job_id=job_id,
            status=entry["status"],
            config_id=entry["config_id"],
            fingerprint=entry["fingerprint"],
            best_fitness=result.best_fitness if result else None,
            error=result.error if result else None,
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OceanTune Node Server")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--gpu-indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--port-pool-start", type=int, default=8000)
    parser.add_argument("--port-pool-end", type=int, default=8099)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--node-host", default=os.getenv("NODE_HOST", "localhost"))
    args = parser.parse_args()

    indices = [int(i) for i in args.gpu_indices.split(",")]
    app = create_app(
        gpu_type=args.gpu_type,
        gpu_indices=indices,
        port_pool_start=args.port_pool_start,
        port_pool_end=args.port_pool_end,
        startup_timeout_sec=args.startup_timeout,
        node_host=args.node_host,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
