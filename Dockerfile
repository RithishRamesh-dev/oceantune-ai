# OceanTune AI — Dockerfile
#
# Extends the official vLLM OpenAI-compatible server image.
# vLLM, CUDA, PyTorch, and all GPU drivers are already included.
# We layer our repo and test dependencies on top.
#
# Build:
#   docker build -t oceantune-ai .
#
# Run tests:
#   docker run --rm --gpus all oceantune-ai pytest tests/ -v
#
# Run server (interactive):
#   docker run --rm --gpus all -p 8000:8000 oceantune-ai \
#       python oceantune.py run --model mistral

FROM vllm/vllm-openai:latest

LABEL maintainer="OceanTune AI"
LABEL description="vLLM inference optimisation system"

# ── System deps ────────────────────────────────────────────────────────────
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        git \
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /workspace/oceantune-ai

# ── Python test dependencies ───────────────────────────────────────────────
# Only install packages not already provided by the vLLM base image.
# We do NOT re-pin pydantic/numpy/openai/httpx — the base image has
# compatible versions already. Pinning them here would downgrade vLLM's deps.
RUN pip install --no-cache-dir \
        pyyaml \
        python-dotenv \
        structlog \
        boto3 \
        pytest==8.2.0 \
        pytest-asyncio==0.23.6

# ── Copy repo ──────────────────────────────────────────────────────────────
COPY . .

# ── Storage directories (tests write logs here) ────────────────────────────
RUN mkdir -p storage/logs storage/results

# ── Smoke test at build time ───────────────────────────────────────────────
# Validates imports and config loading work inside the container.
# Runs without GPU — pure Python only.
RUN python3 oceantune.py validate-config

# ── Clear base image entrypoint (vllm/vllm-openai sets ENTRYPOINT ["vllm"]) ──
ENTRYPOINT []

# ── Default command: run the full test suite ───────────────────────────────
CMD ["python3", "-m", "pytest", "tests/", "-v", "--tb=short", "--asyncio-mode=auto"]
