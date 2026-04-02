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
# Copy requirements first so Docker layer-caches the pip install
COPY requirements.txt .

# Skip vllm — already in base image
RUN pip install --no-cache-dir \
        pyyaml==6.0.1 \
        python-dotenv==1.0.1 \
        pydantic==2.7.1 \
        httpx==0.27.0 \
        requests==2.32.2 \
        pandas==2.2.2 \
        numpy==1.26.4 \
        boto3==1.34.107 \
        rich==13.7.1 \
        structlog==24.1.0 \
        openai==1.30.1 \
        click==8.1.7 \
        pytest==8.2.0 \
        pytest-asyncio==0.23.6

# ── Copy repo ──────────────────────────────────────────────────────────────
COPY . .

# ── Storage directories (tests write logs here) ────────────────────────────
RUN mkdir -p storage/logs storage/results

# ── Smoke test at build time ───────────────────────────────────────────────
# Validates imports and config loading work inside the container.
# Runs without GPU — pure Python only.
RUN python oceantune.py validate-config

# ── Default command: run the full test suite ───────────────────────────────
CMD ["pytest", "tests/", "-v", "--tb=short", "--asyncio-mode=auto"]
