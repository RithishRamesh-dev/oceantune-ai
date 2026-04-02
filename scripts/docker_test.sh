#!/usr/bin/env bash
# =============================================================================
# scripts/docker_test.sh
# OceanTune AI — Droplet bootstrap + test runner
#
# Run this ONCE on a fresh DigitalOcean GPU Droplet.
# Installs Docker + NVIDIA Container Toolkit, clones the repo,
# builds the image, and runs the full test suite inside the container.
#
# Usage:
#   ssh root@YOUR_DROPLET_IP
#   export REPO_URL=https://github.com/RithishRamesh-dev/oceantune-ai.git
#   bash scripts/docker_test.sh
#
#   # Or one-shot via SSH (after setting REPO_URL on your Mac):
#   ssh root@YOUR_DROPLET_IP "export REPO_URL=... && bash -s" < scripts/docker_test.sh
#
# Environment variables:
#   REPO_URL          Git repo URL            (required)
#   REPO_BRANCH       Branch to clone         (default: main)
#   HF_TOKEN          Hugging Face token      (optional)
#   DO_SPACES_KEY     DigitalOcean Spaces key (optional)
#   DO_SPACES_SECRET  DigitalOcean Spaces secret
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
REPO_URL="${REPO_URL:-}"
REPO_BRANCH="${REPO_BRANCH:-main}"
CLONE_DIR="${CLONE_DIR:-/opt/oceantune-ai}"
IMAGE_NAME="oceantune-ai:latest"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}✅ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
fail() { echo -e "${RED}❌ $*${NC}"; exit 1; }

[ -z "$REPO_URL" ] && fail "REPO_URL is not set. Export it before running:\n  export REPO_URL=https://github.com/RithishRamesh-dev/oceantune-ai.git"

log "=== OceanTune AI — Droplet Setup & Test Runner ==="
log "Repo    : $REPO_URL  (branch: $REPO_BRANCH)"
log "Clone to: $CLONE_DIR"
log "Image   : $IMAGE_NAME"

# ── 1. Install Docker ─────────────────────────────────────────────────────────
if command -v docker &>/dev/null; then
    ok "Docker already installed: $(docker --version)"
else
    log "Installing Docker..."
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable --now docker
    ok "Docker installed: $(docker --version)"
fi

# ── 2. Install NVIDIA Container Toolkit ───────────────────────────────────────
if nvidia-smi &>/dev/null; then
    if dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then
        ok "NVIDIA Container Toolkit already installed"
    else
        log "Installing NVIDIA Container Toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt-get update -qq
        apt-get install -y -qq nvidia-container-toolkit
        nvidia-ctk runtime configure --runtime=docker
        systemctl restart docker
        ok "NVIDIA Container Toolkit installed"
    fi
    log "GPU info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        | while IFS= read -r line; do log "  GPU: $line"; done
else
    warn "nvidia-smi not found — GPU tests will be skipped (CPU-only mode)"
fi

# ── 3. Clone / update the repo ────────────────────────────────────────────────
if [ -d "$CLONE_DIR/.git" ]; then
    log "Repo already cloned — pulling latest..."
    git -C "$CLONE_DIR" fetch origin
    git -C "$CLONE_DIR" checkout "$REPO_BRANCH"
    git -C "$CLONE_DIR" pull origin "$REPO_BRANCH"
    ok "Repo updated to $(git -C "$CLONE_DIR" rev-parse --short HEAD)"
else
    log "Cloning $REPO_URL ..."
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$CLONE_DIR"
    ok "Repo cloned: $(git -C "$CLONE_DIR" rev-parse --short HEAD)"
fi

cd "$CLONE_DIR"

# ── 4. Write .env if secrets are available ────────────────────────────────────
if [ ! -f .env ]; then
    log "Creating .env ..."
    cat > .env <<EOF
HF_TOKEN=${HF_TOKEN:-}
DO_SPACES_KEY=${DO_SPACES_KEY:-}
DO_SPACES_SECRET=${DO_SPACES_SECRET:-}
EOF
    ok ".env created"
else
    ok ".env already exists"
fi

# ── 5. Build Docker image ─────────────────────────────────────────────────────
log "Building Docker image $IMAGE_NAME ..."
log "(First run pulls vllm/vllm-openai:latest — may take 5-10 min)"

docker build --tag "$IMAGE_NAME" --file Dockerfile --progress=plain . 2>&1 | tail -20

ok "Docker image built: $IMAGE_NAME"

# ── 6. Verify CUDA inside container ───────────────────────────────────────────
if nvidia-smi &>/dev/null; then
    log "Checking CUDA inside container..."
    docker run --rm --gpus all "$IMAGE_NAME" \
        python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    ok "CUDA check passed"
else
    warn "No GPU — skipping CUDA check"
fi

# ── 7. Run the test suite ─────────────────────────────────────────────────────
log "Running full test suite inside container..."
echo ""

docker run \
    --rm \
    --name oceantune-tests \
    --env-file .env \
    --volume "$CLONE_DIR/storage:/workspace/oceantune-ai/storage" \
    "$IMAGE_NAME" \
    python3 -m pytest tests/ -v --tb=short --asyncio-mode=auto

TEST_EXIT=$?
echo ""

[ $TEST_EXIT -eq 0 ] && ok "=== All tests PASSED ===" || fail "=== Tests FAILED (exit $TEST_EXIT) ==="

# ── 8. Summary ────────────────────────────────────────────────────────────────
echo ""
log "=== Setup complete ==="
log "Repo : $CLONE_DIR"
log "Image: $IMAGE_NAME"
log ""
log "Day-to-day commands (from $CLONE_DIR):"
log "  git pull && docker compose run --rm tests          # pull + test"
log "  docker compose run --rm tests-gpu                  # test with GPU"
log "  docker compose up vllm-server                      # launch vLLM"
log "  docker compose run --rm optimizer                  # full pipeline"
