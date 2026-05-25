#!/bin/bash
# ============================================================
# NexusOps Phase 1 — Local Development Setup Script
# Run this once to get everything working end-to-end
# ============================================================

set -e  # Exit on any error

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

header() { echo -e "\n${CYAN}══════════════════════════════════════════${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}══════════════════════════════════════════${NC}"; }
ok()     { echo -e "${GREEN}  ✓ $1${NC}"; }
warn()   { echo -e "${YELLOW}  ⚠ $1${NC}"; }
info()   { echo -e "  → $1"; }

header "NexusOps — Phase 1 Setup"

# ── Step 1: Python virtual environment ───────────────────────────────────────
header "1/6 — Python Environment"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    ok "Virtual environment created"
else
    warn "venv already exists, skipping creation"
fi
source venv/bin/activate
ok "Virtual environment activated"

# ── Step 2: Install dependencies ─────────────────────────────────────────────
header "2/6 — Installing Dependencies"
pip install --upgrade pip -q
info "Installing PyTorch (CPU version for local dev)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu -q
info "Installing TensorFlow..."
pip install tensorflow -q
info "Installing remaining requirements..."
pip install -r requirements.txt -q
pip install pytest pytest-cov pytest-asyncio httpx ruff -q
ok "All dependencies installed"

# ── Step 3: Environment file ──────────────────────────────────────────────────
header "3/6 — Environment Configuration"
if [ ! -f ".env" ]; then
    cp .env.example .env
    warn ".env created from template — please fill in your API keys!"
    info "Required: OPENAI_API_KEY, LANGSMITH_API_KEY, TAVILY_API_KEY"
else
    ok ".env already exists"
fi

# ── Step 4: Start local services ─────────────────────────────────────────────
header "4/6 — Starting Local Services (Docker)"
if command -v docker &>/dev/null; then
    docker compose up -d
    info "Waiting for PostgreSQL to be ready..."
    sleep 5
    ok "Services started: PostgreSQL, Redis, MLflow, Qdrant"
    info "MLflow UI: http://localhost:5001"
    info "Qdrant UI: http://localhost:6333/dashboard"
else
    warn "Docker not found — skipping service startup"
    warn "Install Docker Desktop: https://docs.docker.com/desktop/mac/"
fi

# ── Step 5: Run tests ─────────────────────────────────────────────────────────
header "5/6 — Running Tests"
python -m pytest tests/ -v --tb=short -q 2>&1 | tail -20
ok "Tests complete"

# ── Step 6: Quick demo ────────────────────────────────────────────────────────
header "6/6 — Quick Demo"
echo ""
echo -e "  ${CYAN}To train the PyTorch anomaly model:${NC}"
echo -e "  ${YELLOW}python -m ml.pytorch.train${NC}"
echo ""
echo -e "  ${CYAN}To train the TensorFlow log classifier:${NC}"
echo -e "  ${YELLOW}python -m ml.tensorflow.train${NC}"
echo ""
echo -e "  ${CYAN}To run the metrics simulator (dry run):${NC}"
echo -e "  ${YELLOW}python -m simulator.generator --scenario cpu_spike${NC}"
echo ""
echo -e "  ${CYAN}To start the FastAPI gateway:${NC}"
echo -e "  ${YELLOW}uvicorn api.main:app --reload --port 8000${NC}"
echo -e "  ${YELLOW}# Then visit: http://localhost:8000/docs${NC}"
echo ""
echo -e "  ${CYAN}To run the simulator sending metrics to the API:${NC}"
echo -e "  ${YELLOW}python -m simulator.generator --scenario cpu_spike --no-dry-run${NC}"
echo ""
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ NexusOps Phase 1 setup complete!      ${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
