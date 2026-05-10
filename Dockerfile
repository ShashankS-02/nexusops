# NexusOps API — Production Dockerfile
# Multi-stage build to minimize final image size

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Production Runtime ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Labels for traceability
LABEL org.opencontainers.image.title="NexusOps API"
LABEL org.opencontainers.image.description="Autonomous AI-SRE FastAPI Gateway"
LABEL org.opencontainers.image.source="https://github.com/your-org/nexusops"

# Create non-root user (security best practice)
RUN groupadd --gid 1001 nexusops && \
    useradd --uid 1001 --gid nexusops --shell /bin/bash --create-home nexusops

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY --chown=nexusops:nexusops nexusops/ ./nexusops/
COPY --chown=nexusops:nexusops agents/ ./agents/
COPY --chown=nexusops:nexusops api/ ./api/
COPY --chown=nexusops:nexusops ml/ ./ml/
COPY --chown=nexusops:nexusops simulator/ ./simulator/

# Create directories for model artifacts
RUN mkdir -p ml/pytorch/saved_models ml/tensorflow/saved_models && \
    chown -R nexusops:nexusops ml/

USER nexusops

# Expose port
EXPOSE 8000

# Health check (Kubernetes liveness probe compatible)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/v1/health')" || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--access-log"]