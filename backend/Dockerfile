# ── Backend Dockerfile ──────────────────────────────────────────────────────
#
# Packages the FastAPI app into a container image.
#
# Build context is the PROJECT ROOT (not backend/), because our Python
# imports use "from backend.services..." — the container needs the backend/
# folder to be one level below the working directory, just like locally.
#
# Why python:3.11-slim?
#   "slim" removes docs, tests, and build tools we don't need at runtime.
#   Result: ~150 MB image instead of ~900 MB for the full image.
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set working directory — all paths inside the container are relative to this
WORKDIR /app

# ── Step 1: Install dependencies ──
# Copy requirements BEFORE source code so Docker can cache this layer.
# If you only change source code (not requirements.txt), Docker reuses
# this cached layer and skips the slow pip install step on rebuild.
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Step 2: Copy backend source code ──
# Copies backend/ into /app/backend/ so imports like
# "from backend.services.rag_chain import ask" resolve correctly.
COPY backend/ ./backend/

# ── Step 3: Expose port ──
# Cloud Run injects $PORT (default 8080) and expects the container to listen on it.
# Locally (docker-compose) PORT is not set, so we default to 8000.
EXPOSE 8080

# ── Step 4: Start command ──
# Shell form (not JSON array) so $PORT is expanded at runtime.
# Cloud Run sets PORT=8080; locally we fall back to 8000.
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}