# ── Frontend Dockerfile ──────────────────────────────────────────────────────
#
# Packages the Streamlit app into a container image.
#
# Much simpler than the backend — only needs streamlit + requests.
# No GCP SDK here: the frontend never talks to GCP directly.
# It only talks to the backend container via HTTP.
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# ── Step 1: Install dependencies ──
COPY frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Step 2: Copy frontend source code ──
COPY frontend/ ./frontend/

# ── Step 3: Expose Streamlit's default port ──
EXPOSE 8501

# ── Step 4: Start Streamlit ──
# --server.address=0.0.0.0  → accept connections from outside the container
# --server.port=8501         → explicit port
# --server.headless=true     → don't try to open a browser (no browser in container)
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]