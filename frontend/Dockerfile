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

# ── Step 3: Expose port ──
# Cloud Run injects $PORT (default 8080). Streamlit must listen on it.
EXPOSE 8080

# ── Step 4: Start Streamlit ──
# Shell form so $PORT is expanded at runtime.
# Cloud Run sets PORT=8080; locally we fall back to 8501.
CMD streamlit run frontend/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true