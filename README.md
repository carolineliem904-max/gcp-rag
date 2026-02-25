# GCP RAG Chatbot

A hands-on learning project: a Retrieval-Augmented Generation (RAG) chatbot
built on Google Cloud Platform, step by step.

## Stack
- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit
- **LLM**: Gemini 1.5 Flash via Vertex AI
- **Embeddings**: Vertex AI text-embedding-004
- **Vector DB**: ChromaDB (local) → Vertex AI Vector Search (later)
- **Chat History**: Firestore
- **Document Storage**: Cloud Storage
- **Deployment**: Cloud Run + Docker + Artifact Registry

## Project Phases
| Phase | Focus |
|-------|-------|
| 1 | Project Setup & GCP Configuration |
| 2 | Document Pipeline (upload, embed, store) |
| 3 | RAG Chain (retrieve + generate) |
| 4 | FastAPI Backend |
| 5 | Streamlit Frontend |
| 6 | Dockerize |
| 7 | Deploy to GCP Cloud Run |
| 8 | Observability & Cleanup |

## Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd gcp-rag

# 2. Copy env template and fill in your values
cp .env.example .env
# Edit .env with your GCP project ID and settings

# 3. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify GCP connection
python scripts/test_vertex_ai.py
```

## Security Notes
- Never commit `.env` or `service-account-key.json`
- Both are excluded in `.gitignore`
