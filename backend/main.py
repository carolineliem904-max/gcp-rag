"""
FastAPI Application Entry Point
================================
This is the main file that starts the web server.

FastAPI is a modern Python web framework that:
  - Automatically generates interactive API docs (Swagger UI at /docs)
  - Validates request and response data using Pydantic
  - Is fast and easy to read

How routing works:
  We split endpoints into separate "router" files (upload.py, chat.py, history.py)
  and register them here. This keeps the code organized as the app grows.

  main.py (entry point)
    └── includes routers:
          ├── POST /upload   (upload.py)
          ├── POST /chat     (chat.py)
          ├── GET  /history  (history.py)
          └── DELETE /history (history.py)

CORS (Cross-Origin Resource Sharing):
  Browsers block requests from one origin (e.g. localhost:8501 Streamlit)
  to a different origin (e.g. localhost:8000 FastAPI) by default.
  We add CORSMiddleware to allow this — required for the frontend to work.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import upload, chat, history

# --- Create the FastAPI app ---
app = FastAPI(
    title="GCP RAG Chatbot API",
    description="A RAG chatbot powered by Vertex AI, ChromaDB, and Gemini.",
    version="1.0.0",
)

# --- CORS Middleware ---
# Allows the Streamlit frontend (running on a different port) to call this API.
# In production on Cloud Run, we'll restrict this to the actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins for local dev
    allow_methods=["*"],   # allow GET, POST, DELETE, etc.
    allow_headers=["*"],   # allow all headers
)

# --- Register routers ---
# Each router brings its own endpoints into the app.
app.include_router(upload.router, tags=["Documents"])
app.include_router(chat.router,   tags=["Chat"])
app.include_router(history.router, tags=["History"])


# --- Health check endpoint ---
# Cloud Run calls this to verify the service is alive.
# Also useful to quickly test if the server is running.
@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok", "service": "rag-chatbot-backend"}


# --- Root endpoint ---
@app.get("/", tags=["System"])
async def root():
    return {
        "message": "GCP RAG Chatbot API is running.",
        "docs": "/docs",
        "health": "/health",
    }