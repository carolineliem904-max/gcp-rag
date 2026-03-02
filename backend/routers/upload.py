"""
Upload Router — POST /upload
=============================
Handles document upload and processing.

Flow when a user uploads a file:
  1. FastAPI receives the file + session_id via multipart form upload
  2. We save it to a temp location on disk
  3. Upload original file to Cloud Storage under documents/{session_id}/{filename}
  4. Extract text from the file
  5. Split text into chunks
  6. Generate embeddings for each chunk (Vertex AI call)
  7. Store chunks + embeddings in ChromaDB, tagged with session_id
  8. Delete the temp file
  9. Return a success response

Why Cloud Storage AND ChromaDB?
  - Cloud Storage = stores the original file (PDF/TXT) permanently
  - ChromaDB = stores the processed vectors for fast similarity search
  These serve different purposes and are both needed.

Why session_id?
  Each session gets its own document namespace. Documents uploaded by
  session A are never visible to session B's searches.
"""

import os
import tempfile
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

from backend.services.storage import upload_file
from backend.utils.pdf_parser import extract_text, chunk_text
from backend.services.embeddings import get_embeddings
from backend.services.vector_store import add_documents, get_collection_stats

# APIRouter lets us group related endpoints.
# In main.py we'll attach this router with a prefix like /upload.
router = APIRouter()


class UploadResponse(BaseModel):
    """What we return after a successful upload."""
    message: str
    filename: str
    chunks_stored: int
    total_chunks_in_db: int


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """
    Upload a PDF or TXT file, process it, and store in the vector database.

    - Accepts: multipart/form-data with 'file' and 'session_id' fields
    - Returns: upload confirmation with chunk count

    You can test this in Swagger UI at http://localhost:8000/docs
    """
    # --- Validate file type ---
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Please upload a .pdf or .txt file."
        )

    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    # --- Save to a temporary file ---
    # UploadFile is a stream — we must save it to disk before processing.
    # tempfile.NamedTemporaryFile creates a file in /tmp that auto-deletes.
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # --- Upload original to Cloud Storage ---
        # Store under documents/{session_id}/ so each session's files are separate
        gcs_blob_name = f"documents/{session_id}/{filename}"
        upload_file(
            local_file_path=tmp_path,
            destination_blob_name=gcs_blob_name,
        )

        # --- Extract text ---
        text = extract_text(tmp_path)
        if not text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from file. The file may be empty or image-based."
            )

        # --- Chunk the text ---
        chunks = chunk_text(text, chunk_size=500, overlap=50)

        # --- Generate embeddings ---
        embeddings = get_embeddings(chunks)

        # --- Store in ChromaDB with session_id ---
        add_documents(chunks, embeddings, source_filename=filename, session_id=session_id)

        # --- Get total DB stats ---
        stats = get_collection_stats()

        return UploadResponse(
            message=f"'{filename}' uploaded and processed successfully.",
            filename=filename,
            chunks_stored=len(chunks),
            total_chunks_in_db=stats["total_chunks"],
        )

    finally:
        # Always clean up the temp file, even if something went wrong
        if os.path.exists(tmp_path):
            os.remove(tmp_path)