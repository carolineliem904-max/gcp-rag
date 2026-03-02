"""
Documents Router — GET /documents, DELETE /documents/{filename}
===============================================================
Lists and deletes documents for a specific session.

Endpoints:
  GET  /documents?session_id=xxx
       → Returns a list of filenames uploaded by this session.

  DELETE /documents/{filename}?session_id=xxx
       → Deletes the document from ChromaDB (vectors) and GCS (original file).

Why two storage locations to delete from?
  - ChromaDB holds the embedded chunks used for search — delete these so
    the document no longer affects any chat responses.
  - GCS holds the original PDF/TXT file — delete this to keep storage clean.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from google.cloud.exceptions import NotFound

from backend.services.vector_store import list_documents, delete_document
from backend.services.storage import delete_file

router = APIRouter()


class DocumentListResponse(BaseModel):
    """Response for GET /documents."""
    session_id: str
    documents: list[str]
    total: int


class DeleteResponse(BaseModel):
    """Response for DELETE /documents/{filename}."""
    message: str
    filename: str
    chunks_deleted: int


@router.get("/documents", response_model=DocumentListResponse)
async def get_documents(session_id: str):
    """
    List all documents uploaded by a session.

    Returns the unique filenames stored in ChromaDB for this session_id.
    Useful for the frontend to show which documents are currently active.

    Example:
        GET /documents?session_id=session-abc123
        → {"session_id": "session-abc123", "documents": ["guide.pdf", "notes.txt"], "total": 2}
    """
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    docs = list_documents(session_id)
    return DocumentListResponse(
        session_id=session_id,
        documents=docs,
        total=len(docs),
    )


@router.delete("/documents/{filename}", response_model=DeleteResponse)
async def remove_document(filename: str, session_id: str):
    """
    Delete a document from this session's knowledge base.

    Steps:
      1. Delete all ChromaDB chunks where session_id + source match
      2. Delete the original file from GCS

    The document will no longer appear in search results or affect
    future chat responses for this session.

    Example:
        DELETE /documents/guide.pdf?session_id=session-abc123
    """
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    # --- Step 1: Delete from ChromaDB ---
    chunks_deleted = delete_document(session_id=session_id, source_filename=filename)

    if chunks_deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found for this session."
        )

    # --- Step 2: Delete from GCS ---
    gcs_blob_name = f"documents/{session_id}/{filename}"
    try:
        delete_file(gcs_blob_name)
    except NotFound:
        # GCS file missing is non-fatal — vectors are already gone, which is what matters
        pass

    return DeleteResponse(
        message=f"'{filename}' deleted successfully.",
        filename=filename,
        chunks_deleted=chunks_deleted,
    )