"""
Chat Router — POST /chat
=========================
Handles user questions and returns RAG-generated answers.

Flow per message:
  1. Receive session_id + user message
  2. Save user message to Firestore
  3. Call RAG chain (embed → retrieve → generate)
  4. Save assistant answer to Firestore
  5. Return answer + sources

Why save to Firestore?
  So the user can see their full conversation history even after
  refreshing the page or coming back later. The frontend calls
  GET /history to reload previous messages.

What is session_id?
  A unique string that identifies one conversation thread.
  The frontend generates it (usually a UUID) when a new chat starts.
  Different users have different session IDs → separate histories.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.rag_chain import ask
from backend.services.firestore_service import save_message

router = APIRouter()


class ChatRequest(BaseModel):
    """What the frontend sends us."""
    session_id: str   # identifies the conversation
    message: str      # the user's question


class SourceChunk(BaseModel):
    """A single retrieved document chunk."""
    text: str
    source: str
    distance: float


class ChatResponse(BaseModel):
    """What we send back to the frontend."""
    answer: str
    sources: list[SourceChunk]
    session_id: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get a RAG-powered response.

    - Accepts: JSON body with session_id and message
    - Returns: answer from Gemini + source chunks used

    Example request body:
        {
            "session_id": "user-abc123",
            "message": "What is supervised learning?"
        }
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    # --- Save user message to Firestore ---
    # Do this BEFORE calling the RAG chain so history is complete even on error
    save_message(
        session_id=request.session_id,
        role="user",
        content=request.message,
    )

    # --- Run RAG pipeline ---
    # This embeds the question, retrieves chunks, and calls Gemini
    result = ask(question=request.message)

    # --- Save assistant answer to Firestore ---
    save_message(
        session_id=request.session_id,
        role="assistant",
        content=result["answer"],
    )

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
        session_id=request.session_id,
    )