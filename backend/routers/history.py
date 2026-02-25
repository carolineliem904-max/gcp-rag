"""
History Router — GET /history
===============================
Returns the full chat history for a given session.

The frontend calls this when:
  - The page loads (to restore previous conversation)
  - After a new message is sent (to refresh the display)

Firestore stores messages as an array inside a session document.
We read that array and return it sorted by timestamp.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.firestore_service import get_history, delete_session

router = APIRouter()


class Message(BaseModel):
    """A single message in the conversation."""
    role: str        # "user" or "assistant"
    content: str     # message text
    timestamp: str   # ISO format datetime string


class HistoryResponse(BaseModel):
    """What we return for a history request."""
    session_id: str
    messages: list[Message]
    total_messages: int


@router.get("/history", response_model=HistoryResponse)
async def get_chat_history(session_id: str):
    """
    Get the full chat history for a session.

    - Accepts: ?session_id=xxx as a query parameter
    - Returns: list of messages in chronological order

    Example: GET /history?session_id=user-abc123
    """
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    messages = get_history(session_id)

    return HistoryResponse(
        session_id=session_id,
        messages=[Message(**m) for m in messages],
        total_messages=len(messages),
    )


@router.delete("/history")
async def clear_chat_history(session_id: str):
    """
    Delete all chat history for a session.

    - Accepts: ?session_id=xxx as a query parameter
    - Returns: confirmation message

    Example: DELETE /history?session_id=user-abc123
    """
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    delete_session(session_id)

    return {"message": f"Chat history for session '{session_id}' deleted."}