"""
Firestore Service
=================
Handles storing and retrieving chat history in Google Cloud Firestore.

What is Firestore?
  A NoSQL cloud database from Google. Unlike SQL databases (rows + tables),
  Firestore stores data as "documents" inside "collections" — like JSON files
  organized in folders.

Why Firestore for chat history?
  - Serverless: no database server to manage
  - Scales automatically
  - Free tier: 50,000 reads + 20,000 writes per day
  - Works great for storing chat sessions (nested JSON structure)

Data structure in Firestore:
  Collection: "chat_history"
      Document: "session-abc123"           ← one per chat session
          Field: messages (array)
              [0]: { role: "user",      content: "What is ML?",  timestamp: "..." }
              [1]: { role: "assistant", content: "ML is ...",     timestamp: "..." }
              [2]: { role: "user",      content: "Tell me more.", timestamp: "..." }

A session_id is like a conversation thread ID. The frontend generates one
per browser session so each user has their own isolated chat history.
"""

import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.cloud import firestore

load_dotenv()

COLLECTION_NAME = os.getenv("FIRESTORE_COLLECTION", "chat_history")


def get_db() -> firestore.Client:
    """
    Create and return a Firestore client.
    Authentication via Application Default Credentials (ADC).
    """
    return firestore.Client()


def save_message(session_id: str, role: str, content: str) -> None:
    """
    Append a single message to a session's chat history.

    Args:
        session_id: unique ID for the conversation (e.g. "session-abc123")
        role:       "user" or "assistant"
        content:    the message text

    How it works:
        Firestore's ArrayUnion adds to an array without overwriting existing items.
        If the document doesn't exist yet, merge=True creates it automatically.
    """
    db = get_db()

    message = {
        "role": role,
        "content": content,
        # Store timestamp as ISO string — easy to sort and display
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Get or create the document for this session, then append the message
    doc_ref = db.collection(COLLECTION_NAME).document(session_id)
    doc_ref.set(
        {"messages": firestore.ArrayUnion([message])},
        merge=True,  # don't overwrite existing messages, just append
    )


def get_history(session_id: str) -> list[dict]:
    """
    Retrieve all messages for a given session, in chronological order.

    Args:
        session_id: the conversation ID to look up

    Returns:
        List of message dicts: [{"role": ..., "content": ..., "timestamp": ...}]
        Returns empty list if session doesn't exist yet.
    """
    db = get_db()

    doc = db.collection(COLLECTION_NAME).document(session_id).get()

    if not doc.exists:
        return []

    data = doc.to_dict()
    messages = data.get("messages", [])

    # Sort by timestamp to ensure chronological order
    messages.sort(key=lambda m: m.get("timestamp", ""))

    return messages


def delete_session(session_id: str) -> None:
    """
    Delete all chat history for a session. Useful for "clear chat" feature.

    Args:
        session_id: the session to delete
    """
    db = get_db()
    db.collection(COLLECTION_NAME).document(session_id).delete()