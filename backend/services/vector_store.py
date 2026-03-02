"""
Vector Store Service
====================
Stores and searches document embeddings using ChromaDB.

What is ChromaDB?
  A local vector database — it stores text chunks + their embeddings,
  and lets you find the most similar chunks to a given query vector.

How it works:
  1. You add chunks with their vectors:
       "Machine learning is..." → [0.02, -0.08, 0.14, ...]

  2. When a user asks a question, you embed the question:
       "What is ML?"            → [0.02, -0.07, 0.13, ...]

  3. ChromaDB finds the stored chunks with the closest vectors:
       → returns "Machine learning is..." (very similar!)

  This is called "nearest neighbor search" or "similarity search".

Session isolation:
  Each upload is tagged with a session_id in metadata.
  Search and delete operations filter by session_id so users only
  see results from their own uploaded documents.

Persistence:
  ChromaDB saves data to disk (./chroma_db/ by default).
  The data survives between script runs — you embed once, search many times.
"""

import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Where ChromaDB stores its data on disk
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Name of the collection (like a table in a regular database)
COLLECTION_NAME = "documents"


def get_collection() -> chromadb.Collection:
    """
    Get (or create) the ChromaDB collection where we store document chunks.

    PersistentClient saves data to disk so it survives between runs.
    get_or_create_collection is safe to call multiple times — won't overwrite.

    The 'cosine' distance metric means we measure similarity by angle,
    not raw distance — this works better for text embeddings.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for text
    )
    return collection


def add_documents(
    chunks: list[str],
    embeddings: list[list[float]],
    source_filename: str,
    session_id: str,
) -> None:
    """
    Add text chunks and their embeddings to ChromaDB, tagged by session.

    Each chunk gets:
      - A unique ID (session_id + source filename + chunk index)
      - The embedding vector (768 floats)
      - The original text (stored so we can return it later)
      - Metadata: source filename, chunk index, session_id

    The session_id prefix in the ID ensures two sessions uploading
    the same filename don't overwrite each other.

    Args:
        chunks:          list of text strings (the actual content)
        embeddings:      list of vectors, one per chunk
        source_filename: name of the original document (for tracking)
        session_id:      the session that uploaded this document
    """
    collection = get_collection()

    # Composite ID: session_id prefix prevents cross-session collisions
    ids = [f"{session_id}__{source_filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": source_filename, "chunk_index": i, "session_id": session_id}
        for i in range(len(chunks))
    ]

    # ChromaDB upsert: insert if new, update if ID already exists
    # This means re-uploading the same file in the same session won't duplicate
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    print(f"  Stored {len(chunks)} chunks from '{source_filename}' (session: {session_id})")


def search(query_embedding: list[float], session_id: str, n_results: int = 3) -> list[dict]:
    """
    Find the most relevant chunks for a given query, scoped to a session.

    Uses ChromaDB's 'where' filter so only documents uploaded by this
    session are considered — different sessions are completely isolated.

    Args:
        query_embedding: the vector for the user's question
        session_id:      only search documents from this session
        n_results:       how many chunks to return (top-k)

    Returns:
        List of dicts, each with:
          - 'text':     the chunk content
          - 'source':   which document it came from
          - 'distance': similarity score (lower = more similar in cosine space)

    Returns [] if no documents are uploaded for this session yet.
    """
    collection = get_collection()

    # Check if this session has any documents at all — ChromaDB raises an
    # error if you query with n_results > number of matching items = 0.
    session_docs = collection.get(
        where={"session_id": session_id},
        include=[],
    )
    if not session_docs["ids"]:
        return []

    # Cap n_results to how many chunks actually exist for this session
    actual_n = min(n_results, len(session_docs["ids"]))

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=actual_n,
        where={"session_id": session_id},  # only this session's documents
        include=["documents", "metadatas", "distances"],
    )

    # Reformat into a clean list of dicts
    output = []
    for i in range(len(results["documents"][0])):
        output.append({
            "text":     results["documents"][0][i],
            "source":   results["metadatas"][0][i].get("source", "unknown"),
            "distance": round(results["distances"][0][i], 4),
        })

    return output


def list_documents(session_id: str) -> list[str]:
    """
    Return a list of unique document filenames uploaded by a session.

    Args:
        session_id: the session to list documents for

    Returns:
        Sorted list of unique source filenames, e.g. ["guide.pdf", "notes.txt"]
    """
    collection = get_collection()

    results = collection.get(
        where={"session_id": session_id},
        include=["metadatas"],
    )

    # Extract unique filenames from metadata
    seen = set()
    for meta in results["metadatas"]:
        seen.add(meta.get("source", "unknown"))

    return sorted(seen)


def delete_document(session_id: str, source_filename: str) -> int:
    """
    Delete all chunks for a specific document within a session.

    Args:
        session_id:      the session that owns the document
        source_filename: the document filename to delete

    Returns:
        Number of chunks deleted
    """
    collection = get_collection()

    # Find all chunk IDs for this session + filename
    results = collection.get(
        where={
            "$and": [
                {"session_id": {"$eq": session_id}},
                {"source": {"$eq": source_filename}},
            ]
        },
        include=[],
    )

    ids_to_delete = results["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f"  Deleted {len(ids_to_delete)} chunks for '{source_filename}' (session: {session_id})")

    return len(ids_to_delete)


def get_collection_stats() -> dict:
    """
    Return basic stats about what's stored in ChromaDB.
    Useful for debugging and verification.
    """
    collection = get_collection()
    count = collection.count()
    return {"total_chunks": count, "collection_name": COLLECTION_NAME}


def clear_collection() -> None:
    """
    Delete all documents from the collection.
    Useful when re-processing documents from scratch.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    client.delete_collection(name=COLLECTION_NAME)
    print(f"  Cleared ChromaDB collection '{COLLECTION_NAME}'")