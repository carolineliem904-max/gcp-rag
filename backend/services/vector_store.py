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
) -> None:
    """
    Add text chunks and their embeddings to ChromaDB.

    Each chunk gets:
      - A unique ID (source filename + chunk index)
      - The embedding vector (768 floats)
      - The original text (stored so we can return it later)
      - Metadata (source file name, chunk index)

    Args:
        chunks:          list of text strings (the actual content)
        embeddings:      list of vectors, one per chunk
        source_filename: name of the original document (for tracking)
    """
    collection = get_collection()

    # Build IDs, embeddings, documents, and metadata lists
    ids = [f"{source_filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": source_filename, "chunk_index": i}
        for i in range(len(chunks))
    ]

    # ChromaDB upsert: insert if new, update if ID already exists
    # This means re-uploading the same file won't create duplicates
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    print(f"  Stored {len(chunks)} chunks from '{source_filename}' in ChromaDB")


def search(query_embedding: list[float], n_results: int = 3) -> list[dict]:
    """
    Find the most relevant chunks for a given query embedding.

    Args:
        query_embedding: the vector for the user's question
        n_results:       how many chunks to return (top-k)

    Returns:
        List of dicts, each with:
          - 'text':     the chunk content
          - 'source':   which document it came from
          - 'distance': similarity score (lower = more similar in cosine space)

    Example:
        query_vec = get_single_embedding("What is machine learning?")
        results = search(query_vec, n_results=3)
        for r in results:
            print(r['text'])
    """
    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],  # wrap in list — ChromaDB expects batch
        n_results=n_results,
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