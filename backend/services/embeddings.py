"""
Embeddings Service
==================
Converts text into vectors (embeddings) using Vertex AI text-embedding-004.

What is an embedding?
  A list of numbers (e.g. 768 floats) that represents the *meaning* of text.
  Similar meanings → similar numbers. This is how semantic search works.

  "What is AI?"        → [0.02, -0.08, 0.14, ...]
  "Define AI"          → [0.02, -0.07, 0.13, ...]  ← very similar!
  "I love pizza"       → [0.91,  0.33, -0.52, ...] ← very different

Why batch?
  - Each API call has some overhead (network latency, auth)
  - Sending 10 texts in one call is much faster than 10 separate calls
  - Vertex AI allows up to 250 texts per request
  - We use batches of 20 to stay safe
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Batch size: how many text chunks to embed in one API call
BATCH_SIZE = 20


def get_genai_client() -> genai.Client:
    """
    Create and return a Gen AI client connected to Vertex AI.
    Authentication via Application Default Credentials (ADC).
    """
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into a list of embedding vectors.

    Args:
        texts: list of strings to embed
               e.g. ["chunk 1 text...", "chunk 2 text...", ...]

    Returns:
        list of float vectors, one per input text
        e.g. [[0.02, -0.08, ...], [0.14, 0.33, ...], ...]

    Example:
        chunks = ["Machine learning is...", "Neural networks are..."]
        vectors = get_embeddings(chunks)
        # vectors[0] is the embedding for "Machine learning is..."
        # vectors[1] is the embedding for "Neural networks are..."
    """
    client = get_genai_client()
    all_embeddings = []

    # Process in batches to avoid hitting API limits
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        # Send this batch to Vertex AI
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
        )

        # Extract the float vector from each embedding result
        batch_vectors = [embedding.values for embedding in result.embeddings]
        all_embeddings.extend(batch_vectors)

        print(f"  Embedded batch {i // BATCH_SIZE + 1}: {len(batch)} chunks")

    return all_embeddings


def get_single_embedding(text: str) -> list[float]:
    """
    Embed a single piece of text. Used when embedding a user's query.

    Args:
        text: the query string to embed

    Returns:
        A single float vector (list of 768 floats)
    """
    client = get_genai_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values