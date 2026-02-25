"""
Phase 2 Verification Script — Document Pipeline
================================================
Tests the full document ingestion pipeline end-to-end:

  Step 1: Upload a document to Cloud Storage
  Step 2: Extract text from the document
  Step 3: Split text into chunks
  Step 4: Generate embeddings for each chunk (calls Vertex AI)
  Step 5: Store chunks + embeddings in ChromaDB
  Step 6: Query ChromaDB with a test question
  Step 7: Print the retrieved results

Run this with:
  python scripts/test_pipeline.py

You should see all steps complete and 3 relevant text chunks printed at the end.
"""

import sys
import os

# Add the project root to Python's import path
# This allows us to import from backend/ as if we're in the root directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.pdf_parser import extract_text, chunk_text
from backend.services.storage import upload_file, list_files
from backend.services.embeddings import get_embeddings, get_single_embedding
from backend.services.vector_store import add_documents, search, get_collection_stats

# Path to the sample document we created
SAMPLE_DOC_PATH = "./sample_docs/sample.txt"
SAMPLE_DOC_NAME = "sample.txt"


def run_pipeline():
    print("=" * 60)
    print("Phase 2 — Document Pipeline Test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Upload document to Cloud Storage
    # ------------------------------------------------------------------
    # Why? In production, users upload PDFs through the web app.
    # We store them in GCS so Cloud Run can access them even after restart.
    # ------------------------------------------------------------------
    print("\n[ Step 1 ] Uploading document to Cloud Storage...")
    gcs_uri = upload_file(
        local_file_path=SAMPLE_DOC_PATH,
        destination_blob_name=f"documents/{SAMPLE_DOC_NAME}",
    )
    print(f"  Done. File is now at: {gcs_uri}")

    # Verify it's there by listing the bucket
    files = list_files(prefix="documents/")
    print(f"  Files in bucket: {files}")

    # ------------------------------------------------------------------
    # Step 2: Extract text from the document
    # ------------------------------------------------------------------
    # For a PDF, this would use pdfplumber to read each page.
    # For a .txt file, it just reads the content directly.
    # ------------------------------------------------------------------
    print(f"\n[ Step 2 ] Extracting text from '{SAMPLE_DOC_NAME}'...")
    text = extract_text(SAMPLE_DOC_PATH)
    print(f"  Extracted {len(text)} characters of text")
    print(f"  Preview (first 200 chars): {text[:200]}...")

    # ------------------------------------------------------------------
    # Step 3: Split text into chunks
    # ------------------------------------------------------------------
    # We can't embed the whole document as one vector — too much info
    # gets compressed into one point. Chunks of ~500 chars work well.
    # ------------------------------------------------------------------
    print(f"\n[ Step 3 ] Splitting text into chunks...")
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"  Created {len(chunks)} chunks")
    print(f"  Chunk 0 preview: '{chunks[0][:100]}...'")
    print(f"  Chunk 1 preview: '{chunks[1][:100]}...'")

    # ------------------------------------------------------------------
    # Step 4: Generate embeddings for all chunks
    # ------------------------------------------------------------------
    # This calls Vertex AI for each batch of chunks.
    # Each chunk → one vector of 768 floats.
    # This is where the actual AI API cost happens (but it's tiny).
    # ------------------------------------------------------------------
    print(f"\n[ Step 4 ] Generating embeddings for {len(chunks)} chunks...")
    embeddings = get_embeddings(chunks)
    print(f"  Generated {len(embeddings)} embedding vectors")
    print(f"  Each vector has {len(embeddings[0])} dimensions")
    print(f"  First 5 values of chunk 0: {[round(v, 4) for v in embeddings[0][:5]]}")

    # ------------------------------------------------------------------
    # Step 5: Store chunks + embeddings in ChromaDB
    # ------------------------------------------------------------------
    # ChromaDB saves everything to ./chroma_db/ on disk.
    # Next time you run this, it won't lose the data.
    # ------------------------------------------------------------------
    print(f"\n[ Step 5 ] Storing chunks in ChromaDB...")
    add_documents(chunks, embeddings, source_filename=SAMPLE_DOC_NAME)
    stats = get_collection_stats()
    print(f"  ChromaDB now has {stats['total_chunks']} total chunks stored")

    # ------------------------------------------------------------------
    # Step 6: Query ChromaDB with a test question
    # ------------------------------------------------------------------
    # This is exactly what happens when a user asks a question:
    #   1. Embed the question
    #   2. Find the 3 most similar stored chunks
    #   3. Return them (Phase 3 will pass these to Gemini)
    # ------------------------------------------------------------------
    test_question = "What is the difference between supervised and unsupervised learning?"
    print(f"\n[ Step 6 ] Querying with: '{test_question}'")

    # Embed the question using the same model
    query_vector = get_single_embedding(test_question)
    print(f"  Query embedded: {len(query_vector)} dimensions")

    # Search ChromaDB for the 3 most similar chunks
    results = search(query_vector, n_results=3)

    # ------------------------------------------------------------------
    # Step 7: Print retrieved results
    # ------------------------------------------------------------------
    print(f"\n[ Step 7 ] Top {len(results)} retrieved chunks:")
    print("-" * 60)
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (distance: {result['distance']}, source: {result['source']})")
        print(f"  {result['text'][:300]}...")

    print("\n" + "=" * 60)
    print("Pipeline test complete! Phase 2 is working correctly.")
    print("=" * 60)
    print("\nWhat you just built:")
    print("  upload → extract → chunk → embed → store → retrieve")
    print("This is the foundation of every RAG system.")


if __name__ == "__main__":
    run_pipeline()