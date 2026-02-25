"""
Phase 1 Verification Script
============================
Tests that your local machine can successfully connect to Vertex AI.

What this script does:
  1. Loads your GCP credentials from .env
  2. Initializes the Vertex AI SDK
  3. Generates a text embedding (vector) using text-embedding-004
  4. Calls Gemini 1.5 Flash for text generation

Run this with:
  python scripts/test_vertex_ai.py

Expected result: All 3 tests print ✓ and "All tests passed!"
"""

import os
import sys
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Load environment variables from .env file
# This reads GCP_PROJECT_ID, GCP_LOCATION, etc. into os.environ
# -------------------------------------------------------------------
load_dotenv()

project_id = os.getenv("GCP_PROJECT_ID")
location = os.getenv("GCP_LOCATION")
embedding_model = os.getenv("EMBEDDING_MODEL")
llm_model = os.getenv("LLM_MODEL")

# Basic validation — fail early with a helpful message
if not project_id or project_id == "your-project-id-here":
    print("ERROR: GCP_PROJECT_ID is not set in your .env file.")
    print("Open .env and replace 'your-project-id-here' with your real project ID.")
    sys.exit(1)

print("=" * 50)
print("GCP RAG Chatbot — Phase 1 Verification")
print("=" * 50)
print(f"  Project ID:       {project_id}")
print(f"  Location:         {location}")
print(f"  Embedding Model:  {embedding_model}")
print(f"  LLM Model:        {llm_model}")
print()


# -------------------------------------------------------------------
# Test 1: Initialize Vertex AI
# vertexai.init() tells the SDK which GCP project and region to use.
# Authentication comes from GOOGLE_APPLICATION_CREDENTIALS in .env,
# which points to your downloaded service-account-key.json file.
# -------------------------------------------------------------------
print("=== Test 1: Initialize Vertex AI ===")
try:
    import vertexai
    vertexai.init(project=project_id, location=location)
    print("✓ Vertex AI initialized successfully\n")
except Exception as e:
    print(f"✗ Failed to initialize Vertex AI: {e}")
    print("  Check that GOOGLE_APPLICATION_CREDENTIALS points to a valid JSON key file.")
    sys.exit(1)


# -------------------------------------------------------------------
# Test 2: Generate an embedding
# Embeddings convert text into a list of numbers (a vector).
# Similar texts produce similar vectors — this is how RAG retrieval works.
# text-embedding-004 produces 768-dimensional vectors.
# -------------------------------------------------------------------
print("=== Test 2: Generate Embeddings ===")
try:
    from vertexai.language_models import TextEmbeddingModel

    # Load the embedding model
    model = TextEmbeddingModel.from_pretrained(embedding_model)

    # Generate an embedding for a test sentence
    test_text = "This is a test sentence to verify embeddings work."
    embeddings = model.get_embeddings([test_text])

    # embeddings[0].values is the list of numbers (the vector)
    vector = embeddings[0].values
    print(f"✓ Embedding generated successfully")
    print(f"  Vector dimensions: {len(vector)}")
    print(f"  First 5 values:    {[round(v, 4) for v in vector[:5]]}\n")
except Exception as e:
    print(f"✗ Failed to generate embedding: {e}")
    print("  Check that roles/aiplatform.user is granted to your service account.")
    sys.exit(1)


# -------------------------------------------------------------------
# Test 3: Call Gemini for text generation
# GenerativeModel wraps the Gemini API on Vertex AI.
# generate_content() sends a prompt and returns the model's response.
# gemini-1.5-flash is the fastest and cheapest Gemini model.
# -------------------------------------------------------------------
print("=== Test 3: Call Gemini LLM ===")
try:
    from vertexai.generative_models import GenerativeModel

    # Load the Gemini model
    gemini = GenerativeModel(llm_model)

    # Send a simple test prompt
    response = gemini.generate_content(
        "Say hello and confirm that you are responding correctly from Vertex AI."
    )

    print(f"✓ Gemini responded successfully")
    print(f"  Response: {response.text.strip()}\n")
except Exception as e:
    print(f"✗ Failed to call Gemini: {e}")
    print("  Check that Vertex AI API is enabled and the model name is correct.")
    sys.exit(1)


# -------------------------------------------------------------------
# All tests passed!
# -------------------------------------------------------------------
print("=" * 50)
print("All tests passed! Phase 1 is complete.")
print("You are ready to move on to Phase 2.")
print("=" * 50)
