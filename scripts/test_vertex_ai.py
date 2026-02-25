"""
Phase 1 Verification Script
============================
Tests that your local machine can successfully connect to Vertex AI.

Uses the NEW Google Gen AI SDK (google-genai package), which replaced the
deprecated vertexai.language_models and vertexai.generative_models classes.

What this script does:
  1. Loads your GCP config from .env
  2. Creates a Gen AI client connected to Vertex AI
  3. Generates a text embedding (vector) using text-embedding-004
  4. Calls Gemini 2.0 Flash for text generation

Authentication: Application Default Credentials (ADC)
  Run once in terminal: gcloud auth application-default login
  Credentials stored at: ~/.config/gcloud/application_default_credentials.json

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
# Test 1: Create Gen AI Client connected to Vertex AI
#
# The new google-genai SDK uses a single Client object.
# Setting vertexai=True tells it to use Vertex AI (your GCP project)
# instead of Google AI Studio (which requires a different API key).
#
# Authentication is handled automatically via ADC — no key file needed.
# -------------------------------------------------------------------
print("=== Test 1: Initialize Vertex AI Client ===")
try:
    from google import genai

    # Create client — vertexai=True routes all calls through your GCP project
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )
    print("✓ Vertex AI client created successfully\n")
except Exception as e:
    print(f"✗ Failed to create Vertex AI client: {e}")
    print("  Make sure you ran: gcloud auth application-default login")
    sys.exit(1)


# -------------------------------------------------------------------
# Test 2: Generate an embedding
#
# Embeddings convert text into a list of numbers (a vector).
# Similar texts produce similar vectors — this is the core of RAG retrieval.
# text-embedding-004 produces 768-dimensional vectors.
# -------------------------------------------------------------------
print("=== Test 2: Generate Embeddings ===")
try:
    test_text = "This is a test sentence to verify embeddings work."

    # embed_content sends text to the Vertex AI Embeddings API
    result = client.models.embed_content(
        model=embedding_model,
        contents=test_text,
    )

    # result.embeddings is a list; [0].values is the actual float vector
    vector = result.embeddings[0].values
    print(f"✓ Embedding generated successfully")
    print(f"  Vector dimensions: {len(vector)}")
    print(f"  First 5 values:    {[round(v, 4) for v in vector[:5]]}\n")
except Exception as e:
    print(f"✗ Failed to generate embedding: {e}")
    print("  Check that roles/aiplatform.user is granted and the API is enabled.")
    sys.exit(1)


# -------------------------------------------------------------------
# Test 3: Call Gemini for text generation
#
# generate_content sends a prompt to Gemini and returns the response.
# gemini-2.0-flash is the latest fast + cheap Gemini model on Vertex AI.
# -------------------------------------------------------------------
print("=== Test 3: Call Gemini LLM ===")
try:
    response = client.models.generate_content(
        model=llm_model,
        contents="Say hello and confirm you are responding correctly from Vertex AI.",
    )

    print(f"✓ Gemini responded successfully")
    print(f"  Response: {response.text.strip()}\n")
except Exception as e:
    print(f"✗ Failed to call Gemini: {e}")
    print("  Check that the model name in .env is correct and the API is enabled.")
    sys.exit(1)


# -------------------------------------------------------------------
# All tests passed!
# -------------------------------------------------------------------
print("=" * 50)
print("All tests passed! Phase 1 is complete.")
print("You are ready to move on to Phase 2.")
print("=" * 50)