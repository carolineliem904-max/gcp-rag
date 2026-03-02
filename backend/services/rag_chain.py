"""
RAG Chain Service
=================
The core of the chatbot — connects retrieval + generation into one pipeline.

RAG = Retrieval-Augmented Generation:
  "Augmented" means we ADD external knowledge (your documents) to the LLM.
  Without RAG, Gemini only knows what it learned during training.
  With RAG, Gemini can answer questions about YOUR specific documents.

Flow:
  question → embed → search ChromaDB → build prompt → call Gemini → answer

Why this order matters:
  1. We MUST embed the question with the same model used for documents
     (text-embedding-004), otherwise the similarity search won't work —
     comparing apples to oranges.
  2. We retrieve BEFORE calling Gemini to keep costs low — Gemini only
     sees 3 relevant chunks, not the entire document.
  3. We put context BEFORE the question in the prompt — LLMs pay more
     attention to what comes first.
"""

import os
import json
from dotenv import load_dotenv
from google import genai

from backend.services.embeddings import get_single_embedding
from backend.services.vector_store import search

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")
LLM_MODEL = os.getenv("LLM_MODEL")

# How many document chunks to retrieve and send to Gemini as context.
# More chunks = more context = better answers, but also higher cost.
# 3-5 is a good balance for most use cases.
TOP_K = 3


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Build the prompt that we'll send to Gemini.

    This is called "prompt engineering" — how you structure the input
    heavily affects the quality of the output.

    We use a simple but effective structure:
      1. Tell Gemini its role (helpful assistant grounded in documents)
      2. Provide the retrieved context
      3. Ask the question
      4. Give clear instructions on how to behave

    Args:
        question:       the user's question
        context_chunks: list of dicts with 'text' and 'source' keys
                        (returned by vector_store.search())

    Returns:
        A formatted string prompt ready to send to Gemini
    """
    # Format each chunk with its source for transparency
    context_text = ""
    for i, chunk in enumerate(context_chunks):
        context_text += f"\n[Chunk {i+1} from '{chunk['source']}']\n"
        context_text += chunk["text"]
        context_text += "\n"

    prompt = f"""You are a helpful assistant that answers questions based strictly on the provided document context.

DOCUMENT CONTEXT:
{context_text}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer the question using ONLY the information in the document context above.
- If the answer cannot be found in the context, say: "I don't have enough information in the provided documents to answer this question."
- Be concise and accurate.
- Do not make up information that is not in the context.
- If relevant, mention which part of the document the answer comes from.

ANSWER:"""

    return prompt


def ask(question: str, session_id: str, n_results: int = TOP_K) -> dict:
    """
    The main RAG function — takes a question, returns a grounded answer.

    This is what the FastAPI /chat endpoint will call in Phase 4.

    Args:
        question:   the user's question as a plain string
        session_id: only retrieve documents uploaded by this session
        n_results:  how many document chunks to retrieve (default 3)

    Returns:
        A dict with:
          - 'answer':   Gemini's response string
          - 'sources':  list of source chunks used for context
          - 'question': the original question (for logging)

    Example:
        result = ask("What is supervised learning?", session_id="session-abc123")
        print(result['answer'])
        print(result['sources'])
    """
    # --- Step 1: Embed the question ---
    # Convert question text into a vector using the SAME embedding model
    # we used for the documents. This is critical — they must match.
    print(f"  [RAG] Embedding question...")
    query_vector = get_single_embedding(question)

    # --- Step 2: Retrieve relevant chunks from ChromaDB ---
    # Find the top-k chunks whose vectors are closest to the question vector,
    # filtered to only this session's uploaded documents.
    print(f"  [RAG] Searching ChromaDB for top {n_results} relevant chunks (session: {session_id})...")
    retrieved_chunks = search(query_vector, session_id=session_id, n_results=n_results)

    if not retrieved_chunks:
        return {
            "answer": "No documents have been uploaded yet. Please upload a document first.",
            "sources": [],
            "question": question,
        }

    # --- Step 3: Build the prompt ---
    # Combine the retrieved chunks + the question into one structured prompt.
    # This is what Gemini will actually read and respond to.
    print(f"  [RAG] Building prompt with {len(retrieved_chunks)} context chunks...")
    prompt = build_prompt(question, retrieved_chunks)

    # --- Step 4: Call Gemini ---
    # Send the prompt to Gemini 2.0 Flash via Vertex AI.
    # Gemini reads the context and generates a grounded answer.
    print(f"  [RAG] Calling Gemini ({LLM_MODEL})...")
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    answer = response.text.strip()

    # --- Step 5: Return answer + sources ---
    # We return the sources so the frontend can show "Based on: sample.txt"
    # This is important for trust — users should know where the answer came from.
    return {
        "answer": answer,
        "sources": retrieved_chunks,
        "question": question,
    }


def stream_ask(question: str, session_id: str, n_results: int = TOP_K):
    """
    Streaming version of ask() — a generator that yields text tokens one by one
    as Gemini produces them, instead of waiting for the full response.

    Why streaming?
      Without streaming: user waits 3-5 seconds staring at a spinner.
      With streaming:    text appears word by word immediately — feels much faster.

    Protocol:
      This generator yields two types of chunks:
        1. Text tokens  — plain strings like "Machine", " learning", " is", "..."
        2. Sources JSON — one final chunk starting with "__SOURCES__" followed by
                          a JSON list of the retrieved chunks used as context.

      The frontend reads text tokens and displays them progressively, then
      parses the final __SOURCES__ chunk to show which document parts were used.

    Args:
        question:   the user's question
        session_id: only retrieve documents uploaded by this session
        n_results:  number of document chunks to retrieve (default 3)

    Yields:
        str: text tokens from Gemini, then a final __SOURCES__ JSON chunk
    """
    # --- Step 1 & 2: Embed + retrieve (fast, not streamed) ---
    query_vector = get_single_embedding(question)
    retrieved_chunks = search(query_vector, session_id=session_id, n_results=n_results)

    if not retrieved_chunks:
        yield "No documents have been uploaded yet. Please upload a document first."
        yield f"__SOURCES__{json.dumps([])}"
        return

    # --- Step 3: Build prompt ---
    prompt = build_prompt(question, retrieved_chunks)

    # --- Step 4: Stream from Gemini token by token ---
    # generate_content_stream() returns an iterator of partial response chunks.
    # Each chunk.text contains a small piece of the answer (a few words).
    # We yield each piece immediately so the frontend can display it.
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )
    for chunk in client.models.generate_content_stream(
        model=LLM_MODEL,
        contents=prompt,
    ):
        if chunk.text:
            yield chunk.text

    # --- Step 5: Send sources as final chunk ---
    # After all text is streamed, send sources so the frontend can display them.
    # The __SOURCES__ marker tells the frontend to stop reading text and parse JSON.
    yield f"__SOURCES__{json.dumps(retrieved_chunks)}"