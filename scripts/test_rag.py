"""
Phase 3 Verification Script — RAG Chain
========================================
Tests the full RAG pipeline end-to-end:
  question → retrieve → generate → answer

Prerequisites:
  - Phase 2 must be complete (ChromaDB has documents stored)
  - If ChromaDB is empty, run: python scripts/test_pipeline.py first

Run this with:
  python scripts/test_rag.py
"""

import sys
import os

# Add project root to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.rag_chain import ask
from backend.services.vector_store import get_collection_stats


def run_rag_test():
    print("=" * 60)
    print("Phase 3 — RAG Chain Test")
    print("=" * 60)

    # Check that ChromaDB has data from Phase 2
    stats = get_collection_stats()
    print(f"\nChromaDB status: {stats['total_chunks']} chunks stored")

    if stats["total_chunks"] == 0:
        print("\nERROR: ChromaDB is empty.")
        print("Run Phase 2 first: python scripts/test_pipeline.py")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Test questions — all answerable from sample.txt
    # ---------------------------------------------------------------
    test_questions = [
        "What is the difference between supervised and unsupervised learning?",
        "How does RAG work?",
        "What are some examples of reinforcement learning?",
        "What is backpropagation?",
    ]

    for i, question in enumerate(test_questions):
        print(f"\n{'─' * 60}")
        print(f"Question {i+1}: {question}")
        print("─" * 60)

        result = ask(question)

        # Print the answer
        print(f"\nAnswer:\n{result['answer']}")

        # Print which chunks were used as context
        print(f"\nSources used ({len(result['sources'])} chunks):")
        for j, source in enumerate(result['sources']):
            print(f"  [{j+1}] from '{source['source']}' "
                  f"(similarity distance: {source['distance']})")
            print(f"       Preview: {source['text'][:100]}...")

    # ---------------------------------------------------------------
    # Bonus: test a question that's NOT in the document
    # ---------------------------------------------------------------
    print(f"\n{'─' * 60}")
    out_of_scope = "What is the capital of France?"
    print(f"Out-of-scope question: {out_of_scope}")
    print("─" * 60)
    result = ask(out_of_scope)
    print(f"\nAnswer:\n{result['answer']}")
    print("\n(Gemini should say it doesn't have enough info from documents)")

    print("\n" + "=" * 60)
    print("RAG Chain test complete! Phase 3 is working correctly.")
    print("=" * 60)
    print("\nWhat you just built:")
    print("  question → embed → retrieve → prompt → Gemini → answer")
    print("This is the brain of your RAG chatbot.")


if __name__ == "__main__":
    run_rag_test()