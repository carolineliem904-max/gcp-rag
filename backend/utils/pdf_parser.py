"""
PDF Parser Utility
==================
Handles two jobs:
  1. extract_text()  — reads a PDF (or .txt) file and returns raw text
  2. chunk_text()    — splits long text into smaller overlapping pieces

Why chunking matters:
  - Embedding models have a token limit (~3000 tokens max)
  - Storing one vector per page is too coarse — retrieval is imprecise
  - Storing one vector per sentence is too granular — loses context
  - Chunks of ~500 characters with 50-char overlap is a good balance:
      * Small enough to be specific
      * Overlap ensures we don't cut a sentence in half and lose meaning
"""

import os
import pdfplumber


def extract_text(file_path: str) -> str:
    """
    Extract all text from a PDF or plain text file.

    Args:
        file_path: path to a .pdf or .txt file

    Returns:
        A single string with all the text from the file.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        # Plain text — just read it directly
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        # PDF — use pdfplumber to extract text page by page
        # pdfplumber handles layout better than PyPDF2 for most PDFs
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:  # some pages may be images with no text
                    text_parts.append(page_text)

        return "\n".join(text_parts)

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split a long text into overlapping chunks of roughly chunk_size characters.

    Why overlap?
        If a key sentence sits at the boundary between two chunks,
        overlap ensures it appears in at least one chunk fully.

    Example with chunk_size=20, overlap=5:
        text = "The quick brown fox jumps over the lazy dog"
        chunks = [
            "The quick brown fox ",   (chars 0-19)
            "fox jumps over the l",   (chars 15-34)  ← overlaps with previous
            "the lazy dog",           (chars 29-end)
        ]

    Args:
        text:       the full document text
        chunk_size: max characters per chunk (default 500)
        overlap:    how many characters to repeat between chunks (default 50)

    Returns:
        A list of text strings (chunks)
    """
    chunks = []
    start = 0

    while start < len(text):
        # Take a slice of chunk_size characters starting at 'start'
        end = start + chunk_size
        chunk = text[start:end].strip()

        # Only add non-empty chunks
        if chunk:
            chunks.append(chunk)

        # Move forward by (chunk_size - overlap) so next chunk overlaps
        start += chunk_size - overlap

    return chunks