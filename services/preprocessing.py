from pathlib import Path
from typing import List


def pdf2chunks(
    filepath: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "fixed",   # "fixed" or "sentence"
) -> List[str]:
    """
    Extract text from a PDF file and split it into overlapping chunks.

    Args:
        filepath:     Path to the PDF file
        chunk_size:   Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        strategy:     "fixed"    — original character-based sliding window
                      "sentence" — split on sentence boundaries, then group
                                   until chunk_size is reached

    Returns:
        List of text chunks as strings
    """
    import PyPDF2

    # ------------------------------------------------------------------ #
    # 1. Extract raw text from every page
    # ------------------------------------------------------------------ #
    full_text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + " "

    if not full_text.strip():
        return []

    # ------------------------------------------------------------------ #
    # 2. Split according to strategy
    # ------------------------------------------------------------------ #
    if strategy == "sentence":
        return _sentence_chunks(full_text, chunk_size, chunk_overlap)
    else:
        return _fixed_chunks(full_text, chunk_size, chunk_overlap)


# ------------------------------------------------------------------ #
# Strategy A: original fixed-size sliding window
# ------------------------------------------------------------------ #
def _fixed_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - chunk_overlap
    return chunks


# ------------------------------------------------------------------ #
# Strategy B: sentence-aware chunking (no mid-sentence cuts)
# ------------------------------------------------------------------ #
def _sentence_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split text into sentences first, then group sentences into chunks
    that stay under chunk_size characters.  Overlap is achieved by
    re-including the last few sentences of the previous chunk.
    """
    import re

    # Split on '.', '!', '?' followed by whitespace or end-of-string
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_sentences: List[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence would exceed chunk_size, save current chunk
        if current_length + sentence_len > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Overlap: keep sentences from the end of the current chunk
            # until their total length is >= chunk_overlap
            overlap_sentences: List[str] = []
            overlap_length = 0
            for s in reversed(current_sentences):
                if overlap_length >= chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_length += len(s)

            current_sentences = overlap_sentences
            current_length = overlap_length

        current_sentences.append(sentence)
        current_length += sentence_len

    # Don't forget the last chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
