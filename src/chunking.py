"""
chunking.py
Module for splitting complaint narratives into manageable text chunks for embedding and retrieval.
"""

from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """
    Splits a long text into overlapping chunks.
    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk (in words).
        overlap (int): The number of words to overlap between chunks.
    Returns:
        List[str]: List of text chunks.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks 