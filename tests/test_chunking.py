# src/tests/test_chunking.py

from src.chunking import split_text

def test_chunk_output():
    text = "This is a test. " * 100
    chunks = split_text(text)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)

