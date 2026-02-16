# src/tests/test_retriever.py

import pytest
from src.retriever import retrieve_documents

def test_retrieve_non_empty():
    """Retriever should return at least one result."""
    docs = retrieve_documents("billing dispute", k=3)
    assert len(docs) > 0

def test_retriever_content():
    """Results should contain text content."""
    result = retrieve_documents("credit card issues", k=3)[0]
    assert hasattr(result, "page_content")
    assert len(result.page_content) > 0
