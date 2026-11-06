"""
Tests for the Document Processing Service.

These tests verify that:
- Text extraction works for TXT files.
- Text chunking produces the expected results.
- Error handling works for invalid inputs.

Note: These tests do NOT require an OpenAI API key.
They test only the local document processing logic.

Run with: pytest tests/ -v
"""

import os
import pytest
from app.services.document_processor import (
    extract_text_from_txt,
    split_text_into_chunks,
)


# --- Fixtures ---
# Fixtures provide reusable test data and setup.

@pytest.fixture
def sample_txt_file(tmp_path):
    """
    Create a temporary TXT file for testing.
    tmp_path is a pytest fixture that provides a temporary directory.
    """
    content = (
        "This is a sample document for testing purposes. "
        "It contains information about company policies. "
        "Employees receive 20 days of annual leave per year. "
        "The standard working hours are 9 AM to 5:30 PM. "
        "Remote work is allowed up to 3 days per week."
    )
    file_path = tmp_path / "test_document.txt"
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


@pytest.fixture
def empty_txt_file(tmp_path):
    """Create an empty TXT file for testing."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    return str(file_path)


@pytest.fixture
def sample_pages():
    """Create sample page data for testing chunking."""
    # Simulate extracted pages from a document
    long_text = " ".join([f"Sentence number {i}." for i in range(100)])
    return [
        {"text": long_text, "page": 1},
        {"text": "Short text on page two.", "page": 2},
    ]


# --- Test: Text Extraction ---

def test_extract_text_from_txt(sample_txt_file):
    """Test that text is correctly extracted from a TXT file."""
    result = extract_text_from_txt(sample_txt_file)
    
    # Should return a list with one entry (TXT files = single page)
    assert len(result) == 1
    
    # The text should contain our content
    assert "20 days of annual leave" in result[0]["text"]
    
    # TXT files don't have page numbers
    assert result[0]["page"] is None


def test_extract_text_from_empty_txt(empty_txt_file):
    """Test that empty TXT files raise an error."""
    with pytest.raises(ValueError, match="empty"):
        extract_text_from_txt(empty_txt_file)


# --- Test: Text Chunking ---

def test_split_text_into_chunks(sample_pages):
    """Test that text is split into chunks correctly."""
    chunks = split_text_into_chunks(sample_pages, "test.txt")
    
    # Should create at least one chunk
    assert len(chunks) > 0
    
    # Each chunk should have text and metadata
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert len(chunk["text"]) > 0
        
        # Check metadata fields
        metadata = chunk["metadata"]
        assert metadata["filename"] == "test.txt"
        assert "chunk_number" in metadata
        assert "file_type" in metadata


def test_chunks_have_sequential_numbers(sample_pages):
    """Test that chunk numbers are sequential starting from 1."""
    chunks = split_text_into_chunks(sample_pages, "test.txt")
    
    for i, chunk in enumerate(chunks, 1):
        assert chunk["metadata"]["chunk_number"] == i


def test_chunk_size_is_reasonable(sample_pages):
    """Test that no chunk exceeds the configured chunk size significantly."""
    from app.config import CHUNK_SIZE
    
    chunks = split_text_into_chunks(sample_pages, "test.txt")
    
    for chunk in chunks:
        # Allow some flexibility (splitter may slightly exceed chunk_size)
        assert len(chunk["text"]) <= CHUNK_SIZE * 1.5


def test_single_page_chunking():
    """Test chunking with a single short page."""
    pages = [{"text": "A short text.", "page": None}]
    chunks = split_text_into_chunks(pages, "short.txt")
    
    # Short text should produce exactly one chunk
    assert len(chunks) == 1
    assert chunks[0]["text"] == "A short text."
    assert chunks[0]["metadata"]["file_type"] == ".txt"
