"""
Tests for the FastAPI API endpoints.

These tests verify that:
- The API endpoints respond correctly.
- Input validation works.
- Error handling returns proper responses.

Note: Tests that would require an OpenAI API key are mocked
so they can run without a real API key.

Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app


# --- Test Client ---
# FastAPI provides a TestClient for testing endpoints
# without actually starting the server.
client = TestClient(app)


# --- Test: Home Page ---

def test_home_page():
    """Test that the home page returns the frontend HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# --- Test: Upload Endpoint ---

def test_upload_unsupported_file_type():
    """Test that uploading an unsupported file type returns 400."""
    # Create a fake .exe file
    response = client.post(
        "/upload",
        files={"file": ("malware.exe", b"fake content", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_empty_file():
    """Test that uploading an empty file returns 400."""
    response = client.post(
        "/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


@patch("app.main.store_document_chunks")
@patch("app.main.split_text_into_chunks")
@patch("app.main.extract_text_from_txt")
@patch("app.main.save_uploaded_file")
def test_upload_txt_success(
    mock_save, mock_extract, mock_split, mock_store
):
    """
    Test successful TXT file upload with mocked services.
    
    We mock the service functions so this test doesn't need
    an OpenAI API key or ChromaDB.
    """
    # Configure mocks
    mock_save.return_value = "/fake/path/test.txt"
    mock_extract.return_value = [{"text": "Test content", "page": None}]
    mock_split.return_value = [
        {"text": "chunk1", "metadata": {"filename": "test.txt", "chunk_number": 1}}
    ]
    mock_store.return_value = 1
    
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"Test content", "text/plain")},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Document uploaded and processed successfully"
    assert data["filename"] == "test.txt"
    assert data["chunks_created"] == 1


# --- Test: Ask Endpoint ---

def test_ask_empty_question():
    """Test that an empty question returns a validation error."""
    response = client.post(
        "/ask",
        json={"question": ""},
    )
    assert response.status_code == 422  # Pydantic validation error


def test_ask_missing_question():
    """Test that a missing question field returns a validation error."""
    response = client.post(
        "/ask",
        json={},
    )
    assert response.status_code == 422


@patch("app.main.ask_question")
def test_ask_question_success(mock_ask):
    """
    Test successful question answering with mocked RAG service.
    
    We mock ask_question so this test doesn't need
    an OpenAI API key or ChromaDB.
    """
    mock_ask.return_value = {
        "question": "What is the leave policy?",
        "answer": "Employees get 20 days of annual leave.",
        "sources": [
            {"filename": "policy.pdf", "page": 4, "chunk": 3}
        ],
    }
    
    response = client.post(
        "/ask",
        json={"question": "What is the leave policy?"},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is the leave policy?"
    assert "20 days" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["filename"] == "policy.pdf"


# --- Test: Documents Endpoint ---

@patch("app.main.get_stored_documents")
def test_list_documents(mock_get_docs):
    """Test listing uploaded documents."""
    mock_get_docs.return_value = ["file1.pdf", "file2.txt"]
    
    response = client.get("/documents")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["documents"]) == 2
    assert "file1.pdf" in data["documents"]


@patch("app.main.get_stored_documents")
def test_list_documents_empty(mock_get_docs):
    """Test listing documents when none are uploaded."""
    mock_get_docs.return_value = []
    
    response = client.get("/documents")
    
    assert response.status_code == 200
    data = response.json()
    assert data["documents"] == []


# --- Test: API Documentation ---

def test_api_docs_accessible():
    """Test that the Swagger documentation page is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_json_accessible():
    """Test that the OpenAPI JSON schema is accessible."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert data["info"]["title"] == "AI Document Q&A Assistant"


# --- Test: Delete Document Endpoint ---

@patch("app.main.delete_document_chunks")
def test_delete_document_success(mock_delete):
    """Test successful document deletion."""
    mock_delete.return_value = True
    
    response = client.delete("/documents/policy.pdf")
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "policy.pdf"
    assert "deleted successfully" in data["message"]


@patch("app.main.delete_document_chunks")
def test_delete_document_not_found(mock_delete):
    """Test deleting non-existent document returns 404."""
    mock_delete.return_value = False
    
    response = client.delete("/documents/nonexistent.pdf")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

