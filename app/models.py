"""
Pydantic models for API request and response validation.

This file defines the data structures (schemas) for our API.
Pydantic models ensure that:
- Incoming requests have the correct format.
- Outgoing responses have a consistent structure.
- Data validation happens automatically.

Why Pydantic?
- FastAPI uses Pydantic models natively for request/response handling.
- They provide automatic validation, serialization, and documentation.
- They make the API self-documenting through OpenAPI/Swagger.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class QuestionRequest(BaseModel):
    """Schema for the question-asking endpoint request.
    
    The user sends a JSON body with their question string.
    """
    question: str = Field(
        ...,  # '...' means this field is required
        min_length=1,  # Question must not be empty
        description="The question to ask about the uploaded documents",
        examples=["What is the annual leave policy?"]
    )


class SourceResponse(BaseModel):
    """Schema for source information returned with an answer.
    
    Each source tells the user where in which document the
    relevant information was found.
    """
    filename: str = Field(description="Name of the source document")
    page: Optional[int] = Field(
        default=None,
        description="Page number (available for PDFs, None for TXT files)"
    )
    chunk: int = Field(description="Chunk number within the document")


class QuestionResponse(BaseModel):
    """Schema for the question-answering endpoint response.
    
    Contains the original question, the LLM-generated answer,
    and a list of sources where the information was found.
    """
    question: str = Field(description="The original question asked")
    answer: str = Field(description="The AI-generated answer")
    sources: List[SourceResponse] = Field(
        description="List of document sources used to generate the answer"
    )


class UploadResponse(BaseModel):
    """Schema for the document upload endpoint response."""
    message: str = Field(description="Status message")
    filename: str = Field(description="Name of the uploaded file")
    chunks_created: int = Field(
        description="Number of text chunks created from the document"
    )


class DocumentListResponse(BaseModel):
    """Schema for the document listing endpoint response."""
    documents: List[str] = Field(
        description="List of uploaded document filenames"
    )


class DeleteResponse(BaseModel):
    """Schema for the document deletion endpoint response."""
    message: str = Field(description="Status message")
    filename: str = Field(description="Name of the deleted document")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    detail: str = Field(description="Error description")
