"""
Main FastAPI Application.

This is the entry point of the AI Document Q&A Assistant API.
It defines all the HTTP endpoints (routes) that the frontend 
and users interact with.

Endpoints:
  GET  /           - Serves the frontend HTML page.
  POST /upload     - Upload a document (PDF or TXT).
  POST /ask        - Ask a question about uploaded documents.
  GET  /documents  - List all uploaded documents.
  DELETE /documents/{filename} - Delete a document.

How it fits in the architecture:
  Frontend/User → THIS FILE (API routes) → Services → Response
"""

import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, DOCUMENTS_DIR
from app.models import (
    QuestionRequest,
    QuestionResponse,
    UploadResponse,
    DocumentListResponse,
    DeleteResponse,
    ErrorResponse,
)
from app.services.document_processor import (
    extract_text_from_pdf,
    extract_text_from_txt,
    split_text_into_chunks,
    save_uploaded_file,
)
from contextlib import asynccontextmanager
from app.services.vector_store import (
    store_document_chunks,
    get_stored_documents,
    delete_document_chunks,
)
from app.services.rag_service import ask_question

# --- Configure Logging ---
# Logging helps us track what's happening in the application.
# We log to the console with timestamps and severity levels.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Application Lifespan Event ---
# Using modern lifespan context manager instead of deprecated @app.on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log startup and shutdown messages."""
    logger.info("=" * 50)
    logger.info("AI Document Q&A Assistant is starting up!")
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("Frontend: http://localhost:8000")
    logger.info("=" * 50)
    yield
    logger.info("AI Document Q&A Assistant is shutting down.")


# --- Create the FastAPI Application ---
# FastAPI automatically generates API documentation at /docs
app = FastAPI(
    title="AI Document Q&A Assistant",
    description=(
        "Upload PDF and TXT documents, then ask questions about them. "
        "Uses Retrieval-Augmented Generation (RAG) with OpenAI and ChromaDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# --- Mount Static Files ---
# This serves the frontend files (HTML, CSS, JS) from the /static URL path.
# For example, /static/style.css serves the CSS file.
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ==========================================
# ENDPOINT: Home Page
# ==========================================
@app.get("/", include_in_schema=False)
async def home():
    """
    Serve the frontend HTML page.
    
    When a user visits http://localhost:8000 in their browser,
    this returns the index.html file which contains our UI.
    """
    return FileResponse(os.path.join(static_dir, "index.html"))


# ==========================================
# ENDPOINT: Upload Document
# ==========================================
@app.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Upload a document",
    description="Upload a PDF or TXT file to be processed and indexed for Q&A.",
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.
    
    The complete upload pipeline:
    1. Validate file type and size.
    2. Save the file to disk.
    3. Extract text from the file.
    4. Split text into chunks.
    5. Generate embeddings and store in ChromaDB.
    6. Return success with chunk count.
    """
    logger.info(f"Upload request received for file: {file.filename}")
    
    # --- Step 1: Validate file type ---
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        logger.warning(f"Rejected file with extension: {file_extension}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_extension}'. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    
    # --- Step 2: Read and validate file size ---
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB.",
        )
    
    if len(file_content) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )
    
    try:
        # --- Step 3: Save the file ---
        file_path = save_uploaded_file(file_content, file.filename)
        
        # --- Step 4: Extract text ---
        if file_extension == ".pdf":
            pages = extract_text_from_pdf(file_path)
        else:  # .txt
            pages = extract_text_from_txt(file_path)
        
        if not pages:
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from the document.",
            )
        
        # --- Step 5: Split into chunks ---
        chunks = split_text_into_chunks(pages, file.filename)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Document produced no text chunks.",
            )
        
        # --- Step 6: Store in ChromaDB ---
        num_chunks = store_document_chunks(chunks)
        
        logger.info(
            f"Successfully processed '{file.filename}': {num_chunks} chunks created"
        )
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            filename=file.filename,
            chunks_created=num_chunks,
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except ValueError as e:
        logger.error(f"Value error processing document: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the document. Please check your API key and try again.",
        )


# ==========================================
# ENDPOINT: Ask Question
# ==========================================
@app.post(
    "/ask",
    response_model=QuestionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Ask a question",
    description="Ask a question about the uploaded documents using RAG.",
)
async def ask_question_endpoint(request: QuestionRequest):
    """
    Ask a question about uploaded documents.
    
    The complete Q&A pipeline:
    1. Validate the question.
    2. Search ChromaDB for relevant document chunks.
    3. Build context from retrieved chunks.
    4. Send context + question to LLM.
    5. Return answer with source references.
    """
    logger.info(f"Question received: '{request.question}'")
    
    try:
        # The ask_question function in rag_service.py handles
        # the entire RAG pipeline: retrieval → context → LLM → answer
        result = ask_question(request.question)
        
        return QuestionResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
        )
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question. Please check your API key and try again.",
        )


# ==========================================
# ENDPOINT: List Documents
# ==========================================
@app.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List uploaded documents",
    description="Get a list of all documents that have been uploaded and indexed.",
)
async def list_documents():
    """
    Return a list of all uploaded document filenames.
    
    This reads from ChromaDB metadata, so it shows documents
    that have been processed and indexed (not just saved to disk).
    """
    try:
        documents = get_stored_documents()
        return DocumentListResponse(documents=documents)
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        # Return empty list on error rather than failing
        return DocumentListResponse(documents=[])


# ==========================================
# ENDPOINT: Delete Document
# ==========================================
@app.delete(
    "/documents/{filename}",
    response_model=DeleteResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Delete a document",
    description="Remove a document and its vectors from the system.",
)
async def delete_document(filename: str):
    """
    Delete a document's chunks from ChromaDB and remove the file.
    
    This removes:
    1. All vector embeddings for this document from ChromaDB.
    2. The physical file from the documents directory.
    """
    logger.info(f"Delete request for document: {filename}")
    
    try:
        # Delete from ChromaDB
        deleted = delete_document_chunks(filename)
        
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{filename}' not found.",
            )
        
        # Delete the physical file if it exists
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
        
        return DeleteResponse(
            message="Document deleted successfully",
            filename=filename,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while deleting the document.",
        )
