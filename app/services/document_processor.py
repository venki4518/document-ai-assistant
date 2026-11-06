"""
Document Processing Service.

This file handles everything related to processing uploaded documents:
1. Extracting text from PDF and TXT files.
2. Splitting text into smaller chunks for embedding.

How it fits in the architecture:
  User uploads file → THIS MODULE extracts text → splits into chunks →
  chunks are sent to vector_store.py for embedding and storage.

Why we split documents into chunks:
- LLMs have a limited context window (maximum input size).
- Smaller chunks allow more precise retrieval of relevant information.
- Embeddings (numerical representations) work better on focused text.
- If we sent an entire 100-page PDF to the LLM, it would be too large
  and the answer quality would suffer.
"""

import os
import logging
from typing import List, Dict
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENTS_DIR

# Set up logging for this module
logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from a PDF file, page by page.
    
    Args:
        file_path: Path to the PDF file.
        
    Returns:
        A list of dictionaries, each containing:
        - 'text': The extracted text from a page.
        - 'page': The page number (1-indexed).
        
    Why page-by-page?
    - So we can track which page each chunk came from.
    - This helps us provide accurate source references in answers.
    """
    logger.info(f"Extracting text from PDF: {file_path}")
    pages = []
    
    try:
        reader = PdfReader(file_path)
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():  # Skip empty pages
                pages.append({
                    "text": text,
                    "page": i + 1  # 1-indexed page numbers
                })
        
        logger.info(f"Extracted text from {len(pages)} pages")
        return pages
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Failed to read PDF file: {e}")


def extract_text_from_txt(file_path: str) -> List[Dict]:
    """
    Extract text from a TXT file.
    
    Args:
        file_path: Path to the TXT file.
        
    Returns:
        A list with a single dictionary containing:
        - 'text': The entire file content.
        - 'page': None (TXT files don't have pages).
    """
    logger.info(f"Extracting text from TXT: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        if not text.strip():
            raise ValueError("The text file is empty.")
        
        return [{"text": text, "page": None}]
        
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()
        return [{"text": text, "page": None}]


def split_text_into_chunks(
    pages: List[Dict],
    filename: str
) -> List[Dict]:
    """
    Split extracted text into smaller chunks with metadata.
    
    This is a critical step in the RAG pipeline. We take the full text
    and break it into smaller, overlapping pieces.
    
    Args:
        pages: List of page dictionaries from extract_text_from_* functions.
        filename: Original filename for metadata.
        
    Returns:
        A list of chunk dictionaries, each containing:
        - 'text': The chunk text.
        - 'metadata': Dictionary with filename, page, chunk_number, file_type.
    
    Why overlap?
    Imagine a sentence that says: "Employees receive 20 days of annual leave."
    If this sentence falls right at the boundary between two chunks (without
    overlap), it might get split into:
      Chunk 1: "...Employees receive 20 days"
      Chunk 2: "of annual leave. ..."
    Neither chunk has the complete information! With overlap, chunk 2 would
    start a bit before its boundary, capturing the full sentence.
    """
    logger.info(f"Splitting document '{filename}' into chunks")
    
    # RecursiveCharacterTextSplitter tries to split on natural boundaries
    # (paragraphs, sentences, words) rather than cutting mid-word.
    # It first tries to split on "\n\n" (paragraphs), then "\n" (lines),
    # then " " (words), and finally individual characters.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,  # Use character count for length
    )
    
    all_chunks = []
    chunk_number = 0  # Global chunk counter across all pages
    file_type = os.path.splitext(filename)[1].lower()  # e.g., ".pdf" or ".txt"
    
    for page_data in pages:
        text = page_data["text"]
        page = page_data["page"]
        
        # Split this page's text into chunks
        chunks = text_splitter.split_text(text)
        
        for chunk_text in chunks:
            chunk_number += 1
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "filename": filename,
                    "page": page,  # None for TXT files
                    "chunk_number": chunk_number,
                    "file_type": file_type,
                }
            })
    
    logger.info(f"Created {len(all_chunks)} chunks from '{filename}'")
    return all_chunks


def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """
    Save an uploaded file to the documents directory.
    
    Args:
        file_content: Raw bytes of the uploaded file.
        filename: Name to save the file as.
        
    Returns:
        The full path where the file was saved.
    """
    # Create documents directory if it doesn't exist
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    logger.info(f"Saved uploaded file to: {file_path}")
    return file_path
