"""
RAG (Retrieval-Augmented Generation) Service.

This is the central orchestrator of the RAG pipeline. It connects
all the pieces together:

1. RETRIEVAL: Search ChromaDB for relevant document chunks.
2. CONTEXT BUILDING: Combine the retrieved chunks into a context string.
3. GENERATION: Send the context + question to the LLM for an answer.

What is RAG?
- RAG stands for Retrieval-Augmented Generation.
- Instead of asking an LLM to answer from its general knowledge,
  we first RETRIEVE relevant information from our documents,
  then AUGMENT the LLM's prompt with that information,
  and let it GENERATE an answer based on the retrieved context.
- This makes the LLM's answers grounded in actual document data.

The RAG Pipeline:
  Question → Embed Question → Search ChromaDB → Get Relevant Chunks →
  Build Context → Send to LLM → Get Answer → Return Answer + Sources
"""

import logging
from typing import Dict, List
from app.services.vector_store import retrieve_relevant_chunks, get_stored_documents
from app.services.llm_service import generate_answer
from app.config import TOP_K

logger = logging.getLogger(__name__)


def ask_question(question: str) -> Dict:
    """
    Process a user's question through the complete RAG pipeline.
    
    This is the main function that orchestrates the entire Q&A process:
    
    Step 1: Check if any documents are uploaded.
    Step 2: Retrieve relevant chunks from ChromaDB (Retrieval).
    Step 3: Build a context string from the chunks (Augmentation).
    Step 4: Send context + question to the LLM (Generation).
    Step 5: Format and return the answer with source references.
    
    Args:
        question: The user's question string.
        
    Returns:
        A dictionary with 'question', 'answer', and 'sources' keys.
    """
    logger.info(f"Processing question: '{question}'")
    
    # Step 1: Check if any documents have been uploaded
    stored_docs = get_stored_documents()
    if not stored_docs:
        return {
            "question": question,
            "answer": "No documents have been uploaded yet. Please upload a document first.",
            "sources": [],
        }
    
    # Step 2: RETRIEVAL - Find the most relevant document chunks
    # This converts the question to an embedding and searches ChromaDB
    relevant_chunks = retrieve_relevant_chunks(question, top_k=TOP_K)
    
    if not relevant_chunks:
        return {
            "question": question,
            "answer": "I couldn't find any relevant information in the uploaded documents.",
            "sources": [],
        }
    
    # Step 3: AUGMENTATION - Build the context from retrieved chunks
    # We combine all relevant chunks into a single context string.
    # Each chunk is separated by a line for clarity.
    context_parts = []
    for i, chunk in enumerate(relevant_chunks, 1):
        context_parts.append(f"--- Chunk {i} ---\n{chunk.page_content}")
    
    context = "\n\n".join(context_parts)
    
    # Step 4: GENERATION - Send context + question to the LLM
    answer = generate_answer(context=context, question=question)
    
    # Step 5: Extract source information from chunk metadata
    sources = extract_sources(relevant_chunks)
    
    logger.info("Question processed successfully")
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


def extract_sources(chunks) -> List[Dict]:
    """
    Extract unique source references from retrieved chunks.
    
    We deduplicate sources so the user doesn't see the same
    file + page + chunk listed multiple times.
    
    Args:
        chunks: List of LangChain Document objects with metadata.
        
    Returns:
        A list of source dictionaries with filename, page, and chunk.
    """
    sources = []
    seen = set()  # Track seen sources to avoid duplicates
    
    for chunk in chunks:
        metadata = chunk.metadata
        filename = metadata.get("filename", "Unknown")
        page = metadata.get("page")  # None for TXT files
        chunk_number = metadata.get("chunk_number", 0)
        
        # Create a unique key for deduplication
        source_key = (filename, page, chunk_number)
        
        if source_key not in seen:
            seen.add(source_key)
            sources.append({
                "filename": filename,
                "page": page,
                "chunk": chunk_number,
            })
    
    return sources
