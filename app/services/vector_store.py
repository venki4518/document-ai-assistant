"""
Vector Store Service (ChromaDB).

This file manages all interactions with ChromaDB, our vector database.

What is a Vector Database?
- A regular database stores text, numbers, dates, etc.
- A vector database stores "embeddings" — numerical representations of text.
- It can quickly find similar items by comparing these numbers.
- Think of it as a smart search engine that understands meaning, not just keywords.

How it fits in the architecture:
  Document chunks → THIS MODULE generates embeddings + stores in ChromaDB
  User question → THIS MODULE searches ChromaDB for similar chunks

Why ChromaDB?
- It's simple to set up (no external server needed).
- It runs locally and persists data to disk.
- It integrates well with LangChain.
- Perfect for learning and small-to-medium projects.
"""

import logging
from typing import List, Dict, Optional

import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import (
    OPENAI_API_KEY,
    NVIDIA_BASE_URL,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    TOP_K,
)

logger = logging.getLogger(__name__)


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Get or create a persistent ChromaDB client.
    
    This client allows direct interaction with the local ChromaDB database
    stored on disk in CHROMA_PERSIST_DIR.
    """
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


class NvidiaEmbeddingsWrapper:
    """Wrapper to supply NVIDIA's required 'passage' and 'query' input_type tags."""
    def __init__(self):
        self.doc_embeddings = OpenAIEmbeddings(
            base_url=NVIDIA_BASE_URL,
            api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
            check_embedding_ctx_length=False,
            model_kwargs={"extra_body": {"input_type": "passage"}},
        )
        self.query_embeddings = OpenAIEmbeddings(
            base_url=NVIDIA_BASE_URL,
            api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
            check_embedding_ctx_length=False,
            model_kwargs={"extra_body": {"input_type": "query"}},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.doc_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.query_embeddings.embed_query(text)


def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_store() -> Chroma:
    """
    Get or create the ChromaDB vector store for LangChain.
    
    This creates a persistent ChromaDB instance. "Persistent" means
    the data is saved to disk (in the chroma_db/ folder) and survives
    application restarts.
    
    Returns:
        A Chroma vector store instance.
    """
    embedding_function = get_embedding_function()
    client = get_chroma_client()
    
    vector_store = Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    
    return vector_store


def store_document_chunks(chunks: List[Dict]) -> int:
    """
    Store document chunks with their embeddings in ChromaDB.
    
    This function:
    1. Takes the text chunks created by document_processor.py.
    2. Converts each chunk into a LangChain Document (with text + metadata).
    3. Generates embeddings for each chunk using OpenAI.
    4. Stores everything in ChromaDB.
    
    Args:
        chunks: List of chunk dictionaries with 'text' and 'metadata' keys.
        
    Returns:
        The number of chunks stored.
    """
    logger.info(f"Storing {len(chunks)} chunks in ChromaDB")
    
    # Convert our chunk dictionaries into LangChain Document objects.
    # LangChain Documents have two fields:
    # - page_content: the actual text
    # - metadata: a dictionary of extra information (filename, page, etc.)
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"],
        )
        documents.append(doc)
    
    # Get the vector store and add documents.
    # ChromaDB automatically generates embeddings using our embedding function
    # and stores them alongside the text and metadata.
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    
    logger.info(f"Successfully stored {len(documents)} chunks in ChromaDB")
    return len(documents)


def retrieve_relevant_chunks(
    question: str,
    top_k: int = TOP_K
) -> List[Document]:
    """
    Search ChromaDB for chunks most relevant to the question.
    
    How similarity search works:
    1. The question is converted into an embedding (a vector of numbers).
    2. ChromaDB compares this question vector with all stored chunk vectors.
    3. It finds the chunks whose vectors are most similar (closest in
       mathematical space) to the question vector.
    4. It returns the top_k most similar chunks.
    
    Think of it like this: if the question is about "annual leave",
    its embedding will be similar to chunks that discuss leave policies,
    vacation days, time off, etc.
    
    Args:
        question: The user's question string.
        top_k: Number of most relevant chunks to retrieve.
        
    Returns:
        A list of LangChain Document objects (most relevant chunks).
    """
    logger.info(f"Searching for chunks relevant to: '{question}'")
    
    vector_store = get_vector_store()
    
    # similarity_search converts the question to an embedding and
    # finds the most similar document chunks in ChromaDB.
    results = vector_store.similarity_search(query=question, k=top_k)
    
    logger.info(f"Retrieved {len(results)} relevant chunks")
    return results


def get_stored_documents() -> List[str]:
    """
    Get a list of all unique document filenames stored in ChromaDB.
    
    This reads metadata directly from ChromaDB without needing
    an embedding model or API key.
    
    Returns:
        A list of unique filenames.
    """
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        results = collection.get()
        
        if not results or not results["metadatas"]:
            return []
        
        # Extract unique filenames from metadata
        filenames = set()
        for metadata in results["metadatas"]:
            if metadata and "filename" in metadata:
                filenames.add(metadata["filename"])
        
        return sorted(list(filenames))
    except Exception as e:
        logger.error(f"Error getting stored documents: {e}")
        return []


def delete_document_chunks(filename: str) -> bool:
    """
    Delete all chunks associated with a specific document from ChromaDB.
    
    This deletes records directly from ChromaDB by metadata filter
    without needing an embedding model or API key.
    
    Args:
        filename: The name of the document to delete.
        
    Returns:
        True if deletion was successful, False otherwise.
    """
    logger.info(f"Deleting chunks for document: {filename}")
    
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        
        # Get IDs of all chunks belonging to this document
        results = collection.get(where={"filename": filename})
        
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for '{filename}'")
            return True
        else:
            logger.warning(f"No chunks found for document: {filename}")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting document chunks: {e}")
        raise
