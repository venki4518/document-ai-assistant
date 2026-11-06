"""
Configuration module for the AI Document Q&A Assistant.

This file centralizes all configuration settings for the application.
It loads environment variables (like the OpenAI API key) and defines
constants used throughout the project.

Why this file exists:
- Keeps all settings in one place instead of scattered across files.
- Makes it easy to change settings without modifying business logic.
- Loads sensitive data (API keys) from environment variables for security.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This reads the .env file in the project root and makes its values
# available through os.getenv(). This way, we never hardcode API keys.
load_dotenv()

# --- NVIDIA NIM Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Gemini API endpoint
NVIDIA_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.5-flash-lite"

# --- Text Chunking Configuration ---
# When we upload a document, we split it into smaller pieces called "chunks".
# This is because:
# 1. LLMs have a limited context window (max input size).
# 2. Smaller chunks allow more precise retrieval of relevant information.
# 3. Embeddings work better on focused, smaller pieces of text.

# Maximum number of characters in each chunk.
CHUNK_SIZE = 1000

# Number of characters that overlap between consecutive chunks.
# Overlap ensures that if important information spans two chunks,
# it won't be lost at the boundary. For example, if a sentence
# starts at the end of chunk 1, it will also appear at the start of chunk 2.
CHUNK_OVERLAP = 200

# --- ChromaDB Configuration ---
# ChromaDB is our vector database. It stores document chunks along with
# their embeddings (numerical representations) for fast similarity search.

# Directory where ChromaDB stores its data persistently.
# "Persistent" means the data survives application restarts.
CHROMA_PERSIST_DIR = "./chroma_db"

# Name of the collection (like a table in a regular database).
CHROMA_COLLECTION_NAME = "documents"

# --- Retrieval Configuration ---
# When a user asks a question, we search for the most relevant chunks.
# TOP_K defines how many chunks to retrieve.
# More chunks = more context for the LLM, but also more noise.
# 4 is a good balance for most use cases.
TOP_K = 4

# --- File Upload Configuration ---
# Allowed file extensions for upload.
ALLOWED_EXTENSIONS = {".txt", ".pdf"}

# Maximum file size in bytes (10 MB).
MAX_FILE_SIZE = 10 * 1024 * 1024

# Directory to store uploaded documents.
DOCUMENTS_DIR = "./documents"
