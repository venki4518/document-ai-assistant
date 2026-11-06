# 📄 AI Document Q&A Assistant

A beginner-friendly **Retrieval-Augmented Generation (RAG)** application that lets you upload PDF and TXT documents and ask questions about them. The app retrieves relevant information from your documents and uses an AI to generate accurate answers — grounded in your data, not imagination.

---

## 📋 Table of Contents

1. [What Problem Does This Solve?](#what-problem-does-this-solve)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Technologies Used](#technologies-used)
5. [Key Concepts Explained](#key-concepts-explained)
6. [Project Structure](#project-structure)
7. [Installation](#installation)
8. [Environment Setup](#environment-setup)
9. [How to Run](#how-to-run)
10. [How to Test](#how-to-test)
11. [API Endpoints](#api-endpoints)
12. [Example Usage](#example-usage)
13. [Demo Questions](#demo-questions)
14. [Limitations](#limitations)
15. [Future Improvements](#future-improvements)

---

## ❓ What Problem Does This Solve?

Imagine you have a 100-page company handbook and need to find information about the leave policy. You could:

1. **Read the entire document** — slow and tedious.
2. **Use Ctrl+F** — only finds exact keyword matches, not meaning.
3. **Use this app** — ask in natural language and get a precise answer with source references!

This application combines **document search** with **AI-powered answering** so you can query your documents conversationally.

---

## ✨ Features

- 📤 **Upload Documents** — Support for PDF and TXT files (up to 10 MB)
- 🔍 **Smart Search** — Finds relevant information using semantic similarity, not just keywords
- 🤖 **AI Answers** — Generates human-readable answers using OpenAI GPT
- 📌 **Source References** — Shows exactly where the answer came from (file, page, chunk)
- 🛡️ **Hallucination Control** — The AI only answers based on your documents, not general knowledge
- 📚 **Document Management** — List and delete uploaded documents
- 🌐 **Web Interface** — Clean, simple frontend (no React needed)
- 📖 **API Documentation** — Auto-generated Swagger docs at `/docs`
- 🧪 **Tested** — Unit tests for core functionality

---

## 🏗️ Architecture

Here's how the application works, step by step:

### Document Upload Flow

```
User uploads PDF/TXT
        ↓
   FastAPI receives file
        ↓
   Validate file type & size
        ↓
   Extract text (PyPDF for PDF, Python for TXT)
        ↓
   Split text into chunks (1000 chars, 200 overlap)
        ↓
   Generate embeddings (OpenAI text-embedding-3-small)
        ↓
   Store chunks + embeddings in ChromaDB
        ↓
   Return success + chunk count
```

### Question-Answering Flow (RAG Pipeline)

```
User asks a question
        ↓
   Generate embedding for the question
        ↓
   Similarity search in ChromaDB
        ↓
   Retrieve top 4 relevant chunks
        ↓
   Build context from chunks
        ↓
   Send context + question to LLM (GPT-4o-mini)
        ↓
   LLM generates answer based ONLY on context
        ↓
   Return answer + source references
```

### Component Diagram

```
┌──────────────┐
│   Frontend   │  (HTML/CSS/JS)
│  index.html  │
└──────┬───────┘
       │ HTTP
       ↓
┌──────────────┐
│   FastAPI     │  (main.py - API routes)
│   REST API    │
└──────┬───────┘
       │
       ├──→ document_processor.py  (extract text, split chunks)
       │
       ├──→ vector_store.py        (ChromaDB + embeddings)
       │
       ├──→ rag_service.py         (orchestrates the RAG pipeline)
       │
       └──→ llm_service.py         (sends prompts to OpenAI GPT)
```

---

## 🛠️ Technologies Used

| Technology | Purpose | Why We Use It |
|-----------|---------|---------------|
| **Python** | Programming language | Most popular for AI/ML applications |
| **FastAPI** | Web framework / REST API | Fast, modern, auto-generates API docs |
| **LangChain** | LLM framework | Simplifies working with LLMs, embeddings, and vector stores |
| **OpenAI API** | Embeddings + LLM | High-quality embeddings and GPT models |
| **ChromaDB** | Vector database | Simple, local, no server setup needed |
| **PyPDF** | PDF text extraction | Reliable, pure-Python PDF reader |
| **Uvicorn** | ASGI server | Runs our FastAPI application |
| **python-dotenv** | Environment variables | Loads API keys from `.env` file securely |
| **Pydantic** | Data validation | Validates API request/response data |

---

## 🧠 Key Concepts Explained

### What is an LLM (Large Language Model)?

An LLM is an AI model trained on massive amounts of text data. It can understand and generate human-like text. Examples include GPT-4, GPT-4o-mini. Think of it as a very smart autocomplete that understands context and meaning.

### What is an Embedding?

An embedding converts text into a list of numbers (a "vector"). Similar texts produce similar numbers. For example:
- "cat" → [0.2, 0.8, 0.1, ...]
- "kitten" → [0.2, 0.7, 0.2, ...]  (very similar!)
- "airplane" → [0.9, 0.1, 0.6, ...]  (very different!)

This allows us to mathematically compare how similar two pieces of text are.

### What is a Vector Database?

A regular database stores text, numbers, and dates. A vector database stores **embeddings** (lists of numbers) and can quickly find the most similar ones. ChromaDB is our vector database — it stores document chunk embeddings and finds relevant chunks when you ask a question.

### What is a Retriever?

A retriever is the component that searches the vector database to find the most relevant document chunks for a given question. It converts the question into an embedding and finds the closest matching chunks.

### What is RAG (Retrieval-Augmented Generation)?

RAG is a technique that combines **retrieval** (finding relevant documents) with **generation** (creating an answer using an LLM). Instead of asking the LLM to answer from its general knowledge, we:

1. **Retrieve** relevant chunks from our documents.
2. **Augment** the LLM's prompt with those chunks.
3. Let the LLM **Generate** an answer based on the retrieved context.

This makes answers accurate and grounded in your actual documents.

### What is LangChain?

LangChain is a Python framework that makes it easy to build applications with LLMs. It provides ready-made components for:
- Text splitting (chunking documents)
- Embeddings (converting text to vectors)
- Vector stores (connecting to databases like ChromaDB)
- LLM chains (sending prompts to models like GPT)

It saves us from writing a lot of boilerplate code.

### What is FastAPI?

FastAPI is a modern Python web framework for building REST APIs. It's:
- **Fast** to run and develop with
- **Auto-documents** your API (Swagger UI at `/docs`)
- **Validates** data automatically using Pydantic models
- **Async-capable** for handling multiple requests

---

## 📁 Project Structure

```
ai-document-qa/
│
├── app/                          # Main application package
│   ├── __init__.py               # Makes 'app' a Python package
│   ├── main.py                   # FastAPI routes (API endpoints)
│   ├── config.py                 # Configuration settings
│   ├── models.py                 # Pydantic request/response models
│   │
│   ├── services/                 # Business logic
│   │   ├── __init__.py
│   │   ├── document_processor.py # Text extraction & chunking
│   │   ├── vector_store.py       # ChromaDB operations & embeddings
│   │   ├── llm_service.py        # OpenAI LLM integration
│   │   └── rag_service.py        # RAG pipeline orchestrator
│   │
│   └── static/                   # Frontend files
│       ├── index.html            # Web interface
│       ├── style.css             # Styles
│       └── script.js             # JavaScript logic
│
├── documents/                    # Uploaded documents storage
│   └── sample_company_policy.txt # Sample document for testing
│
├── chroma_db/                    # ChromaDB persistent storage (auto-created)
│
├── tests/                        # Test files
│   ├── __init__.py
│   ├── test_document_processor.py # Tests for document processing
│   └── test_api.py               # Tests for API endpoints
│
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── run.py                        # Application entry point
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.9+** installed on your system
- **OpenAI API key** (get one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys))

### Step-by-Step Setup (Windows)

```bash
# 1. Navigate to the project directory
cd ai-document-qa

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment (Windows)
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Step-by-Step Setup (macOS/Linux)

```bash
# 1. Navigate to the project directory
cd ai-document-qa

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 🔐 Environment Setup

```bash
# 1. Copy the example environment file
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux

# 2. Edit .env and add your OpenAI API key
# Replace 'your_api_key_here' with your actual key
```

Your `.env` file should look like:

```
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
```

> ⚠️ **IMPORTANT**: Never commit your `.env` file to Git. It's already in `.gitignore`.

---

## ▶️ How to Run

```bash
# Make sure your virtual environment is activated and .env is configured

# Option 1: Using run.py
python run.py

# Option 2: Using uvicorn directly
uvicorn app.main:app --reload
```

The application will start at:
- **Frontend**: [http://localhost:8000](http://localhost:8000)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 How to Test

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_document_processor.py -v
pytest tests/test_api.py -v
```

> **Note**: Tests use mocking so they do NOT require an OpenAI API key. They test local logic and API structure.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the frontend web page |
| `POST` | `/upload` | Upload a PDF or TXT document |
| `POST` | `/ask` | Ask a question about uploaded documents |
| `GET` | `/documents` | List all uploaded documents |
| `DELETE` | `/documents/{filename}` | Delete a document and its vectors |
| `GET` | `/docs` | Swagger API documentation |

### POST /upload

Upload a document for processing.

**Request**: `multipart/form-data` with a `file` field.

**Response**:
```json
{
    "message": "Document uploaded and processed successfully",
    "filename": "company_policy.pdf",
    "chunks_created": 25
}
```

### POST /ask

Ask a question about uploaded documents.

**Request**:
```json
{
    "question": "What is the annual leave policy?"
}
```

**Response**:
```json
{
    "question": "What is the annual leave policy?",
    "answer": "All full-time employees are entitled to 20 days of paid annual leave per calendar year.",
    "sources": [
        {
            "filename": "sample_company_policy.txt",
            "page": null,
            "chunk": 3
        }
    ]
}
```

### GET /documents

**Response**:
```json
{
    "documents": [
        "sample_company_policy.txt",
        "employee_handbook.pdf"
    ]
}
```

---

## 💡 Example Usage

1. **Start the application**: `python run.py`
2. **Open the browser**: Go to `http://localhost:8000`
3. **Upload the sample document**: Click "Choose File", select `documents/sample_company_policy.txt`, click "Upload"
4. **Ask a question**: Type "What is the annual leave policy?" and click "Ask Question"
5. **View the answer**: The AI will respond with information from the document, along with source references

---

## 🎯 Demo Questions

Try these questions after uploading the sample company policy document:

| Question | Expected Behavior |
|----------|-------------------|
| "What is the annual leave policy?" | Answers about 20 days of leave |
| "What are the working hours?" | Answers about 9 AM to 5:30 PM |
| "Can employees work remotely?" | Answers about 3 days/week hybrid |
| "How many holidays are provided?" | Answers about 10 public holidays |
| "What benefits are mentioned?" | Answers about health, dental, 401k, etc. |
| "What is the capital of France?" | Should say it can't find this in the documents |

The last question demonstrates the **hallucination control** — the AI won't use its general knowledge to answer; it will admit the information isn't in the documents.

---

## ⚠️ Limitations

- **Requires OpenAI API key** — the application needs a paid API key for embeddings and LLM.
- **No authentication** — anyone with access to the URL can upload and query documents.
- **English-focused** — performance may vary with other languages.
- **Simple chunking** — uses character-based splitting, not semantic chunking.
- **Single collection** — all documents are stored in one ChromaDB collection.
- **No OCR** — PDFs with scanned images (no selectable text) won't work.
- **Local only** — designed for local development, not production deployment.

---

## 🚀 Future Improvements

- Add user authentication
- Support more file formats (DOCX, CSV, Markdown)
- Implement conversation history (follow-up questions)
- Add OCR support for scanned PDFs
- Deploy to cloud (AWS, GCP, Azure)
- Add streaming responses for real-time answer generation
- Implement semantic chunking for better retrieval
- Add rate limiting for API protection
- Support multiple vector database backends

---

## 📝 License

This project is for educational purposes. Feel free to modify and use it as you wish.
