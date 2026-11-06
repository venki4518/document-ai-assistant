"""
Application entry point.

This file provides a simple way to start the application:
    python run.py

It's equivalent to running:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

The --reload flag automatically restarts the server when you change code.
This is useful during development but should be removed in production.
"""

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting AI Document Q&A Assistant...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🌐 Frontend: http://localhost:8000")
    print("")
    
    uvicorn.run(
        "app.main:app",  # Path to the FastAPI app instance
        host="0.0.0.0",   # Listen on all network interfaces
        port=8000,         # Port number
        reload=True,       # Auto-reload on code changes (dev only)
    )
