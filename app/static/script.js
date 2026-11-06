/**
 * AI Document Q&A Assistant - Frontend JavaScript
 * 
 * This file handles all user interactions:
 * - Uploading documents to the /upload endpoint
 * - Sending questions to the /ask endpoint
 * - Displaying answers and sources
 * - Managing the document list
 * 
 * It uses plain JavaScript (no frameworks like React) for simplicity.
 */

// --- Document Upload ---
async function uploadDocument() {
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const statusDiv = document.getElementById('uploadStatus');

    // Validate that a file is selected
    if (!fileInput.files || fileInput.files.length === 0) {
        showStatus(statusDiv, 'Please select a file first.', 'error');
        return;
    }

    const file = fileInput.files[0];

    // Prepare the file for upload using FormData
    // FormData is the standard way to send files via HTTP
    const formData = new FormData();
    formData.append('file', file);

    // Disable button and show loading state
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading...';
    showLoading(true);

    try {
        // Send the file to our FastAPI backend
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (response.ok) {
            showStatus(
                statusDiv,
                `✅ ${data.message} (${data.chunks_created} chunks created)`,
                'success'
            );
            // Refresh the document list after successful upload
            loadDocuments();
            // Clear the file input
            fileInput.value = '';
        } else {
            showStatus(
                statusDiv,
                `❌ Error: ${data.detail || 'Upload failed'}`,
                'error'
            );
        }
    } catch (error) {
        showStatus(
            statusDiv,
            `❌ Network error: ${error.message}`,
            'error'
        );
    } finally {
        // Re-enable the button
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload Document';
        showLoading(false);
    }
}

// --- Ask Question ---
async function askQuestion() {
    const questionInput = document.getElementById('questionInput');
    const askBtn = document.getElementById('askBtn');
    const answerSection = document.getElementById('answerSection');
    const answerContent = document.getElementById('answerContent');
    const sourcesSection = document.getElementById('sourcesSection');
    const sourcesList = document.getElementById('sourcesList');

    const question = questionInput.value.trim();

    // Validate that a question was entered
    if (!question) {
        alert('Please enter a question.');
        return;
    }

    // Disable button and show loading state
    askBtn.disabled = true;
    askBtn.textContent = 'Thinking...';
    showLoading(true);
    answerSection.classList.add('hidden');

    try {
        // Send the question to our FastAPI backend
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question }),
        });

        const data = await response.json();

        if (response.ok) {
            // Display the answer
            answerContent.textContent = data.answer;
            answerSection.classList.remove('hidden');

            // Display sources if available
            if (data.sources && data.sources.length > 0) {
                sourcesList.innerHTML = '';
                data.sources.forEach(source => {
                    const sourceItem = document.createElement('div');
                    sourceItem.className = 'source-item';
                    
                    let sourceText = `📄 ${source.filename}`;
                    if (source.page) {
                        sourceText += ` — Page ${source.page}`;
                    }
                    sourceText += ` — Chunk ${source.chunk}`;
                    
                    sourceItem.textContent = sourceText;
                    sourcesList.appendChild(sourceItem);
                });
                sourcesSection.classList.remove('hidden');
            } else {
                sourcesSection.classList.add('hidden');
            }
        } else {
            answerContent.textContent = `Error: ${data.detail || 'Something went wrong'}`;
            answerSection.classList.remove('hidden');
            sourcesSection.classList.add('hidden');
        }
    } catch (error) {
        answerContent.textContent = `Network error: ${error.message}`;
        answerSection.classList.remove('hidden');
        sourcesSection.classList.add('hidden');
    } finally {
        // Re-enable the button
        askBtn.disabled = false;
        askBtn.textContent = 'Ask Question';
        showLoading(false);
    }
}

// --- Load Document List ---
async function loadDocuments() {
    const documentList = document.getElementById('documentList');

    try {
        const response = await fetch('/documents');
        const data = await response.json();

        if (data.documents && data.documents.length > 0) {
            documentList.innerHTML = '';
            data.documents.forEach(doc => {
                const docItem = document.createElement('div');
                docItem.className = 'document-item';
                docItem.innerHTML = `
                    <span class="doc-name">📄 ${doc}</span>
                    <button class="delete-btn" onclick="deleteDocument('${doc}')">
                        🗑️ Delete
                    </button>
                `;
                documentList.appendChild(docItem);
            });
        } else {
            documentList.innerHTML = 
                '<p class="empty-state">No documents uploaded yet.</p>';
        }
    } catch (error) {
        documentList.innerHTML = 
            '<p class="empty-state">Could not load documents.</p>';
    }
}

// --- Delete Document ---
async function deleteDocument(filename) {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
        return;
    }

    try {
        const response = await fetch(`/documents/${filename}`, {
            method: 'DELETE',
        });

        const data = await response.json();

        if (response.ok) {
            // Refresh the document list
            loadDocuments();
        } else {
            alert(`Error: ${data.detail || 'Delete failed'}`);
        }
    } catch (error) {
        alert(`Network error: ${error.message}`);
    }
}

// --- Helper: Show Status Message ---
function showStatus(element, message, type) {
    element.textContent = message;
    element.className = `status ${type}`;
    element.classList.remove('hidden');
}

// --- Helper: Show/Hide Loading Spinner ---
function showLoading(show) {
    const loading = document.getElementById('loading');
    if (show) {
        loading.classList.remove('hidden');
    } else {
        loading.classList.add('hidden');
    }
}

// --- Helper: Handle Enter Key in Question Input ---
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        askQuestion();
    }
}

// --- Load documents when the page first loads ---
window.addEventListener('DOMContentLoaded', loadDocuments);
