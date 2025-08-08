# HackRx 6.0 LLM Document Processing API

## Overview
This project is a FastAPI application designed for the HackRx 6.0 Challenge. It processes PDF documents and answers questions based on their content using a Retrieval-Augmented Generation (RAG) pipeline integrated with Gemini API.


## Requirements

- Python 3.8+
- Google Gemini API key
- (Optional) OpenAI API key if using OpenAI embeddings
- The following Python packages:
    - fastapi[all]
    - uvicorn
    - python-multipart
    - requests
    - PyMuPDF
    - numpy
    - faiss-cpu
    - google-generativeai

## Installation

1. **Clone the repository** (if applicable):
   ```
   git clone <repository-url>
   cd HACKRX6.0
   ```

2. **Install necessary packages**:
   ```
   pip install "fastapi[all]" uvicorn python-multipart requests PyMuPDF numpy faiss-cpu google-generativeai
   ```

## Configuration

3. **Set your environment variables** (recommended):
   - For Linux/Mac:
     ```
     export HACKRX_AUTH_TOKEN="b7be0d0c6cb51a6c84e190a66d4542526361d32d3df9035b4c8a00b9198df385"
     ```
   - For Windows (Command Prompt):
     ```
     set HACKRX_AUTH_TOKEN="b7be0d0c6cb51a6c84e190a66d4542526361d32d3df9035b4c8a00b9198df385"
     ```

4. **Run the server**:
   ```
   uvicorn main:app --reload
   ```

## Usage

5. **Send a request using a tool like curl or Postman**:
   Example request using curl:
   ```
   curl -X POST http://127.0.0.1:8000/hackrx/run \
   -H "Content-Type: application/json" \
   -H "Authorization: Bearer b7be0d0c6cb51a6c84e190a66d4542526361d32d3df9035b4c8a00b9198df385" \
   -d '{
       "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
       "questions": [
           "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
           "What is the waiting period for pre-existing diseases (PED) to be covered?"
       ]
   }'
   ```
# Owner 
CoderKP & Flames Team