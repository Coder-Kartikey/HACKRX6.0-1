# Application for the HackRx 6.0 Challenge
# VERSION 3: Using Google Gemini API for high-quality answers.

# --- 1. Imports ---
import os
import time
import requests
import fitz  
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List
from fastapi.middleware.cors import CORSMiddleware


# --- Imports for Google Gemini ---
import google.generativeai as genai

# --- 2. Configuration & API Keys ---

AUTH_TOKEN = os.environ.get("HACKRX_AUTH_TOKEN", "b7be0d0c6cb51a6c84e190a66d4542526361d32d3df9035b4c8a00b9198df385")

# --- Configure Gemini API Key ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyACG4pGMiV1APn0CXTfGxIduUzSmFiY3hc")

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
    print("WARNING: GOOGLE_API_KEY is not set. The application will not work.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Google Gemini API configured.")


# --- 3. FastAPI App Initialization ---
app = FastAPI(
    title="HackRx 6.0 LLM Document Processing System (Google Gemini)",
    description="An API to process documents and answer questions using the Gemini API.",
    version="3.2.0"
)

# CORS Middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document to be processed.")
    questions: List[str] = Field(..., description="A list of questions to be answered based on the document.")

class AnswerResponse(BaseModel):
    answers: List[str]

# --- 5. Security ---
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != AUTH_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# --- 6. Core RAG Pipeline Components ---
class RAGPipeline:
    def __init__(self):
        self.text_chunks = []
        self.vector_store = None
        # Initialize the generative model once
        self.generative_model = genai.GenerativeModel('gemini-1.5-flash-latest')

    def _download_and_extract_text(self, pdf_url: str) -> str:
        try:
            response = requests.get(pdf_url, timeout=20)
            response.raise_for_status()
            pdf_bytes = response.content
            text = ""
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        self.text_chunks = chunks
        return chunks

    def _get_embeddings(self, texts: List[str], task_type: str) -> np.ndarray:
        # --- MODIFIED: Implemented batch processing to prevent timeouts ---
        all_embeddings = []
        # Process the texts in batches of 32 (a safe number for the API)
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            try:
                print(f"  - Processing embedding batch {i//32 + 1}...")
                result = genai.embed_content(
                    model='models/embedding-001',
                    content=batch,
                    task_type=task_type
                )
                all_embeddings.extend(result['embedding'])
                # A small delay to be a good API citizen and avoid rate limits
                time.sleep(0.5) 
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed on batch {i//32 + 1}: {e}")
        
        return np.array(all_embeddings).astype('float32')

    def _build_faiss_index(self, embeddings: np.ndarray):
        try:
            import faiss
        except ImportError:
            raise HTTPException(status_code=500, detail="FAISS library not found. Run 'pip install faiss-cpu'.")
        
        if embeddings.shape[0] == 0:
            self.vector_store = None
            return
        
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
            
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        self.vector_store = index

    def _search_faiss_index(self, query_embedding: np.ndarray, k: int = 5) -> List[str]:
        if self.vector_store is None or self.vector_store.ntotal == 0:
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        distances, indices = self.vector_store.search(query_embedding, k)
        return [self.text_chunks[i] for i in indices[0] if i < len(self.text_chunks)]

    def _ask_llm_with_context(self, question: str, context_chunks: List[str]) -> str:
        context = "\n\n---\n\n".join(context_chunks)
        
        prompt = f"""
        You are an expert insurance assistant. Your goal is to provide clear, helpful, and easy-to-understand answers based on the provided policy document excerpts.

        **Instructions:**
        1.  **IMPORTANT:** Your final output must be clean, plain text. Do not use any special characters, Markdown, bullet points (*), bolding, or asterisks.
        2.  Structure your answer in simple, easy-to-read paragraphs.
        3.  Start with a direct summary sentence that immediately answers the user's core question.
        4.  After the summary, explain the details in a helpful and friendly tone.
        5.  If the information isn't in the provided context, politely state that and suggest the user check the full policy document.
        6.  Rely ONLY on the information given in the context below. Do not use outside knowledge.

        **CONTEXT:**
        {context}

        **QUESTION:**
        {question}

        **CLEAN PLAIN-TEXT ANSWER:**
        """
        
        try:
            # Adding safety settings to prevent the model from refusing to answer
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            response = self.generative_model.generate_content(prompt, safety_settings=safety_settings)
            # Clean up any residual newlines or asterisks just in case
            clean_text = response.text.strip().replace('*', '').replace('  ', ' ')
            return clean_text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get answer from Gemini API: {e}")

    def process_query(self, pdf_url: str, questions: List[str]) -> List[str]:
        print("Step 1: Downloading and extracting text...")
        document_text = self._download_and_extract_text(pdf_url)
        
        print("Step 2: Splitting text into chunks...")
        chunks = self._split_text_into_chunks(document_text)
        if not chunks:
            return ["Could not process the document or it is empty."] * len(questions)
            
        print("Step 3: Generating embeddings for document chunks...")
        chunk_embeddings = self._get_embeddings(chunks, task_type="RETRIEVAL_DOCUMENT")
        
        print("Step 4: Building FAISS index...")
        self._build_faiss_index(chunk_embeddings)
        
        answers = []
        for i, question in enumerate(questions):
            print(f"Step 5.{i+1}: Processing question: '{question}'")
            # For a single question, batching isn't needed
            query_embedding = self._get_embeddings([question], task_type="RETRIEVAL_QUERY")
            context_chunks = self._search_faiss_index(query_embedding, k=5)
            answer = self._ask_llm_with_context(question, context_chunks)
            answers.append(answer)
            print(f" -> Answer found: '{answer[:100]}...'")

        return answers

# --- 7. API Endpoint Definition ---

@app.post("/hackrx/run", response_model=AnswerResponse)
async def run_submission(
    query_request: QueryRequest,
    token: str = Depends(get_current_user)
):
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        raise HTTPException(status_code=500, detail="Google API key is not configured on the server.")
        
    try:
        pipeline = RAGPipeline()
        answers = pipeline.process_query(pdf_url=query_request.documents, questions=query_request.questions)
        return {"answers": answers}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "HackRx 6.0 LLM API (Google Gemini) is running."}
