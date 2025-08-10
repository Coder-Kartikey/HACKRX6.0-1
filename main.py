# main.py
# Complete FastAPI application for the HackRx 6.0 Challenge
# VERSION 13.0: Final version with true API Key Rotation using a direct HTTP client to bypass global state issues.

# --- 1. Imports ---
import os
import time
import asyncio
import requests
import fitz  # PyMuPDF
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict
from itertools import cycle # For API key rotation
from fastapi.middleware.cors import CORSMiddleware
import aiohttp # For async HTTP requests

# --- Imports for Google Gemini (used for embeddings only now) ---
import google.generativeai as genai

# --- Imports for LangChain and Keyword Search ---
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from langchain_community.document_transformers import LongContextReorder

# --- 2. Configuration & API Keys ---

AUTH_TOKEN = os.environ.get("HACKRX_AUTH_TOKEN", "b7be0d0c6cb51a6c84e190a66d4542526361d32d3df9035b4c8a00b9198df385")

# --- NEW: List of API Keys for Rotation ---
GOOGLE_API_KEYS = [
    os.environ.get("GOOGLE_API_KEY_1", "AIzaSyCpci8OJlkTtWAwt99_3wLETP5yIuBsn6E"),
    os.environ.get("GOOGLE_API_KEY_2", "AIzaSyDwV1kqdmc7-FkhaqBqLQfjI-f__Y8Rc3E"),
    os.environ.get("GOOGLE_API_KEY_3", "AIzaSyAag-9G6_aOc6RsxxfEBGZ1WL4gvJcGAGo"),
]

# Filter out placeholder keys
GOOGLE_API_KEYS = [key for key in GOOGLE_API_KEYS if "HERE" not in key]

if not GOOGLE_API_KEYS:
    print("WARNING: No GOOGLE_API_KEYs are set. The application will not work.")
else:
    print(f"Found {len(GOOGLE_API_KEYS)} API keys for rotation.")
    # Configure the SDK with the first key for the initial embedding process
    genai.configure(api_key=GOOGLE_API_KEYS[0])


# --- 3. FastAPI App Initialization ---
app = FastAPI(
    title="HackRx 6.0 Final RAG System",
    description="An API using an Advanced Fusion Retriever with true API Key Rotation for maximum throughput.",
    version="13.0.0"
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
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials.credentials

# --- 6. Core RAG Pipeline Components ---

class RAGPipeline:
    def __init__(self, api_key_pool: List[str]):
        self.text_chunks = []
        self.vector_store = None
        self.keyword_retriever = None
        self.api_key_cycler = cycle(api_key_pool)
        self.safety_settings = [
            {"category": c, "threshold": "BLOCK_NONE"} for c in 
            ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
             "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]

    # --- Document Loading and Chunking ---
    def load_and_chunk_document(self, pdf_url: str):
        print("Step 1.1: Downloading and extracting text...")
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            pdf_bytes = response.content
            full_text = ""
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page in doc:
                    full_text += page.get_text()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {e}")

        print("Step 1.2: Performing Recursive Chunking...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.text_chunks = text_splitter.split_text(full_text)
        print(f"  - Document split into {len(self.text_chunks)} chunks.")

    # --- Indexing for Hybrid Search ---
    def build_indices(self):
        print("Step 1.3: Building indices for Hybrid Search...")
        if not self.text_chunks:
            return

        self.keyword_retriever = BM25Retriever.from_texts(self.text_chunks)
        self.keyword_retriever.k = 10
        print("  - BM25 keyword retriever built.")

        all_embeddings = self._get_embeddings_batched(self.text_chunks, "RETRIEVAL_DOCUMENT")
        try:
            import faiss
            if all_embeddings.ndim == 1:
                all_embeddings = np.expand_dims(all_embeddings, axis=0)
            index = faiss.IndexFlatL2(all_embeddings.shape[1])
            index.add(all_embeddings)
            self.vector_store = index
            print("  - FAISS vector index built.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to build FAISS index: {e}")

    # --- Final RAG Steps ---

    async def _step_1_fusion_retrieval(self, question: str) -> List[str]:
        print(f"  - Step 1: Performing Fusion Retrieval for '{question[:40]}...'")
        
        keyword_docs = self.keyword_retriever.get_relevant_documents(question)
        
        # Embedding call still uses the SDK as it's a one-off and less prone to race conditions
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(None, self._get_embeddings_batched, [question], "RETRIEVAL_QUERY")
        distances, indices = self.vector_store.search(query_embedding, k=10)
        vector_docs = [Document(page_content=self.text_chunks[i]) for i in indices[0] if i < len(self.text_chunks)]
        
        fused_scores = {}
        for i, doc in enumerate(keyword_docs):
            fused_scores.setdefault(doc.page_content, 0)
            fused_scores[doc.page_content] += 1 / (i + 60)

        for i, doc in enumerate(vector_docs):
            fused_scores.setdefault(doc.page_content, 0)
            fused_scores[doc.page_content] += 1 / (i + 60)

        reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_docs = [Document(page_content=doc[0]) for doc in reranked_results[:8]]
        print(f"    - Found and fused {len(final_docs)} unique candidates.")
        
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(final_docs)
        
        return [doc.page_content for doc in reordered_docs]

    # --- NEW: Direct HTTP call for answer generation ---
    async def _step_2_generate_final_answer(self, session: aiohttp.ClientSession, question: str, context_chunks: List[str], api_key: str) -> str:
        print(f"  - Step 2: Generating final answer for '{question[:40]}...' using key ending in ...{api_key[-4:]}")
        if not context_chunks:
            return "I'm sorry, but I could not find any relevant information in the document to answer your question."

        context = "\n\n---\n\n".join(context_chunks)
        
        prompt = f"""
        You are an expert insurance analyst. Your task is to provide a definitive and accurate answer based on the provided context. Follow these steps carefully (Chain-of-Thought):
        1.  **Analyze Context:** First, internally identify all the key facts, conditions, and exclusions from the context that are directly relevant to the user's question.
        2.  **Reasoning:** Second, internally reason step-by-step how these facts combine to form a complete answer.
        3.  **Final Answer:** Finally, synthesize your reasoning into a clear, helpful, and easy-to-understand final answer for the user.
        **IMPORTANT INSTRUCTIONS:**
        - Your final output must be **only the clean, plain-text final answer**.
        - Do not use any special characters, Markdown, asterisks, or bullet points.
        - Start with a direct summary sentence.
        - Rely ONLY on the information given in the context. If the answer cannot be determined from the context, politely state that.
        **CONTEXT:**
        {context}
        **QUESTION:**
        {question}
        **CLEAN PLAIN-TEXT FINAL ANSWER:**
        """
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "safetySettings": self.safety_settings,
            "generationConfig": {"maxOutputTokens": 500, "temperature": 0.0}
        }

        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                clean_text = result['candidates'][0]['content']['parts'][0]['text'].strip().replace('*', '').replace('  ', ' ')
                return clean_text
            else:
                error_text = await response.text()
                print(f"Error from API with key ...{api_key[-4:]}: {response.status} - {error_text}")
                return f"API error: {response.status} - {error_text}"


    # --- Main Orchestrator for a single question ---
    async def process_single_question_async(self, session: aiohttp.ClientSession, question: str) -> str:
        api_key = next(self.api_key_cycler)
        try:
            context = await self._step_1_fusion_retrieval(question)
            final_answer = await self._step_2_generate_final_answer(session, question, context, api_key)
            return final_answer
        except Exception as e:
            print(f"An error occurred processing question '{question[:40]}...': {e}")
            return f"An error occurred while processing the question: {question}"


    # --- Helper for embedding ---
    def _get_embeddings_batched(self, texts: List[str], task_type: str) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            try:
                result = genai.embed_content(model='models/embedding-001', content=batch, task_type=task_type)
                all_embeddings.extend(result['embedding'])
                time.sleep(0.2)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed on embedding batch {i//32 + 1}: {e}")
        return np.array(all_embeddings).astype('float32')

# --- 7. API Endpoint Definition ---
@app.post("/hackrx/run", response_model=AnswerResponse)
async def run_submission(
    query_request: QueryRequest,
    token: str = Depends(get_current_user)
):
    if not GOOGLE_API_KEYS:
        raise HTTPException(status_code=500, detail="No Google API keys are configured on the server.")
        
    try:
        pipeline = RAGPipeline(api_key_pool=GOOGLE_API_KEYS)
        
        pipeline.load_and_chunk_document(query_request.documents)
        pipeline.build_indices()
        
        questions = query_request.questions
        if not pipeline.vector_store or not pipeline.keyword_retriever:
             return {"answers": ["Could not process the document or it is empty."] * len(questions)}

        # --- NEW: Create a single aiohttp session for all requests ---
        async with aiohttp.ClientSession() as session:
            tasks = [pipeline.process_single_question_async(session, q) for q in questions]
            answers = await asyncio.gather(*tasks)

        return {"answers": list(answers)}

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "HackRx 6.0 Final RAG API is running."}

# --- How to Run This Application ---
# 1. Install necessary packages:
#    pip install "fastapi[all]" uvicorn python-multipart requests PyMuPDF numpy "faiss-cpu" google-generativeai "rank-bm25" "langchain" "langchain-community" "aiohttp"
#
# 2. Add your API keys to the `GOOGLE_API_KEYS` list in this script, or set them as environment variables:
#    export GOOGLE_API_KEY_1="key-1..."
#    export GOOGLE_API_KEY_2="key-2..."
#    export GOOGLE_API_KEY_3="key-3..."
