import os
import sys
import logging
import time
from typing import Dict, Any, List, Optional
from functools import lru_cache
from pyngrok import ngrok, conf
from contextlib import asynccontextmanager

# Add the parent directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.retrieval.retriever import Retriever
from src.models.llm import LLMManager
from src.utils.config import (
    API_HOST,
    API_PORT,
    TOP_K_RESULTS,
    MAX_CACHE_SIZE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the retriever and LLM manager lazily
@lru_cache(maxsize=1)
def get_retriever():
    """Get or initialize the retriever."""
    return Retriever()

@lru_cache(maxsize=1)
def get_llm_manager():
    """Get or initialize the LLM manager."""
    return LLMManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models at startup and gracefully shutdown."""
    logger.info("Application startup: Loading models...")
    get_retriever()
    get_llm_manager()
    logger.info("Models loaded successfully.")
    yield
    logger.info("Application shutdown.")

# Initialize the FastAPI app
app = FastAPI(
    title="RAG Q&A API",
    description="API for RAG-based question answering system",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define API models
class QueryRequest(BaseModel):
    """Request model for the query endpoint."""
    query: str = Field(..., description="The question to answer")
    top_k: int = Field(TOP_K_RESULTS, description="Number of documents to retrieve")
    include_sources: bool = Field(True, description="Whether to include sources in the response")
    temperature: float = Field(0.7, description="Temperature for generation")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional filter for retrieval")

class RetrieveRequest(BaseModel):
    """Request model for the retrieve endpoint."""
    query: str = Field(..., description="The query to retrieve documents for")
    top_k: int = Field(TOP_K_RESULTS, description="Number of documents to retrieve")
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional filter for retrieval")

class QueryResponse(BaseModel):
    """Response model for the query endpoint."""
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="The generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="The retrieved sources")
    elapsed_time: float = Field(..., description="Time taken to process the query")

class RetrieveResponse(BaseModel):
    """Response model for the retrieve endpoint."""
    query: str = Field(..., description="The original query")
    results: List[Dict[str, Any]] = Field(..., description="The retrieved documents")
    elapsed_time: float = Field(..., description="Time taken to process the query")

class HealthResponse(BaseModel):
    """Response model for the health endpoint."""
    status: str = Field(..., description="Health status")
    retriever_ready: bool = Field(..., description="Whether the retriever is ready")
    llm_ready: bool = Field(..., description="Whether the LLM is ready")
    document_count: int = Field(..., description="Number of documents in the vector store")

# Define API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the API."""
    try:
        retriever = get_retriever()
        llm_manager = get_llm_manager()
        
        document_count = retriever.vector_store.count()
        
        return {
            "status": "ok",
            "retriever_ready": True,
            "llm_ready": True,
            "document_count": document_count
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """Retrieve relevant documents for a query."""
    start_time = time.time()
    
    try:
        retriever = get_retriever()
        
        # Retrieve documents
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            where=request.filter
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            "query": request.query,
            "results": results,
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Generate an answer for a query."""
    start_time = time.time()
    
    try:
        retriever = get_retriever()
        llm_manager = get_llm_manager()
        
        # Retrieve relevant documents
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            where=request.filter
        )
        
        # Format the context
        context = retriever.format_context(results)
        
        # Generate the response
        answer = llm_manager.generate_response(
            query=request.query,
            context=context,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        elapsed_time = time.time() - start_time
        
        # Prepare the sources
        sources = []
        if request.include_sources:
            for result in results:
                sources.append({
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "similarity_score": result["similarity_score"]
                })
        
        return {
            "query": request.query,
            "answer": answer,
            "sources": sources,
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    # Get ngrok authtoken from environment variable if it exists
    # This is recommended for stable ngrok usage
    ngrok_authtoken = os.getenv("NGROK_AUTHTOKEN")
    if ngrok_authtoken:
        conf.get_default().auth_token = ngrok_authtoken
    else:
        logger.warning("NGROK_AUTHTOKEN not set. Consider setting it for stable ngrok usage.")
        logger.warning("You can get an authtoken from https://dashboard.ngrok.com/get-started/your-authtoken")

    # Uvicorn will run on 0.0.0.0 or 127.0.0.1 (API_HOST) and this port
    # Attempt to use the specified hostname
    # Ensure this hostname is claimed in your ngrok dashboard and your NGROK_AUTHTOKEN is set.
    try:
        public_url_object = ngrok.connect(addr=API_PORT, proto="http", hostname=API_HOST)
        public_url = public_url_object.public_url
    except Exception as e:
        logger.error(f"Could not connect to ngrok with hostname {API_HOST}: {e}")
        logger.info("Falling back to a random ngrok domain.")
        public_url_object = ngrok.connect(addr=API_PORT, proto="http")
        public_url = public_url_object.public_url

    logger.info(f"Ngrok tunnel opened at: {public_url}")
    print(f"Ngrok tunnel opened at: {public_url}") # Also print to console
    try:
        uvicorn.run(app, host="0.0.0.0", port=API_PORT)
    finally:
        if 'public_url_object' in locals() and public_url_object:
            ngrok.disconnect(public_url_object.public_url)
        ngrok.kill()
        logger.info("Ngrok tunnel closed.")