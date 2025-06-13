import logging
import os
from pathlib import Path
from dotenv import load_dotenv

_BASE_DIR_FOR_DOTENV = Path(__file__).resolve().parent.parent.parent
dotenv_path = _BASE_DIR_FOR_DOTENV / '.env'

# Load environment variables from .env file, overriding existing ones if present
# and making sure to specify the path.
logger = logging.getLogger(__name__)
logger.info(f"Attempting to load .env file from: {dotenv_path}")
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f".env file loaded successfully from {dotenv_path}.")
else:
    logger.warning(f".env file not found at {dotenv_path}. Environment variables might not be set as expected.")

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vectordb")

# Text preprocessing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
ENCODING_MODEL = "cl100k_base"  # For OpenAI-like tokenizers

# Embedding settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Vector database settings
VECTOR_DB_TYPE = "chroma"
DISTANCE_METRIC = "cosine"

# Retrieval settings
TOP_K_RESULTS = 3
SIMILARITY_THRESHOLD = 0.2

# LLM settings
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
SYSTEM_PROMPT = """You are a helpful assistant that provides accurate information based on the given context. 
If the answer cannot be found in the context, simply state that you don't have enough information. 
Do not make up information. Always cite your sources."""

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Web app settings
WEBAPP_HOST = os.getenv("WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = int(os.getenv("WEBAPP_PORT", "8501"))
APP_TITLE = "RAG Q&A System"
APP_DESCRIPTION = "Ask questions and get answers based on your documents"

# Cache settings
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
MAX_CACHE_SIZE = 100  # Number of queries to cache
