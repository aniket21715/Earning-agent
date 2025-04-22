# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).resolve().parent / '.env')

# Project paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# API keys and URLs
FMP_API_KEY = os.getenv("FMP_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")  # Get from environment or set default

if not FMP_API_KEY:
    raise ValueError("FMP_API_KEY environment variable not set")

# LLM settings
LLM_MODEL_PATH = MODELS_DIR / "llama-2-7b-chat.Q4_0.gguf"  # Example path, adjust as needed
LLM_CONTEXT_SIZE = 2048

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Size of the embedding vectors

# Chunking settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Vector store settings
VECTOR_STORE_PATH = DATA_DIR / "vectorstore"

# Web scraping settings
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
}

# PDF settings
PDF_OUTPUT_DIR = DATA_DIR / "reports"
PDF_OUTPUT_DIR.mkdir(exist_ok=True)