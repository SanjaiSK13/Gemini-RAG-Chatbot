import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Settings
LLM_MODEL = "gemini-3-flash-preview"  # Fastest for customer support
EMBEDDING_MODEL = "models/gemini-embedding-001"# Best for retrieval accuracy

# Data Settings
RAW_DATA_PATH = "data/raw/kb.txt"
VECTOR_STORE_PATH = "vector_store/faiss_index"
CHUNK_SIZE = 700 
CHUNK_OVERLAP = 100