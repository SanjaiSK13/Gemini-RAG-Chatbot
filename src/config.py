import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# Model Settings
LLM_MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Data Settings
RAW_DATA_PATH = "data/raw/kb.txt"
VECTOR_STORE_PATH = "vector_store/faiss_index"
CHUNK_SIZE = 700 
CHUNK_OVERLAP = 100