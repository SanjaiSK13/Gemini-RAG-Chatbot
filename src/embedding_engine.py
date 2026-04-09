import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from src.config import EMBEDDING_MODEL, VECTOR_STORE_PATH

load_dotenv()

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def create_vector_store(chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store

def load_vector_store():
    embeddings = get_embeddings()
    return FAISS.load_local(
        VECTOR_STORE_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )