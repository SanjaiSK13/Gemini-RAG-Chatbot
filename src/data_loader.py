from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import RAW_DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split_data():
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split by your specific dashes first to keep issues together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["---------------------------------------------------", "\n\n", "\n", " "]
    )
    return text_splitter.split_text(text)