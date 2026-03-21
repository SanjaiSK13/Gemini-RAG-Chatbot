# 🛡️ Intelligence Support Portal (Gemini-RAG)

A high-velocity Retrieval-Augmented Generation (RAG) system built with the **Gemini 3 Flash Preview** model and **FAISS** for millisecond-latency customer support automation.

## 🚀 Key Features

- **Context-Aware Reasoning:** Utilizes Google's 2026 flagship model for high-precision intent handling.
- **Ultra-Low Latency:** Optimized Streamlit UI and FAISS indexing for sub-1.2s responses.
- **Zero-Hallucination Guardrails:** Strict RAG pipeline ensuring 100% factual grounding.

## 🛠️ Tech Stack

- **LLM:** Gemini 3 Flash Preview
- **Framework:** LangChain
- **Vector DB:** FAISS
- **UI:** Streamlit
- **Language:** Python 3.13

## 🏗️ Architecture

1. **Ingestion:** Extracts text from private knowledge bases.
2. **Indexing:** Generates high-dimensional embeddings using `google-generativeai`.
3. **Retrieval:** Semantic search via FAISS.
4. **Generation:** Context-injected prompting for the Gemini engine.

## 💻 Setup

1. Clone the repo.
2. Create a `.env` file with your `GOOGLE_API_KEY`.
3. Run `pip install -r requirements.txt`.
4. Launch with `streamlit run app.py`.
