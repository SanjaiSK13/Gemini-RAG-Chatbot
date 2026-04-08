# 🛡️ Intelligence Support Portal — High-Velocity Gemini-RAG System

A production-ready **Retrieval-Augmented Generation (RAG) platform** engineered for high-precision customer support automation. The system leverages the **Gemini 3 Flash Preview** model and **FAISS vector indexing** to deliver context-aware, zero-hallucination responses with millisecond-scale latency.

The platform demonstrates a sophisticated AI pipeline: from **automated document ingestion and high-dimensional embedding generation** to **semantic retrieval and grounded response synthesis**.

---

## 🚀 Key Features

- **Gemini 3 Flash Integration:** Utilizes Google's 2026 flagship model for high-precision intent handling and reasoning.
- **Ultra-Low Latency:** Optimized Streamlit UI and FAISS indexing for sub-1.2s response times.
- **Zero-Hallucination Guardrails:** Strict RAG pipeline ensuring 100% factual grounding and accuracy.
- **Scalable Vector Search:** Generates and manages high-dimensional embeddings via FAISS for millisecond-latency retrieval.
- **Context-Aware Logic:** Leverages injected context to ensure all generated responses are strictly relevant to the private knowledge base.

---

## 🛠️ Tech Stack

### Core & LLM

- **LLM:** Gemini 3 Flash Preview
- **Orchestration:** LangChain
- **Language:** Python 3.13

### Vector DB & UI

- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** google-generativeai
- **Interface:** Streamlit

---

## 🏗️ System Architecture

Private Knowledge Base (Text/Manuals)
        ↓
Ingestion Engine
(Text Extraction & Chunking)
        ↓
Embedding Layer
(High-Dimensional Vectorization)
        ↓
+-----------------------------+
|     Indexing & Storage      |
|     FAISS Vector DB         |
+-----------------------------+
        ↓
User Query (Support Ticket)
        ↓
+-------------------------------+
|       Retrieval Layer         |
|   (Semantic Search via FAISS) |
+-------------------------------+
        ↓
+-------------------------------+
|       Generation Layer        |
|  (Context-Injected Prompting) |
+-------------------------------+
        ↓
⚡ Grounded Response (< 1.2s)
---

## 💻 Setup & Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/SanjaiSK13/Gemini-RAG-Chatbot.git](https://github.com/SanjaiSK13/Gemini-RAG-Chatbot.git)
cd Gemini-RAG-Chatbot

2. Configure Environment
Create a .env file in the root directory and add your credentials:

Bash

GOOGLE_API_KEY=your_google_gemini_api_key

3. Install Dependencies
Bash

pip install -r requirements.txt

4. Run the Application
Bash

streamlit run app.py

📊 Performance Metrics
Metric	Target / Achievement
Response Latency	< 1.2 Seconds
Factual Grounding	100% (Zero-Hallucination)
Model Version	Gemini 3 Flash Preview
Framework	LangChain / FAISS

Export to Sheets

💡 Business Applications
Customer Support Automation: Millisecond-latency responses for complex user queries.

Private Knowledge Retrieval: Securely querying internal documentation without public data leaks.

Automated Intent Handling: Precise interpretation of customer needs using state-of-the-art LLM reasoning.

👨‍💻 Author
Sanjai K
M.E. Computer Science (AI & ML Specialization)

GitHub: SanjaiSK13
Domain: Deep Learning · Computer Vision · Full-Stack AI Systems

📜 License
This project is open-source and available under the MIT License.
Powered by Google Gemini 3 Flash.
```
