"""
src/evaluator.py
----------------
Handles all evaluation metrics for the RAG chatbot:
  - Contextual Relevance Score  (cosine similarity, query vs retrieved chunks)
  - Latency                     (measured in app.py, summarized here)
  - User Satisfaction Score     (thumbs up/down, stored in eval_log)
  - Precision & Recall (optional, triggered via CLI eval script)
"""

from __future__ import annotations
import re
from typing import List, Any


# ---------------------------------------------------------------------------
# 1. Contextual Relevance Score
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    """Lightweight bag-of-words tokenizer — no external deps required."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return set(text.split())


def cosine_bow(query_tokens: set, doc_tokens: set) -> float:
    """
    Approximate cosine similarity using binary bag-of-words vectors.
    Returns 0.0–1.0.
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    intersection = query_tokens & doc_tokens
    denominator = (len(query_tokens) ** 0.5) * (len(doc_tokens) ** 0.5)
    return len(intersection) / denominator if denominator else 0.0


def compute_contextual_relevance(query: str, source_docs: List[Any]) -> float:
    """
    Compute average cosine similarity between the query and each retrieved chunk.

    Args:
        query:       The user's question string.
        source_docs: List of LangChain Document objects (have .page_content).

    Returns:
        Float between 0.0 and 1.0 — the contextual relevance score.
    """
    if not source_docs:
        return 0.0

    query_tokens = _tokenize(query)
    scores = []
    for doc in source_docs:
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        doc_tokens = _tokenize(content)
        scores.append(cosine_bow(query_tokens, doc_tokens))

    return round(sum(scores) / len(scores), 4)


# ---------------------------------------------------------------------------
# 2. User Satisfaction (Feedback)
# ---------------------------------------------------------------------------

def save_feedback(eval_log: list, message_index: int, feedback: str) -> None:
    """
    Attach thumbs-up/down feedback to the corresponding eval_log entry.

    Args:
        eval_log:      The session's running list of log dicts.
        message_index: Index of the assistant message in st.session_state.messages.
                       Assistant messages are at odd indices (1, 3, 5, …).
        feedback:      "positive" or "negative"
    """
    # Assistant messages are at indices 1, 3, 5, … → log index = message_index // 2
    log_idx = (message_index - 1) // 2
    if 0 <= log_idx < len(eval_log):
        eval_log[log_idx]["feedback"] = feedback


# ---------------------------------------------------------------------------
# 3. Session Metrics Summary
# ---------------------------------------------------------------------------

def get_metrics_summary(eval_log: list) -> dict:
    """
    Compute aggregate metrics from the running eval log.

    Returns a dict with:
        avg_latency       — mean response time in seconds
        avg_relevance     — mean contextual relevance score (0–1)
        satisfaction_rate — fraction of rated responses that were 👍
        total_queries     — total number of queries in session
    """
    if not eval_log:
        return {
            "avg_latency": 0.0,
            "avg_relevance": 0.0,
            "satisfaction_rate": 0.0,
            "total_queries": 0,
        }

    latencies = [e["latency"] for e in eval_log if e.get("latency") is not None]
    relevances = [e["relevance"] for e in eval_log if e.get("relevance") is not None]
    feedbacks = [e["feedback"] for e in eval_log if e.get("feedback") is not None]

    positive = feedbacks.count("positive")
    satisfaction = positive / len(feedbacks) if feedbacks else 0.0

    return {
        "avg_latency": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "avg_relevance": round(sum(relevances) / len(relevances), 4) if relevances else 0.0,
        "satisfaction_rate": round(satisfaction, 4),
        "total_queries": len(eval_log),
    }


# ---------------------------------------------------------------------------
# 4. Optional: Precision & Recall (batch evaluation script)
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k if k else 0.0


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Fraction of all relevant docs captured in the top-k results."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def evaluate_retrieval(test_cases: List[dict], retriever, k: int = 5) -> dict:
    """
    Run precision & recall over a list of test cases.

    Each test case is a dict:
        { "query": "...", "relevant_doc_ids": ["doc1", "doc2", ...] }

    The retriever must be a LangChain retriever that returns Documents
    with metadata["source"] or metadata["id"] as identifiers.

    Usage:
        from src.evaluator import evaluate_retrieval
        results = evaluate_retrieval(test_cases, vs.as_retriever(k=5))
        print(results)
    """
    precisions, recalls = [], []

    for case in test_cases:
        query = case["query"]
        relevant_ids = case["relevant_doc_ids"]
        docs = retriever.get_relevant_documents(query)
        retrieved_ids = [
            d.metadata.get("source", d.metadata.get("id", str(i)))
            for i, d in enumerate(docs)
        ]
        precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))

    return {
        "mean_precision_at_k": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
        "mean_recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "k": k,
        "num_test_cases": len(test_cases),
    }
