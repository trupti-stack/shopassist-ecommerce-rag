import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

category_descriptions = json.loads(
    (DATA / "category_descriptions.json").read_text(encoding="utf-8")
)
chunks = json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))

# Runtime model loading happens once per warm function instance.
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
reranker = CrossEncoder(RERANKER_NAME)

category_names = list(category_descriptions.keys())
category_texts = list(category_descriptions.values())
category_embeddings = embedding_model.encode(
    category_texts, convert_to_numpy=True
).astype("float32")
faiss.normalize_L2(category_embeddings)

# Optional generator for the full academic RAG demonstration.
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"
_llm = None
_tokenizer = None

def _load_llm():
    global _llm, _tokenizer
    if _llm is not None:
        return
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = os.getenv("GENERATOR_MODEL", "google/flan-t5-base")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def rewrite_query(query):
    q = query.lower().strip()
    replacements = {
        "where's": "where is",
        "wheres": "where is",
        "pls": "please",
        "plz": "please",
        "parcel": "package",
        "courier": "delivery",
        "money back": "refund",
        "refund back": "refund",
        "return back": "return",
        "payment unsuccessful": "payment failed",
        "payment didn't go through": "payment failed",
        "payment did not go through": "payment failed",
        "charged but no order": "payment deducted order failed",
        "money deducted but no order": "payment deducted order failed",
    }
    for old, new in replacements.items():
        q = q.replace(old, new)
    return q

def detect_category(query):
    q_emb = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores = np.dot(q_emb, category_embeddings.T)[0]
    best_idx = int(np.argmax(scores))
    return category_names[best_idx], float(scores[best_idx])

def _safe_name(category):
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")

def retrieve(query, top_k=5, candidate_k=10):
    rewritten = rewrite_query(query)
    category, category_score = detect_category(rewritten)

    index = faiss.read_index(str(DATA / f"{_safe_name(category)}.index"))
    mapping = json.loads(
        (DATA / f"{_safe_name(category)}_map.json").read_text(encoding="utf-8")
    )

    q_emb = embedding_model.encode([rewritten], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, min(candidate_k, index.ntotal))

    candidates = []
    for rank, (local_idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        global_idx = mapping[int(local_idx)]
        item = dict(chunks[global_idx])
        item["vector_rank"] = rank
        item["vector_score"] = float(score)
        candidates.append(item)

    # Re-rank the candidate list with the cross encoder.
    pairs = [(rewritten, item["text"]) for item in candidates]
    rerank_scores = reranker.predict(pairs)

    for item, score in zip(candidates, rerank_scores):
        item["rerank_score"] = float(score)

    # Correctly sort by rerank_score (not "ranked_score").
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "category": category,
        "category_score": category_score,
        "results": reranked
    }

def _template_answer(best):
    return best["answer"]

def generate_answer(query, retrieval):
    if not retrieval["results"]:
        return {
            "answer": "I could not find a relevant support article. Please contact customer support with your order or transaction details.",
            "source": None
        }

    best = retrieval["results"][0]

    if not USE_LLM:
        return {
            "answer": _template_answer(best),
            "source": best
        }

    _load_llm()
    context = "\n".join(
        f"- {r['text']} Response guidance: {r['answer']}"
        for r in retrieval["results"]
    )
    prompt = f"""You are an e-commerce customer support assistant.
Answer the customer using ONLY the supplied support context.
Do not invent policies, dates, refunds, or guarantees.
Be concise and helpful.

Context:
{context}

Customer question:
{query}

Answer:"""

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = _llm.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )
    answer = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": answer, "source": best}

def answer_query(query):
    retrieval = retrieve(query)
    generated = generate_answer(query, retrieval)

    best = retrieval["results"][0] if retrieval["results"] else None

    return {
        "query": query,
        "rewritten_query": retrieval["rewritten_query"],
        "category": retrieval["category"],
        "category_similarity": round(retrieval["category_score"], 4),
        "answer": generated["answer"],
        "important_information": best["important_information"] if best else "",
        "top_results": [
            {
                "rank": i + 1,
                "id": r["id"],
                "intent": r["intent"],
                "text": r["text"],
                "vector_score": round(r["vector_score"], 4),
                "rerank_score": round(r["rerank_score"], 4),
            }
            for i, r in enumerate(retrieval["results"])
        ]
    }
