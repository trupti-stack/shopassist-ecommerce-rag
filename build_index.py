import json, re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

def sentence_split(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

def fixed_size_chunking(text, chunk_size=40):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def sentence_based_chunking(text, sentences_per_chunk=2):
    sentences = sentence_split(text)
    return [" ".join(sentences[i:i+sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]

def overlap_chunking(text, chunk_size=40, overlap=10):
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def main():
    df = pd.read_csv(DATA / "corpus.csv")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Use the same chunking family practiced in the coursework.
    # Overlap keeps context from being lost at chunk boundaries.
    rows = []
    for _, r in df.iterrows():
        chunks = overlap_chunking(str(r["text"]), chunk_size=40, overlap=10)
        for cno, chunk in enumerate(chunks):
            rows.append({
                "id": int(r["id"]),
                "category": r["category"],
                "intent": r["intent"],
                "title": r["title"],
                "text": chunk,
                "answer": r["answer"],
                "important_information": r["important_information"],
                "chunk_no": cno + 1,
            })

    meta_path = DATA / "chunks.json"
    meta_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # Full index
    texts = [r["text"] for r in rows]
    emb = model.encode(texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(DATA / "full.index"))

    # Category-specific indexes = metadata filtering.
    categories = sorted(df["category"].unique().tolist())
    for category in categories:
        selected = [i for i, r in enumerate(rows) if r["category"] == category]
        cemb = emb[selected]
        cindex = faiss.IndexFlatIP(cemb.shape[1])
        cindex.add(cemb)
        safe = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
        faiss.write_index(cindex, str(DATA / f"{safe}.index"))
        (DATA / f"{safe}_map.json").write_text(
            json.dumps(selected), encoding="utf-8"
        )

    print(f"Documents: {len(df)}")
    print(f"Chunks: {len(rows)}")
    print(f"Embedding dimension: {emb.shape[1]}")
    print("Index files created in data/")

if __name__ == "__main__":
    main()
