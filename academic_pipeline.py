"""
End-to-end academic pipeline.

Run:
    python build_index.py
    python academic_pipeline.py

Set USE_LLM=true if transformers is installed and you want the
FLAN-T5 generation stage as demonstrated in the coursework.
"""

from rag_engine import answer_query

queries = [
    "Where is my order?",
    "Can you track my parcel?",
    "My package is very late.",
    "The courier says delivered but I did not receive it.",
    "I want to return my product.",
    "Can I exchange this item for another size?",
    "My UPI payment failed.",
    "The money was deducted but there is no order.",
    "My refund is pending.",
    "My refund says completed but I cannot see the money."
]

for q in queries:
    result = answer_query(q)
    print("\n" + "=" * 80)
    print("CUSTOMER QUERY:", result["query"])
    print("REWRITTEN QUERY:", result["rewritten_query"])
    print("CATEGORY:", result["category"])
    print("CATEGORY SIMILARITY:", result["category_similarity"])
    print("ANSWER:", result["answer"])
    print("IMPORTANT INFORMATION:", result["important_information"])
    print("\nTOP RETRIEVED RESULTS:")
    for r in result["top_results"]:
        print(
            r["rank"], "|",
            r["intent"], "|",
            "vector:", r["vector_score"],
            "| rerank:", r["rerank_score"],
            "|", r["text"]
        )
