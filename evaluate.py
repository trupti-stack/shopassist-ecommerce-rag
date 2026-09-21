import pandas as pd
from rag_engine import retrieve

def precision_at_k(results, relevant_ids, k):
    top_results = results[:k]
    relevant_count = sum(1 for r in top_results if r["id"] in relevant_ids)
    return relevant_count / k

def main():
    qdf = pd.read_csv("data/evaluation_queries.csv")
    cdf = pd.read_csv("data/corpus.csv")

    # Relevant IDs are all corpus records belonging to the expected intent.
    precision_before = []
    precision_after = []
    category_correct = []

    for _, row in qdf.iterrows():
        retrieval = retrieve(row["query"], top_k=5, candidate_k=10)

        expected_ids = set(
            cdf.loc[cdf["intent"] == row["expected_intent"], "id"].astype(int).tolist()
        )

        # Initial vector ranking = candidates in vector_rank order.
        before = sorted(
            retrieval["results"],
            key=lambda x: x["vector_rank"]
        )
        after = retrieval["results"]

        precision_before.append(precision_at_k(before, expected_ids, 5))
        precision_after.append(precision_at_k(after, expected_ids, 5))
        category_correct.append(
            int(retrieval["category"] == row["expected_category"])
        )

    print("Average Precision@5 before reranking:",
          round(sum(precision_before) / len(precision_before), 4))
    print("Average Precision@5 after reranking:",
          round(sum(precision_after) / len(precision_after), 4))
    print("Category accuracy:",
          round(sum(category_correct) / len(category_correct), 4))

if __name__ == "__main__":
    main()
