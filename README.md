# ShopAssist — E-commerce Semantic Search + RAG Chatbot

A single end-to-end case-study project for an e-commerce company that receives customer support messages such as:

- Where is my order?
- I want to return my product.
- My payment was unsuccessful.
- Where will I get my refund back?

## What the system does

Customer query
→ query rewriting
→ semantic category detection
→ metadata filtering
→ SentenceTransformer embedding
→ FAISS vector search
→ CrossEncoder re-ranking
→ best support knowledge
→ answer generation
→ chatbot response

## Corpus

The synthetic academic knowledge base contains:

- 128 support documents
- 4 main categories
- 16 sub-intents
- 128 evaluation queries
- order-tracking, returns, payment and refund variations

The four categories are:

1. Order Tracking
2. Returns
3. Payment Issues
4. Refunds

## Local setup

Python 3.13 is suitable.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python build_index.py
python academic_pipeline.py
```

For the optional FLAN-T5 generation stage:

```bash
pip install -r requirements-academic.txt
set USE_LLM=true
python academic_pipeline.py
```

On PowerShell:

```powershell
$env:USE_LLM="true"
python academic_pipeline.py
```

## Evaluation

```bash
python evaluate.py
```

The evaluation reports:

- Precision@5 before re-ranking
- Precision@5 after re-ranking
- semantic category accuracy

## Vercel deployment

The deployment version keeps the semantic retrieval and re-ranking pipeline in the Python backend.

1. Push this folder to GitHub.
2. Import the repository into Vercel.
3. Set the environment variable:
   `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`
   when the deployed function exceeds the standard package path.
4. Deploy.
5. Open the generated Vercel URL.
6. Test all four example queries and additional paraphrases.

For the Vercel demo, `USE_LLM=false` is intentional. The answer is taken from the retrieved support knowledge base, which keeps the chatbot fast and avoids loading the optional FLAN-T5 generator in every serverless deployment. The academic version can run the full FLAN-T5 generation stage locally.

## Important implementation note

The re-ranking sort uses `rerank_score`. Do not accidentally sort by a different field name such as `ranked_score`.

## Project structure

```text
ecommerce_semantic_rag_vercel/
├── api/
│   └── index.py
├── data/
│   ├── corpus.csv
│   ├── evaluation_queries.csv
│   └── category_descriptions.json
├── public/
│   └── index.html
├── academic_pipeline.py
├── build_index.py
├── evaluate.py
├── rag_engine.py
├── requirements.txt
├── requirements-academic.txt
├── vercel.json
└── README.md
```
