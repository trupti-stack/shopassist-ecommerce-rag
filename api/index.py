from flask import Flask, jsonify, request
from rag_engine import answer_query

app = Flask(__name__)

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "ecommerce-semantic-rag"})

@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()

    if not query:
        return jsonify({"error": "Please enter a customer query."}), 400

    try:
        return jsonify(answer_query(query))
    except Exception as exc:
        return jsonify({
            "error": "The chatbot could not process the request.",
            "details": str(exc)
        }), 500
