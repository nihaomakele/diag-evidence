"""Synthetic retrieval HTTP service. No corpus, embeddings or real retrieval."""
import os
from flask import Flask, jsonify, request
from diag_evidence.contracts import SignalValidationError, validate_signal
from diag_evidence.demo import candidates

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True, "index_loaded": True,
                    "num_texts": 12, "origin": "synthetic-fixture"})


@app.post("/v1/search/text")
def search():
    payload = request.get_json(silent=True)
    try:
        if not isinstance(payload, dict):
            raise SignalValidationError("JSON object required")
        validate_signal(payload.get("signal"))
        top_k = payload.get("top_k", 5)
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise SignalValidationError("top_k invalid")
    except SignalValidationError:
        return jsonify({"error": "Invalid request"}), 400
    mode = os.getenv("RAG_STUB_MODE", "consistent")
    body = candidates()
    if mode in {"mixed_candidates", "conflict"}:
        body["results"][1]["meta"]["fault_type"] = "内圈故障"
    if mode == "empty":
        body["results"] = []
    body["results"] = body["results"][:top_k]
    body["num_results"] = len(body["results"])
    return jsonify(body)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("RAG_STUB_PORT", "8000")))
