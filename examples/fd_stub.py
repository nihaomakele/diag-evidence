"""Synthetic classification HTTP service. No trained model or diagnostic accuracy."""
import os
from flask import Flask, jsonify, request
from diag_evidence.contracts import SignalValidationError, validate_signal
from diag_evidence.demo import classifier

app = Flask(__name__)


@app.get("/")
def health():
    return jsonify({"status": "ok", "service": "synthetic FD fixture"})


@app.post("/diagnose")
def diagnose():
    payload = request.get_json(silent=True)
    try:
        if not isinstance(payload, dict):
            raise SignalValidationError("JSON object required")
        validate_signal(payload.get("signal"))
    except SignalValidationError:
        return jsonify({"success": False, "error": "Invalid finite signal window"}), 400
    mode = os.getenv("STUB_MODE", "ok")
    if mode == "unavailable":
        return jsonify({"success": True, "diagnosis": {"predicted_class": -1}})
    return jsonify(classifier(0.2 if mode == "lowconf" else 0.81))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("FD_STUB_PORT", "5001")))
