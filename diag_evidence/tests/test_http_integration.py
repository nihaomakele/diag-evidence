"""Real loopback HTTP boundaries; no external services, Flask or model files."""

from __future__ import annotations

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from unittest.mock import patch

from diag_evidence.agent import COMPLETED, DiagnosisAgent
from diag_evidence.demo import candidates, classifier, synthetic_signal
from diag_evidence.haconfig.settings import Settings
from diag_evidence.hatools import _common, diagnosis, rag


class _Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.routes = {}
        self.calls = []
        self.lock = threading.Lock()
        self.stopping = threading.Event()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        self._respond()

    def do_POST(self):
        self._respond()

    def _respond(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)
        with self.server.lock:
            self.server.calls.append((self.command, self.path, payload))
            responses = self.server.routes.get(self.path, [(404, {"error": "not found"}, 0)])
            status, body, delay = responses[0]
            if len(responses) > 1:
                responses.pop(0)
        self.server.stopping.wait(delay)
        content = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "text/plain" if isinstance(body, bytes) else "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            # Expected when the client has already timed out.
            pass


class HTTPIntegrationTest(unittest.TestCase):
    def setUp(self):
        bypass_proxy = patch.dict(os.environ, {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"})
        bypass_proxy.start()
        self.addCleanup(bypass_proxy.stop)
        self.server = _Server()
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.01}, daemon=False)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self._stop)

    def _stop(self):
        self.server.stopping.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive(), "loopback server did not stop")

    def _route(self, path, *responses):
        self.server.routes[path] = list(responses)

    def _call(self, path, **kwargs):
        return _common.http_call("POST", self.base + path, payload={"synthetic": True},
                                 timeout=kwargs.pop("timeout", 1), backoff=0, **kwargs)

    def _adapter_transport(self, method, url, **kwargs):
        kwargs["max_retries"] = 0
        kwargs["timeout"] = 1
        return _common.http_call(method, self.base + urlsplit(url).path, **kwargs)

    def test_json_round_trip(self):
        self._route("/ok", (200, {"value": 7}, 0))
        self.assertEqual(self._call("/ok"), ("ok", {"value": 7}, ""))
        method, path, payload = self.server.calls[0]
        self.assertEqual((method, path), ("POST", "/ok"))
        self.assertEqual(json.loads(payload), {"synthetic": True})

    def test_non_json_success_is_rejected_by_diagnosis_adapter(self):
        self._route("/diagnose", (200, b"upstream html instead of JSON", 0))
        result = diagnosis.run_fault_diagnosis([(i % 17) / 17 for i in range(1024)], transport=self._adapter_transport)
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["errors"])

    def test_valid_diagnosis_json_crosses_real_http_boundary(self):
        body = {"success": True, "diagnosis": {
            "predicted_class": 0, "confidence": 0.8,
            "predicted_fault": {"name": "内圈故障", "name_en": "Inner Race Fault"},
            "class_probabilities": {"0": 0.8, "1": 0.1, "2": 0.1}}}
        self._route("/diagnose", (200, body, 0))
        result = diagnosis.run_fault_diagnosis([(i % 17) / 17 for i in range(1024)],
                                               transport=self._adapter_transport)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["diagnosis"]["predicted_class"], 0)
        self.assertEqual(len(json.loads(self.server.calls[0][2])["signal"]), 1024)

    def test_valid_retrieval_json_crosses_real_http_boundary(self):
        body = {"results": [{"rank": 1, "score": 0.8, "text": "synthetic case",
                             "meta": {"fault_type": "内圈故障", "record_id": "synthetic-1"}}],
                "num_results": 1, "score_is_normalized": False}
        self._route("/v1/search/text", (200, body, 0))
        result = rag.retrieve_cases([(i % 17) / 17 for i in range(1024)],
                                    transport=self._adapter_transport)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["num_results"], 1)
        self.assertEqual(len(result["data"]["cases"]), 1)

    def test_full_workflow_uses_real_http_without_injected_transport(self):
        self._route("/health", (200, {"status": "ok", "model_loaded": True,
                                     "index_loaded": True, "num_texts": 12}, 0))
        self._route("/", (200, {"status": "ok"}, 0))
        self._route("/diagnose", (200, classifier(), 0))
        self._route("/v1/search/text", (200, candidates(), 0))
        config = Settings(rag_base_url=self.base, fd_base_url=self.base,
                          ui_base_url=self.base, health_timeout=1,
                          rag_search_timeout=1, fd_diagnose_timeout=1,
                          max_retries=0, retry_backoff=0)
        result = DiagnosisAgent(config=config, persist_trace=False).run(
            synthetic_signal(), input_is_synthetic=True)
        self.assertEqual(result["terminal_state"], COMPLETED)
        self.assertEqual(result["support_level"], "high")
        self.assertEqual(result["persistence_errors"], [])
        self.assertEqual([(method, path) for method, path, _ in self.server.calls],
                         [("GET", "/health"), ("GET", "/"),
                          ("POST", "/v1/search/text"), ("POST", "/diagnose")])
        for method, _path, payload in self.server.calls:
            if method == "POST":
                self.assertEqual(json.loads(payload)["signal"], synthetic_signal())
        self.assertIsNone(result["report_path"])

    def test_non_json_transport_body_is_none(self):
        self._route("/text", (200, b"not JSON", 0))
        self.assertEqual(self._call("/text"), ("ok", None, ""))

    def test_not_found_is_not_retried(self):
        kind, _body, error = self._call("/absent", max_retries=2)
        self.assertEqual(kind, "not_found")
        self.assertIn("404", error)
        self.assertEqual(len(self.server.calls), 1)

    def test_client_error_is_not_retried_and_redacts_error(self):
        fake = "sk-TEST_ONLY_NOT_A_REAL_KEY_12345"
        self._route("/client", (400, {"error": "bad request " + fake}, 0))
        kind, _body, error = self._call("/client", max_retries=2)
        self.assertEqual(kind, "client")
        self.assertNotIn(fake, error)
        self.assertEqual(len(self.server.calls), 1)

    def test_retryable_server_error_recovers(self):
        self._route("/retry", (503, {"error": "temporary"}, 0),
                    (502, b"temporary gateway failure", 0), (200, {"ready": True}, 0))
        self.assertEqual(self._call("/retry", max_retries=2), ("ok", {"ready": True}, ""))
        self.assertEqual(len(self.server.calls), 3)

    def test_retry_budget_is_bounded(self):
        self._route("/down", (503, {"error": "temporary"}, 0))
        kind, body, error = self._call("/down", max_retries=1)
        self.assertEqual(kind, "server")
        self.assertIsNone(body)
        self.assertIn("503", error)
        self.assertEqual(len(self.server.calls), 2)

    def test_read_timeout_maps_to_connection_failure(self):
        self._route("/slow", (200, {"late": True}, 0.2))
        kind, body, error = self._call("/slow", timeout=0.02, max_retries=0)
        self.assertEqual(kind, "conn")
        self.assertIsNone(body)
        self.assertTrue(error)
        self.assertEqual(len(self.server.calls), 1)

    def test_malformed_diagnosis_object_fails_closed(self):
        self._route("/diagnose", (200, {"success": True, "diagnosis": ["invalid object"]}, 0))
        result = diagnosis.run_fault_diagnosis([(i % 17) / 17 for i in range(1024)], transport=self._adapter_transport)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["ok"])

    def test_malformed_retrieval_results_fail_closed(self):
        self._route("/v1/search/text", (200, {"results": {"unexpected": "object"}}, 0))
        result = rag.retrieve_cases([(i % 17) / 17 for i in range(1024)], transport=self._adapter_transport)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
