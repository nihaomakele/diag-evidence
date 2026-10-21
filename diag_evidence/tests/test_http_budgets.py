"""Bounded real loopback HTTP, cancellation and worker reclamation tests."""
from contextlib import contextmanager
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
import unittest
from unittest.mock import Mock, patch

from diag_evidence.hatools import _common


class Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.calls = []
        self.redirect = ""
        self.stopping = threading.Event()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self._respond()

    def do_GET(self):
        self._respond()

    def _respond(self):
        payload = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.server.calls.append((self.path, payload))
        try:
            if self.path == "/redirect":
                self.send_response(307)
                self.send_header("Location", self.server.redirect)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path == "/slow-header":
                # Continuous header bytes never reach the read-inactivity limit.
                self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                for _ in range(100):
                    self.wfile.write(b"x"); self.wfile.flush()
                    if self.server.stopping.wait(0.02):
                        break
                self.wfile.write(b"\r\nContent-Length: 2\r\n\r\n{}"); self.wfile.flush()
                return
            content = b'{"ok":true}'
            self.send_response(200)
            if self.path == "/compressed":
                content = gzip.compress(content)
                self.send_header("Content-Encoding", "gzip")
            if self.path in ("/large", "/unannounced-large"):
                content = b"x" * 2048
            if self.path != "/unannounced-large":
                self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if self.path == "/drip":
                for value in content:
                    self.wfile.write(bytes([value])); self.wfile.flush()
                    if self.server.stopping.wait(0.025):
                        break
            else:
                self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass


@contextmanager
def running_server():
    server = Server()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.stopping.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if thread.is_alive():
            raise AssertionError("loopback server did not stop")


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


class HTTPBudgetTest(unittest.TestCase):
    def setUp(self):
        proxy = patch.dict(os.environ, {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"})
        proxy.start()
        self.addCleanup(proxy.stop)

    def test_cross_origin_redirect_never_receives_post(self):
        with running_server() as sink, running_server() as origin:
            origin.redirect = f"http://127.0.0.1:{sink.server_port}/sink"
            result = _common.http_call("POST", f"http://127.0.0.1:{origin.server_port}/redirect",
                                       payload={"signal": [1, 2, 3]}, timeout=1)
            self.assertEqual(result[0], "invalid_response")
            self.assertIn("重定向", result[2])
            self.assertEqual(len(origin.calls), 1)
            self.assertEqual(sink.calls, [])

    def test_announced_and_unannounced_oversize_bodies_are_rejected(self):
        with running_server() as server:
            for path in ("/large", "/unannounced-large"):
                with self.subTest(path=path):
                    result = _common.http_call("GET", f"http://127.0.0.1:{server.server_port}{path}",
                                               timeout=1, max_response_bytes=1024)
                    self.assertEqual(result[0], "invalid_response")
                    self.assertIsNone(result[1])
                    self.assertIn("字节预算", result[2])

    def test_compressed_response_is_rejected_before_decompression(self):
        with running_server() as server:
            result = _common.http_call("GET", f"http://127.0.0.1:{server.server_port}/compressed", timeout=1)
            self.assertEqual(result[0], "invalid_response")
            self.assertIn("压缩响应", result[2])

    def test_deadline_covers_continuous_body_and_header_drips(self):
        for path in ("/drip", "/slow-header"):
            with self.subTest(path=path), running_server() as server:
                begin = time.monotonic()
                result = _common.http_call("GET", f"http://127.0.0.1:{server.server_port}{path}",
                                           timeout=0.3, total_timeout=0.08)
                elapsed = time.monotonic() - begin
                self.assertEqual(result[0], "conn")
                self.assertIn("总时间预算", result[2])
                self.assertLess(elapsed, 0.25)
                self.assertIsNone(result[1])
        self.assertTrue(wait_until(lambda: not any(t.name == "diag-evidence-http"
                                                  for t in threading.enumerate())))

    def test_response_is_closed_and_stream_redirect_options_are_explicit(self):
        response = Mock(status_code=200, headers={"Content-Length": "11"})
        response.raw.read1.side_effect = [b'{"ok":true}', b""]
        with patch.object(_common.requests, "request", return_value=response) as request:
            result = _common.http_call("GET", "http://localhost", timeout=1)
        self.assertEqual(result, ("ok", {"ok": True}, ""))
        response.close.assert_called_once()
        self.assertFalse(request.call_args.kwargs["allow_redirects"])
        self.assertTrue(request.call_args.kwargs["stream"])
        self.assertEqual(request.call_args.kwargs["headers"]["Accept-Encoding"], "identity")

    def test_error_body_is_never_read_and_each_retry_response_is_closed(self):
        responses = [Mock(status_code=503), Mock(status_code=503), Mock(status_code=400)]
        with patch.object(_common.requests, "request", side_effect=responses) as request:
            result = _common.http_call("GET", "http://localhost", timeout=1, max_retries=2, backoff=0)
        self.assertEqual(result[0], "client")
        self.assertIsNone(result[1])
        self.assertEqual(request.call_count, 3)
        for response in responses:
            response.close.assert_called_once()
            response.raw.read1.assert_not_called()
            response.iter_content.assert_not_called()
            response.json.assert_not_called()

    def test_backoff_and_retries_share_one_deadline(self):
        response = Mock(status_code=503)
        begin = time.monotonic()
        with patch.object(_common.requests, "request", return_value=response) as request:
            result = _common.http_call("GET", "http://localhost", timeout=1,
                                       total_timeout=0.05, max_retries=2, backoff=1)
        self.assertEqual(result[0], "conn")
        self.assertLess(time.monotonic() - begin, 0.25)
        self.assertEqual(request.call_count, 1)
        self.assertTrue(wait_until(lambda: not any(t.name == "diag-evidence-http"
                                                  for t in threading.enumerate())))

    def test_timed_out_io_workers_are_bounded_and_reclaimed(self):
        release = threading.Event()
        entered = []
        responses = []
        def blocked(*args, **kwargs):
            entered.append(True)
            release.wait(2)
            response = Mock(status_code=503)
            responses.append(response)
            return response
        # Isolate the limiter so this test can prove its exact capacity.
        limiter = threading.BoundedSemaphore(_common.MAX_HTTP_WORKERS)
        try:
            with patch.object(_common, "_HTTP_SLOTS", limiter), \
                    patch.object(_common.requests, "request", side_effect=blocked):
                for _ in range(_common.MAX_HTTP_WORKERS):
                    result = _common.http_call("GET", "http://localhost", timeout=0.1, total_timeout=0.02)
                    self.assertEqual(result[0], "conn")
                result = _common.http_call("GET", "http://localhost", timeout=0.1, total_timeout=0.02)
                self.assertIn("上限", result[2])
                self.assertEqual(len(entered), _common.MAX_HTTP_WORKERS)
                release.set()
                self.assertTrue(wait_until(lambda: len(responses) == _common.MAX_HTTP_WORKERS
                    and not any(t.name == "diag-evidence-http" for t in threading.enumerate())))
                for response in responses:
                    response.close.assert_called_once()
                with patch.object(_common.requests, "request", return_value=Mock(status_code=404)):
                    self.assertEqual(_common.http_call("GET", "http://localhost", timeout=1)[0], "not_found")
        finally:
            release.set()

    def test_budget_arguments_are_validated_before_network(self):
        with patch.object(_common.requests, "request") as request:
            for kwargs in ({"total_timeout": 0}, {"total_timeout": 301}, {"total_timeout": True},
                           {"total_timeout": float("nan")}, {"max_response_bytes": 0},
                           {"max_response_bytes": _common.MAX_RESPONSE_BYTES + 1},
                           {"max_response_bytes": True}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    _common.http_call("GET", "http://localhost", **kwargs)
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
