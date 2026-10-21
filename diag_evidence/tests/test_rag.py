"""retrieve_cases 的成功/失败测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diag_evidence.tests import _fakes  # noqa: E402
from diag_evidence.hatools import rag  # noqa: E402

from diag_evidence.demo import synthetic_signal

SIGNAL = synthetic_signal()


class RagTest(unittest.TestCase):
    def test_success(self):
        body = {"results": [
            {"rank": 1, "score": 0.73, "text": "滚动体故障描述", "meta": {"fault_type": "滚动体故障"}},
            {"rank": 2, "score": 0.11, "text": "内圈故障描述", "meta": {"fault_type": "内圈故障"}},
            {"rank": 3, "score": 0.09, "text": "外圈故障描述", "meta": {"fault_type": "外圈故障"}},
        ], "num_results": 3}
        result = rag.retrieve_cases(SIGNAL, top_k=3, transport=_fakes.ok(body))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["num_results"], 3)
        self.assertEqual(result["data"]["cases"][0]["fault_type"], "滚动体故障")
        self.assertIsNone(result["data"]["cases"][0]["score_is_normalized"])

    def test_empty_results_is_degraded(self):
        result = rag.retrieve_cases(SIGNAL, transport=_fakes.ok({"results": [], "num_results": 0}))
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["ok"])
        self.assertEqual(result["data"]["cases"], [])
        self.assertTrue(any("不得据此推断" in w for w in result["warnings"]))
        self.assertEqual(result["evidence"][0]["trust"], "unavailable")

    def test_wrong_signal_length_never_calls_service(self):
        def transport(method, url, **kwargs):
            raise AssertionError("不合法输入不应发起请求")
        result = rag.retrieve_cases([0.0] * 512, transport=transport)
        self.assertEqual(result["status"], "invalid_input")

    def test_missing_signal(self):
        def transport(method, url, **kwargs):
            raise AssertionError("缺少输入不应发起请求")
        result = rag.retrieve_cases(None, transport=transport)
        self.assertEqual(result["status"], "invalid_input")

    def test_non_numeric_signal(self):
        result = rag.retrieve_cases(["a"] * 1024, transport=_fakes.ok({}))
        self.assertEqual(result["status"], "invalid_input")

    def test_bad_top_k(self):
        for bad in (0, 101, "3", None):
            result = rag.retrieve_cases(SIGNAL, top_k=bad, transport=_fakes.ok({}))
            self.assertEqual(result["status"], "invalid_input", msg=f"top_k={bad!r}")

    def test_service_unavailable(self):
        result = rag.retrieve_cases(SIGNAL, transport=_fakes.connection_error())
        self.assertEqual(result["status"], "unavailable")
        self.assertTrue(result["errors"])

    def test_service_503(self):
        result = rag.retrieve_cases(SIGNAL, transport=_fakes.server_error({"error": "Model not loaded"}, "Model not loaded"))
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("HTTP", " ".join(result["errors"]))

    def test_service_400(self):
        result = rag.retrieve_cases(SIGNAL, transport=_fakes.client_error({"error": "Signal length must be 1024"}, "Signal length must be 1024"))
        self.assertEqual(result["status"], "invalid_input")

    def test_bad_json(self):
        result = rag.retrieve_cases(SIGNAL, transport=_fakes.ok("not-a-dict"))
        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
