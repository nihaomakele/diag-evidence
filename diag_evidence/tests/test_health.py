"""check_system_status 的成功/失败测试（注入假 transport，无需真实服务）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diag_evidence.tests import _fakes  # noqa: E402
from diag_evidence.hatools import health  # noqa: E402


class HealthTest(unittest.TestCase):
    def test_all_healthy(self):
        transport = _fakes.routed({
            "8000": _fakes.ok({"status": "ok", "model_loaded": True, "index_loaded": True, "num_texts": 12}),
            "5000": _fakes.ok({"status": "ok", "service": "Fault Diagnosis API"}),
            "6006": _fakes.ok({}),
        })
        result = health.check_system_status(transport=transport, check_logs=False)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["ok"])
        self.assertTrue(result["data"]["services"]["rag"]["capable"])
        self.assertEqual(sorted(result["data"]["reachable"]), ["fd", "rag", "ui"])

    def test_small_corpus_warns_but_reachable(self):
        transport = _fakes.routed({
            "8000": _fakes.ok({"status": "ok", "model_loaded": True, "index_loaded": True, "num_texts": 3}),
            "5000": _fakes.ok({"status": "ok"}),
            "6006": _fakes.ok({}),
        })
        result = health.check_system_status(transport=transport, check_logs=False)
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["ok"])
        self.assertTrue(any("3 条" in w for w in result["warnings"]))

    def test_rag_empty_index_is_degraded(self):
        transport = _fakes.routed({
            "8000": _fakes.ok({"status": "ok", "model_loaded": True, "index_loaded": True, "num_texts": 0}),
            "5000": _fakes.ok({"status": "ok"}),
            "6006": _fakes.ok({}),
        })
        result = health.check_system_status(transport=transport, check_logs=False)
        self.assertFalse(result["data"]["services"]["rag"]["capable"])
        self.assertEqual(result["status"], "degraded")

    def test_all_unreachable(self):
        result = health.check_system_status(transport=_fakes.connection_error(), check_logs=False)
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["data"]["unreachable"]), 3)

    def test_partial_failure_does_not_raise(self):
        transport = _fakes.routed({
            "8000": _fakes.ok({"status": "ok", "model_loaded": True, "index_loaded": True, "num_texts": 12}),
            "5000": _fakes.connection_error(),
            "6006": _fakes.connection_error(),
        })
        result = health.check_system_status(transport=transport, check_logs=False)
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["data"]["reachable"], ["rag"])
        self.assertIn("fd", result["data"]["unreachable"])

    def test_unknown_service_name(self):
        result = health.check_system_status(services=["nope"], transport=_fakes.connection_error(), check_logs=False)
        self.assertTrue(any("未知服务" in e for e in result["errors"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
