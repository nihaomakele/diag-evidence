"""DiagnosisAgent 端到端（注入式替身，无需真实服务/GPU）。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diag_evidence.agent import (  # noqa: E402
    ASKING_USER, COMPLETED, DEGRADED, DiagnosisAgent,
)
from diag_evidence.tests import _fakes  # noqa: E402

from diag_evidence.demo import synthetic_signal

SIGNAL = synthetic_signal()


def _case(rank, score, fault):
    return {"rank": rank, "score": score, "text": "%s描述" % fault, "meta": {"fault_type": fault, "record_id": f"synthetic-{rank}"}}


def _fd_degraded():
    return {"success": True, "spectrogram": "AAAA", "statistics": {},
            "diagnosis": {"note": "模型未加载", "predicted_class": -1}}


def _fd_ok(confidence=0.81, fault="滚动体故障"):
    return {"success": True, "spectrogram": "AAAA", "statistics": {},
            "diagnosis": {"predicted_class": 2, "confidence": confidence,
                          "predicted_fault": {"name": fault, "name_en": "Ball Fault"},
                          "class_probabilities": {}}}


def _transport(rag_body=None, fd_body=None, health_ok=True):
    rules = {}
    rules["8000"] = _fakes.ok(rag_body or {"results": [], "num_results": 0})
    rules["5000"] = _fakes.ok(fd_body or _fd_degraded())
    if health_ok:
        rules["6006"] = _fakes.ok({})
    else:
        rules["6006"] = _fakes.connection_error()
    return _fakes.routed(rules)


class AgentTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, transport, signal=SIGNAL, **kwargs):
        agent = DiagnosisAgent(transport=transport, log_dir=self.log_dir)
        return agent.run(signal=signal, sample_info={"source_file": "ib600_1.csv"}, **kwargs)


class DegradedFlowTest(AgentTestBase):
    def test_model_unavailable_ends_degraded(self):
        result = self._run(_transport(rag_body={"results": [_case(1, 0.73, "滚动体故障")], "num_results": 1}))
        self.assertEqual(result["terminal_state"], DEGRADED)
        self.assertFalse(result["ok"])
        self.assertEqual(result["max_allowed_strength"], "no_conclusion")
        self.assertIn("无法给出故障类别判断", result["report_markdown"])
        self.assertNotIn("建议停机", result["report_markdown"])

    def test_evidence_conflict_is_flagged(self):
        rag_body = {"results": [_case(1, 0.73, "滚动体故障"), _case(2, 0.11, "内圈故障")],
                    "num_results": 2}
        result = self._run(_transport(rag_body=rag_body, fd_body=_fd_ok()))
        markdown = result["report_markdown"]
        self.assertIn("候选标签不一致", markdown)
        checks = next(e for e in result["trace"] if e["state"] == "VERIFYING_RESULT")
        self.assertEqual(checks["status"], "degraded")

    def test_service_down_still_produces_report(self):
        result = self._run(_transport(health_ok=False))
        self.assertIn(result["terminal_state"], (DEGRADED, COMPLETED))
        self.assertTrue(result["report_markdown"])


class InputGuardTest(AgentTestBase):
    def test_missing_signal_asks_user(self):
        result = self._run(_transport(), signal=None)
        self.assertEqual(result["terminal_state"], ASKING_USER)
        self.assertIn("1024", result["question"])

    def test_wrong_length_asks_user(self):
        result = self._run(_transport(), signal=[0.0] * 512)
        self.assertEqual(result["terminal_state"], ASKING_USER)
        self.assertIn("512", result["question"])

    def test_non_numeric_asks_user(self):
        result = self._run(_transport(), signal=["a"] * 1024)
        self.assertEqual(result["terminal_state"], ASKING_USER)


class HappyPathTest(AgentTestBase):
    def test_consistent_evidence_completes(self):
        rag_body = {"results": [_case(1, 0.73, "滚动体故障"), _case(2, 0.50, "滚动体故障"),
                                _case(3, 0.40, "滚动体故障")], "num_results": 3}
        result = self._run(_transport(rag_body=rag_body, fd_body=_fd_ok()))
        self.assertEqual(result["terminal_state"], COMPLETED)
        self.assertTrue(result["ok"])
        self.assertEqual(result["trust_level"], "high")
        self.assertTrue(Path(result["report_path"]).exists())


class TraceTest(AgentTestBase):
    def test_trace_states_are_ordered_and_safe(self):
        result = self._run(_transport(rag_body={"results": [_case(1, 0.73, "滚动体故障")], "num_results": 1}))
        states = [entry["state"] for entry in result["trace"]]
        for expected in ("RECEIVED", "ANALYZING_SIGNAL", "RETRIEVING_CASES",
                         "RUNNING_DIAGNOSIS", "VERIFYING_RESULT", "GENERATING_REPORT"):
            self.assertIn(expected, states)

    def test_trace_never_contains_full_signal(self):
        result = self._run(_transport(rag_body={"results": [_case(1, 0.73, "滚动体故障")], "num_results": 1}))
        serialized = json.dumps(result["trace"], ensure_ascii=False)
        self.assertNotIn("0.1, 0.1, 0.1", serialized)
        self.assertNotIn("spectrogram", serialized.replace("spectrogram_present", "")
                         .replace("spectrogram_bytes", ""))
        self.assertTrue(result["trace_path"] and Path(result["trace_path"]).exists())

    def test_trace_file_is_valid_jsonl(self):
        result = self._run(_transport())
        lines = Path(result["trace_path"]).read_text(encoding="utf-8").strip().splitlines()
        self.assertGreater(len(lines), 3)
        for line in lines:
            json.loads(line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
