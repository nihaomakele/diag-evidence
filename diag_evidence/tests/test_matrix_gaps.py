"""T01～T10 覆盖缺口补测（阶段 5）。

补齐：T05（FD 不可用须声明无法判定）、T08（API Key 缺失须明确报错不泄露）、
T09（合成数据须有报告顶部标注）、T10（相同输入重复执行须结构稳定）。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diag_evidence.agent import COMPLETED, DEGRADED, DiagnosisAgent  # noqa: E402
from diag_evidence.haconfig import settings  # noqa: E402
from diag_evidence.tests import _fakes  # noqa: E402

from diag_evidence.demo import synthetic_signal

SIGNAL = synthetic_signal()


def _case(rank, score, fault):
    return {"rank": rank, "score": score, "text": "%s描述" % fault, "meta": {"fault_type": fault, "record_id": f"synthetic-{rank}"}}


def _fd_ok():
    return {"success": True, "spectrogram": "AAAA", "statistics": {},
            "diagnosis": {"predicted_class": 2, "confidence": 0.81,
                          "predicted_fault": {"name": "滚动体故障", "name_en": "Ball Fault"},
                          "class_probabilities": {}}}


def _fd_degraded():
    return {"success": True, "spectrogram": "AAAA", "statistics": {},
            "diagnosis": {"note": "模型未加载", "predicted_class": -1}}


def _health_ok():
    return {"status": "ok", "model_loaded": True, "index_loaded": True, "num_texts": 12}


def _transport(rag=None, fd=None, ui=None, rag_down=False, fd_down=False):
    rules = {}
    rules["8000"] = _fakes.connection_error() if rag_down else (rag or _fakes.ok({"results": [], "num_results": 0}))
    rules["5000"] = _fakes.connection_error() if fd_down else (fd or _fakes.ok(_fd_degraded()))
    rules["6006"] = ui or _fakes.ok({})
    return _fakes.routed(rules)


def _run(transport, signal=SIGNAL, **kwargs):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        agent = DiagnosisAgent(transport=transport, log_dir=tmp)
        return agent.run(signal=signal, sample_info={"source_file": "ib600_1.csv"}, **kwargs)


def _run_ok():
    """构造"三服务全可用"的替身：/health 与 /v1/search/text 分别路由。"""
    rules = {
        "127.0.0.1:8000/health": _fakes.ok({"status": "ok", "model_loaded": True,
                                            "index_loaded": True, "num_texts": 12}),
        "127.0.0.1:8000/v1/search/text": _fakes.ok({
            "results": [_case(1, 0.73, "滚动体故障"), _case(2, 0.50, "滚动体故障"),
                        _case(3, 0.40, "滚动体故障")], "num_results": 3}),
        "127.0.0.1:5000/diagnose": _fakes.ok(_fd_ok()),
        "127.0.0.1:5000/": _fakes.ok({"status": "ok", "service": "Fault Diagnosis API"}),
        "127.0.0.1:6006/": _fakes.ok({}),
    }
    return _fakes.routed(rules)


class MatrixGapsTest(unittest.TestCase):
    def test_t04_rag_unavailable_report_states_it(self):
        result = _run(_transport(rag_down=True, fd=_fakes.ok(_fd_ok())))
        self.assertEqual(result["terminal_state"], DEGRADED, result["terminal_state"])
        markdown = result["report_markdown"] or ""
        assert ("无可用检索证据" in markdown) or ("无法连接" in markdown)

    def test_t05_fd_unavailable_has_no_conclusion(self):
        transport = _transport(fd_down=True, rag=_fakes.ok({"results": [_case(1, 0.7, "滚动体故障")], "num_results": 1}))
        result = _run(transport)
        markdown = result["report_markdown"] or ""
        self.assertEqual(result["terminal_state"], DEGRADED)
        self.assertIn("无法给出故障类别判断", markdown)
        section4 = markdown.split("## 4.")[-1].split("## 5.")[0] if "## 4." in markdown else ""
        self.assertNotIn("**诊断结论**", section4)

    def test_t08_missing_api_key_errors_without_leaking(self):
        original = os.environ.pop(settings.ENV_KEY_NAME, None)
        try:
            self.assertIsNone(settings.get_api_key())
            with self.assertRaises(settings.MissingAPIKeyError) as ctx:
                settings.get_api_key(required=True)
            message = str(ctx.exception)
            self.assertIn(settings.ENV_KEY_NAME, message)
            self.assertNotIn("sk-", message)
        finally:
            if original is not None:
                os.environ[settings.ENV_KEY_NAME] = original

    def test_t09_synthetic_data_banner_present(self):
        result = _run(_transport(fd=_fakes.ok(_fd_degraded())), input_is_synthetic=True)
        markdown = result["report_markdown"] or ""
        self.assertIn("仅用于工程流程验证", markdown)

    def test_t10_repeat_runs_are_structurally_stable(self):
        transport = _run_ok()
        runs = [_run(transport, input_is_synthetic=False) for _ in range(3)]
        self.assertEqual({r["terminal_state"] for r in runs}, {COMPLETED})
        self.assertEqual({r["trust_level"] for r in runs}, {"high"})
        self.assertEqual({r["max_allowed_strength"] for r in runs}, {"strong"})
        sequences = {tuple(e["tool"] for e in r["trace"]) for r in runs}
        self.assertEqual(len(sequences), 1, "工具调用顺序不稳定")


if __name__ == "__main__":
    unittest.main(verbosity=2)
