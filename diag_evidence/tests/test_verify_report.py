"""verify_result 与 generate_report 的测试（含低可信拦截与泄漏检测）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diag_evidence.tests import _fakes  # noqa: E402
from diag_evidence.hatools import diagnosis, rag, report, signal, verify  # noqa: E402

from diag_evidence.demo import synthetic_signal

SIGNAL = synthetic_signal()


def _fd_degraded():
    body = {"success": True, "spectrogram": "AAAA", "statistics": {},
            "diagnosis": {"note": "模型未加载", "predicted_class": -1}}
    return diagnosis.run_fault_diagnosis(SIGNAL, transport=_fakes.ok(body))


def _fd_ok(confidence=0.81):
    body = {"success": True, "spectrogram": "AAAA", "statistics": {},
            "diagnosis": {"predicted_class": 2, "confidence": confidence,
                          "predicted_fault": {"name": "滚动体故障", "name_en": "Ball Fault"},
                          "class_probabilities": {}}}
    return diagnosis.run_fault_diagnosis(SIGNAL, transport=_fakes.ok(body))


def _case(rank, score, fault):
    return {"rank": rank, "score": score, "text": "%s描述" % fault, "meta": {"fault_type": fault, "record_id": f"synthetic-{rank}"}}


def _rag(cases):
    return rag.retrieve_cases(SIGNAL, transport=_fakes.ok({"results": cases, "num_results": len(cases)}))


class VerifyTest(unittest.TestCase):
    def test_model_unavailable_forbids_conclusions(self):
        result = verify.verify_result(diagnosis_result=_fd_degraded())
        self.assertIn(result["data"]["trust_level"], ("low", "unavailable"))
        self.assertEqual(result["data"]["max_allowed_strength"], "no_conclusion")

    def test_leakage_detected(self):
        text = "## 1. 设备信息\n| **真实标签** | 滚动体故障 |\n**判断结果**：✅ 诊断正确"
        hits = verify.detect_label_leakage(text)
        self.assertTrue(any("真实标签" in h for h in hits))
        self.assertTrue(any("诊断正确" in h for h in hits))
        result = verify.verify_result(diagnosis_result=_fd_ok(), llm_prompt_text=text)
        check = next(c for c in result["data"]["checks"] if c["name"] == "label_leakage")
        self.assertFalse(check["passed"])
        self.assertEqual(result["data"]["trust_level"], "low")

    def test_no_leakage_when_clean(self):
        result = verify.verify_result(diagnosis_result=_fd_ok(), llm_prompt_text="仅含统计特征与检索案例")
        check = next(c for c in result["data"]["checks"] if c["name"] == "label_leakage")
        self.assertTrue(check["passed"])

    def test_evidence_conflict(self):
        rag_result = _rag([_case(1, 0.73, "滚动体故障"), _case(2, 0.11, "内圈故障")])
        result = verify.verify_result(diagnosis_result=_fd_ok(), rag_result=rag_result)
        check = next(c for c in result["data"]["checks"] if c["name"] == "evidence_conflict")
        self.assertFalse(check["passed"])
        self.assertTrue(any("候选标签不一致" in w for w in result["warnings"]))

    def test_no_conflict_when_consistent(self):
        rag_result = _rag([_case(1, 0.73, "滚动体故障"), _case(2, 0.50, "滚动体故障"),
                           _case(3, 0.40, "滚动体故障")])
        result = verify.verify_result(diagnosis_result=_fd_ok(), rag_result=rag_result)
        check = next(c for c in result["data"]["checks"] if c["name"] == "evidence_conflict")
        self.assertTrue(check["passed"])

    def test_synthetic_always_warned(self):
        result = verify.verify_result(diagnosis_result=_fd_ok(), input_is_synthetic=True)
        self.assertTrue(any("合成数据" in w for w in result["warnings"]))

    def test_definitive_phrase_detection(self):
        hits = verify.find_definitive_phrases("建议停机排查并更换轴承")
        self.assertIn("建议停机", hits)
        self.assertIn("更换轴承", hits)


class ReportTest(unittest.TestCase):
    def test_degraded_report_forbids_definitive_conclusions(self):
        result = report.generate_report(
            signal_result=signal.analyze_signal(SIGNAL),
            rag_result=_rag([_case(1, 0.73, "滚动体故障")]),
            diagnosis_result=_fd_degraded(),
            sample_info={"source_file": "ib600_1.csv", "input_is_synthetic": True},
            write_file=False,
        )
        markdown = result["data"]["markdown"]
        self.assertIn("仅用于工程流程验证", markdown)
        self.assertIn("无法给出故障类别判断", markdown)
        self.assertIn("模型能力不可用", markdown)
        for phrase in ("建议停机", "拆检", "更换轴承", "备件采购"):
            self.assertNotIn(phrase, markdown, msg="低可信报告不应出现 %s" % phrase)

    def test_report_marks_conflict_and_leakage(self):
        rag_result = _rag([_case(1, 0.73, "滚动体故障"), _case(2, 0.11, "内圈故障")])
        result = report.generate_report(
            signal_result=signal.analyze_signal(SIGNAL),
            rag_result=rag_result,
            diagnosis_result=_fd_ok(),
            sample_info={"source_file": "x.csv", "input_is_synthetic": True,
                         "llm_prompt_text": "| **真实标签** | 滚动体故障 |"},
            write_file=False,
        )
        markdown = result["data"]["markdown"]
        self.assertIn("候选标签不一致", markdown)
        self.assertIn("标签泄漏", markdown)

    def test_report_written_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = report.generate_report(
                signal_result=signal.analyze_signal(SIGNAL),
                diagnosis_result=_fd_degraded(),
                sample_info={"source_file": "x.csv"},
                output_dir=tmp,
                write_file=True,
            )
            path = result["data"]["path"]
            self.assertTrue(path and Path(path).exists())

    def test_ok_report_allows_moderate_advice(self):
        rag_result = _rag([_case(1, 0.73, "滚动体故障"), _case(2, 0.50, "滚动体故障"),
                           _case(3, 0.40, "滚动体故障")])
        result = report.generate_report(
            signal_result=signal.analyze_signal(SIGNAL),
            rag_result=rag_result,
            diagnosis_result=_fd_ok(),
            sample_info={"source_file": "x.csv", "input_is_synthetic": False},
            write_file=False,
        )
        markdown = result["data"]["markdown"]
        self.assertIn("人工复核", markdown)
        self.assertNotIn("仅用于工程流程验证", markdown)


if __name__ == "__main__":
    unittest.main(verbosity=2)
