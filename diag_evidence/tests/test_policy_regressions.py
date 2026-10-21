"""Boundary regressions for explicit policy states and constrained reporting."""

from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from diag_evidence.hatools.report import generate_report
from diag_evidence.hatools.verify import POLICY_VERSION, verify_result


def _diagnosis(**changes):
    diagnosis = {"model_available": True, "predicted_class": 2,
                 "predicted_fault": {"name": "滚动体故障"}, "confidence": 0.8,
                 "conclusion_strength": "strong"}
    diagnosis.update(changes)
    return {"status": "ok", "data": {"diagnosis": diagnosis}}


def _rag(count=3, faults=None):
    cases = [{"rank": index + 1, "score": 0.8 - index * 0.1,
              "fault_type": (faults[index] if faults else "滚动体故障"),
              "text": "合成案例"} for index in range(count)]
    return {"status": "ok", "data": {"cases": cases, "num_results": count}}


def _check(result, name):
    return next(item for item in result["data"]["checks"] if item["name"] == name)


class PolicyBoundaryTest(unittest.TestCase):
    def test_complete_template_has_explicit_not_applicable_and_rule_support(self):
        result = verify_result(_diagnosis(), _rag(), input_is_synthetic=False)
        self.assertEqual(result["data"]["support_level"], "high")
        self.assertEqual(result["data"]["trust_level"], "high")
        self.assertEqual(result["data"]["policy_version"], POLICY_VERSION)
        check = _check(result, "label_leakage")
        self.assertEqual(check["state"], "not_applicable")
        self.assertFalse(check["required"])
        self.assertIsNone(check["passed"])
        self.assertIn("不代表诊断正确概率", result["data"]["support_definition"])
        self.assertIsNone(result["data"]["independent_evidence_count"])

    def test_expected_llm_without_prompt_is_unknown_and_cannot_be_high(self):
        result = verify_result(_diagnosis(), _rag(), llm_expected=True)
        self.assertEqual(_check(result, "label_leakage")["state"], "unknown")
        self.assertNotEqual(result["data"]["support_level"], "high")

    def test_missing_confidence_is_unknown_and_cannot_be_high(self):
        result = verify_result(_diagnosis(confidence=None), _rag())
        self.assertEqual(_check(result, "confidence_valid")["state"], "unknown")
        self.assertEqual(result["data"]["max_allowed_strength"], "weak")

    def test_nonfinite_out_of_range_and_nonnumeric_confidence_fail(self):
        for value in (math.nan, math.inf, -math.inf, 1.5, -0.1, True, "0.9", {}):
            with self.subTest(value=value):
                result = verify_result(_diagnosis(confidence=value), _rag())
                self.assertEqual(_check(result, "confidence_valid")["state"], "failed")
                self.assertEqual(result["data"]["support_level"], "low")

    def test_invalid_class_or_name_cannot_generate_a_conclusion(self):
        for changes in ({"predicted_class": None}, {"predicted_class": 3},
                        {"predicted_class": True}, {"predicted_class": "2"},
                        {"predicted_fault": {"name": "内圈故障"}},
                        {"predicted_fault": "Unknown"}):
            with self.subTest(changes=changes):
                result = verify_result(_diagnosis(**changes), _rag())
                self.assertEqual(result["data"]["max_allowed_strength"], "no_conclusion")
                self.assertNotEqual(result["data"]["support_level"], "high")

    def test_candidate_diversity_is_not_called_semantic_conflict(self):
        result = verify_result(_diagnosis(), _rag(3, ["滚动体故障", "内圈故障", "滚动体故障"]))
        check = _check(result, "evidence_conflict")
        self.assertEqual(check["state"], "failed")
        self.assertEqual(check["semantic"], "candidate_label_agreement")
        self.assertIn("不据此认定语义矛盾", check["detail"])
        self.assertFalse(any("RAG 证据冲突" in warning for warning in result["warnings"]))

    def test_unknown_candidate_labels_block_high(self):
        rag = _rag()
        rag["data"]["cases"][0].pop("fault_type")
        result = verify_result(_diagnosis(), rag)
        self.assertEqual(_check(result, "evidence_conflict")["state"], "unknown")
        self.assertEqual(result["data"]["support_level"], "medium")

    def test_claimed_count_does_not_replace_actual_candidates(self):
        rag = _rag(1)
        rag["data"]["num_results"] = 100
        result = verify_result(_diagnosis(), rag)
        self.assertEqual(result["data"]["retrieval_candidate_count"], 1)
        self.assertEqual(_check(result, "retrieval_scale")["state"], "failed")

    def test_badly_shaped_payloads_fail_closed(self):
        for value in (None, [], {"data": []}, {"data": {"diagnosis": "bad"}}):
            with self.subTest(value=value):
                result = verify_result(value, {"data": {"cases": "bad"}})
                self.assertEqual(result["data"]["max_allowed_strength"], "no_conclusion")
        result = verify_result(_diagnosis(conclusion_strength={}), _rag())
        self.assertEqual(result["data"]["diagnosis_strength"], "no_conclusion")


class ReportBoundaryTest(unittest.TestCase):
    def test_injected_strong_strength_is_actually_clamped(self):
        result = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(1), write_file=False)
        self.assertEqual(result["data"]["requested_strength"], "strong")
        self.assertEqual(result["data"]["max_allowed_strength"], "moderate")
        self.assertEqual(result["data"]["effective_strength"], "moderate")
        self.assertIn("**结论强度**：moderate", result["data"]["markdown"])
        self.assertNotIn("**结论强度**：strong", result["data"]["markdown"])

    def test_missing_or_obsolete_supplied_verification_closes_conclusions(self):
        for verifier in ({}, {"data": {"trust_level": "high", "max_allowed_strength": "strong"}},
                         {"data": {"policy_version": "old", "checks": []}}):
            with self.subTest(verifier=verifier):
                result = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(),
                                         verify_data=verifier, write_file=False)
                self.assertEqual(result["data"]["effective_strength"], "no_conclusion")
                self.assertIn("无法给出故障类别判断", result["data"]["markdown"])

    def test_false_high_declaration_cannot_bypass_failed_checks(self):
        verified = verify_result(_diagnosis(), _rag(), llm_prompt_text="真实标签：滚动体故障")
        verified["data"].update(support_level="high", trust_level="high", max_allowed_strength="strong")
        result = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(),
                                 verify_data=verified, write_file=False)
        self.assertEqual(result["data"]["effective_strength"], "weak")
        self.assertNotEqual(result["data"]["support_level"], "high")

    def test_each_leakage_state_has_accurate_footer(self):
        scenarios = (
            ({}, "不适用"),
            ({"llm_expected": True}, "未获完整检查"),
            ({"llm_prompt_text": "统计特征"}, "未命中本版本泄漏规则"),
            ({"llm_prompt_text": "真实标签：滚动体故障"}, "命中标签泄漏规则"),
        )
        for info, expected in scenarios:
            with self.subTest(info=info):
                result = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(),
                                         sample_info=info, write_file=False)
                self.assertIn(expected, result["data"]["markdown"])
                self.assertNotIn("未使用真实标签作为输入", result["data"]["markdown"])

    def test_external_markdown_and_html_stay_escaped_quotations(self):
        rag = _rag()
        rag["data"]["cases"][0]["text"] = '<script>alert(1)</script>\n# 确诊\n[批准](https://invalid.example)'
        result = generate_report(diagnosis_result=_diagnosis(), rag_result=rag,
                                 sample_info={"source_file": "x|y\n# 外来标题"}, write_file=False)
        markdown = result["data"]["markdown"]
        self.assertNotIn("<script>", markdown)
        self.assertNotIn("\n# 确诊", markdown)
        self.assertNotIn("[批准](https://", markdown)
        self.assertIn("&lt;script&gt;", markdown)
        self.assertIn("外部原文摘录：", markdown)
        self.assertIn("x\\|y", markdown)

    def test_read_only_render_does_not_create_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "must_not_exist"
            result = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(),
                                     output_dir=str(target), write_file=False)
            self.assertFalse(target.exists())
            self.assertIsNone(result["data"]["path"])

    def test_same_run_id_does_not_overwrite_and_uses_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(),
                                    output_dir=tmp, run_id="same-run")
            first_path = Path(first["data"]["path"])
            first_text = first_path.read_text()
            second = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(1),
                                     output_dir=tmp, run_id="same-run")
            self.assertNotEqual(first["data"]["path"], second["data"]["path"])
            self.assertEqual(first_path.read_text(), first_text)
            with patch("diag_evidence.hatools.report.uuid.uuid4") as mocked:
                mocked.return_value.hex = "fixed"
                one = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(),
                                      output_dir=tmp, run_id="collision")
                two = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(1),
                                      output_dir=tmp, run_id="collision")
            self.assertIsNotNone(one["data"]["path"])
            self.assertIsNone(two["data"]["path"])
            self.assertEqual(two["status"], "degraded")
            self.assertTrue(any("FileExistsError" in item for item in two["warnings"]))

    def test_path_like_run_id_is_safe_and_stays_within_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(),
                                     output_dir=tmp, run_id="../../nested/path")
            self.assertEqual(Path(result["data"]["path"]).parent, Path(tmp))
            self.assertNotIn("/", result["data"]["run_id"])

    def test_local_validation_limits_stale_verification(self):
        verified = verify_result(_diagnosis(), _rag())
        result = generate_report(diagnosis_result=_diagnosis(predicted_class=None), rag_result=_rag(),
                                 verify_data=verified, write_file=False)
        self.assertEqual(result["data"]["effective_strength"], "no_conclusion")


if __name__ == "__main__":
    unittest.main()
