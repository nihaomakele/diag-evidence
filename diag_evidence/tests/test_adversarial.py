"""Adversarial integration regressions: one authority for policy and safe errors."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from diag_evidence.agent import DiagnosisAgent
from diag_evidence.demo import SyntheticTransport, classifier, synthetic_signal
from diag_evidence.hatools.diagnosis import run_fault_diagnosis
from diag_evidence.hatools.verify import verify_result
from diag_evidence.hatools.report import generate_report


class ReportIntegrationTest(unittest.TestCase):
    def test_sample_metadata_cannot_override_explicit_template_mode(self):
        result = DiagnosisAgent(transport=SyntheticTransport("consistent"), persist_trace=False).run(
            synthetic_signal(), skip_services_check=True, llm_expected=False,
            sample_info={"llm_expected": True})
        report = result["outcomes"]["report"]["data"]
        self.assertEqual(result["terminal_state"], "COMPLETED")
        self.assertEqual(result["support_level"], report["support_level"])
        self.assertEqual(result["max_allowed_strength"], report["max_allowed_strength"])
        self.assertIn("模板流程未调用 LLM", report["markdown"])

    def test_sample_metadata_cannot_override_the_actual_prompt(self):
        result = DiagnosisAgent(transport=SyntheticTransport("consistent"), persist_trace=False).run(
            synthetic_signal(), skip_services_check=True, llm_expected=True,
            llm_prompt_text="仅含统计特征", sample_info={"llm_prompt_text": "真实标签=滚动体故障"})
        report = result["outcomes"]["report"]["data"]
        self.assertEqual(result["support_level"], "high")
        self.assertEqual(report["support_level"], "high")
        self.assertIn("未命中本版本泄漏规则", report["markdown"])

    def test_report_downgrade_is_reflected_in_terminal_state_and_public_cap(self):
        from diag_evidence.hatools.report import generate_report

        def downgraded_report(**kwargs):
            result = generate_report(**kwargs)
            result["status"] = "degraded"
            result["data"].update(support_level="low", trust_level="low",
                                   max_allowed_strength="weak", effective_strength="weak")
            result["warnings"].append("injected report limitation")
            return result

        with patch("diag_evidence.agent.generate_report", side_effect=downgraded_report):
            result = DiagnosisAgent(transport=SyntheticTransport("consistent"), persist_trace=False).run(
                synthetic_signal(), skip_services_check=True)
        self.assertEqual(result["terminal_state"], "DEGRADED")
        self.assertEqual(result["support_level"], "low")
        self.assertEqual(result["max_allowed_strength"], "weak")
        self.assertEqual(result["effective_strength"], "weak")


class SafeRemoteFailureTest(unittest.TestCase):
    def test_remote_error_does_not_echo_arbitrary_private_prompt(self):
        private_prompt = "PRIVATE_PROMPT_SENTINEL_DO_NOT_EXPORT"

        def transport(method, url, **kwargs):
            return "ok", {"success": False, "error": private_prompt}, ""

        result = run_fault_diagnosis(synthetic_signal(), transport=transport)
        self.assertEqual(result["status"], "error")
        self.assertNotIn(private_prompt, json.dumps(result, ensure_ascii=False))

    def test_client_error_details_are_not_exported(self):
        private_prompt = "PRIVATE_PROMPT_SENTINEL_DO_NOT_EXPORT"

        def transport(method, url, **kwargs):
            return "client", None, private_prompt

        result = run_fault_diagnosis(synthetic_signal(), transport=transport)
        self.assertEqual(result["status"], "invalid_input")
        self.assertNotIn(private_prompt, json.dumps(result, ensure_ascii=False))

    def test_remote_statistics_do_not_export_credentials_or_nonfinite_json(self):
        secret = "STATISTICS_PRIVATE_CREDENTIAL_SENTINEL"

        def transport(method, url, **kwargs):
            body = classifier()
            body["statistics"] = {"api_key": secret,
                                  "nested": {"authorization": secret},
                                  "mean": float("nan")}
            return "ok", body, ""

        result = run_fault_diagnosis(synthetic_signal(), transport=transport)
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        self.assertNotIn(secret, encoded)


class VerifierStatusTest(unittest.TestCase):
    def test_error_envelope_cannot_be_promoted_by_stale_valid_payload(self):
        diagnosis = {"status": "error", "data": {"diagnosis": {
            "model_available": True, "predicted_class": 2,
            "predicted_fault": {"name": "滚动体故障"}, "confidence": 0.9,
            "conclusion_strength": "strong"}}}
        rag = {"status": "ok", "data": {"num_results": 3, "cases": [
            {"fault_type": "滚动体故障", "text": str(index)} for index in range(3)]}}
        result = verify_result(diagnosis, rag)
        self.assertNotEqual(result["data"]["support_level"], "high")
        self.assertEqual(result["data"]["max_allowed_strength"], "no_conclusion")

    def test_bad_envelope_states_override_stale_fields(self):
        from diag_evidence.tests.test_policy_regressions import _diagnosis, _rag
        for status in ("error", "unavailable", "invalid_input"):
            with self.subTest(status=status):
                diagnosis = _diagnosis()
                diagnosis["status"] = status
                result = verify_result(diagnosis, _rag())
                self.assertEqual(result["data"]["max_allowed_strength"], "no_conclusion")
                rendered = generate_report(diagnosis_result=diagnosis, rag_result=_rag(), write_file=False)
                self.assertEqual(rendered["data"]["effective_strength"], "no_conclusion")

    def test_failed_verifier_envelope_cannot_supply_a_cap(self):
        from diag_evidence.tests.test_policy_regressions import _diagnosis, _rag
        verified = verify_result(_diagnosis(), _rag())
        verified["status"] = "error"
        result = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(),
                                 verify_data=verified, write_file=False)
        self.assertEqual(result["data"]["effective_strength"], "no_conclusion")

    def test_failed_retrieval_payload_is_not_used_as_support(self):
        from diag_evidence.tests.test_policy_regressions import _diagnosis, _rag
        rag = _rag()
        rag["status"] = "error"
        result = verify_result(_diagnosis(), rag)
        self.assertNotEqual(result["data"]["support_level"], "high")
        self.assertEqual(result["data"]["retrieval_candidate_count"], 0)

class StaleReportMetadataTest(unittest.TestCase):
    def test_stale_clean_prompt_cannot_hide_current_leakage(self):
        from diag_evidence.demo import SyntheticTransport, synthetic_signal
        from diag_evidence.hatools import run_fault_diagnosis, retrieve_cases, verify_result, generate_report
        transport = SyntheticTransport("consistent")
        fd = run_fault_diagnosis(synthetic_signal(), transport=transport)
        cases = retrieve_cases(synthetic_signal(), transport=transport)
        clean = verify_result(fd, cases, llm_prompt_text="only statistics", input_is_synthetic=False)
        output = generate_report(diagnosis_result=fd, rag_result=cases, verify_data=clean,
                                 sample_info={"llm_prompt_text": "真实标签=滚动体故障", "input_is_synthetic": False},
                                 write_file=False)
        text = output["data"]["markdown"]
        self.assertIn("标签泄漏", text)
        self.assertIn("不能声明输入未使用真实标签", text)
        self.assertNotIn("已提供的提示词未命中", text)
        self.assertEqual(output["data"]["max_allowed_strength"], "weak")

    def test_stale_real_input_flag_cannot_hide_synthetic_marker(self):
        from diag_evidence.demo import SyntheticTransport, synthetic_signal
        from diag_evidence.hatools import run_fault_diagnosis, retrieve_cases, verify_result, generate_report
        transport = SyntheticTransport("consistent")
        fd = run_fault_diagnosis(synthetic_signal(), transport=transport)
        cases = retrieve_cases(synthetic_signal(), transport=transport)
        stale = verify_result(fd, cases, input_is_synthetic=False)
        output = generate_report(diagnosis_result=fd, rag_result=cases, verify_data=stale,
                                 sample_info={"input_is_synthetic": True}, write_file=False)
        self.assertTrue(output["data"]["input_is_synthetic"])
        self.assertIn("仅用于工程流程验证", output["data"]["markdown"])


if __name__ == "__main__":
    unittest.main()
