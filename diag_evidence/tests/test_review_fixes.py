"""Release regressions for rejected controls, safe errors and conservative reporting."""
from __future__ import annotations

from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diag_evidence.agent import DiagnosisAgent, run_agent
from diag_evidence.audit_log import AuditLogger, sanitize
from diag_evidence.cli import main
from diag_evidence.demo import SCENARIOS, SyntheticTransport, assess, synthetic_signal
from diag_evidence.haconfig.settings import Settings
from diag_evidence.hatools.report import generate_report
from diag_evidence.hatools.verify import verify_result
from diag_evidence.tests.test_policy_regressions import _diagnosis, _rag


class ReviewReleaseRegressionTest(unittest.TestCase):
    def test_unknown_convenience_controls_rejected_before_agent_construction(self):
        fake = "REVIEW_FAKE_SECRET_NOT_A_REAL_CREDENTIAL"
        for kwargs in ({"llm_expectd": True}, {"persist_output": False},
                       {"api_key=" + fake: "ignored"}):
            with self.subTest(keys=list(kwargs)), patch("diag_evidence.agent.DiagnosisAgent") as agent:
                with self.assertRaises(TypeError) as caught:
                    run_agent(synthetic_signal(), **kwargs)
                agent.assert_not_called()
                self.assertNotIn(fake, str(caught.exception))

    def test_misspelled_persistence_creates_no_files_or_transport_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "must-not-exist"
            config = Settings(reports_dir=str(target / "reports"), logs_dir=str(target / "logs"))
            transport = SyntheticTransport("consistent")
            with self.assertRaises(TypeError):
                run_agent(synthetic_signal(), config=config, transport=transport,
                          persist_output=False)
            self.assertEqual(transport.calls, [])
            self.assertFalse(target.exists())

    def test_malformed_extra_verifier_checks_close_report_without_crashing(self):
        malformed = (None, [], {}, {"name": []}, {"name": {}},
                     {"name": "", "state": "passed", "required": True},
                     {"name": "extra", "state": [], "required": True},
                     {"name": "extra", "state": "passed", "required": 1})
        for item in malformed:
            with self.subTest(check=item):
                verifier = verify_result(_diagnosis(), _rag())
                verifier["data"]["checks"].append(item)
                output = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(),
                                         verify_data=verifier, write_file=False)
                self.assertEqual(output["data"]["effective_strength"], "no_conclusion")
                self.assertEqual(output["data"]["max_allowed_strength"], "no_conclusion")
                self.assertIn("无法给出故障类别判断", output["data"]["markdown"])

    def test_duplicate_verifier_check_names_are_not_silently_collapsed(self):
        verifier = verify_result(_diagnosis(), _rag())
        verifier["data"]["checks"].append(deepcopy(verifier["data"]["checks"][0]))
        output = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(),
                                 verify_data=verifier, write_file=False)
        self.assertEqual(output["data"]["effective_strength"], "no_conclusion")

    def test_extra_required_checks_are_preserved_and_limit_the_report(self):
        for state in ("failed", "unknown"):
            with self.subTest(state=state):
                verifier = verify_result(_diagnosis(), _rag())
                verifier["data"]["checks"].append({
                    "name": "source_applicability", "state": state, "required": True,
                    "detail": "Extra source review remains incomplete",
                })
                output = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(),
                                         verify_data=verifier, write_file=False)
                self.assertEqual(output["data"]["support_level"], "medium")
                self.assertEqual(output["data"]["max_allowed_strength"], "moderate")
                self.assertEqual(output["data"]["effective_strength"], "moderate")
                self.assertIn("Extra source review remains incomplete", output["data"]["markdown"])

    def test_extra_optional_failed_check_is_shown_without_becoming_required(self):
        verifier = verify_result(_diagnosis(), _rag(), input_is_synthetic=False)
        verifier["data"]["checks"].append({
            "name": "optional_review", "state": "failed", "required": False,
            "detail": "Optional external review",
        })
        output = generate_report(diagnosis_result=_diagnosis(), rag_result=_rag(),
                                 verify_data=verifier,
                                 sample_info={"input_is_synthetic": False}, write_file=False)
        self.assertEqual(output["data"]["support_level"], "high")
        self.assertEqual(output["data"]["max_allowed_strength"], "strong")
        self.assertIn("Optional external review", output["data"]["markdown"])

    def test_huge_integer_confidence_fails_verification_and_report_remains_renderable(self):
        for value in (10 ** 400, -(10 ** 400)):
            with self.subTest(sign=value > 0):
                diagnosis = _diagnosis(confidence=value)
                verified = verify_result(diagnosis, _rag())
                confidence_check = next(item for item in verified["data"]["checks"]
                                        if item["name"] == "confidence_valid")
                self.assertEqual(confidence_check["state"], "failed")
                self.assertEqual(verified["data"]["max_allowed_strength"], "weak")
                output = generate_report(diagnosis_result=diagnosis, rag_result=_rag(), write_file=False)
                self.assertEqual(output["data"]["effective_strength"], "weak")
                self.assertIn("**模型置信度（未经校准）**：N/A", output["data"]["markdown"])

    def test_credential_values_embedded_in_dynamic_keys_are_removed_at_every_depth(self):
        fake = "REVIEW_FAKE_SECRET_NOT_A_REAL_CREDENTIAL"
        metadata = {"nested": [{"api_key=" + fake: "ignored",
                                "password_" + fake: "ignored",
                                "Authorization: Bearer " + fake: "ignored"}]}
        encoded = json.dumps(sanitize(metadata), allow_nan=False)
        self.assertNotIn(fake, encoded)
        logger = AuditLogger(persist=False)
        entry = logger.log("review-fixture", {"status": "error"}, extra=metadata)
        self.assertNotIn(fake, json.dumps(entry, allow_nan=False))
        self.assertIn("REDACTED", encoded)

    def test_cli_rejected_options_and_choice_values_do_not_echo_secrets(self):
        fake = "REVIEW_FAKE_SECRET_NOT_A_REAL_CREDENTIAL"
        for argv in (["demo", "--no-write", "--password=" + fake],
                     ["demo", "--scenario", fake],
                     ["run", fake, "--token=" + fake]):
            with self.subTest(argv=argv):
                errors = StringIO()
                with redirect_stderr(errors), self.assertRaises(SystemExit) as caught:
                    main(argv)
                self.assertEqual(caught.exception.code, 2)
                self.assertNotIn(fake, errors.getvalue())
                self.assertNotIn("Traceback", errors.getvalue())
                self.assertIn("--help", errors.getvalue())

    def low_confidence_run(self):
        transport = SyntheticTransport("low_confidence")
        result = DiagnosisAgent(transport=transport, persist_trace=False).run(
            synthetic_signal(), skip_services_check=True)
        scenario = next(item for item in SCENARIOS if item.name == "low_confidence")
        return scenario, result, transport.calls

    def test_scenario_scoring_rejects_effective_strength_above_the_cap(self):
        scenario, result, calls = self.low_confidence_run()
        self.assertTrue(assess(scenario, result, calls)["passed"])
        result["effective_strength"] = "strong"
        result["report_markdown"] = "**结论强度**：strong"
        result["outcomes"]["report"]["data"].update(
            effective_strength="strong", markdown=result["report_markdown"])
        score = assess(scenario, result, calls)
        self.assertFalse(score["passed"])
        self.assertFalse(score["strength_within_cap"])

    def test_scenario_scoring_rejects_invalid_or_missing_effective_strength(self):
        scenario, original, calls = self.low_confidence_run()
        for value in (None, "unknown", True, [], {}):
            with self.subTest(value=value):
                result = deepcopy(original)
                result["effective_strength"] = value
                score = assess(scenario, result, calls)
                self.assertFalse(score["passed"])
                self.assertFalse(score["strength_within_cap"])

    def test_scenario_scoring_detects_disagreement_with_actual_report_fields(self):
        scenario, result, calls = self.low_confidence_run()
        result["outcomes"]["report"]["data"]["effective_strength"] = "strong"
        score = assess(scenario, result, calls)
        self.assertFalse(score["passed"])
        self.assertFalse(score["report_strength_consistent"])


if __name__ == "__main__":
    unittest.main()
