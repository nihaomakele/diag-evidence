"""Historical-reference decisions, normal controls and report bypass regressions."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from diag_evidence.agent import DiagnosisAgent, run_agent
from diag_evidence.cli import main
from diag_evidence.demo import SyntheticTransport, candidates, synthetic_signal
from diag_evidence.hatools import report, verify, rag
from diag_evidence.reference_policy import assess_references, validate_reference_context
from diag_evidence.tests import _fakes


CONTEXT = {"decision_at": "2026-11-01T10:00:00+08:00", "units": "g",
           "sampling_rate": 50000, "preprocessing_id": "fixture-preprocessing-v1"}
META = {"evidence_role": "historical_reference", "asset_id": "other-asset",
        "units": "g", "sampling_rate": 50000, "preprocessing_id": "fixture-preprocessing-v1",
        "window_start": "2026-10-31T09:00:00+08:00",
        "window_end": "2026-10-31T09:00:01+08:00",
        "available_at": "2026-10-31T10:00:00+08:00"}


class HistoricalTransport(SyntheticTransport):
    def __init__(self, changes=None, missing=(), extra_case=False):
        super().__init__("consistent")
        self.changes, self.missing, self.extra_case = changes or {}, missing, extra_case

    def __call__(self, method, url, **kwargs):
        kind, body, error = super().__call__(method, url, **kwargs)
        if "/v1/search/text" in url:
            for item in body["results"]:
                item["meta"].update(deepcopy(META))
                item["meta"].update(self.changes)
                for key in self.missing:
                    item["meta"].pop(key, None)
            if self.extra_case:
                extra = deepcopy(body["results"][0])
                extra["text"] = "A deliberately incompatible extra candidate"
                extra["meta"].update(units="m/s2", record_id="extra-record", fault_type="内圈故障")
                body["results"].append(extra)
            body["num_results"] = len(body["results"])
        return kind, body, error


class ReferencePolicyTest(unittest.TestCase):
    def assess(self, **changes):
        return assess_references([{"meta": {**META, **changes}}], CONTEXT)["assessments"][0]

    def test_cross_asset_history_is_a_valid_reference_not_current_observation(self):
        outcome = self.assess(asset_id="a-different-machine")
        self.assertEqual(outcome["state"], "passed")
        self.assertEqual(outcome["allowed_uses"], ["historical_reference"])

    def test_no_context_is_explicitly_not_applicable(self):
        result = assess_references([{}], None)
        self.assertEqual(result["state"], "not_applicable")
        self.assertFalse(result["enabled"])

    def test_unit_mismatch_is_not_fixed_by_identical_labels(self):
        self.assertIn("units_mismatch", self.assess(units="m/s2")["reason_codes"])

    def test_sampling_rate_mismatch(self):
        self.assertEqual(self.assess(sampling_rate=64000)["state"], "failed")

    def test_preprocessing_revision_mismatch(self):
        self.assertEqual(self.assess(preprocessing_id="different")["state"], "failed")

    def test_missing_required_metadata_stays_unknown(self):
        for key in ("units", "sampling_rate", "preprocessing_id", "window_end", "available_at"):
            with self.subTest(key=key):
                meta = {k: v for k, v in META.items() if k != key}
                outcome = assess_references([{"meta": meta}], CONTEXT)["assessments"][0]
                self.assertEqual(outcome["state"], "unknown")
                self.assertEqual(outcome["allowed_uses"], [])

    def test_optional_preprocessing_is_not_invented(self):
        context = {k: v for k, v in CONTEXT.items() if k != "preprocessing_id"}
        self.assertEqual(assess_references([{"meta": META}], context)["state"], "passed")

    def test_data_unavailable_at_cutoff_cannot_be_used(self):
        outcome = self.assess(available_at="2026-11-01T10:00:01+08:00")
        self.assertIn("unavailable_at_cutoff", outcome["reason_codes"])

    def test_future_observation_cannot_be_laundered_by_earlier_availability(self):
        outcome = self.assess(window_end="2026-11-01T10:00:01+08:00")
        self.assertIn("observation_after_cutoff", outcome["reason_codes"])
        self.assertIn("available_before_observation_end", outcome["reason_codes"])

    def test_availability_cannot_precede_observation_end(self):
        self.assertIn("available_before_observation_end",
                      self.assess(available_at="2026-10-31T08:00:00+08:00")["reason_codes"])

    def test_reversed_window(self):
        self.assertIn("window_order_invalid",
                      self.assess(window_start="2026-10-31T09:00:02+08:00")["reason_codes"])

    def test_equal_cutoff_and_equivalent_timezones_are_accepted(self):
        outcome = self.assess(window_end="2026-11-01T02:00:00Z",
                              available_at="2026-11-01T10:00:00+08:00")
        self.assertEqual(outcome["state"], "passed")

    def test_naive_or_invalid_case_timestamp_fails(self):
        for value in ("2026-11-01T09:00:00", "not-a-date", 1):
            with self.subTest(value=value):
                self.assertEqual(self.assess(available_at=value)["state"], "failed")

    def test_live_observation_cannot_be_relabelled_as_reference_support(self):
        self.assertEqual(self.assess(evidence_role="current_observation")["state"], "failed")

    def test_invalid_context_is_rejected_without_reflecting_values(self):
        bad = [False, {}, {**CONTEXT, "units": ""}, {**CONTEXT, "sampling_rate": True},
               {**CONTEXT, "sampling_rate": "50000"}, {**CONTEXT, "sampling_rate": float("nan")},
               {**CONTEXT, "decision_at": "2026-11-01"}, {**CONTEXT, "private-secret": "SENSITIVE"}]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as error:
                    validate_reference_context(value)
                self.assertNotIn("SENSITIVE", str(error.exception))
                self.assertNotIn("private-secret", str(error.exception))

    def test_input_objects_are_not_modified(self):
        cases, context = [{"meta": deepcopy(META)}], deepcopy(CONTEXT)
        snapshot = deepcopy((cases, context))
        assess_references(cases, context)
        self.assertEqual((cases, context), snapshot)


class ReferenceWorkflowTest(unittest.TestCase):
    def run_case(self, **transport_options):
        return run_agent(synthetic_signal(), evidence_context=CONTEXT,
                         transport=HistoricalTransport(**transport_options),
                         persist_trace=False, skip_services_check=True)

    def test_compatible_reference_control_keeps_existing_actual_strength(self):
        result = self.run_case()
        self.assertEqual(result["terminal_state"], "COMPLETED")
        self.assertEqual(result["support_level"], "high")
        self.assertEqual(result["effective_strength"], "moderate")
        self.assertEqual(result["outcomes"]["verify"]["data"]["retrieval_candidate_count"], 3)

    def test_incompatible_references_reduce_actual_report_strength(self):
        result = self.run_case(changes={"units": "m/s2"})
        self.assertEqual(result["terminal_state"], "DEGRADED")
        self.assertEqual(result["effective_strength"], "weak")
        self.assertEqual(result["outcomes"]["verify"]["data"]["retrieval_candidate_count"], 0)
        self.assertIn("units_mismatch", result["report_markdown"])

    def test_missing_reference_metadata_is_not_counted_as_support(self):
        result = self.run_case(missing=("available_at",))
        checked = result["outcomes"]["verify"]["data"]
        self.assertEqual(checked["reference_assessments"][0]["state"], "unknown")
        self.assertEqual(result["effective_strength"], "weak")

    def test_excluded_candidate_does_not_corrupt_valid_labels_or_counts(self):
        result = self.run_case(extra_case=True)
        checked = result["outcomes"]["verify"]["data"]
        self.assertEqual(checked["retrieval_received_candidate_count"], 4)
        self.assertEqual(checked["retrieval_candidate_count"], 3)
        self.assertEqual(result["support_level"], "high")
        self.assertEqual(checked["reference_assessments"][3]["state"], "failed")

    def test_report_recomputes_context_instead_of_accepting_stale_success(self):
        success = self.run_case()
        future_cutoff = {**CONTEXT, "decision_at": "2026-10-30T10:00:00+08:00"}
        out = report.generate_report(diagnosis_result=success["outcomes"]["diagnosis"],
                                    rag_result=success["outcomes"]["rag"],
                                    verify_data=success["outcomes"]["verify"],
                                    evidence_context=future_cutoff, write_file=False)
        self.assertEqual(out["data"]["effective_strength"], "weak")
        self.assertIn("unavailable_at_cutoff", out["data"]["markdown"])

    def test_sample_metadata_cannot_enable_or_override_top_level_context(self):
        transport = HistoricalTransport(changes={"units": "m/s2"})
        result = run_agent(synthetic_signal(), transport=transport, persist_trace=False,
                           skip_services_check=True, evidence_context=CONTEXT,
                           sample_info={"evidence_context": None})
        self.assertEqual(result["effective_strength"], "weak")

    def test_invalid_context_is_blocked_before_external_calls(self):
        transport = HistoricalTransport()
        result = DiagnosisAgent(transport=transport, persist_trace=False).run(
            synthetic_signal(), evidence_context={**CONTEXT, "sampling_rate": 64000})
        self.assertEqual(result["terminal_state"], "ASKING_USER")
        self.assertEqual(transport.calls, [])

    def test_case_identity_cannot_hide_conflicting_applicability_metadata(self):
        for field, value in (("sampling_rate", 64000), ("available_at", "2026-11-02T10:00:00+08:00"),
                             ("evidence_role", "current_observation")):
            with self.subTest(field=field):
                body = candidates()
                first = body["results"][0]
                first["case_id"] = "same-case"
                first["meta"].update(deepcopy(META))
                second = deepcopy(first)
                second["meta"][field] = value
                body["results"] = [first, second]
                result = rag.retrieve_cases(synthetic_signal(), transport=_fakes.ok(body))
                self.assertEqual(result["status"], "error")

    def test_cli_accepts_context_and_rejects_mismatch_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"input.json"
            path.write_text(json.dumps({"signal": synthetic_signal(),
                                        "evidence_context": {**CONTEXT, "sampling_rate": 64000}}))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = main(["run", str(path), "--no-write", "--skip-health"])
            self.assertEqual(code, 3)
            self.assertEqual(json.loads(output.getvalue())["terminal_state"], "ASKING_USER")

    def test_explicit_default_role_does_not_manufacture_another_candidate(self):
        first = candidates()["results"][0]
        first["meta"].update({k: v for k, v in META.items() if k != "evidence_role"})
        second = deepcopy(first)
        second["meta"]["evidence_role"] = "historical_reference"
        result = rag.retrieve_cases(synthetic_signal(), transport=_fakes.ok({"results": [first, second]}))
        self.assertEqual(result["data"]["num_results"], 1)

    def test_report_detects_known_signal_sampling_rate_mismatch(self):
        success = self.run_case()
        with self.assertRaises(ValueError):
            report.generate_report(signal_result=success["outcomes"]["signal"],
                                   evidence_context={**CONTEXT, "sampling_rate": 64000}, write_file=False)

    def test_reference_cli_demonstration_checks_eight_scenarios_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)/"must-not-be-created"
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = main(["demo-references", "--no-write", "--output-dir", str(output_dir)])
            result = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(result["matched_expectations"], 8)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
