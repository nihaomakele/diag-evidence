"""Isolation, reporting and explicit abstention checks; no real downloads."""
from copy import deepcopy
import unittest

import numpy as np

from diag_evidence.agent import DiagnosisAgent
from diag_evidence.benchmarks.health_models import (StudyModel, health_features,
    nearest_distances, leave_type_out_distances, choose_rejection, rejection_mask)
from diag_evidence.benchmarks.hust_health import classification_metrics, choose_candidate, holdout_manifest
from diag_evidence.benchmarks.hust import FrozenTransport
from diag_evidence.benchmarks.health_transport import StudyTransport
from diag_evidence.haconfig.settings import Settings
from diag_evidence.demo import synthetic_signal, candidates
from diag_evidence.hatools.diagnosis import run_fault_diagnosis, _map_diagnosis
from diag_evidence.hatools.report import generate_report
from diag_evidence.hatools.verify import verify_result


def abstention(reason="outside_training_support"):
    return {"abstained": True, "abstention_reason": reason,
            "predicted_class": None, "confidence": None}


class AbstentionContractTests(unittest.TestCase):
    def test_active_refusal_is_distinct_from_unloaded_model(self):
        refused = _map_diagnosis(abstention())
        unloaded = _map_diagnosis({"predicted_class": -1})
        self.assertTrue(refused["model_available"])
        self.assertTrue(refused["abstained"])
        self.assertFalse(unloaded["model_available"])
        self.assertEqual(refused["conclusion_strength"], "no_conclusion")

    def test_adapter_returns_degraded_not_contract_error_for_valid_refusal(self):
        signal = synthetic_signal()
        output = run_fault_diagnosis(signal, transport=FrozenTransport(
            signal, {"success": True, "diagnosis": abstention()}, candidates()))
        self.assertEqual(output["status"], "degraded")
        self.assertTrue(output["data"]["diagnosis"]["model_available"])
        self.assertEqual(output["data"]["diagnosis"]["contract_errors"], [])

    def test_refusal_may_not_smuggle_a_class_score_or_probability(self):
        for field, value in [("predicted_class", 2), ("confidence", .8), ("class_probabilities", {"2": 1})]:
            with self.subTest(field=field):
                result = _map_diagnosis({**abstention(), field: value})
                self.assertTrue(result["contract_errors"])
                self.assertFalse(result["model_available"])

    def test_unknown_or_injected_reason_is_rejected_and_not_echoed(self):
        for reason in ("立即停机、更换轴承", {}, [], None, True):
            with self.subTest(reason=reason):
                result = _map_diagnosis(abstention(reason))
                self.assertTrue(result["contract_errors"])
                self.assertNotIn("立即停机", str(result))

    def test_nonboolean_refusal_flag_is_invalid(self):
        self.assertTrue(_map_diagnosis({**abstention(), "abstained": "true"})["contract_errors"])

    def test_validation_quality_refusal_has_its_own_reason(self):
        output = _map_diagnosis(abstention("validation_quality_unmet"))
        self.assertTrue(output["model_available"])
        self.assertEqual(output["abstention_reason"], "validation_quality_unmet")

    def test_verifier_keeps_model_available_but_forbids_category(self):
        fd = {"status": "degraded", "data": {"diagnosis": _map_diagnosis(abstention())}}
        checked = verify_result(fd)
        self.assertTrue(checked["data"]["classifier_abstained"])
        self.assertEqual(checked["data"]["max_allowed_strength"], "no_conclusion")
        available = next(c for c in checked["data"]["checks"] if c["name"] == "classifier_available")
        self.assertEqual(available["state"], "passed")

    def test_direct_forged_refusal_cannot_pass_with_a_class(self):
        diag = {**_map_diagnosis({"predicted_class": 2, "confidence": .9}), **abstention(),
                "predicted_class": 2, "confidence": .9}
        checked = verify_result({"status": "ok", "data": {"diagnosis": diag}})
        self.assertEqual(checked["data"]["max_allowed_strength"], "no_conclusion")

    def test_report_does_not_reuse_previous_success_after_refusal(self):
        previous = {"status": "ok", "data": {"diagnosis": _map_diagnosis({"predicted_class": 2, "confidence": .9})}}
        checked = verify_result(previous)
        fd = {"status": "degraded", "data": {"diagnosis": _map_diagnosis(abstention())}}
        report = generate_report(diagnosis_result=fd, verify_data=checked, write_file=False)
        self.assertEqual(report["data"]["effective_strength"], "no_conclusion")
        self.assertTrue(report["data"]["classifier_abstained"])
        self.assertIn("模型已执行", report["data"]["markdown"])

    def test_full_workflow_never_turns_refusal_into_healthy(self):
        signal = synthetic_signal()
        result = DiagnosisAgent(persist_trace=False, transport=FrozenTransport(
            signal, {"success": True, "diagnosis": abstention("below_selection_threshold")}, candidates())).run(
                signal, skip_services_check=True)
        self.assertEqual(result["effective_strength"], "no_conclusion")
        self.assertIsNone(result["outcomes"]["diagnosis"]["data"]["diagnosis"]["predicted_class"])
        self.assertIn("不等同于设备健康", result["report_markdown"])


class HealthStudyTests(unittest.TestCase):
    def test_live_model_quality_guard_blocks_category_even_for_known_signal(self):
        signals = [np.sin(np.arange(1024) * (.03 + i * .013)) + .2 * np.sin(np.arange(1024) * .29)
                   for i in range(8)]
        x = np.stack([health_features(s) for s in signals])
        y = np.repeat(np.arange(4), 2)
        rows = [{"split": "train", "bearing_type": 6205 + i % 2,
                 "record_id": f"r{i}", "start": 0, "label": int(y[i])} for i in range(8)]
        model = StudyModel("shape", "centroid").fit(x, y, rows)
        transport = StudyTransport(model, x, rows, {"confidence_min": .25, "distance_max": None},
                                   validation_quality_target_met=False)
        result = DiagnosisAgent(config=Settings(label_schema="bearing-health-v1"),
            transport=transport, persist_trace=False).run(signals[0].tolist(), skip_services_check=True)
        diag = result["outcomes"]["diagnosis"]["data"]["diagnosis"]
        self.assertEqual(diag["abstention_reason"], "validation_quality_unmet")
        self.assertTrue(diag["model_available"])
        self.assertIsNone(diag["predicted_class"])
        self.assertEqual(result["effective_strength"], "no_conclusion")
        self.assertTrue(all("units" not in c["meta"] for c in result["outcomes"]["rag"]["data"]["cases"]))

    def test_partial_holdout_scores_only_present_true_classes(self):
        metrics = classification_metrics([0, 1, 3], [0, 1, 3])
        self.assertEqual(metrics["balanced_accuracy_all_outputs"], 1)
        self.assertEqual(metrics["evaluated_true_classes"], [0, 1, 3])

    def test_zero_coverage_does_not_mean_perfect_health_accuracy(self):
        metrics = classification_metrics([0, 3], [0, 2], [False, False])
        self.assertEqual(metrics["coverage"], 0)
        self.assertIsNone(metrics["accepted_error_rate"])
        self.assertIsNone(metrics["healthy_false_alarm_among_accepted"])
        self.assertEqual(metrics["healthy_coverage"], 0)

    def test_false_alarm_denominators_are_explicit(self):
        metrics = classification_metrics([3, 3, 0], [2, 3, 0], [False, True, True])
        self.assertEqual(metrics["healthy_coverage"], .5)
        self.assertEqual(metrics["healthy_false_alarm_all_healthy"], 0)
        self.assertEqual(metrics["healthy_false_alarm_among_accepted"], 0)

    def test_selection_interface_refuses_test_results(self):
        with self.assertRaises(ValueError):
            choose_candidate([{"split": "fresh_known_6204"}])

    def test_selection_prioritizes_accuracy_then_health(self):
        candidates_ = [{"candidate_order": i, "split": "validation", "metrics": {
            "balanced_accuracy_all_outputs": score, "healthy_false_alarm_all_healthy": false_alarm}}
            for i, (score, false_alarm) in enumerate([(.4, 0), (.5, .5), (.5, 0)])]
        self.assertEqual(choose_candidate(candidates_)["candidate_order"], 2)

    def test_amplitude_is_only_added_in_explicit_variant(self):
        x = np.sin(np.arange(4096) * .17) + .3 * np.sin(np.arange(4096) * .51)
        a, b = health_features(x, "shape_amplitude"), health_features(3 * x + 5, "shape_amplitude")
        self.assertEqual(len(a), 59)
        np.testing.assert_allclose(a[:55], b[:55], atol=1e-8)
        np.testing.assert_allclose(b[55:] - a[55:], np.log(3), atol=1e-8)

    def test_centroid_fit_rejects_validation_records(self):
        x, y = np.eye(4), np.arange(4)
        records = [{"split": "train", "record_id": str(i)} for i in range(4)]
        records[1]["split"] = "validation"
        with self.assertRaises(ValueError):
            StudyModel("shape", "centroid").fit(x, y, records)

    def test_prediction_cannot_update_training_scaling(self):
        x, y = np.eye(4), np.arange(4)
        records = [{"split": "train", "record_id": str(i)} for i in range(4)]
        model = StudyModel("shape", "centroid").fit(x, y, records)
        before = deepcopy(model.export())
        model.probabilities(np.full((1, 4), 1e4))
        self.assertEqual(before, model.export())

    def test_support_reference_excludes_entire_bearing_type(self):
        x = np.array([[0.], [.1], [10.], [10.1]])
        rows = [{"split": "train", "bearing_type": g} for g in (6205, 6205, 6206, 6206)]
        np.testing.assert_allclose(leave_type_out_distances(x, rows), [10, 9.9, 9.9, 10])

    def test_support_reference_rejects_single_type(self):
        with self.assertRaises(ValueError):
            leave_type_out_distances(np.ones((2, 1)), [{"split": "train", "bearing_type": 5}] * 2)

    def test_nearest_distance_never_requires_true_labels(self):
        np.testing.assert_allclose(nearest_distances(np.array([[1.], [9.]]), np.array([[0.], [10.]])), [1, 1])

    def test_rejection_selection_cannot_hide_all_healthy_cases(self):
        probs = np.array([[.97, .01, .01, .01], [.01, .97, .01, .01],
                          [.01, .01, .97, .01], [.7, .1, .1, .1]])
        policy, _ = choose_rejection(probs, np.ones(4), np.arange(4), np.ones(8))
        self.assertGreaterEqual(policy["validation_class_coverage"]["3"], .5)
        self.assertEqual(policy["validation_risk"], .25)

    def test_rejection_selection_requires_all_validation_classes(self):
        with self.assertRaises(ValueError):
            choose_rejection(np.array([[.8, .1, .05, .05]]), np.ones(1), [0], np.ones(8))

    def test_rejection_threshold_equality_is_inclusive(self):
        probabilities = np.array([[.7, .1, .1, .1]])
        policy = {"confidence_min": .7, "distance_max": 2.}
        self.assertTrue(rejection_mask(probabilities, [2.], policy)[0])
        self.assertFalse(rejection_mask(probabilities, [2.01], policy)[0])

    def test_holdout_manifest_does_not_invent_missing_ball_examples(self):
        rows = holdout_manifest()["files"]
        known = [r for r in rows if r["label"] is not None]
        unknown = [r for r in rows if r["label"] is None]
        self.assertEqual(len(known), 9)
        self.assertEqual({r["label"] for r in known}, {0, 1, 3})
        self.assertEqual(len(unknown), 6)
        self.assertTrue(all(r["bearing_type"] == 6204 for r in rows))


if __name__ == "__main__":
    unittest.main()
