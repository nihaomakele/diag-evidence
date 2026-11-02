"""Small fixtures test isolation and contracts; real-data scores are separate."""
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from diag_evidence.agent import DiagnosisAgent
from diag_evidence.benchmarks.baseline import (CentroidClassifier, TrainingIndex,
    extract_features, normalize_window)
from diag_evidence.benchmarks.hust import (CONTEXT, SCENARIOS, FrozenTransport,
    classifier_metrics, confidence_mask_at_count, reference_replay, selective_metrics)
from diag_evidence.benchmarks.hust_data import manifest, window_starts, load_windows, verify_file
from diag_evidence.demo import synthetic_signal
from diag_evidence.haconfig.settings import Settings, class_labels
from diag_evidence.hatools.diagnosis import _map_diagnosis
from diag_evidence.hatools.report import generate_report
from diag_evidence.hatools.verify import verify_result
from diag_evidence.reference_policy import assess_references


def healthy_body():
    return {"success": True, "diagnosis": {"predicted_class": 3, "confidence": 0.7,
            "predicted_fault": class_labels("bearing-health-v1")[3],
            "class_probabilities": {"0": 0.1, "1": 0.1, "2": 0.1, "3": 0.7}}}


def history():
    return {"num_results": 5, "results": [{"case_id": f"N500.mat:{i * 4096}",
        "text": f"Training fixture {i}", "score": -float(i), "score_is_normalized": False,
        "meta": {"record_id": f"record-{i}", "source_id": "fixture", "fault_type": "健康",
                 "units": "1", "sampling_rate": 51200,
                 "preprocessing_id": CONTEXT["preprocessing_id"]}} for i in range(5)]}


class OntologyTests(unittest.TestCase):
    def test_default_rejects_health(self):
        self.assertFalse(_map_diagnosis(healthy_body()["diagnosis"])["model_available"])

    def test_explicit_health_schema_accepts_four_complete_probabilities(self):
        result = _map_diagnosis(healthy_body()["diagnosis"], "bearing-health-v1")
        self.assertTrue(result["model_available"])
        self.assertEqual(result["predicted_class"], 3)
        self.assertEqual(result["label_schema"], "bearing-health-v1")

    def test_health_schema_rejects_missing_probability_class(self):
        body = healthy_body()["diagnosis"]
        body["class_probabilities"].pop("2")
        self.assertFalse(_map_diagnosis(body, "bearing-health-v1")["model_available"])

    def test_health_schema_rejects_probability_confidence_disagreement(self):
        body = healthy_body()["diagnosis"]
        body["confidence"] = 0.9
        self.assertFalse(_map_diagnosis(body, "bearing-health-v1")["model_available"])

    def test_unknown_is_not_healthy(self):
        body = healthy_body()["diagnosis"]
        body["predicted_class"] = 4
        self.assertFalse(_map_diagnosis(body, "bearing-health-v1")["model_available"])

    def test_schema_configuration_is_explicit_and_validated(self):
        cfg = Settings.from_env({"DIAG_EVIDENCE_LABEL_SCHEMA": "bearing-health-v1"})
        self.assertEqual(cfg.describe()["label_schema"], "bearing-health-v1")
        for bad in ("other", None, [], True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Settings(label_schema=bad)

    def test_returned_labels_cannot_mutate_default(self):
        labels = class_labels("bearing-health-v1")
        labels[0]["name"] = "bad"
        self.assertEqual(class_labels()[0]["name"], "内圈故障")

    def test_full_agent_accepts_healthy_only_with_explicit_config(self):
        signal = synthetic_signal()
        cfg = Settings(label_schema="bearing-health-v1")
        result = DiagnosisAgent(config=cfg, persist_trace=False,
            transport=FrozenTransport(signal, healthy_body(), history())).run(signal,
                skip_services_check=True, sample_info={"label_schema": "faults-v1"})
        self.assertEqual(result["effective_strength"], "moderate")
        self.assertIn("健康为模型类别", result["report_markdown"])
        denied = DiagnosisAgent(persist_trace=False,
            transport=FrozenTransport(signal, healthy_body(), history())).run(signal,
                skip_services_check=True, sample_info={"label_schema": "bearing-health-v1"})
        self.assertEqual(denied["effective_strength"], "no_conclusion")

    def test_report_rechecks_schema_instead_of_trusting_supplied_success(self):
        diagnosis = _map_diagnosis(healthy_body()["diagnosis"], "bearing-health-v1")
        fd = {"status": "ok", "data": {"diagnosis": diagnosis}}
        cases = [{**case, "fault_type": case["meta"]["fault_type"]} for case in history()["results"]]
        rag = {"status": "ok", "data": {"cases": cases, "num_results": 5}}
        verified = verify_result(fd, rag, label_schema="bearing-health-v1")
        self.assertEqual(verified["data"]["support_level"], "high")
        report = generate_report(diagnosis_result=fd, rag_result=rag, verify_data=verified, write_file=False)
        self.assertEqual(report["data"]["effective_strength"], "no_conclusion")

    def test_verifier_rejects_declared_schema_mismatch_on_shared_class(self):
        diagnosis = _map_diagnosis({"predicted_class": 0, "confidence": 0.8})
        fd = {"status": "ok", "data": {"diagnosis": diagnosis}}
        result = verify_result(fd, label_schema="bearing-health-v1")
        self.assertEqual(result["data"]["max_allowed_strength"], "no_conclusion")


class DataIsolationTests(unittest.TestCase):
    def test_modified_cache_is_rejected_even_at_same_file_size(self):
        data = b"original-data"
        row = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.mat"
            path.write_bytes(data)
            verify_file(path, row)
            path.write_bytes(b"modified-data")
            with self.assertRaises(ValueError):
                verify_file(path, row)

    def test_pinned_manifest_groups_and_four_classes(self):
        data = manifest()
        self.assertEqual(data["license"], "CC-BY-4.0")
        self.assertIsNone(data["acquisition_timestamps"])
        self.assertEqual(len(data["files"]), 48)
        groups = {}
        for row in data["files"]:
            groups.setdefault(row["split"], set()).add(row["bearing_type"])
            self.assertEqual(len(row["sha256"]), 64)
            self.assertTrue(row["url"].startswith("https://data.mendeley.com/public-files/"))
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["train"] & groups["validation"])
        for split in groups:
            self.assertEqual({r["label"] for r in data["files"] if r["split"] == split}, {0, 1, 2, 3})

    def test_windows_cover_original_record_without_overlap(self):
        starts = window_starts(512000)
        self.assertEqual(len(starts), 32)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1] + 4096, 512000)
        self.assertTrue(np.all(np.diff(starts) >= 4096))

    def test_insufficient_record_is_rejected(self):
        with self.assertRaises(ValueError):
            window_starts(4096, count=32)

    def test_loader_ignores_shaft_frequency_as_sampling_rate(self):
        io = ModuleType("scipy.io")
        raw = np.sin(np.arange(512000) * 0.03).reshape(-1, 1)
        calls = []
        def read(path, **kwargs):
            calls.append(kwargs)
            return {"data": raw, "fs": np.array([[24.93]])}
        io.loadmat = read
        row = next(r for r in manifest()["files"] if r["filename"] == "I500.mat")
        with patch.dict(sys.modules, {"scipy": ModuleType("scipy"), "scipy.io": io}), \
                patch("diag_evidence.benchmarks.hust_data.ensure_data", return_value=[row]):
            signals, rows = load_windows("unused")
        self.assertEqual(calls, [{"variable_names": ["data"]}])
        self.assertEqual(signals.shape, (32, 4096))
        self.assertTrue(all(r["split"] == "train" for r in rows))

    def test_features_are_invariant_to_offset_and_positive_scale(self):
        x = np.sin(np.arange(4096) * 0.2) + 0.2 * np.cos(np.arange(4096) * 0.73)
        np.testing.assert_allclose(extract_features(x), extract_features(3 * x + 9), atol=1e-8)

    def test_short_real_record_uses_actual_length_without_padding(self):
        io = ModuleType("scipy.io")
        io.loadmat = lambda *args, **kwargs: {"data": np.sin(np.arange(209000) * .03).reshape(-1, 1)}
        row = next(r for r in manifest()["files"] if r["filename"] == "B804.mat")
        with patch.dict(sys.modules, {"scipy": ModuleType("scipy"), "scipy.io": io}), \
                patch("diag_evidence.benchmarks.hust_data.ensure_data", return_value=[row]):
            signals, rows = load_windows("unused")
        self.assertEqual(signals.shape, (32, 4096))
        self.assertEqual(rows[-1]["start"] + 4096, 209000)
        self.assertEqual(rows[0]["record_samples"], 209000)

    def test_invalid_waveforms_are_rejected(self):
        for x in (np.zeros(64), np.ones(64), np.full(64, np.nan), np.zeros((64, 1))):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                normalize_window(x)

    def fitted(self):
        x = np.random.default_rng(20261101).normal(size=(8, 5))
        y = np.repeat(np.arange(4), 2)
        records = [{"split": "train", "record_id": f"r{i}", "start": 0, "label": int(y[i])} for i in range(8)]
        return x, y, records, CentroidClassifier().fit(x, y, records)

    def test_fitting_rejects_nontraining_records(self):
        x, y, rows, _ = self.fitted()
        rows[-1]["split"] = "test"
        with self.assertRaises(ValueError):
            CentroidClassifier().fit(x, y, rows)

    def test_index_rejects_nontraining_records(self):
        x, _, rows, model = self.fitted()
        rows[-1]["split"] = "validation"
        with self.assertRaises(ValueError):
            TrainingIndex(model, x, rows)

    def test_scaling_is_fitted_from_training_only(self):
        x, _, _, model = self.fitted()
        np.testing.assert_allclose(model.mean, x.mean(axis=0))
        before = model.export()
        model.predict(np.full(5, 1e6))
        self.assertEqual(before, model.export())

    def test_retrieval_caps_each_original_record_to_one_window(self):
        x, _, rows, model = self.fitted()
        duplicate_rows = [dict(row, start=start) for row in rows for start in (0, 4096)]
        index = TrainingIndex(model, np.repeat(x, 2, axis=0), duplicate_rows)
        cases = index.search(x[0])["results"]
        self.assertEqual(len(cases), 5)
        ids = [c["meta"]["record_id"] for c in cases]
        self.assertEqual(len(set(ids)), 5)
        self.assertTrue(set(ids).issubset(model.training_records))


class EvaluationTests(unittest.TestCase):
    def test_zero_coverage_error_is_undefined_not_zero(self):
        result = selective_metrics([0, 1], [0, 0], [False, False])
        self.assertEqual(result["coverage"], 0)
        self.assertIsNone(result["selective_error_rate"])

    def test_matched_coverage_uses_scores_without_labels(self):
        mask = confidence_mask_at_count([0.5, 0.9, 0.8], ["a", "b", "c"], 2)
        self.assertEqual(mask.tolist(), [False, True, True])

    def test_ties_are_stable_at_exact_count(self):
        a = confidence_mask_at_count([0.5] * 4, list("abcd"), 2)
        b = confidence_mask_at_count([0.5] * 4, list("abcd"), 2)
        self.assertEqual(int(a.sum()), 2)
        np.testing.assert_array_equal(a, b)

    def test_healthy_errors_are_reported_separately(self):
        result = classifier_metrics([0, 1, 2, 3], [0, 3, 2, 0])
        self.assertEqual(result["healthy_false_alarm_rate"], 1.0)
        self.assertAlmostEqual(result["fault_predicted_healthy_rate"], 1 / 3)

    def test_native_data_does_not_gain_invented_timestamps(self):
        raw = history()
        output = reference_replay(raw, "native_missing_time")
        self.assertEqual(raw, output)
        self.assertEqual(assess_references(output["results"], CONTEXT)["eligible_indices"], [])
        self.assertNotIn("available_at", raw["results"][0]["meta"])

    def test_controlled_replay_faults_do_not_change_original_tools(self):
        raw, expected = history(), history()
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                output = reference_replay(raw, scenario)
                eligible = assess_references(output["results"], CONTEXT)["eligible_indices"]
                count = 5 if scenario == "replay_compatible" else (4 if scenario == "replay_one_excluded" else 0)
                self.assertEqual(len(eligible), count)
        self.assertEqual(raw, expected)

    def test_frozen_tools_refuse_a_different_query(self):
        transport = FrozenTransport([1, 2], healthy_body(), history())
        with self.assertRaises(ValueError):
            transport("POST", "http://127.0.0.1/diagnose", payload={"signal": [1, 3]})


if __name__ == "__main__":
    unittest.main()
