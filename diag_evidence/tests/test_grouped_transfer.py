"""Group isolation, robust selection, frozen inference and resampling checks."""
from copy import deepcopy
import importlib.util
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from diag_evidence.benchmarks.grouped_validation import (
    nested_type_plan, robust_choice, verify_partition, _quality, deepcopy_metrics_zero)
from diag_evidence.benchmarks.hust_health import classification_metrics
from diag_evidence.benchmarks.ottawa_external import (
    frozen_probabilities, resample_ottawa, ottawa_manifest)


def candidate(name, scores, alarms=None, order=0):
    return {"candidate": name, "candidate_order": order, "folds": [
        {"validation_type": group, "metrics": {"balanced_accuracy_all_outputs": score,
            "healthy_false_alarm_all_healthy": (alarms or [0.] * len(scores))[i]}}
        for i, (group, score) in enumerate(zip([5, 6, 7], scores))]}


class GroupProtocolTests(unittest.TestCase):
    def test_every_type_is_outer_test_once(self):
        groups = np.repeat([5, 6, 7, 8], 3)
        plan = nested_type_plan(groups)
        tests = [i for fold in plan for i in fold["test_indices"]]
        self.assertEqual(sorted(tests), list(range(12)))
        self.assertEqual(len(set(tests)), len(tests))

    def test_inner_training_and_validation_never_see_outer_type(self):
        groups = np.repeat([5, 6, 7, 8], 3)
        for outer in nested_type_plan(groups):
            for inner in outer["inner"]:
                self.assertEqual(len(inner["train_types"]), 2)
                self.assertNotIn(outer["test_type"], inner["train_types"])
                self.assertNotEqual(outer["test_type"], inner["validation_type"])
                self.assertFalse(set(inner["train_indices"]) & set(inner["validation_indices"]))
                self.assertFalse(set(inner["train_indices"]) & set(outer["test_indices"]))

    def test_incomplete_type_set_is_rejected(self):
        with self.assertRaises(ValueError):
            nested_type_plan([5, 6, 7])

    def test_worst_type_takes_priority_over_high_mean(self):
        uneven = candidate("high_mean", [.1, .99, .99])
        stable = candidate("higher_minimum", [.5, .5, .5], order=1)
        selected, _ = robust_choice([uneven, stable], [5, 6, 7])
        self.assertEqual(selected["candidate"], "higher_minimum")

    def test_mean_breaks_equal_minimum_tie(self):
        a, b = candidate("a", [.5, .5, .5]), candidate("b", [.5, .6, .7], order=1)
        self.assertEqual(robust_choice([a, b], [5, 6, 7])[0]["candidate"], "b")

    def test_healthy_alarm_then_fixed_order_break_ties(self):
        a = candidate("a", [.5] * 3, [.8] * 3)
        b = candidate("b", [.5] * 3, [.1] * 3, order=2)
        c = candidate("c", [.5] * 3, [.1] * 3, order=1)
        self.assertEqual(robust_choice([a, b, c], [5, 6, 7])[0]["candidate"], "c")

    def test_outer_test_metric_cannot_enter_selection(self):
        row = candidate("bad", [.5] * 3)
        row["folds"][-1]["validation_type"] = 8
        with self.assertRaises(ValueError):
            robust_choice([row], [5, 6, 7])

    def test_duplicate_validation_type_is_rejected(self):
        row = candidate("bad", [.5] * 3)
        row["folds"][-1]["validation_type"] = 6
        with self.assertRaises(ValueError):
            robust_choice([row], [5, 6, 7])

    def test_same_record_in_different_roles_is_rejected(self):
        rows = [{"record_id": "same", "bearing_type": 5}, {"record_id": "same", "bearing_type": 6}]
        with self.assertRaises(ValueError):
            verify_partition(rows, [0], [1])

    def test_same_type_different_records_cannot_cross_roles(self):
        rows = [{"record_id": "a", "bearing_type": 5}, {"record_id": "b", "bearing_type": 5}]
        with self.assertRaises(ValueError):
            verify_partition(rows, [0], [1])

    def test_quality_screen_requires_every_type_and_class(self):
        good = {"score_0_5": {"accepted_error_rate": .1, "class_coverage": {"0": .5, "3": .5}}}
        self.assertTrue(_quality([good, good]))
        bad = deepcopy(good)
        bad["score_0_5"]["class_coverage"]["3"] = .1
        self.assertFalse(_quality([good, bad]))

    def test_quality_screen_rejects_undefined_zero_coverage_error(self):
        self.assertFalse(_quality([{"score_0_5": {"accepted_error_rate": None, "class_coverage": {"0": 0}}}]))

    def test_zero_guard_preserves_raw_scores_without_claiming_zero_risk(self):
        raw = classification_metrics([0, 1, 2, 3], [0, 1, 2, 0])
        before = deepcopy(raw)
        guard = deepcopy_metrics_zero(raw)
        self.assertEqual(raw, before)
        self.assertEqual(guard["accuracy_all_outputs"], .75)
        self.assertEqual(guard["coverage"], 0)
        self.assertIsNone(guard["accepted_error_rate"])
        self.assertIsNone(guard["healthy_false_alarm_among_accepted"])


class FrozenTransferTests(unittest.TestCase):
    def linear(self):
        return {"candidate": "shape/logistic_C1", "mean": [1., 2.], "scale": [2., 4.],
                "coefficients": [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], "intercept": [0.] * 4}

    def test_frozen_linear_softmax_matches_analytic_result(self):
        p = frozen_probabilities(self.linear(), [3., 2.])[0]
        expected = np.exp([1., 0., -1., 0.]); expected /= expected.sum()
        np.testing.assert_allclose(p, expected)

    def test_target_prediction_does_not_modify_source_scaling(self):
        parameters = self.linear(); before = deepcopy(parameters)
        frozen_probabilities(parameters, [[1., 2.], [1e3, -1e3]])
        self.assertEqual(parameters, before)

    def test_unknown_amplitude_transfer_is_not_accepted(self):
        parameters = self.linear(); parameters["candidate"] = "shape_amplitude/logistic_C1"
        with self.assertRaises(ValueError):
            frozen_probabilities(parameters, [1., 2.])

    def test_invalid_scale_or_weight_dimensions_fail(self):
        for key, value in [("scale", [0., 1.]), ("coefficients", [[1., 2.]]), ("intercept", [float("nan")] * 4)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                frozen_probabilities({**self.linear(), key: value}, [1., 2.])

    def test_centroid_probabilities_use_four_declared_centers(self):
        parameters = {"candidate": "shape/centroid", "mean": [0.], "scale": [1.], "centers": [[0.], [1.], [2.], [3.]]}
        p = frozen_probabilities(parameters, [0.])[0]
        expected = np.exp([0., -1., -2., -3.]); expected /= expected.sum()
        np.testing.assert_allclose(p, expected)

    def leaf(self, probabilities):
        return {"left": [-1], "right": [-1], "feature": [-2], "threshold": [-2.], "value": [[probabilities]]}

    def test_forest_leaf_counts_are_normalized_before_averaging(self):
        parameters = {"candidate": "shape/forest_256_leaf5", "mean": [0.], "scale": [1.],
                      "trees": [self.leaf([2., 3., 0., 0.]), self.leaf([1., 0., 0., 0.])]}
        np.testing.assert_allclose(frozen_probabilities(parameters, [0.])[0], [.7, .3, 0., 0.])

    def test_cyclic_frozen_tree_fails_instead_of_looping(self):
        tree = {"left": [0], "right": [0], "feature": [0], "threshold": [0.], "value": [[[1., 0., 0., 0.]]]}
        parameters = {"candidate": "shape/forest_256_leaf5", "mean": [0.], "scale": [1.], "trees": [tree]}
        with self.assertRaises(ValueError):
            frozen_probabilities(parameters, [0.])

    def test_frozen_tree_uses_library_float32_branch_inputs(self):
        tree = {"left": [1, -1, -1], "right": [2, -1, -1], "feature": [0, -2, -2],
                "threshold": [.1, -2., -2.], "value": [[[1., 1., 0., 0.]],
                    [[1., 0., 0., 0.]], [[0., 1., 0., 0.]]]}
        parameters = {"candidate": "shape/forest_256_leaf5", "mean": [0.], "scale": [1.], "trees": [tree]}
        # float32(0.1) exceeds the float64 threshold 0.1, matching sklearn's cast.
        np.testing.assert_allclose(frozen_probabilities(parameters, [.1])[0], [0., 1., 0., 0.])

    def test_external_manifest_keeps_all_trials_and_four_known_classes(self):
        rows = ottawa_manifest()["files"]
        self.assertEqual(len(rows), 15)
        for label in [0, 1, 2, 3, None]:
            self.assertEqual({r["trial"] for r in rows if r["label"] == label}, {1, 2, 3})
        self.assertTrue(all(r["speed_profile"] == "A" for r in rows))

    def test_resampler_uses_declared_ratio_and_antialias_filter(self):
        signal = ModuleType("scipy.signal")
        calls = []
        def run(x, **kwargs):
            calls.append((x.dtype, kwargs)); return np.array([1.])
        signal.resample_poly = run
        with patch.dict(sys.modules, {"scipy": ModuleType("scipy"), "scipy.signal": signal}):
            resample_ottawa(np.ones((200000, 1), dtype=np.int16))
        self.assertEqual(calls[0][0], np.dtype(float))
        self.assertEqual(calls[0][1], {"up": 32, "down": 125, "window": ("kaiser", 5.0), "padtype": "constant"})

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "optional scipy numeric check")
    def test_resampling_preserves_low_frequency_and_suppresses_alias(self):
        t = np.arange(200000) / 200000
        low = resample_ottawa(np.sin(2 * np.pi * 2000 * t))[5120:-5120]
        high = resample_ottawa(np.sin(2 * np.pi * 60000 * t))[5120:-5120]
        self.assertAlmostEqual(float(np.sqrt(np.mean(low * low))), 2 ** -.5, delta=.01)
        self.assertLess(float(np.sqrt(np.mean(high * high))), .005)


if __name__ == "__main__":
    unittest.main()
