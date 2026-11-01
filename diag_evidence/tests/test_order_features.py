"""Physical synthetic checks and diagnostic identities, without real downloads."""
from copy import deepcopy
import importlib.util
import unittest
import numpy as np

from diag_evidence.benchmarks.feature_diagnostics import linear_margin_audit, legacy_feature_names
from diag_evidence.benchmarks.order_features import (
    FS, constant_phase, window_starts_in_turns, encoder_phase, envelope_vector, paired_features)
from diag_evidence.benchmarks.order_study import SPEC


class DriftAuditTests(unittest.TestCase):
    def parameters(self):
        return {"mean": [0., 0.], "scale": [1., 1.],
                "coefficients": [[0., 0.], [0., 0.], [0., 0.], [2., 0.]], "intercept": [0.] * 4}

    def test_linear_margin_contributions_have_exact_additive_meaning(self):
        report = linear_margin_audit([[0., 0.], [1., 1.]], [[2., 0.]], self.parameters(), ["a", "b"])
        self.assertEqual(report["healthy_margin_median"], 4)
        self.assertEqual(report["features"][0]["healthy_margin_mean_contribution"], 4)
        self.assertTrue(report["attribution_additivity_checked"])

    def test_drift_intervals_use_source_only(self):
        p = self.parameters(); source = np.array([[0., 0.], [1., 1.]])
        before = deepcopy(p)
        a = linear_margin_audit(source, [[2., 0.]], p, ["a", "b"])
        b = linear_margin_audit(source, [[100., 0.]], p, ["a", "b"])
        self.assertEqual(a["features"][0]["source_z_p99"], b["features"][0]["source_z_p99"])
        self.assertEqual(p, before)

    def test_legacy_names_match_frozen_feature_count(self):
        self.assertEqual(len(legacy_feature_names()), 55)
        self.assertEqual(legacy_feature_names()[4], "zero_crossing")

    def test_empty_reference_or_invalid_scale_fails(self):
        with self.assertRaises(ValueError):
            linear_margin_audit(np.empty((0, 2)), [[1., 1.]], self.parameters(), ["a", "b"])
        with self.assertRaises(ValueError):
            linear_margin_audit([[1., 1.]], [[1., 1.]], {**self.parameters(), "scale": [0., 1.]}, ["a", "b"])


class OrderContractTests(unittest.TestCase):
    def test_constant_phase_uses_shaft_hz_not_sampling_rate(self):
        p = constant_phase(FS, 25.)
        self.assertAlmostEqual(p[-1], 25. - 25. / FS)

    def test_invalid_shaft_metadata_does_not_fall_back_to_nominal(self):
        for rate in (0, float("nan"), 200., -1):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                constant_phase(1000, rate)

    def test_eight_windows_do_not_overlap_and_keep_margins(self):
        phase = np.linspace(0, 100, 10001)
        starts = window_starts_in_turns(phase)
        self.assertEqual(len(starts), 8)
        self.assertTrue(np.all(np.diff(starts) >= 8))
        self.assertGreaterEqual(starts[0], 2)
        self.assertLessEqual(starts[-1] + 8, phase[-1] - 2)

    def test_short_or_reversing_phase_is_rejected(self):
        for phase in ([], [0, 60], [100, 0], [0, np.nan]):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                window_starts_in_turns(phase)

    def test_constant_envelope_has_finite_zero_modulation_features(self):
        x = np.ones(4096); axis = np.fft.rfftfreq(4096, d=1 / 512)
        features = envelope_vector(x, axis)
        self.assertEqual(len(features), 44)
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(features[:4], np.zeros(4))

    def test_even_constant_envelope_needs_a_valid_axis(self):
        with self.assertRaises(ValueError):
            envelope_vector(np.ones(4096), np.arange(10))

    def test_missing_or_multichannel_encoder_is_not_assumed_healthy(self):
        for x in (np.ones(200000), np.zeros((2, 200000)), np.ones(200000, dtype=complex)):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                encoder_phase(x, 51200)

    def test_known_encoder_pulses_recover_relative_rotations(self):
        t = np.arange(400000) / 200000
        pulse = (np.sin(2 * np.pi * 1024 * 40 * t) > 0).astype(float)
        phase, meta = encoder_phase(pulse, 102400)
        self.assertAlmostEqual(phase[-1] - phase[0], 80, delta=.01)
        self.assertAlmostEqual(meta["speed_min_hz"], 40, delta=.01)
        self.assertTrue(np.all(np.diff(phase) >= 0))

    def test_out_of_range_speed_and_length_mismatch_fail(self):
        t = np.arange(400000) / 200000
        fast = (np.sin(2 * np.pi * 1024 * 60 * t) > 0).astype(float)
        with self.assertRaises(ValueError): encoder_phase(fast, 102400)
        normal = (np.sin(2 * np.pi * 1024 * 40 * t) > 0).astype(float)
        with self.assertRaises(ValueError): encoder_phase(normal, 102401)

    def test_protocol_is_offline_and_lowpass_is_below_angular_nyquist(self):
        self.assertTrue(SPEC["offline_noncausal"])
        self.assertLess(SPEC["envelope_lowpass_hz"], min(SPEC["speed_range_hz"]) * SPEC["samples_per_turn"] / 2)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "optional scipy signal tests")
class OrderNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = np.arange(FS * 4) / FS
        cls.phase = 20 * cls.t + 2 * cls.t ** 2
        cls.signal = (1 + .5 * np.cos(2 * np.pi * 5 * cls.phase)) * np.cos(2 * np.pi * 5000 * cls.t)
        cls.a, cls.b, cls.crops, cls.windows = paired_features(cls.signal, cls.phase)

    def test_known_five_order_modulation_stays_in_the_same_band(self):
        peaks = self.a[:, 4:].argmax(axis=1)
        np.testing.assert_array_equal(peaks, np.full(8, 10))

    def test_nominal_time_control_shifts_when_speed_changes(self):
        self.assertGreater(np.ptp(self.b[:, 4:].argmax(axis=1)), 1)

    def test_same_windows_and_legacy_context_are_recorded(self):
        self.assertEqual(self.crops.shape, (8, 4096))
        for row in self.windows:
            self.assertAlmostEqual(row["end_turn"] - row["start_turn"], 8)
            self.assertGreater(row["duration_seconds"], .2)

    def test_envelope_modulation_depth_is_recovered(self):
        np.testing.assert_allclose(self.a[:, 0], .5 / np.sqrt(2), atol=.02)

    def test_positive_gain_and_dc_offset_do_not_change_features(self):
        a, _, _, _ = paired_features(3 * self.signal + 7, self.phase)
        np.testing.assert_allclose(a, self.a, atol=1e-5, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
