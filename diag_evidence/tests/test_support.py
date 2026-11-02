import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from diag_evidence.benchmarks import support, hit_data
from diag_evidence.benchmarks.support_study import summarize


def records(names, split='train'):
    return [{'record_id': name, 'split': split} for name in names]


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0] * 55, [2] * 55, [10] * 55, [12] * 55], dtype=float)
        self.rows = records(['a', 'a', 'b', 'b'])
        self.state = support.fit_support(self.x, self.rows)

    def test_original_record_median_prototypes(self):
        self.assertEqual(self.state['prototype_records'], ['a', 'b'])
        np.testing.assert_array_equal(self.state['prototypes'], [[1] * 55, [11] * 55])

    def test_source_scaling_and_query_do_not_mutate(self):
        before = copy.deepcopy(self.state)
        support.support_scores(np.full((1, 55), 100), self.state)
        self.assertEqual(before, self.state)
        np.testing.assert_array_equal(self.state['mean'], self.x.mean(axis=0))

    def test_tie_uses_sorted_original_record(self):
        _, nearest = support.support_scores(np.full((1, 55), 6.), self.state)
        self.assertEqual(nearest, ['a'])

    def test_malformed_features_rejected(self):
        for x in [np.zeros((0, 55)), np.zeros((2, 54)), np.full((2, 55), np.nan),
                  np.zeros((2, 55), complex), np.zeros((2, 55), bool)]:
            with self.subTest(shape=x.shape, dtype=x.dtype), self.assertRaises(ValueError):
                support.fit_support(x, records(['a'] * len(x)))

    def test_training_roles_enforced(self):
        with self.assertRaises(ValueError):
            support.fit_support(self.x, records(['a'] * 4, 'external'))

    def test_query_dimensions_checked(self):
        with self.assertRaises(ValueError):
            support.support_scores(np.zeros((2, 54)), self.state)

    def test_frozen_parameters_checked(self):
        for key, value in [('scale', [0] * 55), ('mean', [0] * 54),
                           ('format', 'unknown'), ('prototype_records', ['a', 'a'])]:
            bad = {**self.state, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                support.support_scores(self.x, bad)

    def test_query_overflow_rejected(self):
        with self.assertRaises(ValueError):
            support.support_scores(np.full((1, 55), 1e308), self.state)

    def test_fit_overflow_rejected(self):
        with self.assertRaises(ValueError):
            support.fit_support(np.full((2, 55), 1e308), records(['a', 'b']))

    def test_calibration_uses_record_maxima(self):
        threshold = support.calibrate_record_threshold([0, 1, 2, 3, 100],
            records(['a', 'a', 'b', 'c', 'd'], 'validation'), alpha=.25)
        self.assertEqual(threshold['threshold'], 3)
        self.assertEqual(threshold['rank_1based'], 3)
        self.assertEqual(len(threshold['calibration_records']), 4)

    def test_calibration_roles_and_parameters(self):
        for alpha in [0, 1, True, float('nan')]:
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                support.calibrate_record_threshold([1], records(['a'], 'validation'), alpha)
        with self.assertRaises(ValueError):
            support.calibrate_record_threshold([1], records(['a']))

    def test_fixed_acceptance_and_boundary(self):
        p = np.array([[.5, .3, .1, .1], [.9, .05, .04, .01], [.4, .3, .2, .1]])
        masks = support.acceptance_masks(p, [5, 6, 4], 5)
        np.testing.assert_array_equal(masks['score_only'], [True, True, False])
        np.testing.assert_array_equal(masks['support_only'], [True, False, True])
        np.testing.assert_array_equal(masks['both'], [True, False, False])

    def test_invalid_probability_and_distance(self):
        for p, d, t in [([[1, 1, 0, 0]], [1], 5), ([[1, 0, 0, 0]], [-1], 5),
                         ([[1, 0, 0, 0]], [1], float('nan'))]:
            with self.subTest(p=p, d=d), self.assertRaises(ValueError):
                support.acceptance_masks(p, d, t)

    def test_json_roundtrip_preserves_scores(self):
        state = json.loads(json.dumps(self.state))
        np.testing.assert_array_equal(support.support_scores(self.x, state)[0],
                                      support.support_scores(self.x, self.state)[0])

    def test_record_aggregation_and_zero_acceptance(self):
        rows = [{'record_id': 'a', 'label': 0}, {'record_id': 'a', 'label': 0}]
        p = np.array([[.4, .3, .2, .1], [.9, .05, .04, .01]])
        result = summarize(rows, p, np.array([1., 6.]), 5)
        self.assertEqual(result['arms']['both']['windows']['accepted'], 0)
        self.assertIsNone(result['arms']['both']['records']['accepted_error_rate'])
        self.assertFalse(result['per_record'][0]['record_accepted']['support_only'])


class HITDataTests(unittest.TestCase):
    def test_vector_orientation_and_scalar_metadata(self):
        for shape in [(4096,), (1, 4096), (4096, 1)]:
            name, x = hit_data.vibration_vector({'__header__': b'ignored', 'fs': 51200,
                                                 'signal': np.arange(4096).reshape(shape)})
            self.assertEqual(name, 'signal'); self.assertEqual(x.shape, (4096,))

    def test_ambiguous_and_multichannel_rejected(self):
        for data in [{'x': np.zeros((4096, 2))}, {'x': np.zeros(4096), 'y': np.ones(4096)}]:
            with self.assertRaises(ValueError):
                hit_data.vibration_vector(data)

    def test_complex_short_and_nonfinite_rejected(self):
        for x in [np.ones(4096, complex), np.ones(20), np.full(4096, np.inf)]:
            with self.assertRaises(ValueError):
                hit_data.vibration_vector({'x': x})

    def test_manifest_covers_both_rigs_and_three_true_classes(self):
        data = hit_data.manifest()
        self.assertEqual(data['sampling_rate_hz'], 51200)
        self.assertEqual(len(data['files']), 42)
        self.assertEqual(len({r['filename'] for r in data['files']}), 42)
        for rig in ['selfbuilt', 'spectraquest']:
            rows = [r for r in data['files'] if r['rig'] == rig]
            self.assertEqual(len(rows), 21)
            self.assertEqual({r['label'] for r in rows}, {0, 1, 3})

    def test_blob_identity_checked_before_parsing(self):
        data = b'matlab test bytes'
        row = {'bytes': len(data), 'git_blob_sha': hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.mat'; path.write_bytes(data)
            self.assertEqual(hit_data.verify_file(path, row), hashlib.sha256(data).hexdigest())
            path.write_bytes(b'!' * len(data))
            with self.assertRaises(ValueError):
                hit_data.verify_file(path, row)
            path.write_bytes(b'x')
            with self.assertRaises(ValueError):
                hit_data.verify_file(path, row)
