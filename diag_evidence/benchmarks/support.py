"""Source-record support screening in fixed feature coordinates."""
import math
import numpy as np

from .baseline import FEATURE_VERSION

DIMENSIONS = 55


def _matrix(value):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu' or raw.ndim != 2 or raw.shape[1] != DIMENSIONS or len(raw) == 0:
        raise ValueError('Expected nonempty real 55-dimensional features')
    x = raw.astype(float)
    if not np.isfinite(x).all():
        raise ValueError('Support features must be finite')
    return x


def _records(records, count, split):
    if len(records) != count or any(r.get('split') != split or
            not isinstance(r.get('record_id'), str) or not r['record_id'] for r in records):
        raise ValueError('Support records have invalid roles or identifiers')
    return np.array([r['record_id'] for r in records])


def fit_support(features, records):
    x = _matrix(features)
    ids = _records(records, len(x), 'train')
    with np.errstate(over='ignore', invalid='ignore'):
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('Support fitting overflow')
    names = sorted(set(ids.tolist()))
    prototypes = np.stack([np.median(x[ids == name], axis=0) for name in names])
    return {'format': 'record-support/v1', 'feature_version': FEATURE_VERSION,
            'mean': mean.tolist(), 'scale': scale.tolist(),
            'prototype_records': names, 'prototypes': prototypes.tolist()}


def support_scores(features, parameters):
    x = _matrix(features)
    mean, scale = np.asarray(parameters['mean']), np.asarray(parameters['scale'])
    prototypes = _matrix(parameters['prototypes'])
    names = parameters['prototype_records']
    if (parameters.get('format') != 'record-support/v1' or parameters.get('feature_version') != FEATURE_VERSION
            or mean.shape != (DIMENSIONS,) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not 1 <= len(names) <= 4096 or len(names) != len(prototypes)
            or any(not isinstance(n, str) or not n for n in names) or names != sorted(set(names))):
        raise ValueError('Invalid frozen support parameters')
    with np.errstate(over='ignore', invalid='ignore'):
        z, p = (x - mean) / scale, (prototypes - mean) / scale
    if not np.isfinite(z).all() or not np.isfinite(p).all():
        raise ValueError('Support scaling overflow')
    distances, nearest = [], []
    with np.errstate(over='ignore', invalid='ignore'):
        for block in np.array_split(z, max(1, (len(z) + 31) // 32)):
            d = np.linalg.norm(block[:, None, :] - p[None, :, :], axis=2)
            indices = d.argmin(axis=1)
            distances.extend(d[np.arange(len(block)), indices].tolist())
            nearest.extend(names[i] for i in indices)
    if not np.isfinite(distances).all():
        raise ValueError('Support distance overflow')
    return np.array(distances), nearest


def calibrate_record_threshold(distances, records, alpha=0.05):
    d = np.asarray(distances)
    if (d.ndim != 1 or len(d) == 0 or not np.isfinite(d).all() or np.any(d < 0)
            or isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError('Invalid support calibration')
    ids = _records(records, len(d), 'validation')
    maxima = [{'record_id': name, 'maximum_distance': float(d[ids == name].max())}
              for name in sorted(set(ids.tolist()))]
    rank = math.ceil((1 - alpha) * len(maxima))
    threshold = sorted(r['maximum_distance'] for r in maxima)[rank - 1]
    return {'threshold': threshold, 'alpha': alpha, 'rank_1based': rank,
            'calibration_records': maxima, 'aggregation': 'maximum per original record'}


def acceptance_masks(probabilities, distances, threshold):
    p, d = np.asarray(probabilities), np.asarray(distances)
    if (p.ndim != 2 or p.shape[1] != 4 or d.shape != (len(p),)
            or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0)
            or not np.isfinite(d).all() or np.any(d < 0)
            or isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0):
        raise ValueError('Invalid support decision inputs')
    score, support = p.max(axis=1) >= .5, d <= threshold
    return {'score_only': score, 'support_only': support, 'both': score & support}
