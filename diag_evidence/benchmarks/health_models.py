"""Bounded exploratory baselines; optional scikit-learn imports occur in fit()."""
from __future__ import annotations

import numpy as np

from .baseline import extract_features

SEED = 20261101
CANDIDATES = tuple((variant, method) for variant in ("shape", "shape_amplitude")
                   for method in ("centroid", "logistic_C1", "forest_256_leaf5"))


def health_features(signal, variant="shape"):
    shape = extract_features(signal)
    if variant == "shape":
        return shape
    if variant != "shape_amplitude":
        raise ValueError("Unsupported feature variant")
    x = np.asarray(signal, dtype=np.float64)
    centered = x - x.mean()
    # Native numerical amplitude, not an asserted physical unit or calibration.
    amplitudes = [np.sqrt(np.mean(centered ** 2)), np.max(np.abs(centered)),
                  np.quantile(np.abs(centered), .95), np.quantile(np.abs(centered), .99)]
    return np.concatenate([shape, np.log(np.maximum(amplitudes, 1e-12))])


class StudyModel:
    def __init__(self, variant, method):
        if (variant, method) not in CANDIDATES:
            raise ValueError("Unplanned model candidate")
        self.variant, self.method = variant, method

    @property
    def name(self):
        return self.variant + "/" + self.method

    def fit(self, x, y, records):
        x, y = np.asarray(x, dtype=float), np.asarray(y)
        if (x.ndim != 2 or len(x) != len(records) or len(x) != len(y)
                or not np.isfinite(x).all() or set(y.tolist()) != {0, 1, 2, 3}
                or any(r["split"] != "train" for r in records)):
            raise ValueError("Model fitting requires finite four-class training records only")
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-6)
        z = self.transform(x)
        self.training_records = sorted({r["record_id"] for r in records})
        if self.method == "centroid":
            self.centers = np.stack([z[y == label].mean(axis=0) for label in range(4)])
        elif self.method == "logistic_C1":
            from sklearn.linear_model import LogisticRegression
            self.estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                                random_state=SEED).fit(z, y)
            if np.any(self.estimator.n_iter_ >= 2000):
                raise RuntimeError("Logistic baseline did not converge")
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.estimator = RandomForestClassifier(n_estimators=256, min_samples_leaf=5,
                max_features="sqrt", random_state=SEED, n_jobs=1).fit(z, y)
        return self

    def transform(self, x):
        return (np.asarray(x) - self.mean) / self.scale

    def probabilities(self, x):
        z = np.atleast_2d(self.transform(x))
        if self.method == "centroid":
            logits = -np.linalg.norm(z[:, None, :] - self.centers[None, :, :], axis=2)
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
        else:
            probs = self.estimator.predict_proba(z)
        return probs

    def export(self):
        result = {"candidate": self.name, "seed": SEED, "confidence_calibrated": False,
                  "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                  "training_records": self.training_records}
        if self.method == "centroid":
            result["centers"] = self.centers.tolist()
        elif self.method == "logistic_C1":
            result.update(coefficients=self.estimator.coef_.tolist(),
                          intercept=self.estimator.intercept_.tolist(), C=1.0)
        else:
            result["trees"] = [{"left": e.tree_.children_left.tolist(),
                "right": e.tree_.children_right.tolist(), "feature": e.tree_.feature.tolist(),
                "threshold": e.tree_.threshold.tolist(), "value": e.tree_.value.tolist()}
                for e in self.estimator.estimators_]
        return result


def nearest_distances(query, training):
    query, training = np.asarray(query), np.asarray(training)
    if query.ndim != 2 or training.ndim != 2 or query.shape[1] != training.shape[1] or len(training) == 0:
        raise ValueError("Invalid support distance arrays")
    # A bounded batch avoids allocating an entire data-by-data tensor.
    return np.concatenate([np.linalg.norm(block[:, None, :] - training[None, :, :], axis=2).min(axis=1)
                           for block in np.array_split(query, max(1, (len(query) + 31) // 32))])


def leave_type_out_distances(features, records):
    groups = np.array([r["bearing_type"] for r in records])
    if any(r["split"] != "train" for r in records) or len(set(groups)) < 2:
        raise ValueError("Support reference needs at least two training bearing types")
    output = np.empty(len(groups))
    for group in sorted(set(groups)):
        mask = groups == group
        output[mask] = nearest_distances(features[mask], features[~mask])
    return output


def rejection_mask(probabilities, distances, policy):
    probabilities, distances = np.asarray(probabilities), np.asarray(distances)
    if (probabilities.ndim != 2 or probabilities.shape[1] != 4 or len(probabilities) != len(distances)
            or not np.isfinite(probabilities).all() or not np.isfinite(distances).all()):
        raise ValueError("Invalid rejection inputs")
    mask = probabilities.max(axis=1) >= policy["confidence_min"]
    if policy["distance_max"] is not None:
        mask &= distances <= policy["distance_max"]
    return mask


def choose_rejection(probabilities, distances, labels, training_distances):
    """Validation-only heuristic; class coverage constraint avoids rejecting all healthy cases."""
    probabilities, labels = np.asarray(probabilities), np.asarray(labels)
    if set(labels.tolist()) != {0, 1, 2, 3}:
        raise ValueError("Rejection selection needs all four validation classes")
    predicted = probabilities.argmax(axis=1)
    candidates = []
    for score in (.25, .4, .5, .6, .7, .8, .9, .95):
        for quantile in (.9, .95, .99, 1.0, None):
            policy = {"confidence_min": score, "training_distance_quantile": quantile,
                      "distance_max": None if quantile is None else float(np.quantile(training_distances, quantile))}
            mask = rejection_mask(probabilities, distances, policy)
            coverage = {str(k): float(mask[labels == k].mean()) for k in range(4)}
            if min(coverage.values()) < .5:
                continue
            risk = float(np.mean(predicted[mask] != labels[mask]))
            candidates.append({**policy, "validation_risk": risk, "validation_coverage": float(mask.mean()),
                               "validation_class_coverage": coverage})
    if not candidates:
        raise RuntimeError("No policy meets the prespecified per-class coverage floor")
    selected = min(candidates, key=lambda r: (r["validation_risk"], -r["validation_coverage"],
                                             r["confidence_min"], r["training_distance_quantile"] or 2))
    return selected, candidates
