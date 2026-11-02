"""Small NumPy classifier and training-only retrieval, without pretrained weights."""
from __future__ import annotations

import numpy as np

from ..haconfig.settings import class_labels

PREPROCESSING_ID = "hust-window4096-demean-rms-v1"
FEATURE_VERSION = "shape-spectrum-envelope-v1"


def normalize_window(signal) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or len(x) < 64 or not np.isfinite(x).all():
        raise ValueError("Expected a finite one-dimensional signal")
    peak = np.max(np.abs(x))
    if peak <= 0:
        raise ValueError("Zero signal")
    x = x / peak
    x = x - x.mean()
    rms = np.sqrt(np.mean(x * x))
    if rms <= 1e-12:
        raise ValueError("Constant or nearly constant signal")
    return x / rms


def _bands(power: np.ndarray, count: int) -> np.ndarray:
    total = float(power.sum())
    if total <= 0:
        return np.full(count, np.log(1e-12))
    return np.log([float(block.sum()) / total + 1e-12
                   for block in np.array_split(power, count)])


def extract_features(signal) -> np.ndarray:
    """Fixed features; all learned scaling is in fit(), using training data only."""
    x = normalize_window(signal)
    n = len(x)
    taper = np.hanning(n)
    power = np.abs(np.fft.rfft(x * taper)) ** 2
    power[0] = 0
    multiplier = np.zeros(n)
    multiplier[0] = 1
    multiplier[1:(n + 1) // 2] = 2
    if n % 2 == 0:
        multiplier[n // 2] = 1
    envelope = np.abs(np.fft.ifft(np.fft.fft(x) * multiplier))
    centered = envelope - envelope.mean()
    envelope_power = np.abs(np.fft.rfft(centered * taper)) ** 2
    envelope_power[0] = 0
    moments = np.array([np.mean(x ** 3), np.mean(x ** 4),
                        np.log(np.max(np.abs(x))), np.quantile(np.abs(x), 0.95),
                        np.mean(x[1:] * x[:-1] < 0), np.std(envelope),
                        np.dot(np.arange(len(power)), power) / (power.sum() * len(power))])
    features = np.concatenate([moments, _bands(power[1:], 32),
                               _bands(envelope_power[1:max(17, n * 4096 // 51200)], 16)])
    if not np.isfinite(features).all():
        raise ValueError("Non-finite feature vector")
    return features


class CentroidClassifier:
    """Fixed temperature 1 softmax(-distance); scores are NOT calibrated accuracy."""
    def fit(self, features, labels, records):
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels)
        if (x.ndim != 2 or len(x) != len(y) or len(records) != len(y)
                or not np.isfinite(x).all() or any(r["split"] != "train" for r in records)
                or set(y.tolist()) != {0, 1, 2, 3}):
            raise ValueError("Classifier must fit finite training-only four-class data")
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-6)
        z = (x - self.mean) / self.scale
        self.centroids = np.stack([z[y == label].mean(axis=0) for label in range(4)])
        self.training_records = sorted({r["record_id"] for r in records})
        return self

    def transform(self, features):
        return (np.asarray(features) - self.mean) / self.scale

    def predict(self, features) -> dict:
        distances = np.linalg.norm(self.centroids - self.transform(features), axis=1)
        logits = -distances
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        predicted = int(probabilities.argmax())
        return {"success": True, "diagnosis": {
            "predicted_class": predicted, "confidence": float(probabilities[predicted]),
            "predicted_fault": class_labels("bearing-health-v1")[predicted],
            "class_probabilities": {str(i): float(p) for i, p in enumerate(probabilities)}}}

    def export(self) -> dict:
        return {"method": "standardized nearest centroid, softmax(-Euclidean distance), T=1",
                "feature_version": FEATURE_VERSION, "preprocessing_id": PREPROCESSING_ID,
                "label_schema": "bearing-health-v1", "confidence_calibrated": False,
                "training_records": self.training_records, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "centroids": self.centroids.tolist()}


class TrainingIndex:
    """Retrieve features from training only, keeping at most one window per record.

    Distinct records are not proof of distinct physical bearings or independent
    evidence. Retrieval and classifier share features and are correlated.
    """
    def __init__(self, model, features, records):
        if not records or any(row["split"] != "train" for row in records):
            raise ValueError("Retrieval index may contain training records only")
        if len(features) != len(records):
            raise ValueError("Index features and records disagree")
        self.model, self.records = model, list(records)
        self.features = model.transform(features)

    def search(self, features, top_k=5) -> dict:
        if type(top_k) is not int or not 1 <= top_k <= len(self.records):
            raise ValueError("Invalid top_k")
        distances = np.linalg.norm(self.features - self.model.transform(features), axis=1)
        results, seen = [], set()
        for i in np.argsort(distances, kind="stable"):
            row = self.records[int(i)]
            if row["record_id"] in seen:
                continue
            seen.add(row["record_id"])
            results.append({"case_id": f'{row["record_id"]}:{row["start"]}',
                "score": -float(distances[i]), "score_type": "negative_euclidean_feature_distance",
                "score_is_normalized": False,
                "text": f'HUST v3 {row["record_id"]}, window start {row["start"]}; training reference.',
                "meta": {"source_id": "hust-cbv7jyx4p9-v3", "record_id": row["record_id"],
                         "source_url": "https://data.mendeley.com/datasets/cbv7jyx4p9/3",
                         "fault_type": class_labels("bearing-health-v1")[row["label"]]["name"],
                         "units": "1", "sampling_rate": 51200,
                         "preprocessing_id": PREPROCESSING_ID,
                         "evidence_role": "historical_reference"}})
            if len(results) == top_k:
                break
        return {"results": results, "num_results": len(results)}
