"""Validation-selected health study with a new, explicitly partial 6204 holdout.

Candidate and rejection selection never receive 6208 or 6204 measurements or
outcomes. The local selection receipt is frozen before either final evaluation.
This is an exploratory study, not external preregistration or deployment proof.
"""
from __future__ import annotations

from importlib.resources import files
import hashlib
import json
from pathlib import Path
import platform
import time
from datetime import datetime, timezone

import numpy as np

from .health_models import (CANDIDATES, SEED, StudyModel, health_features,
    nearest_distances, leave_type_out_distances, choose_rejection, rejection_mask)
from .hust_data import ensure_data, load_windows, window_starts, SAMPLE_RATE, WINDOW_SIZE


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def holdout_manifest():
    return json.loads(files(__package__).joinpath("hust_holdout_manifest.json").read_text(encoding="utf-8"))


def classification_metrics(labels, predicted, accepted=None):
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    if labels.ndim != 1 or len(labels) == 0 or labels.shape != predicted.shape:
        raise ValueError("Invalid classification metric inputs")
    mask = np.ones(len(labels), dtype=bool) if accepted is None else np.asarray(accepted, dtype=bool)
    if mask.shape != labels.shape:
        raise ValueError("Acceptance shape mismatch")
    true_classes = sorted(set(labels.tolist()))
    matrix = np.zeros((4, 4), dtype=int)
    for true, guess in zip(labels[mask], predicted[mask]):
        matrix[int(true), int(guess)] += 1
    count = int(mask.sum())
    healthy, faults = labels == 3, labels != 3
    health_accepted = mask & healthy
    return {"samples": len(labels), "accepted": count, "coverage": float(mask.mean()),
        "accepted_error_rate": float(np.mean(labels[mask] != predicted[mask])) if count else None,
        "balanced_accuracy_all_outputs": float(np.mean([np.mean(predicted[labels == k] == k) for k in true_classes])),
        "accuracy_all_outputs": float(np.mean(labels == predicted)),
        "evaluated_true_classes": true_classes, "confusion_accepted_true_rows": matrix.tolist(),
        "class_coverage": {str(k): float(mask[labels == k].mean()) for k in true_classes},
        "healthy_coverage": float(mask[healthy].mean()) if healthy.any() else None,
        "healthy_false_alarm_all_healthy": float(np.mean(mask[healthy] & (predicted[healthy] != 3))) if healthy.any() else None,
        "healthy_false_alarm_among_accepted": float(np.mean(predicted[health_accepted] != 3)) if health_accepted.any() else None,
        "fault_predicted_healthy_all_faults": float(np.mean(mask[faults] & (predicted[faults] == 3))) if faults.any() else None}


def choose_candidate(candidates):
    """Only validation records may enter this selection interface."""
    if not candidates or any(row["split"] != "validation" for row in candidates):
        raise ValueError("Candidate selection accepts validation results only")
    return min(candidates, key=lambda row: (-row["metrics"]["balanced_accuracy_all_outputs"],
        row["metrics"]["healthy_false_alarm_all_healthy"], row["candidate_order"]))


def _feature_matrix(signals, variant):
    return np.stack([health_features(signal, variant) for signal in signals])


def _new_holdout(cache_dir, *, download):
    from scipy.io import loadmat
    data = holdout_manifest()
    rows = ensure_data(cache_dir, download=download, records=data["files"])
    signals, windows = [], []
    for row in rows:
        matrix = np.asarray(loadmat(Path(cache_dir) / row["filename"], variable_names=["data"])["data"], dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != 1 or not np.isfinite(matrix).all():
            raise ValueError("Invalid final holdout signal")
        for start in window_starts(len(matrix), count=min(32, len(matrix) // WINDOW_SIZE)):
            windows.append({"sample_id": f"new6204-{len(windows):05d}", "record_id": row["filename"],
                "bearing_type": 6204, "split": row["split"], "label": row["label"],
                "start": int(start), "record_samples": len(matrix), "source_sha256": row["sha256"]})
            signals.append(matrix[start:start + WINDOW_SIZE, 0].copy())
    return np.stack(signals), windows


def _support_audit(signals, rows):
    result = []
    for record_id in sorted({r["record_id"] for r in rows}):
        indices = [i for i, row in enumerate(rows) if row["record_id"] == record_id]
        x = signals[indices]
        rms = np.sqrt(np.mean((x - x.mean(axis=1, keepdims=True)) ** 2, axis=1))
        result.append({"record_id": record_id, "split": rows[indices[0]]["split"],
                       "label": rows[indices[0]]["label"], "median_native_rms": float(np.median(rms)),
                       "minimum_rms": float(rms.min()), "maximum_rms": float(rms.max()),
                       "physical_units_verified": False})
    return result


def run_health_study(cache_dir, holdout_cache_dir, output_dir, *, download=False):
    try:
        import sklearn
        import scipy
    except ImportError:
        raise RuntimeError("Install diag-evidence[study] for the optional health study") from None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    protocol = {"created_at": datetime.now(timezone.utc).isoformat(),
        "train": [6205, 6206], "selection_validation": 6207,
        "already_seen_exploratory_only": 6208, "new_final_partial_holdout": 6204,
        "new_holdout_true_classes": [0, 1, 3], "new_unsupported_combinations": ["IO", "OB"],
        "candidates": [list(c) for c in CANDIDATES], "seed": SEED,
        "window_size": WINDOW_SIZE, "training_validation_windows_per_record": 32,
        "final_windows_per_record_max": 32,
        "model_selection": "validation balanced accuracy, then healthy false alarm, then fixed order",
        "rejection_selection": "validation error minimized subject to each class coverage >= 0.5; then higher coverage",
        "confidence_grid": [.25, .4, .5, .6, .7, .8, .9, .95],
        "training_distance_quantiles": [.9, .95, .99, 1.0, None],
        "score_calibration_claim": False, "distance_method": "training leave-one-bearing-type nearest distance",
        "quality_target": "validation accepted error <= 0.10, each class coverage >= 0.5",
        "refit_on_validation": False, "external_preregistration": False}
    _write(output / "protocol.json", protocol)
    signals, rows = load_windows(cache_dir, normalize=False, splits=("train", "validation"))
    train = np.array([r["split"] == "train" for r in rows])
    val = ~train
    training_rows = [r for r in rows if r["split"] == "train"]
    labels = np.array([r["label"] for r in rows])
    feature_sets = {v: _feature_matrix(signals, v) for v in ("shape", "shape_amplitude")}
    fitted, candidates = {}, []
    for order, (variant, method) in enumerate(CANDIDATES):
        x = feature_sets[variant]
        model = StudyModel(variant, method).fit(x[train], labels[train], training_rows)
        probs = model.probabilities(x[val])
        candidate = {"candidate": model.name, "candidate_order": order, "split": "validation",
                     "metrics": classification_metrics(labels[val], probs.argmax(axis=1))}
        candidates.append(candidate)
        fitted[model.name] = model
    chosen = choose_candidate(candidates)
    model = fitted[chosen["candidate"]]
    features = feature_sets[model.variant]
    ztrain = model.transform(features[train])
    support_reference = leave_type_out_distances(ztrain, training_rows)
    val_distance = nearest_distances(model.transform(features[val]), ztrain)
    val_probs = model.probabilities(features[val])
    rejection, rejection_candidates = choose_rejection(val_probs, val_distance, labels[val], support_reference)
    selection = {"frozen_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": _digest(protocol),
        "candidate": chosen, "rejection": rejection,
        "validation_quality_target_met": rejection["validation_risk"] <= .1,
        "training_records": model.training_records,
        "training_model_sha256": _digest(model.export()),
        "study_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_code_sha256": hashlib.sha256(Path(__file__).with_name("health_models.py").read_bytes()).hexdigest(),
        "new_holdout_measurements_loaded": False, "old6208_used_for_selection": False,
        "refit_on_validation": False}
    _write(output / "validation_candidates.json", candidates)
    _write(output / "validation_rejection_candidates.json", rejection_candidates)
    _write(output / "frozen_selection.json", selection)
    _write(output / "trained_model.json", model.export())
    # No measurements or outcomes from either final set are loaded until the
    # model and rejection choice are frozen above.
    selection_bytes = (output / "frozen_selection.json").read_bytes()
    old_signals, old_rows = load_windows(cache_dir, normalize=False, splits=("test",))
    old_rows = [{**r, "split": "seen6208_exploratory"} for r in old_rows]
    new_signals, new_rows = _new_holdout(holdout_cache_dir, download=download)
    holdout_loaded_at = datetime.now(timezone.utc).isoformat()
    final_signals = np.concatenate([old_signals, new_signals])
    final_rows = old_rows + new_rows
    baseline = fitted["shape/centroid"]
    final_features = _feature_matrix(final_signals, model.variant)
    baseline_features = final_features if model.variant == "shape" else _feature_matrix(final_signals, "shape")
    probs = model.probabilities(final_features)
    base_probs = baseline.probabilities(baseline_features)
    distance = nearest_distances(model.transform(final_features), ztrain)
    accepted = rejection_mask(probs, distance, rejection)
    predictions = []
    for i, row in enumerate(final_rows):
        predictions.append({**row, "baseline_predicted": int(base_probs[i].argmax()),
            "selected_predicted": int(probs[i].argmax()), "selected_probabilities": probs[i].tolist(),
            "selected_confidence": float(probs[i].max()), "support_distance": float(distance[i]),
            "accepted_by_classifier_gate": bool(accepted[i]),
            "abstention_reason": None if accepted[i] else (
                "outside_training_support" if rejection["distance_max"] is not None and distance[i] > rejection["distance_max"]
                else "below_selection_threshold")})
    results = {}
    for split in ("seen6208_exploratory", "fresh_known_6204", "fresh_unsupported_combination_6204"):
        indices = [i for i, r in enumerate(final_rows) if r["split"] == split]
        selected_mask = accepted[indices]
        if split == "fresh_unsupported_combination_6204":
            results[split] = {"samples": len(indices), "records": len({final_rows[i]["record_id"] for i in indices}),
                "accepted_unsupported": int(selected_mask.sum()),
                "unsupported_rejection_rate": float(np.mean(~selected_mask)),
                "baseline_closed_set_forced_output_rate": 1.0,
                "interpretation": "near-OOD combination stress test; not general unknown-state validation"}
            record_rejection = [{"record_id": record, "samples": sum(final_rows[i]["record_id"] == record for i in indices),
                "rejection_rate": float(np.mean([not accepted[i] for i in indices if final_rows[i]["record_id"] == record]))}
                for record in sorted({final_rows[i]["record_id"] for i in indices})]
            results[split]["per_record"] = record_rejection
            results[split]["record_mean_rejection_rate"] = float(np.mean([r["rejection_rate"] for r in record_rejection]))
        else:
            y = np.array([final_rows[i]["label"] for i in indices])
            results[split] = {"records": len({final_rows[i]["record_id"] for i in indices}),
                "baseline": classification_metrics(y, base_probs[indices].argmax(axis=1)),
                "selected_without_rejection": classification_metrics(y, probs[indices].argmax(axis=1)),
                "selected_with_rejection": classification_metrics(y, probs[indices].argmax(axis=1), selected_mask),
                "classifier_gate_and_workflow_score_0_5_eligible": int(np.sum(selected_mask & (probs[indices].max(axis=1) >= .5)))}
    if selection_bytes != (output / "frozen_selection.json").read_bytes() or _digest(model.export()) != selection["training_model_sha256"]:
        raise RuntimeError("Final evaluation mutated the frozen selection or model")
    _write(output / "predictions.json", predictions)
    _write(output / "training_validation_windows.json", rows)
    _write(output / "native_amplitude_audit.json", _support_audit(signals, rows))
    _write(output / "holdout_data_manifest.json", holdout_manifest())
    _write(output / "distance_reference.json", {"method": "train-only leave-one-bearing-type nearest distance",
        "samples": support_reference.tolist(), "training_bearing_types": [6205, 6206],
        "no_conformal_or_probability_guarantee": True})
    summary = {"scope": "exploratory validation-selected health/generalization study",
        "selected": model.name, "rejection": rejection,
        "validation_quality_target_met": selection["validation_quality_target_met"],
        "selection_frozen_at": selection["frozen_at"], "holdout_loaded_at": holdout_loaded_at,
        "selection_file_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "unchanged_selection_after_evaluation": True, "default_baseline_replaced": False,
        "score_probabilities_calibrated": False,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "seconds": time.perf_counter() - start, "results": results,
        "limitations": ["Only two training types and one selection-validation type",
            "6208 already seen, exploratory only; 6204 new but lacks true ball-fault examples",
            "Combined faults are a near-OOD stress test, not coverage of unknown failures",
            "Native amplitude physical units remain unverified; no cross-sensor amplitude claim",
            "Validation model/threshold selection is optimistic; no calibrated risk guarantee",
            "Per-window counts are correlated within original records; no independent-device confidence intervals",
            "Classifier acceptance is separate from the workflow's 0.5 score rule and reference requirements"]}
    _write(output / "metrics.json", summary)
    _write(output / "checksums.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file()})
    return summary
