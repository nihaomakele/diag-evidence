"""Fixed offline envelope/order study; known A is development, B is a profile holdout."""
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json
from pathlib import Path

import numpy as np

from .baseline import extract_features
from .grouped_validation import digest, write_json, robust_choice, _quality
from .health_models import StudyModel
from .hust_data import ensure_data, manifest
from .hust_health import classification_metrics
from .ottawa_external import ottawa_manifest, resample_ottawa, frozen_probabilities
from .order_features import (FEATURE_VERSION, FS, encoder_phase, constant_phase, paired_features)

METHODS = ("centroid", "logistic_C1", "forest_256_leaf5")
SPEC = {"feature_version": FEATURE_VERSION, "sampling_hz": FS,
        "carrier_hz": [2000, 10000], "envelope_lowpass_hz": 1000,
        "butterworth_order": 4, "turns_per_window": 8, "samples_per_turn": 512,
        "windows_per_record": 8, "edge_margin_turns": 2, "order_range": [0, 20], "bands": 40,
        "encoder_cpr": 1024, "speed_range_hz": [8, 45], "nominal_control_hz": 25,
        "offline_noncausal": True}


def _fit(method, features, labels, rows, indices):
    records = [{**rows[i], "split": "train"} for i in indices]
    return StudyModel("shape", method).fit(features[indices], labels[indices], records)


def _score(labels, probabilities):
    return {"metrics": classification_metrics(labels, probabilities.argmax(axis=1)),
            "score_0_5": classification_metrics(labels, probabilities.argmax(axis=1), probabilities.max(axis=1) >= .5)}


def run_order_development(cache_dir, output_dir, legacy_selection_dir=None):
    try:
        from scipy.io import loadmat
        import sklearn
    except ImportError:
        raise RuntimeError("Install diag-evidence[study] for the order experiment") from None
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {"recorded_at": datetime.now(timezone.utc).isoformat(), "spec": SPEC,
        "A_previously_used_to_diagnose_feature_drift": True, "HUST_previously_seen": True,
        "primary": "angular envelope", "paired_control": "same measured windows, uniform time, nominal25Hz",
        "classifier_selection": "HUST leave-type-out worst balanced accuracy, mean, healthy alarm, fixed order",
        "target_model_fitting_or_threshold_selection": False, "default_replaced": False})
    angular, nominal, rows, record_metadata = [], [], [], []
    for row in ensure_data(cache_dir):
        matrix = loadmat(Path(cache_dir) / row["filename"], variable_names=["data", "fs"])
        raw = np.asarray(matrix["data"], dtype=float)
        if raw.ndim != 2 or raw.shape[1] != 1 or np.asarray(matrix["fs"]).size != 1:
            raise ValueError("Invalid HUST vibration or shaft-frequency metadata")
        shaft = float(np.asarray(matrix["fs"]).item())
        phase = constant_phase(len(raw), shaft)
        a, b, _, windows = paired_features(raw[:, 0], phase)
        angular.extend(a); nominal.extend(b)
        for window in windows:
            rows.append({**window, "record_id": row["filename"], "bearing_type": row["bearing_type"],
                         "label": row["label"], "split": "development", "source_sha256": row["sha256"]})
        record_metadata.append({"record_id": row["filename"], "shaft_hz": shaft,
                                "phase_source": "recorded constant scalar, not measured instantaneous angle"})
    angular, nominal = np.stack(angular), np.stack(nominal)
    y = np.array([r["label"] for r in rows]); groups = np.array([r["bearing_type"] for r in rows])
    candidates = []
    for order, method in enumerate(METHODS):
        folds = []
        for group in sorted(set(groups)):
            train, validation = np.flatnonzero(groups != group), np.flatnonzero(groups == group)
            model = _fit(method, angular, y, rows, train)
            scored = _score(y[validation], model.probabilities(angular[validation]))
            folds.append({"validation_type": int(group), **scored, "training_records": model.training_records})
        candidates.append({"candidate": method, "candidate_order": order, "folds": folds})
    selected, ranking = robust_choice(candidates, sorted(set(groups)))
    method = selected["candidate"]
    model = _fit(method, angular, y, rows, np.arange(len(rows)))
    control = _fit(method, nominal, y, rows, np.arange(len(rows)))
    chosen_folds = next(c["folds"] for c in candidates if c["candidate"] == method)
    legacy = None
    if legacy_selection_dir is not None:
        prior = Path(legacy_selection_dir)
        legacy = json.loads((prior / "external_model.json").read_text())
        prior_selection = json.loads((prior / "external_selection.json").read_text())
        if digest(legacy) != prior_selection["model_parameter_sha256"]:
            raise ValueError("Legacy frozen control does not match its source receipt")
    checkpoint = {"format": "offline-order-study/v1", "spec": SPEC, "classifier": method,
                  "angular_numeric_model": model.export(), "time_numeric_model": control.export(),
                  "legacy_numeric_model": legacy}
    selection = {"frozen_at": datetime.now(timezone.utc).isoformat(), "spec": SPEC,
        "selected": selected, "ranking": ranking, "quality_screen_passed": _quality(chosen_folds),
        "checkpoint_sha256": digest(checkpoint), "target_B_measurements_loaded": False,
        "A_is_development_diagnostic": True, "target_labels_used_to_fit": False,
        "source_records": sorted({r["record_id"] for r in rows}), "source_types": sorted(set(groups.tolist())),
        "features_code_sha256": hashlib.sha256(Path(__file__).with_name("order_features.py").read_bytes()).hexdigest(),
        "study_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "default_replaced": False}
    write_json(output / "source_cv.json", candidates)
    write_json(output / "source_windows.json", rows)
    write_json(output / "speed_metadata.json", record_metadata)
    write_json(output / "checkpoint.json", checkpoint)
    write_json(output / "frozen_selection.json", selection)
    write_json(output / "checksums.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file()})
    return selection


def _profile_manifest(profile):
    if profile == "A":
        return ottawa_manifest()
    if profile == "B":
        return json.loads(files(__package__).joinpath("ottawa_b_manifest.json").read_text(encoding="utf-8"))
    raise ValueError("Only predeclared A diagnostic and B profile holdout are supported")


def evaluate_order_profile(selection_dir, cache_dir, output_dir, profile, *, download=False):
    from scipy.io import loadmat
    directory, output = Path(selection_dir), Path(output_dir)
    receipt_bytes = (directory / "frozen_selection.json").read_bytes()
    selection = json.loads(receipt_bytes)
    checkpoint = json.loads((directory / "checkpoint.json").read_text())
    if checkpoint["format"] != "offline-order-study/v1" or checkpoint["spec"] != SPEC or digest(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("Frozen envelope/order checkpoint does not match its protocol")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {"recorded_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile, "role": "seen A development diagnostic" if profile == "A" else "previously unread B operating condition",
        "selection_sha256_before_loading": hashlib.sha256(receipt_bytes).hexdigest(),
        "checkpoint_sha256": digest(checkpoint), "no_target_training_or_threshold_tuning": True,
        "same_physical_devices_not_independent_source": True, "spec": SPEC})
    dataset = _profile_manifest(profile)
    prediction_rows, speeds = [], []
    for row in ensure_data(cache_dir, download=download, records=dataset["files"]):
        mat = loadmat(Path(cache_dir) / row["filename"], variable_names=["Channel_1", "Channel_2"])
        raw = resample_ottawa(mat["Channel_1"])
        phase, speed = encoder_phase(mat["Channel_2"], len(raw))
        a, b, crops, windows = paired_features(raw, phase)
        probabilities = {"angular": frozen_probabilities(checkpoint["angular_numeric_model"], a),
                         "time_control": frozen_probabilities(checkpoint["time_numeric_model"], b)}
        if checkpoint["legacy_numeric_model"] is not None:
            probabilities["legacy_short_context"] = frozen_probabilities(checkpoint["legacy_numeric_model"],
                np.stack([extract_features(x) for x in crops]))
        speeds.append({"record_id": row["filename"], **speed})
        for i, window in enumerate(windows):
            prediction_rows.append({"record_id": row["filename"], "label": row["label"],
                "source_sha256": row["sha256"], **window,
                "predictions": {name: {"class": int(p[i].argmax()), "probabilities": p[i].tolist(),
                                       "confidence": float(p[i].max())} for name, p in probabilities.items()}})
    known = [r for r in prediction_rows if r["label"] is not None]
    unsupported = [r for r in prediction_rows if r["label"] is None]
    y = np.array([r["label"] for r in known])
    outcomes = {}
    for name in known[0]["predictions"]:
        p = np.array([r["predictions"][name]["class"] for r in known])
        confidence = np.array([r["predictions"][name]["confidence"] for r in known])
        records = []
        for record in sorted({r["record_id"] for r in known}):
            part = [r for r in known if r["record_id"] == record]
            guesses = [r["predictions"][name]["class"] for r in part]
            records.append({"record_id": record, "label": part[0]["label"],
                "majority_prediction": int(np.bincount(guesses, minlength=4).argmax()),
                "window_accuracy": float(np.mean(np.array(guesses) == part[0]["label"]))})
        outcomes[name] = {"all_outputs": classification_metrics(y, p),
            "score_0_5": classification_metrics(y, p, confidence >= .5),
            "record_majority_accuracy": float(np.mean([r["label"] == r["majority_prediction"] for r in records])),
            "per_record": records,
            "unsupported_score_accepted": sum(r["predictions"][name]["confidence"] >= .5 for r in unsupported)}
    if receipt_bytes != (directory / "frozen_selection.json").read_bytes():
        raise RuntimeError("Target evaluation changed frozen selection")
    summary = {"profile": profile, "selected_classifier": checkpoint["classifier"],
        "feature_version": FEATURE_VERSION, "known_records": 12, "known_windows": len(known),
        "unsupported_records": 3, "unsupported_windows": len(unsupported),
        "source_quality_screen_passed": selection["quality_screen_passed"], "outcomes": outcomes,
        "selection_unchanged": True, "default_replaced": False,
        "limitations": ["Offline zero-phase filtering and complete-record speed extraction, not causal real-time",
            "A informed feature design; B is another condition on the same rig/prototypes, not new independent devices",
            "Primary and time control share physical windows and selected learner; legacy uses shorter context",
            "Classifier scores uncalibrated; no general unknown-state detection or deployment proof"]}
    write_json(output / "metrics.json", summary)
    write_json(output / "predictions.json", prediction_rows)
    write_json(output / "speed_metadata.json", speeds)
    write_json(output / "data_manifest.json", dataset)
    write_json(output / "checksums.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file()})
    return summary
