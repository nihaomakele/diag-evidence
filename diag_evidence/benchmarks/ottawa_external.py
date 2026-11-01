"""Frozen source transfer to Ottawa profile A; no target fitting or threshold tuning."""
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json
from pathlib import Path

import numpy as np

from .baseline import extract_features
from .grouped_validation import digest, write_json
from .hust_data import ensure_data, window_starts, WINDOW_SIZE
from .hust_health import classification_metrics


def ottawa_manifest():
    return json.loads(files(__package__).joinpath("ottawa_manifest.json").read_text(encoding="utf-8"))


def _json_file(path):
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Frozen model metadata exceeds its local size budget")
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_probabilities(parameters, features):
    """Interpret numeric JSON parameters only, never pickle or executable code."""
    x = np.atleast_2d(np.asarray(features, dtype=float))
    mean, scale = np.asarray(parameters["mean"]), np.asarray(parameters["scale"])
    if (mean.ndim != 1 or scale.shape != mean.shape or x.shape[1] != len(mean)
            or not np.isfinite(x).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Invalid frozen feature scaling")
    z = (x - mean) / scale
    if not np.isfinite(z).all():
        raise ValueError("Frozen feature transform overflowed")
    candidate = parameters["candidate"]
    if candidate == "shape/centroid":
        centers = np.asarray(parameters["centers"])
        if centers.shape != (4, len(mean)) or not np.isfinite(centers).all():
            raise ValueError("Invalid frozen centers")
        logits = -np.linalg.norm(z[:, None, :] - centers[None, :, :], axis=2)
    elif candidate == "shape/logistic_C1":
        coef, intercept = np.asarray(parameters["coefficients"]), np.asarray(parameters["intercept"])
        if (coef.shape != (4, len(mean)) or intercept.shape != (4,)
                or not np.isfinite(coef).all() or not np.isfinite(intercept).all()):
            raise ValueError("Invalid frozen linear weights")
        logits = z @ coef.T + intercept
    elif candidate == "shape/forest_256_leaf5":
        # sklearn trees predict on float32 input. Finite input has no missing-value path.
        z = z.astype(np.float32)
        trees = parameters["trees"]
        if not 1 <= len(trees) <= 4096:
            raise ValueError("Invalid frozen forest size")
        result = np.zeros((len(z), 4))
        for tree in trees:
            left, right, feature, threshold = (np.asarray(tree[k]) for k in ("left", "right", "feature", "threshold"))
            values = np.asarray(tree["value"])
            count = len(left)
            if (not 1 <= count <= 1000000 or any(a.shape != (count,) for a in (right, feature, threshold))
                    or values.shape != (count, 1, 4) or not np.isfinite(values).all()):
                raise ValueError("Invalid frozen tree arrays")
            for i, sample in enumerate(z):
                node = 0
                for step in range(count):
                    if left[node] == -1:
                        value = values[node, 0]
                        if np.any(value < 0) or value.sum() <= 0:
                            raise ValueError("Invalid frozen leaf probabilities")
                        result[i] += value / value.sum()
                        break
                    column = int(feature[node])
                    if not 0 <= column < len(mean) or not np.isfinite(threshold[node]):
                        raise ValueError("Invalid frozen split")
                    node = int(left[node] if sample[column] <= threshold[node] else right[node])
                    if not 0 <= node < count:
                        raise ValueError("Invalid frozen child index")
                else:
                    raise ValueError("Cycle in frozen decision tree")
        return result / len(trees)
    else:
        raise ValueError("External transfer requires a frozen shape-only candidate")
    if not np.isfinite(logits).all():
        raise ValueError("Frozen model logits overflowed")
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def resample_ottawa(signal):
    from scipy.signal import resample_poly
    x = np.asarray(signal)
    if x.ndim == 2 and 1 in x.shape:
        x = x.reshape(-1)
    if x.ndim != 1 or len(x) < 200000 or not np.isfinite(x).all() or np.iscomplexobj(x):
        raise ValueError("Invalid Ottawa vibration channel")
    return resample_poly(x.astype(float), up=32, down=125, window=("kaiser", 5.0), padtype="constant")


def run_external_ottawa(selection_dir, cache_dir, output_dir, *, download=False):
    try:
        from scipy.io import loadmat
        import scipy
    except ImportError:
        raise RuntimeError("Install diag-evidence[benchmark] for Ottawa MATLAB input") from None
    selection_dir, output = Path(selection_dir), Path(output_dir)
    selection_path = selection_dir / "external_selection.json"
    receipt_bytes = selection_path.read_bytes()
    selection = _json_file(selection_path)
    parameters = _json_file(selection_dir / "external_model.json")
    if (digest(parameters) != selection["model_parameter_sha256"]
            or parameters["candidate"] != selection["selected"]["candidate"]
            or selection["feature_variant"] != "shape" or selection["external_outcomes_used"] is not False):
        raise ValueError("Frozen transfer model and selection do not match")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"recorded_at": datetime.now(timezone.utc).isoformat(),
        "selection_sha256_before_target_loading": hashlib.sha256(receipt_bytes).hexdigest(),
        "frozen_model_parameter_sha256": selection["model_parameter_sha256"],
        "inference_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_candidate": parameters["candidate"], "target_fit_or_threshold_tuning": False,
        "target": "Ottawa v2, profile A, all three trials of H/I/O/B and combined C",
        "resampling": {"from_hz": 200000, "to_hz": 51200, "up": 32, "down": 125,
                       "window": ["kaiser", 5.0], "padtype": "constant"},
        "window_size": WINDOW_SIZE, "max_windows_per_record": 32,
        "features": "demean/RMS normalized shape only; no native amplitude transfer",
        "score_cutoff": selection["score_cutoff"], "quality_screen_passed": selection["quality_screen_passed"],
        "calibrated_probability_or_general_unknown_detection_claim": False}
    write_json(output / "protocol.json", protocol)
    data = ottawa_manifest()
    raw_files = ensure_data(cache_dir, download=download, records=data["files"])
    predictions = []
    for row in raw_files:
        matrix = loadmat(Path(cache_dir) / row["filename"], variable_names=["Channel_1"])
        if "Channel_1" not in matrix:
            raise ValueError("Pinned Ottawa file lacks the declared vibration channel")
        source = matrix["Channel_1"]
        target = resample_ottawa(source)
        for start in window_starts(len(target), count=min(32, len(target) // WINDOW_SIZE)):
            feature = extract_features(target[start:start + WINDOW_SIZE])
            probability = frozen_probabilities(parameters, feature)[0]
            accepted = bool(probability.max() >= selection["score_cutoff"])
            predictions.append({"sample_id": f"ottawa-{len(predictions):05d}", "record_id": row["filename"],
                "split": row["split"], "label": row["label"], "trial": row["trial"],
                "source_sha256": row["sha256"], "source_samples": int(np.asarray(source).size),
                "resampled_samples": len(target), "start": int(start),
                "predicted": int(probability.argmax()), "probabilities": probability.tolist(),
                "score_0_5_accepted": accepted,
                "quality_guard_accepted": accepted and selection["quality_screen_passed"]})
    known = [row for row in predictions if row["label"] is not None]
    unknown = [row for row in predictions if row["label"] is None]
    y = np.array([row["label"] for row in known])
    p = np.array([row["predicted"] for row in known])
    accepted = np.array([row["score_0_5_accepted"] for row in known])
    guarded = np.array([row["quality_guard_accepted"] for row in known])
    if receipt_bytes != selection_path.read_bytes() or digest(parameters) != selection["model_parameter_sha256"]:
        raise RuntimeError("Target evaluation changed the frozen source model")
    per_record = []
    for record in sorted({row["record_id"] for row in known}):
        subset = [r for r in known if r["record_id"] == record]
        per_record.append({"record_id": record, "samples": len(subset), "label": subset[0]["label"],
            "accuracy": float(np.mean([r["predicted"] == r["label"] for r in subset])),
            "score_acceptance": float(np.mean([r["score_0_5_accepted"] for r in subset]))})
    summary = {"scope": "frozen cross-source transfer, no target training or tuning",
        "source_doi": selection["source_doi"], "target_doi": data["doi"], "target_license": data["license"],
        "candidate": parameters["candidate"], "known_records": len(per_record), "known_windows": len(known),
        "unsupported_records": len({r["record_id"] for r in unknown}), "unsupported_windows": len(unknown),
        "all_outputs": classification_metrics(y, p), "score_0_5": classification_metrics(y, p, accepted),
        "quality_guarded": classification_metrics(y, p, guarded),
        "unsupported_score_accepted": sum(r["score_0_5_accepted"] for r in unknown),
        "unsupported_quality_guard_accepted": sum(r["quality_guard_accepted"] for r in unknown),
        "per_record": per_record, "source_selection_unchanged": True, "default_replaced": False,
        "runtime": {"numpy": np.__version__, "scipy": scipy.__version__},
        "limitations": ["One speed profile and three trials per condition, not all Ottawa operating conditions",
            "Trial files are not proof of independent physical bearings; windows within trials are correlated",
            "Cross-source shift includes sensor, bearing geometry, speed and acquisition differences",
            "Resampling and RMS normalization align input representation, not physical domain distributions",
            "Combined-condition rejection is a stress test, not general unknown-class coverage",
            "A quality guard rejecting all inputs has zero coverage, not perfect accuracy"]}
    write_json(output / "predictions.json", predictions)
    write_json(output / "data_manifest.json", data)
    write_json(output / "metrics.json", summary)
    write_json(output / "checksums.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file()})
    return summary
