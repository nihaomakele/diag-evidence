"""Nested leave-type-out validation; all HUST types have been seen previously.

Outer test outcomes never enter their own inner candidate selection. Full-data
LOTO scores are separately used to freeze a shape-only model for a new source;
its selected development score is NOT an unbiased performance estimate.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .health_models import CANDIDATES, SEED, StudyModel, health_features
from .hust_data import load_windows, manifest
from .hust_health import classification_metrics


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def nested_type_plan(groups):
    groups = np.asarray(groups)
    types = sorted(set(groups.tolist()))
    if len(types) != 4:
        raise ValueError("This fixed nested protocol requires four complete bearing types")
    plan = []
    for outer in types:
        inner = []
        for validation in types:
            if validation == outer:
                continue
            training = [t for t in types if t not in (outer, validation)]
            inner.append({"train_types": training, "validation_type": validation,
                          "train_indices": np.flatnonzero(np.isin(groups, training)).tolist(),
                          "validation_indices": np.flatnonzero(groups == validation).tolist()})
        plan.append({"test_type": outer, "train_types": [t for t in types if t != outer],
                     "test_indices": np.flatnonzero(groups == outer).tolist(), "inner": inner})
    return plan


def robust_choice(candidates, validation_types):
    expected = set(validation_types)
    if not candidates or len(expected) < 2:
        raise ValueError("Robust selection needs multiple validation types")
    summary = []
    for row in candidates:
        folds = row["folds"]
        if len(folds) != len(expected) or {f["validation_type"] for f in folds} != expected:
            raise ValueError("Candidate contains wrong, duplicate or missing validation types")
        scores = [f["metrics"]["balanced_accuracy_all_outputs"] for f in folds]
        alarms = [f["metrics"]["healthy_false_alarm_all_healthy"] for f in folds]
        summary.append({"candidate": row["candidate"], "candidate_order": row["candidate_order"],
                        "minimum_type_accuracy": min(scores), "mean_type_accuracy": float(np.mean(scores)),
                        "mean_healthy_false_alarm": float(np.mean(alarms))})
    selected = min(summary, key=lambda r: (-r["minimum_type_accuracy"], -r["mean_type_accuracy"],
                                          r["mean_healthy_false_alarm"], r["candidate_order"]))
    return selected, summary


def verify_partition(rows, train, validation, outer_test=()):
    roles = [list(train), list(validation), list(outer_test)]
    for a in range(3):
        for b in range(a + 1, 3):
            if set(roles[a]) & set(roles[b]):
                raise ValueError("Window indices cross partitions")
            if ({rows[i]["record_id"] for i in roles[a]} & {rows[i]["record_id"] for i in roles[b]}):
                raise ValueError("Original recordings cross partitions")
            if ({rows[i]["bearing_type"] for i in roles[a]} & {rows[i]["bearing_type"] for i in roles[b]}):
                raise ValueError("Bearing types cross partitions")


def _fit(variant, method, features, labels, rows, indices):
    training = [{**rows[i], "original_split": rows[i]["split"], "split": "train"} for i in indices]
    return StudyModel(variant, method).fit(features[indices], labels[indices], training)


def _score(labels, probabilities):
    all_output = classification_metrics(labels, probabilities.argmax(axis=1))
    thresholded = classification_metrics(labels, probabilities.argmax(axis=1), probabilities.max(axis=1) >= .5)
    return {"all_outputs": all_output, "score_0_5": thresholded}


def _quality(folds):
    # A fixed engineering screen, not a calibrated error guarantee.
    return all(f["score_0_5"]["accepted_error_rate"] is not None
               and f["score_0_5"]["accepted_error_rate"] <= .1
               and min(f["score_0_5"]["class_coverage"].values()) >= .5 for f in folds)


def run_grouped_validation(cache_dir, output_dir):
    try:
        import sklearn
        import scipy
    except ImportError:
        raise RuntimeError("Install diag-evidence[study] for grouped validation") from None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = {"created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "exploratory nested type-group validation; all HUST outcomes previously inspected",
        "external_preregistration": False, "types": [6205, 6206, 6207, 6208],
        "candidates": [list(c) for c in CANDIDATES], "random_seed": SEED,
        "selection": "max worst inner-type balanced accuracy; max mean; min healthy alarm; fixed order",
        "outer": "one held-out type", "inner": "train two types, validate third within outer training",
        "refit": "selected model on all three outer training types",
        "fixed_score_cutoff": .5, "cutoff_tuning": False,
        "quality_screen": "every inner type has accepted error <=0.1 and every class coverage >=0.5",
        "external_model": "shape-only candidate, robust full-development LOTO selection; refit all four types",
        "default_replaced": False}
    write_json(output / "protocol.json", protocol)
    signals, rows = load_windows(cache_dir, normalize=False)
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["bearing_type"] for r in rows])
    plan = nested_type_plan(groups)
    feature_sets = {variant: np.stack([health_features(signal, variant) for signal in signals])
                    for variant in ("shape", "shape_amplitude")}
    write_json(output / "split_plan.json", plan)
    write_json(output / "window_manifest.json", rows)
    outer_results, prediction_rows, selection_receipts = [], [], []
    development = [{"candidate": v + "/" + m, "candidate_order": order, "folds": []}
                   for order, (v, m) in enumerate(CANDIDATES)]
    for outer in plan:
        candidate_rows, inner_predictions = [], {}
        for order, (variant, method) in enumerate(CANDIDATES):
            inner_folds = []
            name = variant + "/" + method
            inner_predictions[name] = []
            for fold in outer["inner"]:
                train, val = fold["train_indices"], fold["validation_indices"]
                verify_partition(rows, train, val, outer["test_indices"])
                model = _fit(variant, method, feature_sets[variant], labels, rows, train)
                probs = model.probabilities(feature_sets[variant][val])
                scored = _score(labels[val], probs)
                inner_folds.append({"validation_type": fold["validation_type"],
                    "train_types": fold["train_types"], "metrics": scored["all_outputs"],
                    "score_0_5": scored["score_0_5"], "training_records": model.training_records,
                    "model_parameter_sha256": digest(model.export())})
                inner_predictions[name].append(scored)
            candidate_rows.append({"candidate": name, "candidate_order": order, "folds": inner_folds})
        selected, ranking = robust_choice(candidate_rows, outer["train_types"])
        quality = _quality(inner_predictions[selected["candidate"]])
        receipt = {"frozen_at": datetime.now(timezone.utc).isoformat(), "test_type": outer["test_type"],
                   "selected": selected, "quality_screen_passed": quality,
                   "selection_validation_types": outer["train_types"],
                   "outer_test_used_for_selection": False, "rankings": ranking, "inner_folds": candidate_rows}
        receipt_path = output / f'outer_{outer["test_type"]}_selection.json'
        write_json(receipt_path, receipt)
        receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        selection_receipts.append(receipt)
        # Score outer outcomes only after this fold's selection has been frozen.
        test = outer["test_indices"]
        train = [i for i in range(len(rows)) if groups[i] != outer["test_type"]]
        per_candidate = {}
        for order, (variant, method) in enumerate(CANDIDATES):
            verify_partition(rows, train, test)
            model = _fit(variant, method, feature_sets[variant], labels, rows, train)
            probs = model.probabilities(feature_sets[variant][test])
            scored = _score(labels[test], probs)
            name = variant + "/" + method
            development[order]["folds"].append({"validation_type": outer["test_type"],
                "metrics": scored["all_outputs"], "score_0_5": scored["score_0_5"]})
            per_candidate[name] = {**scored, "model_parameter_sha256": digest(model.export())}
            if name in {selected["candidate"], "shape/centroid", "shape_amplitude/forest_256_leaf5"}:
                for position, i in enumerate(test):
                    prediction_rows.append({"sample_id": rows[i]["sample_id"], "record_id": rows[i]["record_id"],
                        "test_type": outer["test_type"], "label": rows[i]["label"], "candidate": name,
                        "is_nested_selected": name == selected["candidate"],
                        "predicted": int(probs[position].argmax()), "probabilities": probs[position].tolist(),
                        "quality_guard_allows": quality if name == selected["candidate"] else None})
        if hashlib.sha256(receipt_path.read_bytes()).hexdigest() != receipt_hash:
            raise RuntimeError("Outer evaluation mutated its selection receipt")
        primary = per_candidate[selected["candidate"]]
        guarded = deepcopy_metrics_zero(primary["all_outputs"]) if not quality else primary["score_0_5"]
        outer_results.append({"test_type": outer["test_type"], "selected": selected["candidate"],
            "quality_screen_passed": quality, "nested_selected": primary,
            "quality_guarded": guarded, "fixed_centroid": per_candidate["shape/centroid"],
            "fixed_previous_forest": per_candidate["shape_amplitude/forest_256_leaf5"]})
    # This is development model selection for a NEW source, distinct from the
    # nested estimate above. Never present its selected LOTO score as unbiased.
    shape_rows = [row for row in development if row["candidate"].startswith("shape/")]
    external_choice, external_ranking = robust_choice(shape_rows, sorted(set(groups.tolist())))
    variant, method = external_choice["candidate"].split("/")
    external_model = _fit(variant, method, feature_sets[variant], labels, rows, list(range(len(rows))))
    chosen_folds = next(r["folds"] for r in shape_rows if r["candidate"] == external_choice["candidate"])
    external_selection = {"frozen_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "frozen zero-shot source-transfer model, not a new unbiased HUST result",
        "selected": external_choice, "shape_only_rankings": external_ranking,
        "source_doi": manifest()["doi"], "training_records": external_model.training_records,
        "training_types": sorted(set(groups.tolist())), "feature_variant": "shape",
        "training_sampling_rate": 51200, "native_amplitude_units_used": False,
        "quality_screen_passed": _quality(chosen_folds), "score_cutoff": .5,
        "model_parameter_sha256": digest(external_model.export()),
        "grouped_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "health_models_code_sha256": hashlib.sha256(Path(__file__).with_name("health_models.py").read_bytes()).hexdigest(),
        "external_outcomes_used": False}
    write_json(output / "external_selection.json", external_selection)
    write_json(output / "external_model.json", external_model.export())
    write_json(output / "development_loto.json", development)
    write_json(output / "outer_results.json", outer_results)
    with (output / "outer_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    selected_predictions = [row for row in prediction_rows if row["is_nested_selected"]]
    if len(selected_predictions) != len(rows) or len({r["sample_id"] for r in selected_predictions}) != len(rows):
        raise RuntimeError("Each window must be selected-model outer-tested exactly once")
    aggregate = {}
    for key in ("nested_selected", "fixed_centroid", "fixed_previous_forest"):
        scores = [f[key]["all_outputs"]["balanced_accuracy_all_outputs"] for f in outer_results]
        alarms = [f[key]["all_outputs"]["healthy_false_alarm_all_healthy"] for f in outer_results]
        aggregate[key] = {"mean_type_accuracy": float(np.mean(scores)), "worst_type_accuracy": min(scores),
                          "mean_healthy_false_alarm": float(np.mean(alarms))}
    summary = {"scope": protocol["scope"], "outer_folds": 4, "inner_folds_per_outer": 3,
        "train_fit_count": 72 + 24 + 1, "outer_tested_windows": len(selected_predictions),
        "outer_tested_records": len({row["record_id"] for row in rows}), "selection_strategy": protocol["selection"],
        "threshold_tuning": False, "aggregate": aggregate, "outer_results": outer_results,
        "external_model_candidate": external_choice["candidate"], "default_replaced": False,
        "runtime": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["All HUST types previously inspected; exploratory procedure audit only",
            "Four type groups from one test bench are not four independent deployment populations",
            "Windows are correlated within recordings; no independent-sample confidence intervals",
            "Robust candidate selection and quality screen have no calibrated risk guarantee",
            "Full-development LOTO selection is for external transfer, not an unbiased selected-model estimate"]}
    write_json(output / "metrics.json", summary)
    write_json(output / "checksums.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file()})
    return summary


def deepcopy_metrics_zero(metrics):
    """Explicit zero-coverage gate result; raw classifier scores remain separate."""
    result = json.loads(json.dumps(metrics))
    result.update(accepted=0, coverage=0., accepted_error_rate=None,
                  class_coverage={k: 0. for k in result["class_coverage"]},
                  healthy_coverage=0., healthy_false_alarm_all_healthy=0.,
                  healthy_false_alarm_among_accepted=None, fault_predicted_healthy_all_faults=0.,
                  confusion_accepted_true_rows=[[0] * 4 for _ in range(4)])
    return result
