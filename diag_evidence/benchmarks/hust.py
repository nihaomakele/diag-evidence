"""Real classifier evaluation, with separate controlled reference-metadata replay.

Ground truth is used for fitting training centroids and for offline metrics. It
is never supplied to a test-sample classifier, retrieval query or report prompt.
All tool outputs are computed once per query and frozen across policy variants.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from ..agent import DiagnosisAgent
from ..haconfig.settings import Settings
from ..hatools.diagnosis import run_fault_diagnosis
from ..hatools.rag import retrieve_cases
from ..hatools.report import generate_report
from ..hatools.verify import verify_result, POLICY_VERSION
from .baseline import CentroidClassifier, TrainingIndex, extract_features, PREPROCESSING_ID
from .hust_data import load_windows, manifest, SAMPLE_RATE, WINDOW_SIZE

SCHEMA = "bearing-health-v1"
# A declared SIMULATED clock, never claimed to come from the real recordings.
CONTEXT = {"decision_at": "2000-01-02T00:00:00Z", "units": "1",
           "sampling_rate": SAMPLE_RATE, "preprocessing_id": PREPROCESSING_ID}
SCENARIOS = ("native_missing_time", "replay_compatible", "replay_unit_mismatch",
             "replay_rate_mismatch", "replay_preprocessing_mismatch", "replay_future_available",
             "replay_missing_available", "replay_wrong_role", "replay_one_excluded")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def reference_replay(raw: dict, scenario: str) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError("Unknown metadata replay")
    body = deepcopy(raw)
    if scenario == "native_missing_time":
        return body
    base = datetime(2000, 1, 1, tzinfo=timezone.utc)
    for case in body["results"]:
        offset = int(case["case_id"].rsplit(":", 1)[1]) / SAMPLE_RATE
        meta = case["meta"]
        meta.update(window_start=(base + timedelta(seconds=offset)).isoformat(),
                    window_end=(base + timedelta(seconds=offset + WINDOW_SIZE / SAMPLE_RATE)).isoformat(),
                    available_at="2000-01-01T01:00:00Z")
        case["text"] += " SIMULATED clock replay; not acquisition timestamps."
        if scenario == "replay_unit_mismatch":
            meta["units"] = "g"
        elif scenario == "replay_rate_mismatch":
            meta["sampling_rate"] = 64000
        elif scenario == "replay_preprocessing_mismatch":
            meta["preprocessing_id"] = "other-preprocessing"
        elif scenario == "replay_future_available":
            meta["available_at"] = "2000-01-03T00:00:00Z"
        elif scenario == "replay_missing_available":
            meta.pop("available_at")
        elif scenario == "replay_wrong_role":
            meta["evidence_role"] = "current_observation"
    if scenario == "replay_one_excluded":
        body["results"][0]["meta"]["units"] = "g"
    return body


class FrozenTransport:
    """Replay actual computed outputs in-process, with a checked query digest."""
    def __init__(self, signal, classifier, references):
        self.signal_digest = digest(signal)
        self.classifier, self.references = classifier, references

    def __call__(self, method, url, **kwargs):
        if method != "POST" or digest(kwargs["payload"]["signal"]) != self.signal_digest:
            raise ValueError("Frozen tool output used for a different query")
        if url.endswith("/diagnose"):
            return "ok", deepcopy(self.classifier), ""
        if url.endswith("/v1/search/text"):
            return "ok", deepcopy(self.references), ""
        raise ValueError("Unexpected benchmark route")


class LiveTransport:
    """Run the trained NumPy model and real feature index through public adapters.

    In-process transport, no HTTP server or LLM. The ordinary HTTP integration
    tests cover network handling independently.
    """
    def __init__(self, model, index):
        self.model, self.index = model, index
        self.outputs = {}

    def __call__(self, method, url, **kwargs):
        if method != "POST":
            raise ValueError("Unexpected benchmark method")
        features = extract_features(kwargs["payload"]["signal"])
        if url.endswith("/diagnose"):
            result = self.model.predict(features)
            self.outputs["classifier"] = deepcopy(result)
        elif url.endswith("/v1/search/text"):
            result = self.index.search(features, kwargs["payload"]["top_k"])
            self.outputs["references"] = deepcopy(result)
        else:
            raise ValueError("Unexpected benchmark route")
        return "ok", result, ""


def selective_metrics(target, predicted, accepted) -> dict:
    target, predicted, accepted = np.asarray(target), np.asarray(predicted), np.asarray(accepted, dtype=bool)
    if target.ndim != 1 or predicted.shape != target.shape or accepted.shape != target.shape or len(target) == 0:
        raise ValueError("Invalid evaluation arrays")
    count = int(accepted.sum())
    errors = int(np.sum(target[accepted] != predicted[accepted]))
    return {"total": len(target), "accepted": count, "errors": errors,
            "coverage": count / len(target), "selective_error_rate": errors / count if count else None}


def classifier_metrics(target, predicted) -> dict:
    target, predicted = np.asarray(target), np.asarray(predicted)
    confusion = np.zeros((4, 4), dtype=int)
    for true, guess in zip(target, predicted):
        confusion[int(true), int(guess)] += 1
    recalls, f1 = [], []
    for label in range(4):
        tp = confusion[label, label]
        support = confusion[label].sum()
        recalls.append(float(tp / support) if support else 0.0)
        denom = confusion[label].sum() + confusion[:, label].sum()
        f1.append(float(2 * tp / denom) if denom else 0.0)
    healthy = target == 3
    faults = ~healthy
    return {"accuracy": float(np.mean(target == predicted)),
            "balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(np.mean(f1)),
            "per_class_recall": {str(i): r for i, r in enumerate(recalls)},
            "confusion_matrix_true_rows": confusion.tolist(),
            "healthy_false_alarm_rate": float(np.mean(predicted[healthy] != 3)) if healthy.any() else None,
            "fault_predicted_healthy_rate": float(np.mean(predicted[faults] == 3)) if faults.any() else None}


def confidence_mask_at_count(scores, sample_ids, count):
    """Label-free ranking; hashed IDs break score ties, no test-label tuning."""
    scores = np.asarray(scores)
    if type(count) is not int or not 0 <= count <= len(scores) or len(sample_ids) != len(scores):
        raise ValueError("Invalid matched coverage")
    tie = np.array([hashlib.sha256(s.encode()).hexdigest() for s in sample_ids])
    indices = np.lexsort((tie, -scores))
    selected = np.zeros(len(scores), dtype=bool)
    selected[indices[:count]] = True
    return selected


def evaluate_policy(fd, rag, cfg, *, strict: bool, simulated: bool):
    context = CONTEXT if strict else None
    tick = time.perf_counter()
    verified = verify_result(fd, rag, input_is_synthetic=False, label_schema=SCHEMA,
                             evidence_context=context)
    report = generate_report(diagnosis_result=fd, rag_result=rag, verify_data=verified,
                             signal_result={"data": {"quality": {
                                 "points": cfg.signal_window_size, "sampling_rate": cfg.sampling_rate}}},
                             sample_info={"input_is_synthetic": False,
                                          "reference_clock": "SIMULATED" if simulated else "UNAVAILABLE"},
                             write_file=False, label_schema=SCHEMA, evidence_context=context)
    if report["status"] == "error":
        raise RuntimeError("Benchmark report failed")
    actual = report["data"]
    return {"effective_strength": actual["effective_strength"],
            "accepted": actual["effective_strength"] in ("moderate", "strong"),
            "support_level": actual["support_level"],
            "counted_references": verified["data"]["retrieval_candidate_count"],
            "policy_and_report_ms": (time.perf_counter() - tick) * 1000}


def run_benchmark(cache_dir: str | Path, output_dir: str | Path, *, download=False) -> dict:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    signals, rows = load_windows(cache_dir, download=download)
    train = [i for i, row in enumerate(rows) if row["split"] == "train"]
    evaluation = [i for i, row in enumerate(rows) if row["split"] != "train"]
    groups = {split: sorted({r["bearing_type"] for r in rows if r["split"] == split})
              for split in ("train", "validation", "test")}
    if any(set(groups[a]) & set(groups[b]) for a, b in (("train", "test"), ("train", "validation"), ("validation", "test"))):
        raise ValueError("Bearing groups overlap across splits")
    tick = time.perf_counter()
    features = np.stack([extract_features(signal) for signal in signals])
    feature_seconds = time.perf_counter() - tick
    training_rows = [rows[i] for i in train]
    model = CentroidClassifier().fit(features[train], [r["label"] for r in training_rows], training_rows)
    index = TrainingIndex(model, features[train], training_rows)
    _write_json(folder / "model.json", model.export())
    _write_json(folder / "split_manifest.json", rows)
    _write_json(folder / "dataset_manifest.json", manifest())
    cfg = Settings(signal_window_size=WINDOW_SIZE, sampling_rate=SAMPLE_RATE, label_schema=SCHEMA,
                   reports_dir=str(folder / "reports"), logs_dir=str(folder / "logs"))
    results, costs = [], []
    frozen_path = folder / "frozen_tools.jsonl"
    with frozen_path.open("w", encoding="utf-8") as frozen:
        for idx in evaluation:
            signal = signals[idx].tolist()
            tick = time.perf_counter()
            body_fd = model.predict(features[idx])
            body_rag = index.search(features[idx])
            costs.append((time.perf_counter() - tick) * 1000)
            transport = FrozenTransport(signal, body_fd, body_rag)
            fd = run_fault_diagnosis(signal, transport=transport, config=cfg)
            if fd["status"] not in ("ok", "degraded"):
                raise RuntimeError("Real classifier violated its declared contract")
            if any(case["meta"]["record_id"] not in model.training_records for case in body_rag["results"]):
                raise RuntimeError("Non-training record retrieved")
            entry = {**rows[idx], "predicted": body_fd["diagnosis"]["predicted_class"],
                     "confidence": body_fd["diagnosis"]["confidence"], "scenarios": {}}
            frozen.write(json.dumps({"sample_id": rows[idx]["sample_id"], "signal_digest": digest(signal),
                                     "classifier": body_fd, "references": body_rag},
                                    ensure_ascii=False, allow_nan=False) + "\n")
            for scenario in SCENARIOS:
                body = reference_replay(body_rag, scenario)
                transport.references = body
                rag = retrieve_cases(signal, top_k=5, transport=transport, config=cfg)
                if rag["status"] not in ("ok", "degraded"):
                    raise RuntimeError("Real retrieval violated its declared contract")
                frozen_pair = digest({"classifier": fd["data"], "references": rag["data"]})
                old = evaluate_policy(fd, rag, cfg, strict=False, simulated=scenario != "native_missing_time")
                new = evaluate_policy(fd, rag, cfg, strict=True, simulated=scenario != "native_missing_time")
                if digest({"classifier": fd["data"], "references": rag["data"]}) != frozen_pair:
                    raise RuntimeError("Policy mutated frozen tool outputs")
                entry["scenarios"][scenario] = {"tool_pair_digest": frozen_pair, "legacy": old, "strict": new}
            results.append(entry)
    with (folder / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for entry in results:
            handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    metrics = {}
    for split in ("validation", "test"):
        subset = [r for r in results if r["split"] == split]
        y = np.array([r["label"] for r in subset])
        p = np.array([r["predicted"] for r in subset])
        scores = np.array([r["confidence"] for r in subset])
        ids = [r["sample_id"] for r in subset]
        modes = {"classifier_direct": np.ones(len(y), dtype=bool),
                 "schema_only": np.ones(len(y), dtype=bool),
                 "confidence_0.5": scores >= 0.5, "confidence_0.8": scores >= 0.8}
        base = {name: selective_metrics(y, p, mask) for name, mask in modes.items()}
        comparisons = {}
        for scenario in SCENARIOS:
            compared = {}
            for policy in ("legacy", "strict"):
                mask = np.array([r["scenarios"][scenario][policy]["accepted"] for r in subset])
                matched = confidence_mask_at_count(scores, ids, int(mask.sum()))
                outcome = selective_metrics(y, p, mask)
                outcome["confidence_ranking_same_count"] = selective_metrics(y, p, matched)
                eligible = np.array([r["scenarios"][scenario]["strict"]["counted_references"] for r in subset])
                outcome["accepted_with_zero_eligible_references"] = int(np.sum(mask & (eligible == 0)))
                outcome["median_policy_and_report_ms"] = float(np.median([
                    r["scenarios"][scenario][policy]["policy_and_report_ms"] for r in subset]))
                compared[policy] = outcome
            legacy = np.array([r["scenarios"][scenario]["legacy"]["accepted"] for r in subset])
            strict = np.array([r["scenarios"][scenario]["strict"]["accepted"] for r in subset])
            compared["new_rejections_relative_to_legacy"] = int(np.sum(legacy & ~strict))
            comparisons[scenario] = compared
        record_summary = []
        for record_id in sorted({r["record_id"] for r in subset}):
            items = [r for r in subset if r["record_id"] == record_id]
            counts = np.bincount([r["predicted"] for r in items], minlength=4)
            record_summary.append({"record_id": record_id, "label": items[0]["label"],
                                   "majority_prediction": int(counts.argmax()),
                                   "window_accuracy": float(np.mean([r["predicted"] == r["label"] for r in items]))})
        metrics[split] = {"windows": len(subset), "records": len(record_summary),
                          "classifier": classifier_metrics(y, p), "baselines": base,
                          "compatible_replay_support_audit": {
                              level: {**selective_metrics(y, p, np.array([
                                  r["scenarios"]["replay_compatible"]["legacy"]["support_level"] == level
                                  for r in subset])),
                                  "healthy_misclassified": sum(
                                      r["label"] == 3 and r["predicted"] != 3
                                      and r["scenarios"]["replay_compatible"]["legacy"]["support_level"] == level
                                      for r in subset)}
                              for level in ("high", "medium", "low", "unavailable")},
                          "policy_comparisons": comparisons, "per_record": record_summary,
                          "record_majority_accuracy": float(np.mean([
                              r["label"] == r["majority_prediction"] for r in record_summary])),
                          "risk_coverage": [selective_metrics(y, p, confidence_mask_at_count(
                              scores, ids, round(len(y) * fraction))) for fraction in (0, .1, .25, .5, .75, 1)]}
    # Run the complete public workflow with live computations, one test query per
    # true class. This stratifies an integration smoke test, not the accuracy set.
    smoke = []
    for label in range(4):
        idx = next(i for i in evaluation if rows[i]["split"] == "test" and rows[i]["label"] == label)
        pair = []
        for strict in (False, True):
            live = LiveTransport(model, index)
            result = DiagnosisAgent(transport=live, config=cfg, persist_trace=False).run(
                signals[idx].tolist(), sample_info={"sample_id": rows[idx]["sample_id"]},
                input_is_synthetic=False, skip_services_check=True,
                evidence_context=CONTEXT if strict else None)
            if result["terminal_state"] in ("FAILED", "ASKING_USER"):
                raise RuntimeError("Full real-model workflow smoke failed")
            pair.append({"strict": strict, "raw_tools": live.outputs, "result": result})
        if pair[0]["raw_tools"] != pair[1]["raw_tools"]:
            raise RuntimeError("Live workflow variants received different predictions or references")
        smoke.append({"sample_id": rows[idx]["sample_id"], "variants": pair})
    _write_json(folder / "full_workflow_smoke.json", smoke)
    for scene in SCENARIOS:
        if scene not in ("replay_compatible", "replay_one_excluded"):
            if any(r["scenarios"][scene]["strict"]["accepted"] for r in results):
                raise RuntimeError("No eligible references must cap the result below moderate")
    if any(r["scenarios"]["replay_compatible"]["legacy"]["accepted"] !=
           r["scenarios"]["replay_compatible"]["strict"]["accepted"] for r in results):
        raise RuntimeError("Compatible replay changed acceptance")
    import scipy
    summary = {"scope": "real measured signals and trained classifier; reference clock replay separately simulated",
               "policy_version": POLICY_VERSION, "label_schema": SCHEMA,
               "dataset_doi": manifest()["doi"], "dataset_license": "CC-BY-4.0",
               "split_groups": groups, "train_records": len(model.training_records), "train_windows": len(train),
               "window_size": WINDOW_SIZE, "windows_per_record": 32, "sampling_rate": SAMPLE_RATE,
               "features": features.shape[1], "tuning": "none; validation descriptive, not used to fit or tune",
               "source_units": None, "processed_units": "1 (demean/RMS dimensionless)",
               "native_acquisition_timestamps": None, "clock_replay_context_SIMULATED": CONTEXT,
               "frozen_pair_checks": len(results) * len(SCENARIOS),
               "full_workflow_live_queries": 8, "full_workflow_matching_pairs": 4,
               "feature_extraction_seconds": feature_seconds,
               "median_cached_feature_predict_and_retrieve_ms": float(np.median(costs)),
               "elapsed_seconds": time.perf_counter() - started,
               "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                           "platform": platform.platform()},
               "limitations": ["One held-out bearing type per evaluation split; laboratory artificial faults only",
                               "Windows from the same recording are correlated; window count is not independent sample count",
                               "No healthy/unknown equivalence, open-set or combined-fault validation",
                               "Classifier scores uncalibrated; no accuracy improvement follows from metadata rejection",
                               "Virtual-clock faults are controlled software tests, not observed deployment incidents",
                               "No cross-device generalization beyond this fixed subset; no LLM or real HTTP model server"],
               "metrics": metrics}
    _write_json(folder / "metrics.json", summary)
    _write_json(folder / "artifact_checksums.json", {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.iterdir()) if path.is_file()})
    return summary
