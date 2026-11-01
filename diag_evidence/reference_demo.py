"""Deterministic applicability scenarios; no real signals or model evaluation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from uuid import uuid4

from .agent import DiagnosisAgent
from .demo import SyntheticTransport, synthetic_signal
from .haconfig.settings import Settings

CONTEXT = {"decision_at": "2026-11-01T10:00:00+08:00", "units": "g",
           "sampling_rate": 50000, "preprocessing_id": "synthetic-preprocessing-v1"}
REFERENCE_META = {"evidence_role": "historical_reference", "asset_id": "synthetic-other-asset",
                  "units": "g", "sampling_rate": 50000, "preprocessing_id": "synthetic-preprocessing-v1",
                  "window_start": "2026-10-31T09:00:00+08:00",
                  "window_end": "2026-10-31T09:00:01+08:00",
                  "available_at": "2026-10-31T10:00:00+08:00"}
SCENARIOS = (
    ("compatible_history", {}, (), False, "high", "moderate", 3),
    ("unit_mismatch", {"units": "m/s2"}, (), False, "low", "weak", 0),
    ("sampling_rate_mismatch", {"sampling_rate": 64000}, (), False, "low", "weak", 0),
    ("preprocessing_mismatch", {"preprocessing_id": "other"}, (), False, "low", "weak", 0),
    ("unavailable_at_cutoff", {"available_at": "2026-11-02T10:00:00+08:00"}, (), False, "low", "weak", 0),
    ("missing_availability", {}, ("available_at",), False, "low", "weak", 0),
    ("unsupported_role", {"evidence_role": "current_observation"}, (), False, "low", "weak", 0),
    ("valid_with_excluded_candidate", {}, (), True, "high", "moderate", 3),
)


class ReferenceTransport(SyntheticTransport):
    def __init__(self, changes, missing, extra):
        super().__init__("consistent")
        self.changes, self.missing, self.extra = changes, missing, extra

    def __call__(self, method, url, **kwargs):
        kind, body, error = super().__call__(method, url, **kwargs)
        if "/v1/search/text" in url:
            for case in body["results"]:
                case["meta"].update(REFERENCE_META)
                case["meta"].update(self.changes)
                for key in self.missing:
                    case["meta"].pop(key, None)
            if self.extra:
                case = deepcopy(body["results"][0])
                case["meta"].update(units="m/s2", fault_type="内圈故障", record_id="synthetic-extra")
                case["text"] = "自主构造的不兼容候选，仅用于工程测试。"
                body["results"].append(case)
            body["num_results"] = len(body["results"])
        return kind, body, error


def evaluate_references(output_dir: str | Path, write: bool = False) -> dict:
    folder = Path(output_dir) / ("reference-evaluation-" + uuid4().hex)
    results = []
    for name, changes, missing, extra, support, strength, count in SCENARIOS:
        case_folder = folder / name
        config = replace(Settings.from_env(), logs_dir=str(case_folder / "logs"),
                         reports_dir=str(case_folder / "reports"))
        result = DiagnosisAgent(config=config, persist_trace=write,
                                transport=ReferenceTransport(changes, missing, extra)).run(
            synthetic_signal(), skip_services_check=True, evidence_context=CONTEXT,
            input_is_synthetic=True)
        checked = result["outcomes"]["verify"]["data"]
        expected = {"support_level": support, "effective_strength": strength,
                    "eligible_references": count}
        actual = {"support_level": result["support_level"], "effective_strength": result["effective_strength"],
                  "eligible_references": checked["retrieval_candidate_count"]}
        report_data = result["outcomes"]["report"]["data"]
        order = {"no_conclusion": 0, "weak": 1, "moderate": 2, "strong": 3}
        report_consistent = (report_data["effective_strength"] == actual["effective_strength"]
                             and result["report_markdown"] == report_data["markdown"]
                             and order[actual["effective_strength"]] <= order[result["max_allowed_strength"]])
        passed = (actual == expected and report_consistent
                  and result["terminal_state"] == ("COMPLETED" if support == "high" else "DEGRADED"))
        results.append({"scenario": name, "passed": passed, "expected": expected, "actual": actual,
                        "report_consistent": report_consistent, "result": result})
    summary = {"scope": "synthetic reference applicability behavior; not diagnostic accuracy",
               "total": len(results), "matched_expectations": sum(x["passed"] for x in results),
               "passed": all(x["passed"] for x in results), "scenarios": results}
    if write:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "evaluation.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        summary["evaluation_path"] = str(path)
    return summary
