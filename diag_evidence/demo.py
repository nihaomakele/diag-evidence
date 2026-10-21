"""Self-authored synthetic fault injections. These are contract tests, not accuracy data."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any
from uuid import uuid4
import json

from .agent import DiagnosisAgent
from .haconfig.settings import Settings


def synthetic_signal() -> list[float]:
    """1024 finite samples, mathematical waveform, no measured device data."""
    return [math.sin(2 * math.pi * 37 * i / 1024) + 0.1 * math.cos(2 * math.pi * 91 * i / 1024)
            for i in range(1024)]


def classifier(confidence: Any = 0.81) -> dict:
    return {"success": True, "diagnosis": {
        "predicted_class": 2, "confidence": confidence,
        "predicted_fault": {"name": "滚动体故障", "name_en": "Ball Fault"},
        "class_probabilities": {},
    }, "model_revision": "synthetic-stub-v1"}


def candidates() -> dict:
    return {"results": [
        {"rank": i + 1, "score": score, "score_type": "synthetic_raw_score",
         "score_is_normalized": False, "text": f"自主构造候选 {i + 1}，仅用于接口测试。",
         "meta": {"record_id": f"synthetic-record-{i + 1}", "fault_type": "滚动体故障",
                  "source": "synthetic-fixture-v1"}}
        for i, score in enumerate([0.73, 0.50, 0.40])], "num_results": 3}


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    expected_state: str
    expected_support: str | None
    expected_cap: str


SCENARIOS = (
    Scenario("consistent", "合法分类与一致候选，规则全部通过", "COMPLETED", "high", "strong"),
    Scenario("mixed_candidates", "候选标签多样，不声明语义矛盾", "DEGRADED", "medium", "moderate"),
    Scenario("low_confidence", "合法但低置信度分类", "DEGRADED", "low", "weak"),
    Scenario("no_results", "检索服务正常但无结果", "DEGRADED", "medium", "moderate"),
    Scenario("service_down", "分类服务不可达", "DEGRADED", "low", "no_conclusion"),
    Scenario("missing_class", "分类响应缺少类别字段", "DEGRADED", "low", "no_conclusion"),
    Scenario("nan_confidence", "分类响应包含NaN置信度", "DEGRADED", "low", "no_conclusion"),
    Scenario("out_of_range", "分类置信度超出[0,1]", "DEGRADED", "low", "no_conclusion"),
    Scenario("invalid_signal", "非有限输入在任何网络调用前被拒绝", "ASKING_USER", None, "no_conclusion"),
    Scenario("duplicate_results", "相同候选重复不能增加有效数量", "DEGRADED", "medium", "moderate"),
    Scenario("label_leakage", "提示词命中真实标签启发式规则", "DEGRADED", "low", "weak"),
    Scenario("missing_prompt", "声明使用LLM但未提供待检提示词", "DEGRADED", "low", "weak"),
)


class SyntheticTransport:
    """In-memory HTTP-shaped fixture; never creates a socket or loads a model."""
    def __init__(self, scenario: str):
        self.scenario = scenario
        self.calls: list[dict] = []

    def __call__(self, method, url, **kwargs):
        self.calls.append({"method": method, "service": kwargs.get("service")})
        if method == "GET":
            return "ok", {"model_loaded": True, "index_loaded": True, "num_texts": 12}, ""
        if "/diagnose" in url:
            if self.scenario == "service_down":
                return "conn", None, "synthetic service unavailable"
            body = classifier()
            if self.scenario == "low_confidence":
                body = classifier(0.2)
            elif self.scenario == "nan_confidence":
                body = classifier(float("nan"))
            elif self.scenario == "out_of_range":
                body = classifier(1.5)
            elif self.scenario == "missing_class":
                del body["diagnosis"]["predicted_class"]
            return "ok", body, ""
        if "/v1/search/text" in url:
            body = candidates()
            if self.scenario == "mixed_candidates":
                body["results"][1]["meta"]["fault_type"] = "内圈故障"
            elif self.scenario == "no_results":
                body = {"results": [], "num_results": 0}
            elif self.scenario == "duplicate_results":
                body["results"] = [deepcopy(body["results"][0]) for _ in range(3)]
            return "ok", body, ""
        raise AssertionError("Unexpected synthetic route")


def run_scenario(name: str, *, output_dir: str | Path | None = None,
                 write: bool = False) -> tuple[Scenario, dict, list]:
    scenario = next((s for s in SCENARIOS if s.name == name), None)
    if scenario is None:
        raise ValueError(f"Unknown scenario: {name}")
    cfg = Settings.from_env()
    if output_dir is not None:
        folder = Path(output_dir)
        cfg = replace(cfg, logs_dir=str(folder / "logs"), reports_dir=str(folder / "reports"))
    transport = SyntheticTransport(name)
    values = synthetic_signal()
    if name == "invalid_signal":
        values[0] = float("nan")
    result = DiagnosisAgent(transport=transport, config=cfg, persist_trace=write).run(
        values, sample_info={"source_file": "self-authored mathematical fixture"},
        skip_services_check=True, input_is_synthetic=True,
        llm_prompt_text="真实标签=滚动体故障" if name == "label_leakage" else None,
        llm_expected=name in {"missing_prompt", "label_leakage"})
    return scenario, result, transport.calls


def assess(scenario: Scenario, result: dict, calls: list) -> dict:
    """Check declared fields, strength bounds and literal report agreement.

    Matching Markdown strings links the returned text to the report payload;
    it does not evaluate that text's meaning, correctness or diagnostic accuracy.
    """
    expected = {"terminal_state": scenario.expected_state, "support_level": scenario.expected_support,
                "max_allowed_strength": scenario.expected_cap}
    actual = {key: result.get(key) for key in expected}
    no_network_on_invalid = scenario.name != "invalid_signal" or not calls
    strength_order = {"no_conclusion": 0, "weak": 1, "moderate": 2, "strong": 3}
    effective = result.get("effective_strength")
    cap = result.get("max_allowed_strength")
    strength_within_cap = (isinstance(effective, str) and effective in strength_order
                           and isinstance(cap, str) and cap in strength_order
                           and strength_order[effective] <= strength_order[cap])
    outcomes = result.get("outcomes")
    report = outcomes.get("report") if isinstance(outcomes, dict) else None
    if report is None:
        report_consistent = (result.get("terminal_state") == "ASKING_USER"
                             and result.get("report_markdown") is None)
    else:
        report_data = report.get("data") if isinstance(report, dict) else None
        report_consistent = (isinstance(report_data, dict)
                             and report_data.get("effective_strength") == effective
                             and report_data.get("max_allowed_strength") == cap
                             and report_data.get("markdown") == result.get("report_markdown"))
    passed = (actual == expected and no_network_on_invalid and not result["persistence_errors"]
              and strength_within_cap and report_consistent)
    return {"scenario": scenario.name, "description": scenario.description,
            "passed": passed, "expected": expected, "actual": actual,
            "transport_calls": len(calls), "run_id": result["run_id"],
            "strength_within_cap": strength_within_cap,
            "report_strength_consistent": report_consistent,
            "effective_strength": result.get("effective_strength"),
            "report_path": result["report_path"], "errors": result["errors"]}


def evaluate(*, output_dir: str | Path | None = None, write: bool = False,
             names: list[str] | None = None) -> dict:
    rows = []
    for name in names or [s.name for s in SCENARIOS]:
        scenario, result, calls = run_scenario(name, output_dir=output_dir, write=write)
        rows.append(assess(scenario, result, calls))
    data = {"schema_version": "1.0", "suite": "synthetic-contract-v1",
            "scope": "工程契约与失败处理；不含真实诊断标签，不评价诊断准确率。",
            "passed": all(row["passed"] for row in rows), "total": len(rows),
            "matched_expectations": sum(row["passed"] for row in rows), "scenarios": rows}
    if write:
        folder = Path(output_dir or "artifacts")
        folder.mkdir(parents=True, exist_ok=True)
        report_id = uuid4().hex
        path = folder / f"evaluation_{report_id}.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        markdown = folder / f"evaluation_{report_id}.md"
        lines = ["# 合成契约场景验证\n\n", data["scope"] + "\n\n",
                 "| 场景 | 匹配预期 | 终态 | 规则支持 | 强度上限 |\n",
                 "|---|---|---|---|---|\n"]
        for row in rows:
            actual = row["actual"]
            lines.append(f"| {row['scenario']} | {row['passed']} | {actual['terminal_state']} | "
                         f"{actual['support_level']} | {actual['max_allowed_strength']} |\n")
        with markdown.open("x", encoding="utf-8") as handle:
            handle.write("".join(lines))
        data["evaluation_path"] = str(path)
        data["evaluation_markdown_path"] = str(markdown)
    return data
