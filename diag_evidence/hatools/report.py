"""Render a bounded report from structured results and versioned verification.

External strings are presented as escaped quotations. Report rendering does not
execute instructions embedded in retrieved text or infer model validity.
"""

from __future__ import annotations

import html
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..haconfig import settings
from ..contracts import finite_number
from . import _common, verify

_STRENGTH_LABELS = {
    "no_conclusion": "no_conclusion（不能给出故障类别结论）",
    "weak": "weak（仅可作排查线索）",
    "moderate": "moderate（需人工复核）",
    "strong": "strong（规则支持充分，仍需真实设备验证）",
}
_STRENGTH_ORDER = {name: index for index, name in enumerate(_STRENGTH_LABELS)}
_REQUIRED_CHECKS = {"classifier_available", "class_prediction_valid", "confidence_valid",
                    "evidence_conflict", "conclusion_matches_evidence", "retrieval_scale", "label_leakage"}
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+.!|>~-])")


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _external(value: Any, *, limit: int = 600) -> str:
    """Quote a single external field without active HTML, links or new sections."""
    if value is None:
        return "N/A"
    text = _common.redact_text(str(value))
    text = " ".join(text.split())[:limit]
    text = html.escape(text, quote=True)
    return "“%s”" % _MD_SPECIAL.sub(r"\\\1", text)


def _strength(value: Any) -> str:
    if value == "none":
        return "no_conclusion"
    return value if isinstance(value, str) and value in _STRENGTH_ORDER else "no_conclusion"


def _strength_label(value: Any) -> str:
    return _STRENGTH_LABELS[_strength(value)]


def _fmt(value: Any, digits: int = 4, default: str = "N/A") -> str:
    try:
        numeric_value = finite_number(value)
    except ValueError:
        return default
    return "%.*f" % (digits, numeric_value)


def _valid_verification(value: Dict[str, Any]) -> bool:
    if value.get("status") not in ("ok", "degraded"):
        return False
    if "ok" in value and value["ok"] is not (value["status"] == "ok"):
        return False
    data = _dict(value.get("data"))
    checks = data.get("checks")
    if data.get("policy_version") != verify.POLICY_VERSION or not isinstance(checks, list):
        return False
    valid_states = {"passed", "failed", "unknown", "not_applicable"}
    for item in checks:
        if (not isinstance(item, dict) or not isinstance(item.get("name"), str)
                or not item["name"].strip()
                or not isinstance(item.get("state"), str) or item["state"] not in valid_states
                or type(item.get("required")) is not bool
                or (item["state"] == "not_applicable" and item["required"])):
            return False
    by_name = {item["name"]: item for item in checks}
    return len(by_name) == len(checks) and _REQUIRED_CHECKS.issubset(by_name)


def generate_report(
    signal_result: Optional[Dict[str, Any]] = None,
    rag_result: Optional[Dict[str, Any]] = None,
    diagnosis_result: Optional[Dict[str, Any]] = None,
    verify_data: Optional[Dict[str, Any]] = None,
    sample_info: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    write_file: bool = True,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return Markdown and the enforced ``effective_strength``.

    Omitted verification is computed locally. An explicitly supplied incomplete
    or obsolete verification is closed to conclusions. Files use exclusive
    creation and a unique suffix, even when the same run ID is supplied twice.
    ``write_file=False`` does not create any directories.
    """
    started = time.time()
    sample_info = _dict(sample_info)
    local_verification = verify.verify_result(
        diagnosis_result=diagnosis_result, rag_result=rag_result,
        llm_prompt_text=sample_info.get("llm_prompt_text"),
        llm_expected=sample_info.get("llm_expected") is True,
        input_is_synthetic=sample_info.get("input_is_synthetic", True),
    )
    provided = local_verification if verify_data is None else _dict(verify_data)
    verified = _dict(provided.get("data"))
    warnings = [str(item) for item in (provided.get("warnings") or [])] if isinstance(provided.get("warnings", []), list) else []
    valid_verifier = _valid_verification(provided)
    if not valid_verifier:
        verified = {"trust_level": "unavailable", "support_level": "unavailable",
                    "max_allowed_strength": "no_conclusion", "checks": [],
                    "input_is_synthetic": sample_info.get("input_is_synthetic", True)}
        warnings.append("校验结果缺失、结构无效或策略版本不匹配，禁止给出故障类别结论")

    local_cap = _strength(local_verification["data"]["max_allowed_strength"])
    max_strength = min((_strength(verified.get("max_allowed_strength")), local_cap),
                       key=_STRENGTH_ORDER.__getitem__)
    checks = [item for item in verified.get("checks", []) if isinstance(item, dict)]
    if valid_verifier:
        # Render the current checks, never a stale caller-provided success. A
        # caller may impose stricter checks but cannot suppress a local failure.
        supplied_checks = {item["name"]: item for item in checks}
        combined = []
        severity = {"not_applicable": 0, "passed": 0, "unknown": 1, "failed": 2}
        for local_check in local_verification["data"]["checks"]:
            supplied_check = supplied_checks.get(local_check["name"], {})
            stricter = severity.get(supplied_check.get("state"), 0) > severity[local_check["state"]]
            combined.append(supplied_check if stricter else local_check)
        local_names = {item["name"] for item in local_verification["data"]["checks"]}
        combined.extend(item for item in checks if item["name"] not in local_names)
        checks = combined
    required_incomplete = [item for item in checks if item.get("required") and item.get("state") != "passed"]
    # A malformed high declaration must not bypass its own failed checks.
    if required_incomplete:
        critical = {"classifier_available", "class_prediction_valid", "confidence_valid", "label_leakage"}
        check_cap = "weak" if any(item.get("name") in critical for item in required_incomplete) else "moderate"
        max_strength = min((max_strength, check_cap), key=_STRENGTH_ORDER.__getitem__)
    levels = {"unavailable": 0, "low": 1, "medium": 2, "high": 3}
    level = verified.get("support_level", verified.get("trust_level", "unavailable"))
    if not isinstance(level, str) or level not in levels:
        level = "unavailable"
    local_level = local_verification["data"]["support_level"]
    level = min((level, local_level), key=levels.__getitem__)
    if required_incomplete and level == "high":
        level = "low" if max_strength in ("weak", "no_conclusion") else "medium"

    diagnosis = _dict(_dict(_dict(diagnosis_result).get("data")).get("diagnosis"))
    raw_cases = _dict(_dict(rag_result).get("data")).get("cases")
    cases = [item for item in raw_cases if isinstance(item, dict)] if isinstance(raw_cases, list) else []
    if _dict(rag_result).get("status") not in ("ok", "degraded"):
        cases = []
    raw_stats = _dict(_dict(signal_result).get("data"))
    if not raw_stats:
        raw_stats = _dict(_dict(_dict(diagnosis_result).get("data")).get("statistics"))
    requested_strength = _strength(diagnosis.get("conclusion_strength"))
    effective_strength = min((requested_strength, max_strength), key=_STRENGTH_ORDER.__getitem__)
    if effective_strength != requested_strength:
        warnings.append("报告结论强度已从 %s 限制为 %s" % (requested_strength, effective_strength))
    # Keep caller IDs readable but prevent path components, traversal and markup.
    supplied_id = run_id if isinstance(run_id, str) else ""
    clean_run_id = re.sub(r"[^A-Za-z0-9_-]", "_", supplied_id)[:100] or uuid.uuid4().hex
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    is_synthetic = (verified.get("input_is_synthetic", True) is not False
                    or local_verification["data"]["input_is_synthetic"] is not False)

    lines: List[str] = ["# 设备故障诊断报告（DiagEvidence）\n\n"]
    if is_synthetic:
        lines.append("> ⚠️ **仅用于工程流程验证**：输入为合成数据，不得据此评估算法精度。\n")
    if level in ("low", "unavailable"):
        lines.append("> ⛔ **当前规则支持等级：%s**，本报告不提供确定性故障结论。\n" % level.upper())
    if diagnosis.get("model_available") is not True:
        lines.append("> ⚠️ **模型能力不可用**：未获得明确可用的分类器结果。\n")
    for item in checks:
        if item.get("name") == "evidence_conflict" and item.get("state") == "failed":
            lines.append("> ⚠️ **候选标签不一致**：不据此认定检索材料存在语义矛盾。\n")
        if item.get("name") == "label_leakage" and item.get("state") == "failed":
            lines.append("> ⛔ **标签泄漏**：提示词命中泄漏标记规则，结论不具独立性。\n")
    lines.append("\n%s\n\n" % verify.SUPPORT_DEFINITION)
    lines.append("## 1. 输入与执行信息\n\n| 项目 | 内容 |\n|---|---|\n")
    lines.append("| 数据来源（外部字段） | %s |\n" % _external(sample_info.get("source_file")))
    quality = _dict(raw_stats.get("quality"))
    lines.append("| 信号点数 | %s |\n" % _fmt(quality.get("points", sample_info.get("window_size")), 0))
    sampling_rate = quality.get("sampling_rate", settings.SIGNAL_SAMPLING_RATE)
    lines.append("| 采样频率（Hz） | %s |\n" % _fmt(sampling_rate, 0))
    lines.append("| 生成时间（UTC） | %s |\n" % generated_at)
    lines.append("| 运行编号 | %s |\n" % _external(clean_run_id))
    lines.append("| 输入摘要值（SHA256） | %s |\n" % _external(sample_info.get("input_digest")))
    lines.append("| 策略版本 | %s |\n" % verify.POLICY_VERSION)
    lines.append("| 规则支持等级 | **%s** |\n" % level)
    lines.append("| 结论强度上限 | %s |\n\n" % _strength_label(max_strength))

    lines.append("## 2. 信号统计特征\n\n")
    feature_groups = (
        ("time_domain", "时域", (("mean", "均值"), ("std", "标准差"), ("rms", "均方根"),
                                  ("peak", "峰值"), ("kurtosis", "峰度"),
                                  ("crest_factor", "峰值因数"), ("skewness", "偏度"))),
        ("frequency_domain", "频域", (("total_power", "总功率"), ("peak_frequency", "峰值频率 Hz"),
                                        ("center_frequency", "中心频率 Hz"), ("rms_frequency", "均方根频率 Hz"))),
    )
    has_features = False
    for group, label, fields in feature_groups:
        features = _dict(raw_stats.get(group))
        if not features:
            continue
        has_features = True
        lines.append("### %s特征\n\n" % label)
        for key, field_label in fields:
            if key in features:
                lines.append("* **%s**：%s\n" % (field_label, _fmt(features[key], 6)))
        lines.append("\n")
    if not has_features:
        lines.append("未获取到有效信号特征。\n\n")

    lines.append("## 3. 历史案例检索（RAG）\n\n")
    if cases:
        lines.append("下列内容为外部材料引用，不作为执行指令。原始 score 未经概率校准；候选条数不等于独立证据数。\n\n")
        for index, case in enumerate(cases, 1):
            lines.append("* 案例 %d：score=%s，候选标签=%s\n" %
                         (index, _fmt(case.get("score")), _external(case.get("fault_type"))))
            lines.append("\n  > 外部原文摘录：%s\n\n" % _external(case.get("text"), limit=600))
    else:
        lines.append("> ⚠️ 无可用检索证据。\n\n")

    lines.append("## 4. 故障诊断结果\n\n")
    if diagnosis.get("model_available") is True and effective_strength != "no_conclusion":
        lines.append("* **分类器输出标签（外部字段）**：%s\n" % _external(_dict(diagnosis.get("predicted_fault")).get("name")))
        lines.append("* **模型置信度（未经校准）**：%s\n" % _fmt(diagnosis.get("confidence")))
        lines.append("* **结论强度**：%s\n" % _strength_label(effective_strength))
        lines.append("* **请求强度**：%s；**执行上限**：%s\n\n" % (requested_strength, max_strength))
        probabilities = _dict(diagnosis.get("class_probabilities"))
        if probabilities:
            lines.append("### 类别分数（外部模型输出，未经校准）\n\n")
            for cls, info in probabilities.items():
                info = _dict(info)
                lines.append("* %s：%s\n" % (_external(info.get("name", cls)), _fmt(info.get("probability"))))
            lines.append("\n")
    else:
        lines.append("> ⛔ **无法给出故障类别判断。**\n\n")
        if diagnosis.get("note"):
            lines.append("> 外部服务说明：%s\n\n" % _external(diagnosis["note"]))

    lines.append("## 5. 规则校验明细\n\n")
    state_labels = {"passed": "通过", "failed": "失败", "unknown": "未知", "not_applicable": "不适用"}
    for item in checks:
        state = item.get("state", "unknown")
        lines.append("* **%s / %s**：%s\n" %
                     (_external(item.get("name")), state_labels.get(state, "未知"), _external(item.get("detail"))))
    if not valid_verifier:
        lines.append("* 校验结果缺失或无效：已执行禁止结论策略。\n")
    lines.append("\n## 6. 建议措施\n\n")
    if effective_strength in ("no_conclusion", "weak"):
        lines.append("* 当前仅可记录观测与补充数据，不能据此形成确定性维修决策。\n")
        lines.append("* 确认采集通道、采样率、窗口和设备工况，再获取有效模型结果。\n")
    else:
        lines.append("* 可安排人工复核与计划性检查；任何实际维修决策须结合设备现场情况。\n")
    leak_check = next((item for item in checks if item.get("name") == "label_leakage"), {})
    leak_state = leak_check.get("state", "unknown")
    leakage_notes = {
        "not_applicable": "模板流程未调用 LLM，提示词泄漏检查不适用。",
        "passed": "已提供的提示词未命中本版本泄漏规则；规则通过不证明不存在标签泄漏。",
        "failed": "提示词命中标签泄漏规则，不能声明输入未使用真实标签。",
        "unknown": "提示词未获完整检查，不能声明输入未使用真实标签。",
    }
    lines.append("\n---\n\n%s\n" % leakage_notes.get(leak_state, leakage_notes["unknown"]))
    lines.append("本报告由固定模板与结构化结果生成，外部引用仅供复核。\n")
    markdown = "".join(lines)
    path: Optional[str] = None
    if write_file:
        target_dir = output_dir if output_dir is not None else settings.resolve_config().reports_dir
        try:
            os.makedirs(target_dir, exist_ok=True)
            name = "diag_evidence_report_%s_%s.md" % (clean_run_id, uuid.uuid4().hex)
            candidate = os.path.join(target_dir, name)
            with open(candidate, "x", encoding="utf-8") as handle:
                handle.write(markdown)
            path = candidate
        except OSError as exc:
            warnings.append("报告落盘失败（%s），已返回正文" % type(exc).__name__)
    data = {"markdown": markdown, "path": path, "trust_level": level,
            "support_level": level, "policy_version": verify.POLICY_VERSION,
            "max_allowed_strength": max_strength, "requested_strength": requested_strength,
            "effective_strength": effective_strength, "run_id": clean_run_id,
            "input_is_synthetic": is_synthetic}
    return _common.make_response(
        _common.STATUS_DEGRADED if warnings or level != "high" else _common.STATUS_OK,
        data, warnings=warnings, duration_ms=_common.elapsed_ms(started))
