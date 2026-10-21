"""Versioned, deterministic evidence-support policy.

``high`` means the declared checks passed. It is neither calibrated accuracy nor
proof of independent evidence, model validity, or suitability for maintenance.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from ..haconfig import settings
from ..contracts import finite_number
from . import _common

POLICY_VERSION = "diagnosis-evidence/1.0"
SUPPORT_DEFINITION = (
    "规则证据支持等级：high 仅表示本版本必需检查均通过；"
    "不代表诊断正确概率、真实设备有效性或检索案例相互独立。"
)
_STRENGTH_ORDER = {"no_conclusion": 0, "weak": 1, "moderate": 2, "strong": 3}
_LEGACY_STRENGTH = {"none": "no_conclusion"}
DEFINITIVE_PHRASES = (
    "建议停机", "立即停机", "紧急停机", "拆检", "拆解检查", "更换轴承", "立即更换",
    "备件采购", "小时内完成更换", "确定故障为", "确诊",
)
_LEAK_PATTERNS = (
    re.compile(r"真实标签"),
    re.compile(r"ground[_ ]?truth", re.IGNORECASE),
    re.compile(r"true[_ ]?label", re.IGNORECASE),
    re.compile(r"诊断正确|诊断错误"),
)


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def detect_label_leakage(text: Optional[str]) -> List[str]:
    """Find heuristic leakage markers; an empty match is not proof of absence."""
    if not isinstance(text, str) or not text:
        return []
    return [match.group(0) for pattern in _LEAK_PATTERNS
            if (match := pattern.search(text))]


def find_definitive_phrases(text: Optional[str]) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    return [phrase for phrase in DEFINITIVE_PHRASES if phrase in text]


def verify_result(
    diagnosis_result: Optional[Dict[str, Any]] = None,
    rag_result: Optional[Dict[str, Any]] = None,
    llm_prompt_text: Optional[str] = None,
    input_is_synthetic: bool = True,
    min_retrieval: int = 3,
    llm_expected: bool = False,
) -> Dict[str, Any]:
    """Return explicit check states, a support level, and a conclusion-strength cap.

    A template-only workflow has ``label_leakage=not_applicable``. When an LLM
    is expected, a missing prompt is ``unknown`` and cannot pass the policy.
    The legacy ``trust_level`` and boolean/None ``passed`` fields remain aliases.
    """
    started = time.time()
    checks: List[Dict[str, Any]] = []
    warnings: List[str] = []

    def check(name: str, state: str, detail: str, *, required: bool = True,
              **extra: Any) -> None:
        checks.append({"name": name, "state": state, "required": required,
                       "passed": {"passed": True, "failed": False}.get(state),
                       "detail": detail, **extra})

    diagnosis = _dict(_dict(_dict(diagnosis_result).get("data")).get("diagnosis"))
    model_flag = diagnosis.get("model_available")
    diagnosis_envelope = _dict(diagnosis_result)
    diagnosis_envelope_valid = (diagnosis_envelope.get("status") in ("ok", "degraded")
                                and ("ok" not in diagnosis_envelope or diagnosis_envelope["ok"] is
                                     (diagnosis_envelope["status"] == "ok")))
    model_available = model_flag is True and diagnosis_envelope_valid
    predicted_class = diagnosis.get("predicted_class")
    predicted_name = _dict(diagnosis.get("predicted_fault")).get("name")
    confidence = diagnosis.get("confidence")
    strength = diagnosis.get("conclusion_strength")
    if isinstance(strength, str):
        strength = _LEGACY_STRENGTH.get(strength, strength)
    if not isinstance(strength, str) or strength not in _STRENGTH_ORDER:
        strength = "no_conclusion"

    if diagnosis and not diagnosis_envelope_valid:
        check("classifier_available", "failed", "诊断响应执行失败、不可用、输入无效或响应状态无效；不采信其残留分类字段")
        warnings.append("诊断执行状态不允许使用分类结论")
    elif not diagnosis or model_flag not in (True, False):
        check("classifier_available", "unknown", "未提供明确的模型可用状态")
    elif not model_available:
        check("classifier_available", "failed", "分类器不可用")
        warnings.append("分类器不可用，不能给出故障类别结论")
    else:
        check("classifier_available", "passed", "上游声明模型可用；不代表模型经过独立验证")

    class_valid = (
        type(predicted_class) is int and predicted_class in settings.FAULT_LABELS
        and isinstance(predicted_name, str)
        and predicted_name == settings.FAULT_LABELS[predicted_class]["name"]
    )
    if not diagnosis or (model_available and predicted_class is None):
        check("class_prediction_valid", "unknown", "分类类别或名称缺失")
    elif not model_available:
        check("class_prediction_valid", "not_applicable", "分类器不可用，类别检查不适用", required=False)
    elif class_valid:
        check("class_prediction_valid", "passed", "类别编号与当前三类故障契约一致")
    else:
        check("class_prediction_valid", "failed", "类别编号未知，或类别名称与编号不一致")
        warnings.append("分类类别无效，禁止给出故障类别结论")

    try:
        numeric_confidence = finite_number(confidence)
        valid_confidence = 0 <= numeric_confidence <= 1
    except ValueError:
        valid_confidence = False
    if not model_available:
        check("confidence_valid", "not_applicable", "分类器不可用，置信度检查不适用", required=False)
    elif confidence is None:
        check("confidence_valid", "unknown", "分类置信度缺失")
    elif not valid_confidence:
        check("confidence_valid", "failed", "置信度必须为 [0, 1] 内的有限数值")
        warnings.append("分类置信度无效")
    elif confidence < 0.5:
        check("confidence_valid", "failed", "置信度低于本策略阈值 0.5（工程规则，未经校准）")
        warnings.append("分类置信度偏低，仅可作为排查线索")
    else:
        check("confidence_valid", "passed", "置信度通过有限数值、范围和 0.5 阈值检查（未经校准）")

    rag_envelope = _dict(rag_result)
    rag_envelope_valid = (rag_envelope.get("status") in ("ok", "degraded")
                         and ("ok" not in rag_envelope or rag_envelope["ok"] is
                              (rag_envelope["status"] == "ok")))
    rag_data = _dict(rag_envelope.get("data")) if rag_envelope_valid else {}
    raw_cases = rag_data.get("cases")
    cases = [case for case in raw_cases if isinstance(case, dict)] if isinstance(raw_cases, list) else []
    fault_types = [case.get("fault_type") for case in cases
                   if isinstance(case.get("fault_type"), str) and case.get("fault_type")]
    candidate_agreement = bool(fault_types) and len(set(fault_types)) == 1 and len(fault_types) == len(cases)
    # Retain the old check name for response compatibility. Multiple retrieved
    # labels express candidate diversity, not a demonstrated semantic conflict.
    if not cases or len(fault_types) != len(cases):
        check("evidence_conflict", "unknown", "缺少完整候选标签，无法判断标签一致性",
              semantic="candidate_label_agreement")
    elif candidate_agreement:
        check("evidence_conflict", "passed", "候选标签一致；未验证同设备、同工况或独立来源",
              semantic="candidate_label_agreement")
    else:
        check("evidence_conflict", "failed", "检索候选标签不一致；不据此认定语义矛盾",
              semantic="candidate_label_agreement")
        warnings.append("检索候选标签不一致，候选多样性不等于语义冲突")

    if model_available and class_valid and cases and len(fault_types) == len(cases):
        consistent = predicted_name in fault_types
        check("conclusion_matches_evidence", "passed" if consistent else "failed",
              "分类名称与至少一个候选标签一致" if consistent else "分类名称未匹配任何候选标签")
        if not consistent:
            warnings.append("分类结论未得到候选标签匹配支持")
    else:
        check("conclusion_matches_evidence", "unknown", "缺少有效类别或完整候选标签，无法比对")

    if llm_prompt_text is None and not llm_expected:
        check("label_leakage", "not_applicable", "模板流程未调用 LLM，提示词泄漏检查不适用", required=False)
    elif not isinstance(llm_prompt_text, str) or not llm_prompt_text.strip():
        check("label_leakage", "unknown", "预期使用 LLM，但待发送提示词缺失或无效")
        warnings.append("LLM 提示词未获检查，不能声明未使用真实标签")
    else:
        leak_hits = detect_label_leakage(llm_prompt_text)
        check("label_leakage", "failed" if leak_hits else "passed",
              "检测到真实标签或对错判定标记" if leak_hits else "已提供文本未命中本版本泄漏标记规则；不证明无泄漏")
        if leak_hits:
            warnings.append("送入 LLM 的文本命中标签泄漏规则，结论不具独立性")

    if type(min_retrieval) is not int or min_retrieval < 1:
        raise ValueError("min_retrieval 必须为正整数")
    reported_count = rag_data.get("num_results")
    actual_count = len(cases)
    malformed_cases = isinstance(raw_cases, list) and len(raw_cases) != actual_count
    count_consistent = type(reported_count) is int and reported_count == actual_count
    scale_passes = actual_count >= min_retrieval and count_consistent and not malformed_cases
    if not isinstance(raw_cases, list) or reported_count is None:
        check("retrieval_scale", "unknown", "未获得可核对的案例列表与返回数量")
    else:
        check("retrieval_scale", "passed" if scale_passes else "failed",
              "实际有效候选 %d 条，策略阈值 %d；数量%s，候选条数不代表独立证据数" %
              (actual_count, min_retrieval, "一致" if count_consistent and not malformed_cases else "不一致或结构无效"))
    if not scale_passes:
        warnings.append("检索数量不足或不可核对，规则支持上限降低")
    if not cases:
        warnings.append("无检索证据，结论缺乏外部候选支撑")

    required_incomplete = [item for item in checks
                           if item["required"] and item["state"] != "passed"]
    hard_names = {"classifier_available", "class_prediction_valid", "confidence_valid", "label_leakage"}
    hard_incomplete = [item for item in required_incomplete if item["name"] in hard_names]
    if not model_available and not cases:
        level = "unavailable"
    elif not model_available or hard_incomplete:
        level = "low"
    elif required_incomplete:
        level = "medium"
    else:
        level = "high"

    if not model_available or not class_valid:
        max_strength = "no_conclusion"
    else:
        max_strength = {"unavailable": "no_conclusion", "low": "weak", "medium": "moderate", "high": "strong"}[level]
    if _STRENGTH_ORDER[strength] > _STRENGTH_ORDER[max_strength]:
        warnings.append("请求的结论强度超过策略上限，报告必须执行限制")
    if input_is_synthetic:
        warnings.append("输入为合成数据，仅用于工程流程验证，不得据此评估算法精度")

    data = {"trust_level": level, "support_level": level,
            "support_definition": SUPPORT_DEFINITION, "policy_version": POLICY_VERSION,
            "checks": checks, "max_allowed_strength": max_strength,
            "input_is_synthetic": bool(input_is_synthetic), "diagnosis_strength": strength,
            "retrieval_candidate_count": actual_count, "independent_evidence_count": None}
    evidence = [_common.make_evidence("system", "verification", "规则支持等级 %s；结论强度上限 %s" % (level, max_strength),
                                     {"checks": checks, "policy_version": POLICY_VERSION}, trust=level)]
    return _common.make_response(
        _common.STATUS_DEGRADED if warnings or level != "high" else _common.STATUS_OK,
        data, warnings=warnings, evidence=evidence, duration_ms=_common.elapsed_ms(started))
