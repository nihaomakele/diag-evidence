"""Validated adapter for the optional HTTP fault classifier."""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Optional

from ..contracts import SignalValidationError, finite_number, validate_signal
from ..haconfig import settings
from . import _common

_Transport = Callable[..., Any]
ABSTENTION_REASONS = {"outside_training_support", "below_selection_threshold", "validation_quality_unmet"}


def _map_diagnosis(diagnosis: Any, label_schema: str = "faults-v1") -> Dict[str, Any]:
    """Validate the classifier result before making it available to the verifier."""
    labels = settings.class_labels(label_schema)
    unavailable = {"predicted_class": None, "confidence": None,
                   "predicted_fault": {"name": "模型不可用", "name_en": "Model Unavailable"},
                   "class_probabilities": {}, "model_available": False,
                   "conclusion_strength": "no_conclusion", "contract_errors": []}
    if not isinstance(diagnosis, dict):
        unavailable["contract_errors"] = ["diagnosis 必须为 JSON 对象"]
        return unavailable
    if "abstained" in diagnosis and type(diagnosis["abstained"]) is not bool:
        unavailable["contract_errors"] = ["abstained 必须为布尔值"]
        return unavailable
    if diagnosis.get("abstained") is True:
        reason = diagnosis.get("abstention_reason")
        if (not isinstance(reason, str) or reason not in ABSTENTION_REASONS) or any(
                diagnosis.get(key) is not None for key in ("predicted_class", "confidence")) or (
                diagnosis.get("class_probabilities") not in (None, {})):
            unavailable["contract_errors"] = ["主动拒绝须提供已知原因，且不得同时提供类别、置信度或概率"]
            return unavailable
        return {"predicted_class": None, "confidence": None,
                "predicted_fault": {"name": "主动拒绝分类", "name_en": "Classification Abstained"},
                "class_probabilities": {}, "model_available": True, "abstained": True,
                "abstention_reason": reason, "label_schema": label_schema,
                "conclusion_strength": "no_conclusion", "contract_errors": []}
    # The documented no-model sentinel is not a malformed classification.
    if diagnosis.get("predicted_class") == -1 and not isinstance(diagnosis.get("predicted_class"), bool):
        unavailable["note"] = "服务报告模型未加载"
        return unavailable

    errors = []
    predicted = diagnosis.get("predicted_class")
    valid_predicted = isinstance(predicted, int) and not isinstance(predicted, bool) and predicted in labels
    if not valid_predicted:
        errors.append("predicted_class 缺失或不属于当前 label_schema")
    try:
        confidence = finite_number(diagnosis.get("confidence"))
        if not 0 <= confidence <= 1:
            raise ValueError("超出 0..1")
    except ValueError:
        confidence = None
        errors.append("confidence 必须为 0..1 的有限实数")

    fault = diagnosis.get("predicted_fault")
    if fault is not None and not isinstance(fault, dict):
        errors.append("predicted_fault 必须为对象")
    elif isinstance(fault, dict) and valid_predicted:
        for key in ("name", "name_en"):
            value = fault.get(key)
            if value is not None and value != labels[predicted][key]:
                errors.append("predicted_fault 与 predicted_class 不一致")
                break

    probabilities = diagnosis.get("class_probabilities")
    if probabilities is None:
        probabilities = {}
    if probabilities != {}:
        try:
            if not isinstance(probabilities, dict) or len(probabilities) != len(labels):
                raise ValueError
            normalized = {}
            for key, value in probabilities.items():
                # JSON converts integer map keys into strings.
                if isinstance(key, bool) or str(key) not in {str(k) for k in labels}:
                    raise ValueError
                if isinstance(value, dict):
                    probability = finite_number(value.get("probability"))
                    label = value.get("name")
                    if label is not None and label != labels[int(key)]["name"]:
                        raise ValueError
                else:
                    probability = finite_number(value)
                if not 0 <= probability <= 1 or int(key) in normalized:
                    raise ValueError
                normalized[int(key)] = probability
            if set(normalized) != set(labels) or not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-5):
                raise ValueError
            if valid_predicted and confidence is not None and not math.isclose(normalized[predicted], confidence, abs_tol=1e-5):
                raise ValueError
            probabilities = {str(key): {**labels[key], "probability": value}
                             for key, value in normalized.items()}
        except (ValueError, TypeError, OverflowError):
            errors.append("class_probabilities 必须覆盖全部类别、有限非负且和为 1，并与 confidence 一致")

    if errors:
        unavailable["contract_errors"] = errors
        return unavailable
    return {"predicted_class": predicted, "confidence": confidence,
            "predicted_fault": dict(labels[predicted]),
            "note": None,
            "class_probabilities": probabilities, "model_available": True,
            "contract_errors": [],
            "label_schema": label_schema,
            "conclusion_strength": "no_conclusion" if confidence == 0 else ("weak" if confidence < 0.5 else "moderate")}


def run_fault_diagnosis(signal: Any, transport: Optional[_Transport] = None,
                        keep_spectrogram: bool = False,
                        config: settings.Settings | None = None) -> Dict[str, Any]:
    """Call POST /diagnose only after strict input validation.

    A malformed remote prediction has model_available=False and status=error.
    No confidence value is a calibrated probability of diagnostic correctness.
    """
    started = time.time()
    cfg = settings.resolve_config(config)
    try:
        values = validate_signal(signal, expected_length=cfg.signal_window_size)
    except SignalValidationError as exc:
        return _common.invalid_input_response([str(exc)], duration_ms=_common.elapsed_ms(started))
    call = transport or _common.http_call
    kind, body, error = call(
        "POST", cfg.fd_service.url(settings.FD_DIAGNOSE_PATH),
        payload={"signal": values}, timeout=cfg.fd_diagnose_timeout,
        max_retries=0, backoff=cfg.retry_backoff, service=cfg.fd_service.name)
    duration = _common.elapsed_ms(started)
    if kind == "invalid_response":
        return _common.error_response(["FD 响应违反本地重定向、编码或字节预算策略"], duration_ms=duration)
    if kind == "conn":
        return _common.unavailable_response(["FD 服务不可达或请求超时"], duration_ms=duration)
    if kind in ("server", "not_found"):
        return _common.error_response(["FD 服务返回 HTTP 错误"], duration_ms=duration)
    if kind == "client":
        return _common.invalid_input_response(["FD 服务拒绝了请求（HTTP 4xx）"], duration_ms=duration)
    if kind != "ok" or not isinstance(body, dict):
        return _common.error_response(["FD 响应不是合法 JSON 对象"], duration_ms=duration)
    if body.get("success") is not True:
        return _common.error_response(["FD 返回 success=false 或缺少有效 success 状态"], duration_ms=duration)

    diagnosis = _map_diagnosis(body.get("diagnosis"), cfg.label_schema)
    spectrogram = body.get("spectrogram")
    # Arbitrary raw response fields are not exported; they can contain credentials.
    raw = {"success": True, "diagnosis": diagnosis}
    statistics = {}
    if isinstance(body.get("statistics"), dict):
        for key in ("mean", "std", "variance", "rms", "peak", "max", "min", "kurtosis", "skewness", "crest_factor"):
            try:
                statistics[key] = finite_number(body["statistics"][key])
            except (KeyError, ValueError):
                pass
    data = {"diagnosis": diagnosis,
            "statistics": statistics,
            "spectrogram_present": bool(spectrogram),
            "spectrogram_bytes": len(spectrogram) if isinstance(spectrogram, str) else 0,
            "raw": raw, "signal_summary": _common.summarize_signal(values)}
    if keep_spectrogram:
        data["spectrogram"] = spectrogram
    if diagnosis["contract_errors"]:
        return _common.error_response(diagnosis["contract_errors"], data=data, duration_ms=duration)
    warnings = []
    if diagnosis.get("abstained") is True:
        warnings.append("分类器已执行并主动拒绝本次分类，未输出类别结论；这不同于模型未加载")
    elif not diagnosis["model_available"]:
        warnings.append("FD 模型未加载，分类能力不可用，不得据此输出确定故障类别")
    elif diagnosis["conclusion_strength"] == "no_conclusion":
        warnings.append("分类置信度为 0，不具备判别力")
    elif diagnosis["conclusion_strength"] == "weak":
        warnings.append(f"分类置信度偏低（{diagnosis['confidence']:.4f}），仅可作参考")
    evidence = [_common.make_evidence(
        "fd", "classifier_output",
        f"FD 返回 predicted_class={diagnosis['predicted_class']} confidence={diagnosis['confidence']}",
        {"diagnosis": diagnosis, "spectrogram_present": data["spectrogram_present"]},
        trust="unavailable" if not diagnosis["model_available"] else ("low" if diagnosis["conclusion_strength"] != "moderate" else "medium"))]
    if warnings:
        return _common.degraded_response(data, warnings, evidence=evidence, duration_ms=duration)
    return _common.ok_response(data, evidence=evidence, duration_ms=duration)
