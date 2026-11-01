"""Declared applicability of historical vibration references, not fault truth.

Time checks use a caller-supplied cutoff, never the wall clock. A passing check
does not authenticate metadata, establish independence, or validate a model.
"""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from .contracts import finite_number

_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$")
_FIELDS = {"decision_at", "units", "sampling_rate", "preprocessing_id"}


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not _TIME.fullmatch(value):
        raise ValueError("A timezone-aware RFC3339 timestamp is required")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _label(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError("A bounded nonempty declaration is required")
    value.encode("utf-8")
    return value.strip()


def validate_reference_context(value: Any) -> dict | None:
    """Validate an opt-in request. Missing/misspelled fields are caller errors."""
    if value is None:
        return None
    try:
        if not isinstance(value, dict) or set(value) - _FIELDS:
            raise ValueError
        rate = finite_number(value.get("sampling_rate"))
        if rate <= 0:
            raise ValueError
        result = {
            "decision_at": _timestamp(value.get("decision_at")).isoformat(),
            "units": _label(value.get("units")), "sampling_rate": rate,
        }
        if "preprocessing_id" in value:
            result["preprocessing_id"] = _label(value["preprocessing_id"])
        return result
    except (ValueError, TypeError, OverflowError, UnicodeError):
        # Neither unknown keys nor caller values may be reflected into errors.
        raise ValueError("evidence_context 的字段、单位、采样率或时点格式无效") from None


def assess_references(cases: list[dict], context: Any) -> dict:
    """Select references whose declarations support the requested comparison.

    Asset equality is deliberately not required for historical reference use.
    Other evidence roles, live freshness and forecast inputs are outside scope.
    """
    context = validate_reference_context(context)
    if context is None:
        return {"enabled": False, "eligible_indices": list(range(len(cases))),
                "assessments": [], "state": "not_applicable"}
    cutoff = _timestamp(context["decision_at"])
    assessments = []
    eligible = []
    for index, case in enumerate(cases):
        meta = case.get("meta") if isinstance(case.get("meta"), dict) else {}
        failures, missing = [], []
        if meta.get("evidence_role", "historical_reference") != "historical_reference":
            failures.append("unsupported_reference_role")
        for field in ("units", "sampling_rate", "preprocessing_id"):
            if field not in context:
                continue
            if field not in meta or meta[field] is None:
                missing.append(field + "_missing")
                continue
            try:
                found = finite_number(meta[field]) if field == "sampling_rate" else _label(meta[field])
                if found != context[field]:
                    failures.append(field + "_mismatch")
            except (ValueError, TypeError, OverflowError, UnicodeError):
                failures.append(field + "_invalid")
        times = {}
        for field in ("window_end", "available_at"):
            if field not in meta or meta[field] is None:
                missing.append(field + "_missing")
                continue
            try:
                times[field] = _timestamp(meta[field])
            except (ValueError, TypeError, OverflowError):
                failures.append(field + "_invalid")
        if "window_start" in meta:
            try:
                times["window_start"] = _timestamp(meta["window_start"])
            except (ValueError, TypeError, OverflowError):
                failures.append("window_start_invalid")
        if "window_end" in times and times["window_end"] > cutoff:
            failures.append("observation_after_cutoff")
        if "available_at" in times and times["available_at"] > cutoff:
            failures.append("unavailable_at_cutoff")
        if ("window_end" in times and "available_at" in times
                and times["available_at"] < times["window_end"]):
            failures.append("available_before_observation_end")
        if ("window_start" in times and "window_end" in times
                and times["window_start"] > times["window_end"]):
            failures.append("window_order_invalid")
        state = "failed" if failures else "unknown" if missing else "passed"
        if state == "passed":
            eligible.append(index)
        assessments.append({
            "candidate_index": index + 1, "state": state,
            "reason_codes": failures + missing,
            "allowed_uses": ["historical_reference"] if state == "passed" else [],
        })
    state = ("passed" if eligible else
             "failed" if any(x["state"] == "failed" for x in assessments) else "unknown")
    return {"enabled": True, "eligible_indices": eligible,
            "assessments": assessments, "state": state}
