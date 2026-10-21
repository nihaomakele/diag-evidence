"""Bounded recursive audit summaries; no raw signals or credential fields."""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .haconfig import settings
from .hatools._common import redact_text

AUDIT_FILENAME = "audit.jsonl"
_PRIVATE = re.compile(r"key|token|secret|password|authorization|cookie|credential", re.I)
_OMIT = {"signal", "values", "spectrogram", "plot", "markdown", "content",
         "class_probabilities", "embedding", "embeddings", "prompt", "llm_prompt_text"}


def sanitize(value: Any, *, depth: int = 0) -> Any:
    """Filter every nesting level and bound collections. Not a general DLP system."""
    if depth > 8:
        return "[depth limit]"
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:40]:
            name = str(key)
            private_match = _PRIVATE.search(name)
            if private_match:
                # A dynamic field name can contain the credential itself.
                # Keep only its category, never the original sensitive name.
                result[private_match.group(0).lower() + "_redacted"] = "[REDACTED]"
            elif name.lower() in _OMIT:
                result[name + "_omitted"] = True
            elif name.lower() in {"source_file", "path", "report_path"}:
                result[name] = "[path omitted]"
            else:
                result[redact_text(name)[:80]] = sanitize(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize(item, depth=depth + 1) for item in value[:5]]
    if isinstance(value, str):
        return redact_text(value)[:200]
    if isinstance(value, float) and not math.isfinite(value):
        return "[non-finite]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return "[unsupported type]"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def summary(result: Optional[dict]) -> dict:
    """Deliberate output allowlist; never log retrieved free text."""
    data = (result or {}).get("data") or {}
    if not isinstance(data, dict):
        return {}
    fields = ("num_results", "trust_level", "support_level", "max_allowed_strength",
              "effective_strength", "spectrogram_present", "quality", "signal_summary",
              "reachable", "unreachable", "policy_version", "input_digest")
    output = {k: data[k] for k in fields if k in data}
    diagnosis = data.get("diagnosis")
    if isinstance(diagnosis, dict):
        output["diagnosis"] = {k: diagnosis.get(k) for k in (
            "predicted_class", "confidence", "model_available", "conclusion_strength")}
    return sanitize(output)


class AuditLogger:
    def __init__(self, log_dir: Optional[str] = None, enabled: bool = True,
                 persist: bool = True, run_id: Optional[str] = None):
        self.enabled = enabled
        self.persist = enabled and persist
        self.run_id = run_id or uuid4().hex
        self.entries: list[dict] = []
        self.errors: list[str] = []
        self.path: Optional[str] = None
        if self.persist:
            try:
                directory = Path(log_dir or settings.resolve_config().logs_dir)
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / f"audit_{self.run_id}.jsonl"
                with target.open("x", encoding="utf-8"):
                    pass
                self.path = str(target)
            except OSError as exc:
                self.errors.append(f"audit persistence failed: {type(exc).__name__}")
                self.persist = False

    def log(self, tool: str, result: Optional[dict], input_summary: Optional[dict] = None,
            extra: Optional[dict] = None) -> dict:
        entry = sanitize({
            "run_id": self.run_id, "ts": utc_now(), "tool": tool,
            "status": (result or {}).get("status"), "ok": (result or {}).get("ok"),
            "duration_ms": (result or {}).get("duration_ms"),
            "input_summary": input_summary or {}, "output_summary": summary(result),
            "warnings": (result or {}).get("warnings") or [],
            "errors": (result or {}).get("errors") or [], "extra": extra or {},
        })
        if self.enabled:
            self.entries.append(entry)
        if self.persist and self.path:
            try:
                with Path(self.path).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            except OSError as exc:
                self.errors.append(f"audit persistence failed: {type(exc).__name__}")
                self.persist = False
        return entry

    _sanitize = staticmethod(sanitize)


def read_audit(path: Optional[str] = None) -> list[dict]:
    """Read one run JSONL, ignoring damaged lines for legacy compatibility."""
    target = Path(path or (Path(settings.resolve_config().logs_dir) / AUDIT_FILENAME))
    if not target.exists():
        return []
    entries = []
    for line in target.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries
