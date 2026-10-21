"""Deterministic diagnostic workflow with isolated run records."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from threading import Lock
import time
from typing import Any, Callable, Optional
from uuid import uuid4

from .audit_log import AuditLogger, sanitize, summary, utc_now
from .contracts import SignalValidationError, validate_signal
from .haconfig import settings
from .hatools._common import invalid_input_response
from .hatools import (analyze_signal, check_system_status, generate_report,
                     retrieve_cases, run_fault_diagnosis, verify_result)

RECEIVED = "RECEIVED"
VALIDATING_INPUT = "VALIDATING_INPUT"
CHECKING_SERVICES = "CHECKING_SERVICES"
ANALYZING_SIGNAL = "ANALYZING_SIGNAL"
RETRIEVING_CASES = "RETRIEVING_CASES"
RUNNING_DIAGNOSIS = "RUNNING_DIAGNOSIS"
VERIFYING_RESULT = "VERIFYING_RESULT"
GENERATING_REPORT = "GENERATING_REPORT"
COMPLETED = "COMPLETED"
DEGRADED = "DEGRADED"
FAILED = "FAILED"
ASKING_USER = "ASKING_USER"
STATE_SEQUENCE = (RECEIVED, VALIDATING_INPUT, CHECKING_SERVICES, ANALYZING_SIGNAL,
                  RETRIEVING_CASES, RUNNING_DIAGNOSIS, VERIFYING_RESULT, GENERATING_REPORT)


class TraceRecorder:
    def __init__(self, log_dir: Optional[str] = None, persist: bool = True,
                 run_id: Optional[str] = None):
        self.run_id = run_id or uuid4().hex
        self.entries: list[dict] = []
        self.errors: list[str] = []
        self.path: Optional[str] = None
        self.persist = persist
        if persist:
            try:
                directory = Path(log_dir or settings.resolve_config().logs_dir)
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"agent_trace_{self.run_id}.jsonl"
                with path.open("x", encoding="utf-8"):
                    pass
                self.path = str(path)
            except OSError as exc:
                self.errors.append(f"trace persistence failed: {type(exc).__name__}")
                self.persist = False

    def record(self, state: str, tool: str, result: Optional[dict] = None,
               input_summary: Optional[dict] = None, note: str = "") -> dict:
        entry = sanitize({
            "run_id": self.run_id, "ts": utc_now(), "state": state, "tool": tool,
            "status": (result or {}).get("status"), "ok": (result or {}).get("ok"),
            "duration_ms": (result or {}).get("duration_ms"),
            "input_summary": input_summary or {}, "output_summary": summary(result),
            "warnings": (result or {}).get("warnings") or [],
            "errors": (result or {}).get("errors") or [], "note": note,
        })
        self.entries.append(entry)
        if self.persist and self.path:
            try:
                with Path(self.path).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            except OSError as exc:
                self.errors.append(f"trace persistence failed: {type(exc).__name__}")
                self.persist = False
        return entry

    _summarize = staticmethod(summary)


class DiagnosisAgent:
    """Fixed workflow. Concurrent calls on one instance are rejected.

    persist_trace=False disables all report, trace and audit writes by default.
    Use separate instances for concurrent requests. No LLM/model is loaded here.
    """
    def __init__(self, transport: Optional[Callable] = None, persist_trace: bool = True,
                 log_dir: Optional[str] = None, config: Optional[Any] = None,
                 persist_outputs: Optional[bool] = None):
        self.transport = transport
        self.config = config or settings.Settings.from_env()
        self.persist_outputs = persist_trace if persist_outputs is None else persist_outputs
        self.persist_trace = persist_trace and self.persist_outputs
        self.log_dir = log_dir or self.config.logs_dir
        self.trace: Optional[TraceRecorder] = None
        self.audit: Optional[AuditLogger] = None
        self.state = RECEIVED
        self._lock = Lock()

    def _step(self, state: str, tool: str, func: Callable, input_summary=None, **kwargs):
        self.state = state
        tick = time.monotonic()
        try:
            result = func(**kwargs)
            if not isinstance(result, dict) or "status" not in result:
                raise TypeError("Invalid tool response")
        except Exception as exc:
            result = {"ok": False, "status": "error", "data": None,
                      "warnings": [], "errors": [f"{tool}: {type(exc).__name__}"],
                      "evidence": [], "duration_ms": int((time.monotonic() - tick) * 1000)}
        self.trace.record(state, tool, result, input_summary)
        self.audit.log(tool, result, input_summary, extra={"state": state})
        return result

    def run(self, signal: Any = None, sample_info: Optional[dict] = None,
            llm_prompt_text: Optional[str] = None, input_is_synthetic: bool = True,
            skip_services_check: bool = False, llm_expected: bool = False) -> dict:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("DiagnosisAgent is already running; use a separate instance")
        try:
            return self._run(signal, sample_info, llm_prompt_text, input_is_synthetic,
                             skip_services_check, llm_expected)
        finally:
            self._lock.release()

    def _run(self, signal, sample_info, llm_prompt_text, input_is_synthetic,
             skip_services_check, llm_expected):
        self.run_id = uuid4().hex
        self.trace = TraceRecorder(self.log_dir, self.persist_trace, self.run_id)
        self.audit = AuditLogger(self.log_dir, persist=self.persist_trace, run_id=self.run_id)
        tick = time.monotonic()
        outcomes = {}
        self.trace.record(RECEIVED, "agent", input_summary={"has_signal": signal is not None})
        if any(type(value) is not bool for value in (
                input_is_synthetic, skip_services_check, llm_expected)):
            return self._finish(outcomes, ASKING_USER, tick, question="流程控制参数必须为布尔值。")
        if sample_info is not None and not isinstance(sample_info, dict):
            return self._finish(outcomes, ASKING_USER, tick, question="sample_info 必须是对象。")
        sample_info = deepcopy(sample_info or {})
        # Metadata cannot override controls supplied to run(). The report performs
        # an independent check against the very same declared inputs.
        for control in ("llm_prompt_text", "llm_expected", "input_is_synthetic"):
            sample_info.pop(control, None)
        sample_info.update(llm_prompt_text=llm_prompt_text, llm_expected=llm_expected,
                           input_is_synthetic=input_is_synthetic)
        try:
            # Materialize valid iterators once, so validation cannot consume an
            # input that later tools would see as empty.
            signal_list = validate_signal(signal, expected_length=self.config.signal_window_size)
        except SignalValidationError as exc:
            failure = invalid_input_response([str(exc)])
            outcomes["signal"] = self._step(VALIDATING_INPUT, "validate_signal", lambda: failure)
            return self._finish(outcomes, ASKING_USER, tick,
                                question=f"请提供 {self.config.signal_window_size} 点的一维有限有效信号；{exc}")
        # Complete local numeric/quality validation before any external call.
        outcomes["signal"] = self._step(
            VALIDATING_INPUT, "validate_signal", analyze_signal, signal=signal_list,
            expected_length=self.config.signal_window_size,
            sampling_rate=self.config.sampling_rate, generate_plot=False)
        if outcomes["signal"]["status"] != "ok":
            return self._finish(outcomes, ASKING_USER, tick,
                                question=f"请提供 {self.config.signal_window_size} 点的一维有限有效信号；"
                                         + "; ".join(outcomes["signal"].get("errors") or
                                                     outcomes["signal"].get("warnings") or []))
        digest = hashlib.sha256(json.dumps(signal_list, allow_nan=False,
                                           separators=(",", ":")).encode()).hexdigest()
        self.trace.record(ANALYZING_SIGNAL, "analyze_signal", outcomes["signal"],
                          {"points": len(signal_list), "input_digest": digest})
        if not skip_services_check:
            outcomes["health"] = self._step(
                CHECKING_SERVICES, "check_system_status", check_system_status,
                transport=self.transport, config=self.config, services=["rag", "fd"],
                check_logs=False)
        outcomes["rag"] = self._step(
            RETRIEVING_CASES, "retrieve_cases", retrieve_cases,
            input_summary={"points": len(signal_list), "top_k": 5}, signal=signal_list,
            top_k=5, transport=self.transport, config=self.config)
        outcomes["diagnosis"] = self._step(
            RUNNING_DIAGNOSIS, "run_fault_diagnosis", run_fault_diagnosis,
            input_summary={"points": len(signal_list)}, signal=signal_list,
            transport=self.transport, config=self.config)
        outcomes["verify"] = self._step(
            VERIFYING_RESULT, "verify_result", verify_result,
            diagnosis_result=outcomes["diagnosis"], rag_result=outcomes["rag"],
            llm_prompt_text=llm_prompt_text, llm_expected=llm_expected,
            input_is_synthetic=input_is_synthetic)
        verified = outcomes["verify"].get("data") or {}
        terminal = COMPLETED if verified.get("trust_level") == "high" else DEGRADED
        outcomes["report"] = self._step(
            GENERATING_REPORT, "generate_report", generate_report,
            signal_result=outcomes["signal"], rag_result=outcomes["rag"],
            diagnosis_result=outcomes["diagnosis"], verify_data=outcomes["verify"],
            sample_info={**sample_info, "run_id": self.run_id, "input_digest": digest},
            write_file=self.persist_outputs, output_dir=self.config.reports_dir,
            run_id=self.run_id)
        if outcomes["report"]["status"] == "error" or outcomes["verify"]["status"] == "error":
            terminal = FAILED
        elif self.persist_outputs and not (outcomes["report"].get("data") or {}).get("path") and terminal == COMPLETED:
            terminal = DEGRADED
        return self._finish(outcomes, terminal, tick, input_digest=digest)

    def _finish(self, outcomes: dict, terminal: str, tick: float,
                question: Optional[str] = None, input_digest: Optional[str] = None) -> dict:
        verified = dict(outcomes.get("verify", {}).get("data") or {})
        report = outcomes.get("report", {}).get("data") or {}
        if report:
            levels = {"unavailable": 0, "low": 1, "medium": 2, "high": 3}
            strengths = {"no_conclusion": 0, "weak": 1, "moderate": 2, "strong": 3}
            verified["trust_level"] = min(
                (verified.get("trust_level", "unavailable"), report.get("trust_level", "unavailable")),
                key=lambda value: levels.get(value, 0))
            verified["support_level"] = verified["trust_level"]
            verified["max_allowed_strength"] = min(
                (verified.get("max_allowed_strength", "no_conclusion"),
                 report.get("max_allowed_strength", "no_conclusion")),
                key=lambda value: strengths.get(value, 0))
            if verified["trust_level"] != "high" and terminal == COMPLETED:
                terminal = DEGRADED
        persistence_errors = self.trace.errors + self.audit.errors
        if persistence_errors and terminal == COMPLETED:
            terminal = DEGRADED
        self.state = terminal
        self.trace.record(terminal, "agent", input_summary={"trust_level": verified.get("trust_level")})
        persistence_errors = list(dict.fromkeys(self.trace.errors + self.audit.errors))
        if persistence_errors and terminal == COMPLETED:
            terminal = self.state = DEGRADED
            # The final append itself may have failed. Keep the returned in-memory
            # trace consistent even when the persisted trace is incomplete.
            self.trace.entries[-1]["state"] = DEGRADED
        warnings = [w for r in outcomes.values() for w in r.get("warnings", [])]
        errors = [e for r in outcomes.values() for e in r.get("errors", [])]
        return deepcopy({
            "schema_version": "1.0", "run_id": self.run_id, "input_digest": input_digest,
            "terminal_state": terminal, "state": terminal, "ok": terminal == COMPLETED,
            "trust_level": verified.get("trust_level"),
            "support_level": verified.get("support_level", verified.get("trust_level")),
            "max_allowed_strength": verified.get("max_allowed_strength", "no_conclusion"),
            "effective_strength": report.get("effective_strength", "no_conclusion"),
            "report_path": report.get("path"), "report_markdown": report.get("markdown"),
            "question": question, "trace": self.trace.entries, "trace_path": self.trace.path,
            "audit_path": self.audit.path, "warnings": list(dict.fromkeys(warnings)),
            "errors": errors + persistence_errors, "persistence_errors": persistence_errors,
            "elapsed_ms": int((time.monotonic() - tick) * 1000), "outcomes": outcomes,
        })


def run_agent(signal: Any, sample_info: Optional[dict] = None, **kwargs) -> dict:
    constructor_fields = {"transport", "persist_trace", "log_dir", "config", "persist_outputs"}
    run_fields = {"llm_prompt_text", "input_is_synthetic", "skip_services_check", "llm_expected"}
    if set(kwargs) - constructor_fields - run_fields:
        # Reject typos before constructing an agent or creating any artifacts.
        # Argument names may themselves contain credentials; do not echo them.
        raise TypeError("Unknown run_agent keyword argument; check the API documentation")
    constructor = {k: v for k, v in kwargs.items() if k in constructor_fields}
    run_options = {k: v for k, v in kwargs.items() if k in run_fields}
    return DiagnosisAgent(**constructor).run(signal, sample_info, **run_options)
