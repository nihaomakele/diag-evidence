"""工具层公共设施：统一响应封装、HTTP 调用、错误映射、脱敏处理。

所有工具必须通过这里返回结果，保证响应结构一致：
    {ok, status, data, warnings, errors, evidence, duration_ms}
"""

from __future__ import annotations

import json
import math
import re
import time
import threading
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Dict, Iterable, List, Optional

import requests
from urllib3.exceptions import HTTPError as UrllibHTTPError

# Public status values; see docs/contracts.md.
STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_UNAVAILABLE = "unavailable"
STATUS_INVALID_INPUT = "invalid_input"
STATUS_ERROR = "error"

_SEVERITY = {
    STATUS_OK: 0,
    STATUS_DEGRADED: 1,
    STATUS_INVALID_INPUT: 2,
    STATUS_UNAVAILABLE: 3,
    STATUS_ERROR: 4,
}

# 形似 API Key 的串，日志与响应中一律打码
_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9\-_]{12,}")
_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_QUOTED_ASSIGNMENT_PATTERN = re.compile(
    r'''(?i)((?:api[_-]?key|access[_-]?token|password|secret|authorization)["']?\s*[:=]\s*)(?:"[^"\n]*"|'[^'\n]*')'''
)
_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|access[_-]?token|password|secret|authorization)[\"']?\s*[:=]\s*[\"']?)([^\s\"',;}&]+)"
)


_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_TIMEOUT = 300.0
MAX_HTTP_WORKERS = 4
_HTTP_SLOTS = threading.BoundedSemaphore(MAX_HTTP_WORKERS)


def safe_source_url(value: str) -> Optional[str]:
    """Keep HTTP(S) host/path only; signed queries and URL credentials are private."""
    if not isinstance(value, str) or any(ord(char) < 32 for char in value):
        return None
    try:
        value.encode("utf-8")
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
            return None
        host = parsed.hostname
        if ":" in host:
            host = "[" + host + "]"
        if parsed.port is not None:
            host += ":" + str(parsed.port)
        return urlunsplit((parsed.scheme.lower(), host, parsed.path, "", ""))
    except (ValueError, UnicodeError):
        return None


def redact_text(text: str) -> str:
    """把任何形似 Key 的串替换为 sk-***，用于日志与错误信息。"""
    if text is None:
        return ""
    text = str(text)
    text = _URL_PATTERN.sub(lambda match: safe_source_url(match.group()) or "[URL omitted]", text)
    text = _KEY_PATTERN.sub("sk-***REDACTED***", text)
    text = _BEARER_PATTERN.sub(r"\1***REDACTED***", text)
    text = _QUOTED_ASSIGNMENT_PATTERN.sub(r"\1***REDACTED***", text)
    return _ASSIGNMENT_PATTERN.sub(r"\1***REDACTED***", text)


def make_response(
    status: str,
    data: Any = None,
    warnings: Optional[Iterable[str]] = None,
    errors: Optional[Iterable[str]] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    duration_ms: int = 0,
) -> Dict[str, Any]:
    """构造统一响应。ok 仅当 status == 'ok'。"""
    if status not in _SEVERITY:
        raise ValueError(f"未知 status: {status!r}")
    return {
        "ok": status == STATUS_OK,
        "status": status,
        "data": data,
        "warnings": [redact_text(w) for w in (warnings or [])],
        "errors": [redact_text(e) for e in (errors or [])],
        "evidence": evidence or [],
        "duration_ms": int(duration_ms),
    }


def make_evidence(
    source: str,
    kind: str,
    summary: str,
    payload: Any = None,
    trust: str = "medium",
) -> Dict[str, Any]:
    return {
        "source": source,
        "kind": kind,
        "summary": redact_text(summary),
        "payload": payload,
        "trust": trust,
    }


def elapsed_ms(started: float) -> int:
    return int((time.time() - started) * 1000)


def summarize_signal(signal: Any) -> Dict[str, Any]:
    """只产出信号的统计摘要，绝不记录完整数组（日志安全要求）。"""
    try:
        values = list(signal)
        count = len(values)
        if count == 0:
            return {"points": 0}
        numeric = [float(v) for v in values]
        if not all(math.isfinite(v) for v in numeric):
            return {"points": count, "note": "信号含非有限数值，未计算统计量"}
        total = math.fsum(numeric)
        mean = total / count
        variance = sum((float(v) - mean) ** 2 for v in values) / count
        return {
            "points": count,
            "mean": round(mean, 6),
            "std": round(variance ** 0.5, 6),
            "min": round(min(float(v) for v in values), 6),
            "max": round(max(float(v) for v in values), 6),
        }
    except (TypeError, ValueError, OverflowError):
        return {"points": None, "note": "信号无法统计（非数值序列）"}


class _ResponseLimitError(ValueError):
    pass


class _DeadlineExceeded(TimeoutError):
    pass


def _read_json_bounded(response, maximum: int, deadline: float,
                       cancelled: threading.Event) -> Any:
    # Request identity encoding and reject compressed responses. This keeps the
    # byte limit meaningful without allocating an unbounded decompression result.
    encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
    if encoding not in ("", "identity"):
        raise _ResponseLimitError("不接受压缩响应，请使用 identity 编码")
    declared = response.headers.get("Content-Length")
    if declared is not None:
        try:
            length = int(declared)
        except (TypeError, ValueError):
            raise _ResponseLimitError("响应 Content-Length 无效") from None
        if length < 0 or length > maximum:
            raise _ResponseLimitError("响应超过字节预算")
    content = bytearray()
    # read1 yields available bytes instead of waiting for a full large chunk,
    # so a continuous trickle cannot delay cancellation until 16 KiB fill up.
    read1 = getattr(response.raw, "read1", None)
    if callable(read1):
        def chunks():
            while True:
                chunk = read1(16 * 1024, decode_content=False)
                if not chunk:
                    break
                yield chunk
        source = chunks()
    else:
        # Older urllib3: one-byte chunks preserve cancellation responsiveness.
        source = response.iter_content(chunk_size=1)
    for chunk in source:
        if cancelled.is_set() or time.monotonic() >= deadline:
            raise _DeadlineExceeded
        if not chunk:
            continue
        if len(content) + len(chunk) > maximum:
            raise _ResponseLimitError("响应超过字节预算")
        content.extend(chunk)
    if cancelled.is_set() or time.monotonic() >= deadline:
        raise _DeadlineExceeded
    try:
        return json.loads(content)
    except (ValueError, UnicodeError, RecursionError):
        return None


def _http_exchange(method, url, payload, timeout, attempts, backoff, service,
                   maximum, deadline, cancelled):
    kind, last_error = "conn", f"{service} 请求超过总时间预算"
    for attempt in range(attempts):
        remaining = deadline - time.monotonic()
        if cancelled.is_set() or remaining <= 0:
            return "conn", None, f"{service} 请求超过总时间预算"
        response = None
        try:
            response = requests.request(
                method, url, json=payload, timeout=min(timeout, remaining),
                stream=True, allow_redirects=False,
                headers={"Accept-Encoding": "identity"},
            )
            code = response.status_code
            if 300 <= code < 400:
                return "invalid_response", None, f"{service} 重定向被本地策略拒绝"
            if 200 <= code < 300:
                body = _read_json_bounded(response, maximum, deadline, cancelled)
                return "ok", body, ""
            # Error bodies are intentionally never read or exported. Even a huge
            # 4xx/5xx body cannot bypass the response budget or echo private input.
            if code == 404:
                return "not_found", None, f"{service} 返回 404"
            if code >= 500:
                kind, last_error = "server", f"{service} 返回 HTTP {code}"
            else:
                return "client", None, f"{service} 返回 HTTP {code}"
        except _ResponseLimitError as exc:
            return "invalid_response", None, f"{service} 本地响应策略：{exc}"
        except _DeadlineExceeded:
            return "conn", None, f"{service} 请求超过总时间预算"
        except requests.exceptions.Timeout:
            kind, last_error = "conn", f"{service} 请求超时（{timeout}s）"
        except requests.exceptions.ConnectionError:
            kind, last_error = "conn", f"{service} 无法连接（连接被拒绝或网络不可达）"
        except (requests.exceptions.RequestException, UrllibHTTPError, OSError) as exc:
            kind, last_error = "conn", f"{service} 请求异常（{type(exc).__name__}）"
        finally:
            if response is not None:
                response.close()
        if attempt < attempts - 1:
            wait_for = min(backoff * (attempt + 1), max(0, deadline - time.monotonic()))
            if cancelled.wait(wait_for):
                return "conn", None, f"{service} 请求超过总时间预算"
    return kind, None, last_error


def http_call(
    method: str,
    url: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
    max_retries: int = 0,
    backoff: float = 1.5,
    service: str = "service",
    total_timeout: Optional[float] = None,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
) -> tuple[str, Any, str]:
    """Bounded HTTP exchange returning (kind, body, local error message).

    No redirect is followed, compressed/error bodies are not read. Success JSON
    is streamed with a 2 MiB hard ceiling. ``total_timeout`` covers attempts and
    backoff; by default it is the old attempt budgets plus backoff, capped at
    300 s. The caller returns at that deadline even during DNS or header reads.

    At most four daemon workers can remain in underlying I/O. A deadline signals
    cancellation, but Python cannot force-stop DNS/socket calls: a worker closes
    its response when that operation yields or times out. Saturation fails fast.
    This is a bounded client, not a general network cancellation mechanism.

    Kinds: ok/not_found/server/client/conn, plus invalid_response for a local
    redirect/encoding/size policy violation. Only successful JSON is returned.
    """
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or not 0 <= max_retries <= 10:
        raise ValueError("max_retries 必须是 0..10 的整数")
    for name, value in (("timeout", timeout), ("backoff", backoff)):
        try:
            valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))
        except (OverflowError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"{name} 必须为有限数值")
    if timeout <= 0 or not 0 <= backoff <= 60:
        raise ValueError("timeout 必须为正数，backoff 必须在 0..60 秒之间")
    if (isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int)
            or not 1 <= max_response_bytes <= MAX_RESPONSE_BYTES):
        raise ValueError("max_response_bytes 必须在 1..2097152 之间")
    attempts = max_retries + 1
    if total_timeout is None:
        budget = min(MAX_TOTAL_TIMEOUT, timeout * attempts + backoff * max_retries * attempts / 2)
    else:
        try:
            valid = (not isinstance(total_timeout, bool) and isinstance(total_timeout, (int, float))
                     and math.isfinite(float(total_timeout)) and 0 < total_timeout <= MAX_TOTAL_TIMEOUT)
        except (OverflowError, ValueError):
            valid = False
        if not valid:
            raise ValueError("total_timeout 必须为 0..300 秒内的正有限数值")
        budget = float(total_timeout)
    if not _HTTP_SLOTS.acquire(blocking=False):
        return "conn", None, f"{service} 活动网络请求已达本地上限"
    done, cancelled = threading.Event(), threading.Event()
    outcome = {}
    deadline = time.monotonic() + budget

    def worker():
        try:
            outcome["value"] = _http_exchange(
                method, url, payload, timeout, attempts, backoff, service,
                max_response_bytes, deadline, cancelled)
        except Exception as exc:
            outcome["value"] = ("conn", None, f"{service} 请求异常（{type(exc).__name__}）")
        finally:
            _HTTP_SLOTS.release()
            done.set()

    thread = threading.Thread(target=worker, name="diag-evidence-http", daemon=True)
    try:
        thread.start()
    except RuntimeError:
        _HTTP_SLOTS.release()
        return "conn", None, f"{service} 无法启动本地网络请求"
    if not done.wait(max(0, deadline - time.monotonic())):
        cancelled.set()
        return "conn", None, f"{service} 请求超过总时间预算"
    return outcome["value"]


def ok_response(data: Any, evidence=None, duration_ms: int = 0, warnings=None) -> Dict[str, Any]:
    return make_response(STATUS_OK, data, warnings=warnings, evidence=evidence, duration_ms=duration_ms)


def degraded_response(data: Any, warnings, evidence=None, duration_ms: int = 0) -> Dict[str, Any]:
    return make_response(STATUS_DEGRADED, data, warnings=warnings, evidence=evidence, duration_ms=duration_ms)


def unavailable_response(errors, data: Any = None, duration_ms: int = 0) -> Dict[str, Any]:
    return make_response(STATUS_UNAVAILABLE, data, errors=errors, duration_ms=duration_ms)


def invalid_input_response(errors, data: Any = None, duration_ms: int = 0) -> Dict[str, Any]:
    return make_response(STATUS_INVALID_INPUT, data, errors=errors, duration_ms=duration_ms)


def error_response(errors, data: Any = None, duration_ms: int = 0) -> Dict[str, Any]:
    return make_response(STATUS_ERROR, data, errors=errors, duration_ms=duration_ms)
