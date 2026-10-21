"""工具 1：check_system_status —— 检查三个服务的可达性与能力状态。"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional

from ..haconfig import settings
from . import _common

_Transport = Callable[..., Any]


def _log_hint(service_name: str, config: settings.Settings | None = None) -> Optional[str]:
    """读取服务日志，提取已知失败原因（只回传白名单关键字，不泄露日志正文）。"""
    cfg = settings.resolve_config(config)
    log_path = os.path.join(cfg.logs_dir, f"{service_name}.log")
    _shared_dir = os.getenv("DIAG_EVIDENCE_SHARED_LOGS_DIR", "")
    shared = os.path.join(_shared_dir, "%s.log" % service_name) if _shared_dir else ""
    for candidate in (log_path, shared):
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8", errors="ignore") as handle:
                    text = handle.read()
            except OSError:
                continue
            if "模型加载失败" in text or "Incorrect path_or_model_id" in text:
                return "FD 日志包含历史模型加载失败标记（不代表当前能力）"
            if "检查点不存在" in text:
                return "FD 检查点不存在"
            if "swigfaiss_avx2" in text:
                return "RAG faiss 回退到非 AVX2 版本（非致命）"
    return None


def check_system_status(
    services: Optional[List[str]] = None,
    transport: Optional[_Transport] = None,
    check_logs: bool = True,
    config: settings.Settings | None = None,
) -> Dict[str, Any]:
    """检查 RAG / FD / UI 三个服务状态。

    transport 可注入，便于无卡环境下用测试替身覆盖（签名同 requests.request）。
    返回统一响应；任一服务不可用时**不抛异常**，只降低 status。
    """
    started = time.time()
    cfg = settings.resolve_config(config)
    call = transport or _common.http_call
    wanted = {endpoint.name: endpoint for endpoint in cfg.services}
    names = list(wanted.keys()) if services is None else services
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(name, str) or name not in wanted for name in names)):
        return _common.invalid_input_response(["未知服务名称或空服务列表"], duration_ms=_common.elapsed_ms(started))
    names = list(dict.fromkeys(names))

    reports: Dict[str, Any] = {}
    warnings: List[str] = []
    errors: List[str] = []
    evidence: List[Dict[str, Any]] = []

    for name in names:
        endpoint = wanted.get(name)
        if endpoint is None:
            errors.append(f"未知服务: {name}")
            continue

        tick = time.time()
        kind, body, error = call(
            "GET",
            endpoint.health_url,
            timeout=endpoint.health_timeout,
            max_retries=1,
            backoff=0.5,
            service=endpoint.name,
        )
        entry: Dict[str, Any] = {
            "reachable": kind == "ok",
            "latency_ms": _common.elapsed_ms(tick),
            "url": endpoint.health_url,
        }

        if kind != "ok":
            entry["http"] = "unreachable"
            entry["detail"] = "服务不可达或返回非成功 HTTP 状态"
            errors.append(f"{name} 服务不可达或返回非成功 HTTP 状态")
            evidence.append(_common.make_evidence(
                "system", "health", f"{name} 服务不可达", entry, trust="unavailable"))
            reports[name] = entry
            continue

        entry["http"] = 200
        entry["body"] = {"status": "reachable"}

        if name == "rag":
            body = body if isinstance(body, dict) else {}
            model_loaded = body.get("model_loaded") is True
            index_loaded = body.get("index_loaded") is True
            num_texts = body.get("num_texts")
            if isinstance(num_texts, bool) or not isinstance(num_texts, int) or num_texts < 0:
                num_texts = 0
                warnings.append("RAG 健康响应缺少有效 num_texts，能力状态受限")
            entry.update({"model_loaded": model_loaded, "index_loaded": index_loaded,
                          "num_texts": num_texts})
            entry["capable"] = model_loaded and index_loaded and num_texts > 0
            if not entry["capable"]:
                warnings.append("RAG 已加载但索引/文本库为空，检索能力受限")
            if num_texts and num_texts < 10:
                warnings.append(f"RAG 文本库仅 {num_texts} 条，检索规模有限")
        elif name == "fd":
            hint = _log_hint("fd", cfg) if check_logs else None
            entry["note"] = "GET / 仅表示进程在线，不代表分类能力可用"
            entry["capable"] = None
            if hint:
                entry["capability_hint"] = hint
                warnings.append(hint)
        else:
            entry["capable"] = True

        trust = "high" if entry.get("capable") else "medium"
        evidence.append(_common.make_evidence(
            "system", "health", f"{name} {endpoint.health_url} 可达", entry, trust=trust))
        reports[name] = entry

    reachable = [n for n, r in reports.items() if r.get("reachable")]
    unreachable = [n for n, r in reports.items() if not r.get("reachable")]

    if not unreachable:
        status = _common.STATUS_DEGRADED if warnings else _common.STATUS_OK
    elif len(unreachable) == len(reports):
        status = _common.STATUS_UNAVAILABLE
    else:
        status = _common.STATUS_DEGRADED

    data = {
        "services": reports,
        "reachable": reachable,
        "unreachable": unreachable,
        "all_capable": all(r.get("capable") for r in reports.values()) if reports else False,
    }
    return _common.make_response(
        status, data, warnings=warnings, errors=errors,
        evidence=evidence, duration_ms=_common.elapsed_ms(started))
