"""Validated retrieval adapter with explicit score provenance and local counts."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional

from ..contracts import SignalValidationError, finite_number, validate_signal
from ..haconfig import settings
from . import _common

_Transport = Callable[..., Any]
MAX_RAG_RESULTS = 100
MAX_CASE_TEXT_CHARS = 16_384
MAX_METADATA_CHARS = 2_000
_CONTEXT_FIELDS = ("asset_id", "component", "channel", "units", "window_start",
                   "window_end", "preprocessing_id", "sampling_rate",
                   "available_at", "evidence_role")


class RAGResponseError(ValueError):
    """Local shape, budget or identity failure; never contains remote text."""


def _identifier(value: Any) -> Optional[str]:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    try:
        text = str(value).strip()
    except ValueError:
        raise RAGResponseError("案例身份超出支持范围") from None
    if len(text) > MAX_METADATA_CHARS:
        raise RAGResponseError("案例身份超过 2000 字符预算")
    try:
        text.encode("utf-8")
    except UnicodeError:
        raise RAGResponseError("案例身份包含无效 Unicode 字符") from None
    return _common.redact_text(text) if text else None


def _first_identifier(*values: Any) -> Optional[str]:
    for value in values:
        result = _identifier(value)
        if result is not None:
            return result
    return None


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _normalize_results(results: List[Dict[str, Any]], score_metadata: dict | None = None) -> List[Dict[str, Any]]:
    """Deduplicate stable content and source-scoped IDs, never scoring metadata.

    A reused source/case ID with different content or label fails the response
    instead of silently discarding a potentially relevant alternative.
    """
    if not isinstance(results, list) or len(results) > MAX_RAG_RESULTS:
        raise RAGResponseError("检索响应超过 100 条原始条目预算")
    metadata = score_metadata if isinstance(score_metadata, dict) else {}
    normalized = []
    seen_ids = {}
    seen_content = {}
    for index, item in enumerate(results, 1):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue
        if len(item["text"]) > MAX_CASE_TEXT_CHARS:
            raise RAGResponseError("案例文本超过 16384 字符预算")
        try:
            item["text"].encode("utf-8")
        except UnicodeError:
            raise RAGResponseError("案例文本包含无效 Unicode 字符") from None
        if not item["text"].strip():
            continue
        raw_meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        if "evidence_role" in raw_meta and not isinstance(raw_meta["evidence_role"], str):
            raise RAGResponseError("案例证据角色类型无效")
        allowed_meta = ("fault_type", "label", "record_id", "source_id", "case_id", "asset_id",
                        "component", "channel", "units", "window_start", "window_end", "as_of",
                        "available_at", "preprocessing_id", "score_type", "evidence_role")
        for key in allowed_meta:
            if isinstance(raw_meta.get(key), str):
                if len(raw_meta[key]) > MAX_METADATA_CHARS:
                    raise RAGResponseError("案例元信息超过 2000 字符预算")
                try:
                    raw_meta[key].encode("utf-8")
                except UnicodeError:
                    raise RAGResponseError("案例元信息包含无效 Unicode 字符") from None
        meta = {key: _common.redact_text(raw_meta[key]) for key in allowed_meta
                if isinstance(raw_meta.get(key), str)}
        for key in ("record_id", "case_id", "source_id", "asset_id"):
            if isinstance(raw_meta.get(key), int) and not isinstance(raw_meta.get(key), bool):
                meta[key] = _identifier(raw_meta[key])
        if isinstance(raw_meta.get("source_url"), str):
            # Remove all query/fragment/userinfo, including unknown signature names.
            source_url = _common.safe_source_url(raw_meta["source_url"])
            if source_url is not None and len(source_url) <= MAX_METADATA_CHARS:
                meta["source_url"] = _common.redact_text(source_url)
        if "sampling_rate" in raw_meta:
            try:
                rate = finite_number(raw_meta["sampling_rate"])
                if rate > 0:
                    meta["sampling_rate"] = rate
            except ValueError:
                pass
        if isinstance(raw_meta.get("score_is_normalized"), bool):
            meta["score_is_normalized"] = raw_meta["score_is_normalized"]
        identity = _first_identifier(item.get("case_id"), item.get("id"), meta.get("case_id"))
        source_id = _first_identifier(item.get("source_id"), meta.get("source_id"))
        record_id = _first_identifier(item.get("record_id"), meta.get("record_id"))
        if source_id is not None:
            meta["source_id"] = source_id
        if record_id is not None:
            meta["record_id"] = record_id
        text = _common.redact_text(item["text"])
        context = {key: meta[key] for key in _CONTEXT_FIELDS if key in meta}
        # Explicit and implicit historical-reference roles share an identity.
        context.setdefault("evidence_role", "historical_reference")
        stable_content = {"source_id": source_id, "record_id": record_id,
                          "context": context, "text": " ".join(text.split())}
        content_key = _digest(stable_content)
        fault_type = meta.get("fault_type") or meta.get("label")
        snapshot = _digest({"content": stable_content, "fault_type": fault_type})
        key = "case:" + _digest([source_id, identity]) if identity is not None else "content:" + content_key
        if key in seen_ids and seen_ids[key] != snapshot:
            raise RAGResponseError("同一来源/案例身份对应不同内容、记录或标签")
        if content_key in seen_content and seen_content[content_key] != snapshot:
            raise RAGResponseError("同一案例内容对应不一致标签")
        duplicate = key in seen_ids or content_key in seen_content
        seen_ids[key] = snapshot
        seen_content[content_key] = snapshot
        if duplicate:
            continue
        score = item.get("score")
        try:
            score = finite_number(score)
            score_valid = True
        except ValueError:
            score, score_valid = None, False
        flag = item.get("score_is_normalized", meta.get("score_is_normalized", metadata.get("score_is_normalized")))
        if not isinstance(flag, bool):
            flag = None
        score_type = item.get("score_type", meta.get("score_type", metadata.get("score_type", "unknown")))
        if not isinstance(score_type, str) or not score_type.strip():
            score_type = "unknown"
        if len(score_type) > MAX_METADATA_CHARS:
            raise RAGResponseError("评分元信息超过 2000 字符预算")
        try:
            score_type.encode("utf-8")
        except UnicodeError:
            raise RAGResponseError("评分元信息包含无效 Unicode 字符") from None
        normalized.append({
            "rank": len(normalized) + 1, "source_rank": item.get("rank") if isinstance(item.get("rank"), int) and not isinstance(item.get("rank"), bool) else index,
            "case_id": key, "source_case_id": identity, "source_id": source_id, "record_id": record_id,
            "score": score, "score_valid": score_valid,
            "score_is_normalized": flag, "score_type": _common.redact_text(score_type),
            "text": text, "fault_type": fault_type, "meta": meta,
        })
    return normalized


def retrieve_cases(signal: Any, top_k: int = 5, transport: Optional[_Transport] = None,
                   normalize: bool = True, normalize_scores: bool = False,
                   config: settings.Settings | None = None) -> Dict[str, Any]:
    """Retrieve cases without interpreting a retrieval score as a fault probability."""
    started = time.time()
    cfg = settings.resolve_config(config)
    try:
        values = validate_signal(signal, expected_length=cfg.signal_window_size)
    except SignalValidationError as exc:
        return _common.invalid_input_response([str(exc)], duration_ms=_common.elapsed_ms(started))
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        return _common.invalid_input_response(["top_k 必须是 1..100 之间的整数"], duration_ms=_common.elapsed_ms(started))
    if not isinstance(normalize, bool) or not isinstance(normalize_scores, bool):
        return _common.invalid_input_response(["normalize 和 normalize_scores 必须为布尔值"], duration_ms=_common.elapsed_ms(started))

    call = transport or _common.http_call
    kind, body, error = call(
        "POST", cfg.rag_service.url(settings.RAG_SEARCH_PATH),
        payload={"signal": values, "top_k": top_k, "normalize": normalize, "normalize_scores": normalize_scores},
        timeout=cfg.rag_search_timeout, max_retries=cfg.max_retries,
        backoff=cfg.retry_backoff, service=cfg.rag_service.name)
    duration = _common.elapsed_ms(started)
    if kind == "invalid_response":
        return _common.error_response(["RAG 响应违反本地重定向、编码或字节预算策略"], duration_ms=duration)
    if kind == "conn":
        return _common.unavailable_response(["RAG 服务不可达或请求超时"], duration_ms=duration)
    if kind in ("client", "server", "not_found"):
        return _common.make_response(
            _common.STATUS_UNAVAILABLE if kind == "server" else _common.STATUS_INVALID_INPUT,
            errors=[f"RAG 服务返回 HTTP 错误（{kind}）"], duration_ms=duration)
    if kind != "ok" or not isinstance(body, dict):
        return _common.error_response(["RAG 响应不是合法 JSON 对象"], duration_ms=duration)
    raw_results = body.get("results")
    if not isinstance(raw_results, list):
        return _common.error_response(["RAG results 必须为列表"], duration_ms=duration)
    try:
        cases = _normalize_results(raw_results, body)[:top_k]
    except RAGResponseError as exc:
        return _common.error_response([str(exc)], duration_ms=_common.elapsed_ms(started))
    warnings = []
    if len(cases) < len(raw_results):
        warnings.append("已移除重复、无效或超出 top_k 的检索条目；条目数由本地重新计算")
    declared = body.get("num_results")
    if declared is not None and (isinstance(declared, bool) or not isinstance(declared, int) or declared != len(cases)):
        warnings.append("服务声明的 num_results 与本地有效条目数不一致")
    if any(not case["score_valid"] for case in cases):
        warnings.append("部分检索 score 缺失或不是有限数值，已标记为不可用")
    num_results = len(cases)
    data = {"cases": cases, "num_results": num_results,
            "returned_count": len(raw_results), "reported_num_results": declared if isinstance(declared, int) and not isinstance(declared, bool) and declared >= 0 else None,
            "independent_record_count": len({(c["source_id"], c["record_id"]) for c in cases if c["record_id"] is not None}),
            "score_normalization_requested": normalize_scores,
            "signal_summary": _common.summarize_signal(values)}
    if not cases:
        warnings.append("RAG 无检索结果，不得据此推断故障")
        return _common.degraded_response(data, warnings,
            evidence=[_common.make_evidence("rag", "retrieved_case", "检索返回 0 条案例",
                                          {"num_results": 0}, trust="unavailable")],
            duration_ms=duration)
    if num_results < 3:
        data["retrieval_scale_note"] = f"文本库仅返回 {num_results} 条，规模有限"
    evidence = [_common.make_evidence("rag", "retrieved_case",
        f"检索到 {num_results} 条去重案例，分数含义以服务元数据为准",
        {"top": cases[:3], "num_results": num_results, "score_is_normalized": None},
        trust="medium")]
    if warnings:
        return _common.degraded_response(data, warnings, evidence=evidence, duration_ms=duration)
    return _common.ok_response(data, evidence=evidence, duration_ms=duration)
