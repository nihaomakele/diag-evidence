"""Shared input contracts. Never coerce strings or booleans to numbers."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np


class SignalValidationError(ValueError):
    """A signal cannot be safely used by this diagnostic workflow."""


def finite_number(value: Any) -> float:
    """Return a finite real number, rejecting booleans and numeric strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("必须为有限实数，不能为布尔值或字符串")
    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError("数值超出可表示范围") from exc
    if not math.isfinite(result):
        raise ValueError("数值必须有限，不能为 NaN/Inf")
    return result


def validate_signal(signal: Any, expected_length: int | None = 1024,
                    min_points: int = 16) -> list[float]:
    """Validate a one dimensional vibration window before analysis or network I/O.

    This contract rejects std < 1e-9, mean square < 1e-12, and magnitude > 1e100.
    These are engineering bounds, not calibrated physical thresholds. Callers
    must use the units expected by their own model.
    """
    if expected_length is not None and (isinstance(expected_length, bool)
            or not isinstance(expected_length, int) or expected_length < min_points):
        raise SignalValidationError("expected_length 必须是至少 16 的整数或 None")
    if signal is None:
        raise SignalValidationError("缺少 signal 参数")
    if isinstance(signal, (str, bytes, dict)):
        raise SignalValidationError("signal 必须是一维有限数值序列")
    if isinstance(signal, np.ndarray) and signal.ndim != 1:
        raise SignalValidationError(f"signal 必须是一维序列，实际维度 {signal.ndim}")
    try:
        items = list(signal)
    except (TypeError, ValueError) as exc:
        raise SignalValidationError("signal 必须是一维有限数值序列") from exc
    if expected_length is not None and len(items) != expected_length:
        raise SignalValidationError(f"信号长度必须为 {expected_length}，实际 {len(items)}")
    if len(items) < min_points:
        raise SignalValidationError(f"信号点数过少（{len(items)}），至少需要 {min_points} 点")
    try:
        values = [finite_number(value) for value in items]
    except ValueError as exc:
        raise SignalValidationError(f"signal 必须是一维有限数值序列：{exc}") from exc
    if max(abs(value) for value in values) > 1e100:
        raise SignalValidationError("信号幅值超出安全计算范围（1e100）")
    array = np.asarray(values, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            std = float(np.std(array))
            energy = float(np.mean(np.square(array)))
        except FloatingPointError as exc:
            raise SignalValidationError("信号统计量溢出，无法分析") from exc
    if std < 1e-9:
        raise SignalValidationError("信号标准差接近 0，恒定信号无法用于该振动诊断流程")
    if energy < 1e-12:
        raise SignalValidationError("信号能量接近 0，无法用于该振动诊断流程")
    return values
