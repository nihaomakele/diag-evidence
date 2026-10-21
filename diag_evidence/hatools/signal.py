"""工具 3：analyze_signal —— 对 1024 点振动信号做质量检查与特征提取。

设计原则（对应详细计划 7.3）：
- 第一版不重写原项目算法，只用统计量刻画信号；
- 输入不满足点数或格式时返回 invalid_input；
- 非有限、恒定值或近零能量输入均返回 invalid_input。
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..haconfig import settings
from ..contracts import SignalValidationError, finite_number, validate_signal
from . import _common

# Engineering bounds shared with the input contract.
MIN_STD = 1e-9
MIN_ENERGY = 1e-12


SignalQualityError = SignalValidationError


def _to_array(signal: Any) -> np.ndarray:
    return np.asarray(validate_signal(signal, expected_length=None), dtype=np.float64)


def analyze_signal(
    signal: Any,
    sampling_rate: float | None = None,
    expected_length: int | None = None,
    generate_plot: bool = False,
    config: settings.Settings | None = None,
) -> Dict[str, Any]:
    """分析振动信号，返回统一响应。

    generate_plot=True 时尝试输出 CWT 时频图（需要 pywt/matplotlib，缺失则降级）。
    """
    started = time.time()
    cfg = settings.resolve_config(config)
    sampling_rate = cfg.sampling_rate if sampling_rate is None else sampling_rate
    expected_length = cfg.signal_window_size if expected_length is None else expected_length
    try:
        rate = finite_number(sampling_rate)
        if not 0 < rate <= 1e12:
            raise ValueError("sampling_rate 必须在 0..1e12 Hz 之间")
        sampling_rate = rate
        values = np.asarray(validate_signal(signal, expected_length=expected_length), dtype=np.float64)
    except (SignalQualityError, ValueError) as exc:
        return _common.invalid_input_response([str(exc)], duration_ms=_common.elapsed_ms(started))

    warnings: List[str] = []

    std = float(np.std(values))
    energy = float(np.mean(values ** 2))

    centered = values - float(np.mean(values))
    safe_std = std if std > MIN_STD else 1.0
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(energy))
    crest = float(peak / rms) if rms > MIN_ENERGY else 0.0
    kurtosis = float(np.mean((centered / safe_std) ** 4))
    skewness = float(np.mean((centered / safe_std) ** 3))

    frequencies = np.fft.rfftfreq(len(values), d=1.0 / sampling_rate)
    magnitude = np.abs(np.fft.rfft(centered))
    total_power = float(np.sum(magnitude ** 2))
    if magnitude.size > 1 and total_power > MIN_ENERGY:
        peak_index = int(np.argmax(magnitude[1:]) + 1)
        peak_frequency = float(frequencies[peak_index])
        spectrum = magnitude[1:]
        freq_axis = frequencies[1:]
        weight = float(np.sum(spectrum))
        center_frequency = float(np.sum(freq_axis * spectrum) / weight) if weight > MIN_ENERGY else 0.0
        rms_frequency = float(math.sqrt(np.sum((freq_axis ** 2) * spectrum) / weight)) if weight > MIN_ENERGY else 0.0
    else:
        peak_frequency = center_frequency = rms_frequency = 0.0
        warnings.append("频谱无有效分量")

    data: Dict[str, Any] = {
        "time_domain": {
            "mean": round(float(np.mean(values)), 6),
            "variance": round(float(np.var(values)), 6),
            "std": round(std, 6),
            "max": round(float(np.max(values)), 6),
            "min": round(float(np.min(values)), 6),
            "peak": round(peak, 6),
            "kurtosis": round(kurtosis, 6),
            "skewness": round(skewness, 6),
            "rms": round(rms, 6),
            "crest_factor": round(crest, 6),
        },
        "frequency_domain": {
            "total_power": round(total_power, 6),
            "peak_frequency": round(peak_frequency, 4),
            "center_frequency": round(center_frequency, 4),
            "rms_frequency": round(rms_frequency, 4),
        },
        "quality": {
            "points": int(len(values)),
            "sampling_rate": sampling_rate,
            "finite": True,
            "constant_like": False,
            "empty_like": False,
        },
        "signal_summary": _common.summarize_signal(values),
    }

    if generate_plot:
        data["plot"] = _try_generate_plot(values, sampling_rate, warnings)

    evidence = [_common.make_evidence(
        "signal", "statistics",
        f"信号 {len(values)} 点，std={data['time_domain']['std']}，峰值频率={data['frequency_domain']['peak_frequency']} Hz",
        data["quality"],
        trust="medium" if warnings else "high")]

    if warnings:
        return _common.degraded_response(data, warnings, evidence=evidence,
                                        duration_ms=_common.elapsed_ms(started))
    return _common.ok_response(data, evidence=evidence, duration_ms=_common.elapsed_ms(started))


def _try_generate_plot(values: np.ndarray, sampling_rate: int, warnings: List[str]) -> Optional[Dict[str, Any]]:
    """尽力生成时频图；依赖缺失或失败时只记警告，不影响特征结果。"""
    try:
        import base64
        from io import BytesIO

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pywt

        wavelet = "cmor"
        central = pywt.central_frequency(wavelet)
        scales = np.linspace(central / (sampling_rate / 4), central / (1 / sampling_rate), 64)
        coeffs, _ = pywt.cwt(values, scales, wavelet, sampling_period=1.0 / sampling_rate)
        figure = plt.figure(figsize=(6.4, 6.4), dpi=100)
        plt.imshow(np.abs(coeffs), cmap="jet", aspect="auto", origin="lower")
        plt.axis("off")
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(figure)
        return {"format": "png-base64", "content": encoded}
    except Exception as exc:  # noqa: BLE001 - 图像失败不应影响特征
        warnings.append(f"时频图生成失败（{type(exc).__name__}），仅返回统计特征")
        return None
