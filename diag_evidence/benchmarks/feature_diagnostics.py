"""Descriptive drift and exact linear-margin attribution; not an OOD detector."""
import numpy as np


def legacy_feature_names():
    return (["skewness", "kurtosis", "log_peak", "abs_q95", "zero_crossing", "envelope_std", "spectral_centroid"]
        + [f"power_{i * 800}_{(i + 1) * 800}Hz" for i in range(32)]
        + [f"envelope_band_{i}" for i in range(16)])


def linear_margin_audit(source, target, parameters, names):
    source, target = np.asarray(source), np.asarray(target)
    mean, scale = np.asarray(parameters["mean"]), np.asarray(parameters["scale"])
    coef, bias = np.asarray(parameters["coefficients"]), np.asarray(parameters["intercept"])
    if (source.ndim != 2 or target.ndim != 2 or len(source) == 0 or len(target) == 0 or source.shape[1] != target.shape[1]
            or source.shape[1] != len(names) or mean.shape != scale.shape or len(mean) != len(names)
            or coef.shape != (4, len(names)) or bias.shape != (4,) or np.any(scale <= 0)
            or not all(np.isfinite(a).all() for a in [source, target, mean, scale, coef, bias])):
        raise ValueError("Incompatible frozen linear feature audit")
    zs, zt = (source - mean) / scale, (target - mean) / scale
    logits = zt @ coef.T + bias
    competitor = logits[:, :3].argmax(axis=1)
    contributions = zt * (coef[3] - coef[competitor])
    intercepts = bias[3] - bias[competitor]
    margin = logits[:, 3] - logits[np.arange(len(logits)), competitor]
    if not np.allclose(contributions.sum(axis=1) + intercepts, margin, atol=1e-10):
        raise ValueError("Linear attribution does not sum to its declared margin")
    rows = []
    for i, name in enumerate(names):
        low, high = np.quantile(zs[:, i], [.01, .99])
        rows.append({"feature": name, "index": i, "source_z_p01": float(low), "source_z_p99": float(high),
            "target_z_median": float(np.median(zt[:, i])),
            "target_outside_source_interval_fraction": float(np.mean((zt[:, i] < low) | (zt[:, i] > high))),
            "healthy_margin_mean_contribution": float(np.mean(contributions[:, i]))})
    rows.sort(key=lambda r: abs(r["healthy_margin_mean_contribution"]), reverse=True)
    return {"scope": "post-hoc descriptive association, not causal proof or calibrated unknown detection",
            "features": rows, "healthy_margin_median": float(np.median(margin)),
            "intercept_mean": float(intercepts.mean()), "attribution_additivity_checked": True}
