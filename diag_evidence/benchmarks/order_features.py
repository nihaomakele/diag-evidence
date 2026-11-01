"""Offline envelope/order features. These filters and windows are noncausal.

Encoder phase is measured relative rotation, not a wall-clock timestamp. HUST
uses its recorded constant shaft-Hz scalar; no instantaneous phase is invented.
"""
import numpy as np

from .baseline import normalize_window

FS = 51200
CPR = 1024
TURNS = 8
POINTS_PER_TURN = 512
WINDOW_COUNT = 8
FEATURE_VERSION = "envelope-8turns-40orderbands-v1"


def encoder_phase(channel, target_length):
    raw = np.asarray(channel)
    if np.iscomplexobj(raw) or (raw.ndim != 1 and not (raw.ndim == 2 and 1 in raw.shape)):
        raise ValueError("Encoder must be one real-valued channel")
    x = raw.astype(float).reshape(-1)
    if len(x) < 200000 or not np.isfinite(x).all() or type(target_length) is not int or target_length < 1:
        raise ValueError("Invalid encoder input")
    lo, hi = np.quantile(x, [.1, .9])
    if hi - lo <= 1e-9:
        raise ValueError("Encoder has no usable pulse levels")
    high = x > (lo + hi) / 2
    edges = np.flatnonzero(high[1:] & ~high[:-1]) + 1
    if len(edges) < CPR * 68 or np.mean(np.diff(edges) < 3) > .001:
        raise ValueError("Insufficient or inconsistent encoder pulses")
    bins = np.arange(0, len(x) + 1, 100000)
    speed = np.histogram(edges, bins=bins)[0] / (.5 * CPR)
    if not len(speed) or speed.min() < 8 or speed.max() > 45:
        raise ValueError("Measured speed outside fixed 8..45 Hz study range")
    target_positions = np.arange(target_length) * 125 / 32
    if target_positions[-1] >= len(x):
        raise ValueError("Encoder and resampled vibration lengths disagree")
    phase = np.interp(target_positions, edges, np.arange(len(edges)) / CPR)
    return phase, {"cpr": CPR, "rising_edges": len(edges), "half_second_shaft_hz": speed.tolist(),
                   "speed_min_hz": float(speed.min()), "speed_max_hz": float(speed.max())}


def constant_phase(length, shaft_hz):
    if not np.isfinite(shaft_hz) or not 8 <= shaft_hz <= 45:
        raise ValueError("Recorded shaft frequency outside study range")
    return np.arange(length) * float(shaft_hz) / FS


def window_starts_in_turns(phase):
    phase = np.asarray(phase)
    if (phase.ndim != 1 or len(phase) < 2 or not np.isfinite(phase).all() or np.any(np.diff(phase) < 0)
            or phase[-1] - phase[0] < WINDOW_COUNT * TURNS + 4):
        raise ValueError("Insufficient rotation for eight separated windows and margins")
    starts = np.linspace(phase[0] + 2, phase[-1] - 2 - TURNS, WINDOW_COUNT)
    if np.any(np.diff(starts) < TURNS - 1e-10):
        raise ValueError("Angular windows overlap")
    return starts


def offline_envelope(signal):
    from scipy.signal import butter, hilbert, sosfiltfilt
    x = normalize_window(signal)
    carrier = sosfiltfilt(butter(4, [2000, 10000], btype="bandpass", fs=FS, output="sos"), x)
    envelope = np.abs(hilbert(carrier))
    # Protect angular interpolation from aliasing. At >=8 Hz, 512 samples/turn
    # gives a >=2048-Hz Nyquist frequency; the fixed envelope cutoff is 1000 Hz.
    return sosfiltfilt(butter(4, 1000, btype="lowpass", fs=FS, output="sos"), envelope)


def envelope_vector(envelope, axis):
    """44 features: four normalized modulation statistics plus forty order bands."""
    envelope, axis = np.asarray(envelope, dtype=float), np.asarray(axis, dtype=float)
    if (envelope.ndim != 1 or len(envelope) < 64 or axis.shape != (len(envelope) // 2 + 1,)
            or not np.isfinite(envelope).all() or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0)):
        raise ValueError("Invalid envelope or spectral axis")
    centered = envelope - envelope.mean()
    rms = float(np.sqrt(np.mean(centered ** 2)))
    mean = float(np.mean(np.abs(envelope)))
    if rms <= max(mean, 1e-12) * 1e-12:
        return np.array([0., 0., 0., 0.] + [np.log(1e-12)] * 40)
    u = centered / rms
    power = np.abs(np.fft.rfft(u * np.hanning(len(u)))) ** 2
    power[0] = 0
    if axis.shape != power.shape:
        raise ValueError("Spectrum axis and envelope length disagree")
    edges = np.linspace(0, 20, 41)
    bands = np.array([power[(axis >= a) & (axis < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
    bands = np.log(bands / max(float(bands.sum()), 1e-30) + 1e-12)
    stats = [rms / max(mean, 1e-12), float(np.mean(u ** 3)), float(np.mean(u ** 4)),
             float(np.log(max(float(np.abs(u).max()), 1.)))]
    result = np.concatenate([stats, bands])
    if not np.isfinite(result).all():
        raise ValueError("Envelope features are nonfinite")
    return result


def paired_features(signal, phase):
    signal, phase = np.asarray(signal), np.asarray(phase)
    if signal.ndim != 1 or phase.shape != signal.shape:
        raise ValueError("Vibration and relative phase must align")
    starts = window_starts_in_turns(phase)
    envelope = offline_envelope(signal)
    sample_positions = np.arange(len(signal))
    angular, nominal, crops, windows = [], [], [], []
    size = TURNS * POINTS_PER_TURN
    for start in starts:
        left, right = np.interp([start, start + TURNS], phase, sample_positions)
        angular_phase = start + np.arange(size) / POINTS_PER_TURN
        angular_positions = np.interp(angular_phase, phase, sample_positions)
        angular_envelope = np.interp(angular_positions, sample_positions, envelope)
        order_axis = np.fft.rfftfreq(size, d=1 / POINTS_PER_TURN)
        time_positions = np.linspace(left, right, size, endpoint=False)
        time_envelope = np.interp(time_positions, sample_positions, envelope)
        duration = (right - left) / FS
        nominal_axis = np.fft.rfftfreq(size, d=duration / size) / 25.
        angular.append(envelope_vector(angular_envelope, order_axis))
        nominal.append(envelope_vector(time_envelope, nominal_axis))
        center = int(round((left + right) / 2))
        crop = signal[center - 2048:center + 2048]
        if len(crop) != 4096:
            raise ValueError("Legacy center crop falls outside the measured record")
        crops.append(crop)
        windows.append({"start_turn": float(start), "end_turn": float(start + TURNS),
                        "start_sample": float(left), "end_sample": float(right), "duration_seconds": float(duration)})
    return np.stack(angular), np.stack(nominal), np.stack(crops), windows
