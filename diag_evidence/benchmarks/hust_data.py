"""Pinned CC BY 4.0 HUST subset, split by bearing type BEFORE windowing."""
from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 51200
WINDOW_SIZE = 4096
WINDOWS_PER_RECORD = 32
SPLITS = {6205: "train", 6206: "train", 6207: "validation", 6208: "test"}


def manifest() -> dict:
    return json.loads(files(__package__).joinpath("hust_manifest.json").read_text(encoding="utf-8"))


def verify_file(path: Path, row: dict) -> None:
    if path.stat().st_size != row["bytes"]:
        raise ValueError("HUST file size differs from the pinned manifest")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != row["sha256"]:
        raise ValueError("HUST file checksum differs from the pinned manifest")


def ensure_data(cache_dir: str | Path, *, download: bool = False,
                records: list[dict] | None = None) -> list[dict]:
    """Download only with explicit opt-in; verify every file before parsing."""
    cache = Path(cache_dir)
    rows = manifest()["files"] if records is None else records
    if download:
        cache.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = cache / row["filename"]
        if not path.exists():
            if not download:
                raise FileNotFoundError("Pinned HUST data missing; use --download explicitly")
            import requests
            part = path.with_suffix(".part")
            try:
                with requests.Session() as session:
                    # Public data do not need .netrc credentials or proxy auth.
                    session.trust_env = False
                    with session.get(row["url"], stream=True, timeout=(15, 60)) as response:
                        response.raise_for_status()
                        if not response.url.startswith("https://"):
                            raise ValueError("Dataset download must remain HTTPS")
                        size = 0
                        with part.open("wb") as target:
                            for chunk in response.iter_content(256 * 1024):
                                size += len(chunk)
                                if size > row["bytes"]:
                                    raise ValueError("Dataset exceeds pinned size")
                                target.write(chunk)
                verify_file(part, row)
                part.replace(path)
            finally:
                part.unlink(missing_ok=True)
        verify_file(path, row)
    return rows


def window_starts(length: int, size: int = WINDOW_SIZE,
                  count: int = WINDOWS_PER_RECORD) -> np.ndarray:
    if (type(size) is not int or type(count) is not int or size < 64 or count < 1
            or length < size * count):
        raise ValueError("Insufficient samples for nonoverlapping windows")
    starts = np.linspace(0, length - size, count, dtype=np.int64)
    if count > 1 and np.any(np.diff(starts) < size):
        raise ValueError("Overlapping windows are prohibited")
    return starts


def load_windows(cache_dir: str | Path, *, download: bool = False,
                 normalize: bool = True, splits: tuple[str, ...] | None = None):
    """Read only `data`; the MAT `fs` is SHAFT frequency, not sample rate."""
    try:
        from scipy.io import loadmat
    except ImportError:
        raise RuntimeError("Install diag-evidence[benchmark] for MATLAB data") from None
    from .baseline import normalize_window
    if type(normalize) is not bool or (splits is not None and (
            not isinstance(splits, tuple) or not splits or set(splits) - set(SPLITS.values()))):
        raise ValueError("Invalid normalization or split selection")
    selected = [row for row in manifest()["files"] if splits is None or row["split"] in splits]
    rows = ensure_data(cache_dir, download=download, records=selected)
    samples, records = [], []
    for row in rows:
        split = SPLITS[row["bearing_type"]]
        if split != row["split"]:
            raise ValueError("Manifest bearing group and split disagree")
        matrix = loadmat(Path(cache_dir) / row["filename"], variable_names=["data"])
        raw = np.asarray(matrix["data"], dtype=np.float64)
        if (raw.ndim != 2 or raw.shape[1] != 1 or raw.shape[0] < WINDOW_SIZE * WINDOWS_PER_RECORD
                or not np.isfinite(raw).all()):
            raise ValueError("Unexpected HUST signal shape or values")
        raw = raw[:, 0]
        for start in window_starts(len(raw)):
            records.append({"sample_id": f"sample-{len(records):05d}",
                            "record_id": row["filename"], "split": split,
                            "bearing_type": row["bearing_type"], "load_w": row["load_w"],
                            "start": int(start), "label": row["label"],
                            "record_samples": len(raw), "record_duration_seconds": len(raw) / SAMPLE_RATE,
                            "source_sha256": row["sha256"]})
            window = raw[start:start + WINDOW_SIZE]
            samples.append(normalize_window(window) if normalize else window.copy())
    return np.stack(samples), records
