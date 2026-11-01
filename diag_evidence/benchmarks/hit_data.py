"""HIT-SM original-source files, pinned by commit, byte count and Git blob SHA."""
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re

import numpy as np

from .hust_data import window_starts


def manifest():
    return json.loads(files(__package__).joinpath('hit_manifest.json').read_text(encoding='utf-8'))


def verify_file(path, row):
    path = Path(path)
    if path.stat().st_size != row['bytes']:
        raise ValueError('HIT byte count differs from pinned source')
    data = path.read_bytes()
    sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\x00' + data).hexdigest()
    if sha != row['git_blob_sha']:
        raise ValueError('HIT Git blob SHA differs from pinned source')
    return hashlib.sha256(data).hexdigest()


def ensure_data(cache_dir, *, download=False):
    directory = Path(cache_dir)
    if download:
        directory.mkdir(parents=True, exist_ok=True)
    receipts = []
    for row in manifest()['files']:
        if not re.fullmatch(r'(selfbuilt|spectraquest)__(IR[258]|OR[258]|Normal)_(600|900|1200)\.mat', row['filename']):
            raise ValueError('Invalid HIT cache filename')
        path = directory / row['filename']
        if not path.exists():
            if not download:
                raise FileNotFoundError('Pinned HIT files missing; use --download')
            import requests
            temporary = path.with_suffix('.part')
            try:
                with requests.Session() as session:
                    session.trust_env = False
                    with session.get(row['url'], stream=True, timeout=(15, 60)) as response:
                        response.raise_for_status()
                        if not response.url.startswith('https://raw.githubusercontent.com/'):
                            raise ValueError('HIT download left the declared source host')
                        size = 0
                        with temporary.open('wb') as target:
                            for chunk in response.iter_content(256 * 1024):
                                size += len(chunk)
                                if size > row['bytes']:
                                    raise ValueError('HIT download exceeds pinned byte count')
                                target.write(chunk)
                verify_file(temporary, row)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        receipts.append({**row, 'observed_sha256': verify_file(path, row)})
    return receipts


def vibration_vector(matrix):
    candidates = []
    for name, value in matrix.items():
        if name.startswith('__'):
            continue
        raw = np.asarray(value)
        if raw.size <= 1:
            continue
        if (raw.dtype.kind not in 'fiu' or raw.ndim not in (1, 2) or
                (raw.ndim == 2 and 1 not in raw.shape)):
            raise ValueError('HIT file must contain one real vibration vector')
        candidates.append((name, raw.astype(float).reshape(-1)))
    if len(candidates) != 1:
        raise ValueError('HIT signal variable is ambiguous or absent')
    name, signal = candidates[0]
    if len(signal) < 4096 or not np.isfinite(signal).all():
        raise ValueError('Invalid HIT vibration vector')
    return name, signal


def load_windows(cache_dir, *, download=False):
    from scipy.io import loadmat
    receipts = ensure_data(cache_dir, download=download)
    samples, rows, layouts = [], [], []
    for record in receipts:
        variable, signal = vibration_vector(loadmat(Path(cache_dir) / record['filename']))
        starts = window_starts(len(signal), count=min(32, len(signal) // 4096))
        layouts.append({'record_id': record['filename'], 'variable': variable,
                        'samples': len(signal), 'windows': len(starts)})
        for start in starts:
            samples.append(signal[start:start + 4096].copy())
            rows.append({'record_id': record['filename'], 'rig': record['rig'],
                         'label': record['label'], 'nominal_rpm': record['nominal_rpm'],
                         'split': 'external', 'start': int(start),
                         'source_sha256': record['observed_sha256']})
    return np.stack(samples), rows, receipts, layouts
