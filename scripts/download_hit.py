"""Fetch pinned HIT-SM bytes through GitHub's public blob API."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import requests

from diag_evidence.benchmarks.hit_data import manifest, verify_file


def download_record(row, directory):
    name = row['filename']
    if Path(name).name != name:
        raise ValueError('Invalid cache name')
    path = directory / name
    if path.exists():
        return {'filename': name, 'sha256': verify_file(path, row), 'cached': True}
    url = 'https://api.github.com/repos/hitwzc/Bearing-datasets/git/blobs/' + row['git_blob_sha']
    temporary = path.with_suffix('.api-part')
    for attempt in range(3):
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.get(url, stream=True, timeout=(10, 30)) as response:
                    response.raise_for_status()
                    chunks, size = [], 0
                    for chunk in response.iter_content(256 * 1024):
                        size += len(chunk)
                        if size > 2 * row['bytes'] + 8192:
                            raise ValueError('Blob response exceeds declared bound')
                        chunks.append(chunk)
            blob = json.loads(b''.join(chunks))
            if blob['sha'] != row['git_blob_sha'] or blob['size'] != row['bytes'] or blob['encoding'] != 'base64':
                raise ValueError('Blob metadata differs from pinned source')
            data = base64.b64decode(''.join(blob['content'].split()), validate=True)
            temporary.write_bytes(data)
            sha = verify_file(temporary, row)
            temporary.replace(path)
            print(name + ' verified', flush=True)
            return {'filename': name, 'sha256': sha, 'cached': False}
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(1)
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError('Public download retries exhausted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', required=True, type=Path)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda row: download_record(row, args.cache_dir), manifest()['files']))
    print(json.dumps({'verified': len(receipts), 'cache': str(args.cache_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
