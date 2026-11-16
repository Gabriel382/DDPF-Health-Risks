#!/usr/bin/env python3
"""Explicit downloads from the archived, immutable Zenodo records.

No 2.2 GB dump is downloaded unless --include-dump is provided.
Preserves downloaded files separately from existing repository ontology files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

FILES = (
    {'record': '10214515', 'name': 'PropaPhen.owl', 'md5': '1283bbe91596384af58e17395db200bd', 'size': '587.1 kB', 'large': False},
    {'record': '10987743', 'name': 'BioPropaPhen.rdf', 'md5': '1d211f9532efe1e6633dd33860abe08c', 'size': '2.1 MB', 'large': False},
    {'record': '10987743', 'name': 'BioPropaPhenKG-COVIDvsMPOX.dump', 'md5': '4aced6b0b71bda7f301b81a60d84d307', 'size': '2.2 GB', 'large': True},
)


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # nosec: archive integrity only, never security authentication
    with path.open('rb') as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def fetch(item: dict, dest: Path, retries: int = 2) -> dict:
    url = f"https://zenodo.org/records/{item['record']}/files/{item['name']}?download=1"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and md5_file(dest) == item['md5']:
        return {'file': str(dest), 'status': 'already_verified', 'md5': item['md5']}
    if dest.exists():
        raise RuntimeError(f'Checksum mismatch for existing file {dest}. Refusing to overwrite.')
    part = dest.with_name(dest.name + '.part')
    for attempt in range(retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {'User-Agent': 'PreDiViD-Reproduction/1.0'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=90) as response:
                status = getattr(response, 'status', 200)
                if offset and status != 206:
                    # Server ignored Range; restart safely, don't append duplicate bytes.
                    offset = 0
                with part.open('ab' if offset else 'wb') as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            actual = md5_file(part)
            if actual != item['md5']:
                raise RuntimeError(f'MD5 mismatch for {item["name"]}: {actual} != {item["md5"]}. Kept .part for inspection.')
            part.replace(dest)
            return {'file': str(dest), 'status': 'downloaded_verified', 'md5': actual}
        except (HTTPError, URLError, TimeoutError) as err:
            if attempt == retries:
                raise RuntimeError(f'Download failed after retries: {url}: {err}') from err
            time.sleep(attempt + 1)
    raise AssertionError('Unreachable')


def main() -> int:
    parser = argparse.ArgumentParser(description='Download exact Zenodo file snapshots with MD5 integrity checks')
    parser.add_argument('--root', default='.', help='Repository root')
    parser.add_argument('--include-dump', action='store_true', help='Explicitly download the 2.2 GB Neo4j 4.4.3 dump')
    parser.add_argument('--check', action='store_true', help='Verify locally available files only, no network')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    records = []
    for item in FILES:
        if item['large'] and not args.include_dump:
            continue
        dest = root / 'data' / 'zenodo' / item['record'] / item['name']
        if args.check:
            if not dest.exists():
                result = {'file': str(dest), 'status': 'missing'}
            else:
                actual = md5_file(dest)
                result = {'file': str(dest), 'status': 'verified' if actual == item['md5'] else 'MD5_MISMATCH', 'md5': actual}
        else:
            result = fetch(item, dest)
        result['record'] = item['record']
        result['archive_name'] = item['name']
        result['expected_md5'] = item['md5']
        records.append(result)
        print(f"{item['record']} {item['name']}: {result['status']}")
    local_owl = root / 'data' / 'ontologies' / 'PropaPhen' / 'PropaPhen.owl'
    if local_owl.exists():
        actual = md5_file(local_owl)
        print('Existing repository PropaPhen.owl MD5:', actual,
              '(matches Zenodo:' + str(actual == FILES[0]['md5']) + ')')
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'records': records, 'source': 'Zenodo 10214515 and 10987743'}
    target = root / 'outputs' / 'realdata' / 'zenodo_manifest.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if all(r['status'] not in ('MD5_MISMATCH',) for r in records) else 1

if __name__ == '__main__':
    sys.exit(main())
