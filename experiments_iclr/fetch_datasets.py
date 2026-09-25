"""Fetch missing public benchmark graph files from a pinned source commit.

Existing files are verified, never silently overwritten. Downloads are written
inside this repository's data directory and renamed only after SHA256 matches.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
MANIFEST = ROOT / 'experiments_iclr' / 'data_manifest.json'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    DATA.mkdir(parents=True, exist_ok=True)
    for filename, info in manifest['files'].items():
        path = DATA / filename
        if not path.exists():
            url = manifest['download_url_template'].format(filename=filename)
            part = DATA / (filename + '.part')
            try:
                with urllib.request.urlopen(url, timeout=60) as source, part.open('wb') as target:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        target.write(block)
                if part.stat().st_size != info['bytes'] or digest(part) != info['sha256']:
                    raise ValueError(f'Checksum or size mismatch for downloaded {filename}')
                part.replace(path)
            finally:
                part.unlink(missing_ok=True)
            print(f'Downloaded and verified {filename}')
        elif path.stat().st_size != info['bytes'] or digest(path) != info['sha256']:
            raise ValueError(f'Existing data file does not match manifest: {path}')
        else:
            print(f'Verified {filename}')


if __name__ == '__main__':
    main()
