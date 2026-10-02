"""Stage the official OGB archive inside the authorized repository, without opening members."""
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
URL = 'https://snap.stanford.edu/ogb/data/linkproppred/collab.zip'
OUT = REPO / 'experiments_iclr/postsubmission_20260930/buddy_official_archive_staging_v1/root_stage_v1'


def main():
    if Path.cwd().resolve() != REPO or REPO.resolve() != REPO:
        raise RuntimeError('Authorized actual repository is required')
    if os.environ.get('GNNM_SSH_DESTINATION') != LOGIN:
        raise RuntimeError('Explicit authorized transport binding required')
    git = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=REPO,
                         check=True, capture_output=True, text=True)
    if Path(git.stdout.strip()).resolve() != REPO:
        raise RuntimeError('Actual project Git root differs')
    actual = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True)
    if actual.stdout.strip().splitlines() != [GPU]:
        raise RuntimeError('Authorized one-GPU inventory differs')
    metadata = REPO / '.venv/lib/python3.11/site-packages/ogb/linkproppred/master.csv'
    if not metadata.resolve().is_relative_to(REPO) or not metadata.is_file():
        raise RuntimeError('Repo-confined installed OGB metadata is required')
    metadata_bytes = metadata.read_bytes()
    rows = list(csv.DictReader(metadata_bytes.decode().splitlines()))
    url_rows = [r for r in rows if next(iter(r.values())) == 'url']
    advertised_url = url_rows[0].get('ogbl-collab') if len(url_rows) == 1 else None
    if advertised_url not in (URL, URL.replace('https:', 'http:', 1)):
        raise RuntimeError('Pinned official release URL differs from installed OGB metadata')
    OUT.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema='official-ogbl-collab-archive-staging-v1',
                   UTC=datetime.now(timezone.utc).isoformat(), ssh_destination=LOGIN,
                   GPU_uuid=GPU, GPU_compute=False, dataset_members_opened=False,
                   heldout_labels_or_pairs_opened=False, archive_extracted=False,
                   numerical_or_predictive_result=False, URL=URL,
                   advertised_OGB_URL=advertised_url,
                   OGB_metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    started = time.monotonic()
    archive = OUT / 'collab.zip'
    try:
        digest = hashlib.sha256()
        count = 0
        request = urllib.request.Request(URL, headers={'User-Agent': 'GNNM-research-archive-staging'})
        with urllib.request.urlopen(request, timeout=30) as response, archive.open('xb') as stream:
            if response.geturl() != URL:
                raise RuntimeError('Unexpected archive redirect')
            receipt['HTTP_headers'] = {k: response.headers.get(k) for k in
                                       ('Content-Length', 'Content-Type', 'ETag', 'Last-Modified')}
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > 2 * 1024**3:
                    raise RuntimeError('Archive exceeds fixed 2-GiB acquisition cap')
                stream.write(block)
                digest.update(block)
        expected_size = receipt['HTTP_headers']['Content-Length']
        if expected_size is not None and int(expected_size) != count:
            raise RuntimeError('Incomplete download')
        # Only central-directory metadata is read. No member stream is opened,
        # decompressed, deserialized, or hashed individually at this stage.
        with zipfile.ZipFile(archive) as packed:
            members = []
            for member in packed.infolist():
                path = Path(member.filename)
                if path.is_absolute() or '..' in path.parts:
                    raise RuntimeError('Unsafe official archive member path')
                members.append(dict(name=member.filename, bytes=member.file_size,
                                    compressed_bytes=member.compress_size, CRC=member.CRC))
        receipt.update(status='OFFICIAL_ARCHIVE_STAGED_MEMBERS_UNOPENED',
                       archive_sha256=digest.hexdigest(), archive_bytes=count, members=members)
    except Exception as error:
        receipt.update(status='ACQUISITION_FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt['seconds'] = time.monotonic() - started
        (OUT / 'STAGING_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({k: receipt.get(k) for k in ('status', 'archive_bytes', 'seconds', 'error')}, sort_keys=True))


if __name__ == '__main__':
    main()
