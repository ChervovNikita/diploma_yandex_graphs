"""Acquire the exact public archive; inspect ZIP metadata, never decode arrays."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import urllib.request
import zipfile

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO/'experiments_iclr/postsubmission_20260930'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
URL = 'https://zenodo.org/records/16895532/files/tolokers-2.zip?download=1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--supervisor', required=True)
    args = parser.parse_args()
    assert Path.cwd().resolve() == REPO and os.environ['GNNM_SSH_DESTINATION'] == LOGIN
    assert os.environ['GNNM_PHASE_ROOT'] == str(PHASE)
    assert subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        capture_output=True, text=True, check=True).stdout.splitlines() == [UUID]
    request_path = Path(args.request)
    assert request_path.is_relative_to(PHASE) and '..' not in request_path.parts
    request = json.loads(request_path.read_text())
    assert request['root_admitted'] and request['dataset'] == 'tolokers-2'
    assert not request['scientific_fit_admitted'] and not request['label_decoding_admitted']
    assert request['archive_url'] == URL and request['entry_sha256'] == sha(Path(__file__))
    assert request['whole_cap_seconds'] == 300
    metadata = Path(request['provider_metadata']['path'])
    assert metadata.is_relative_to(PHASE) and sha(metadata) == request['provider_metadata']['sha256']
    provider = json.loads(metadata.read_text())
    row = next(row for row in provider['datasets'] if row['name'] == 'tolokers-2')
    assert row['archive_metadata']['bytes'] == 3361114
    assert row['archive_metadata']['checksum'] == 'md5:f453141143f3a93a61502e461a03d222'
    supervisor = Path(args.supervisor)
    assert supervisor == Path(request['supervisor_directory']) and supervisor.is_relative_to(PHASE)
    command = json.loads((supervisor/'command.json').read_text())
    assert command['cwd'] == str(REPO) and command['argv'] == [str(REPO/'.venv/bin/python'),
        str(Path(__file__)), '--request', str(request_path), '--supervisor', str(supervisor)]
    output = PHASE/'industrial_dataset_acquisition_root_v1/tolokers2_run01'
    assert str(output) == request['output'] and not output.exists()
    output.mkdir()
    started = time.monotonic()
    write(output/'START.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
        request_sha256=sha(request_path), archive_url=URL, label_decoding=False))
    try:
        path = output/'tolokers-2.zip'
        req = urllib.request.Request(URL, headers={'User-Agent': 'GNNM-research-dataset-acquisition/1.0'})
        with urllib.request.urlopen(req, timeout=60) as response, path.open('xb') as stream:
            total = 0
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                assert total <= 20*1024**2, 'Archive exceeded declared download bound'
                stream.write(chunk)
        assert path.stat().st_size == 3361114
        assert hashlib.md5(path.read_bytes()).hexdigest() == 'f453141143f3a93a61502e461a03d222'
        inventory = []
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                relative = PurePosixPath(item.filename)
                assert not relative.is_absolute() and '..' not in relative.parts
                inventory.append(dict(name=item.filename, uncompressed_bytes=item.file_size,
                    compressed_bytes=item.compress_size, CRC=item.CRC, directory=item.is_dir()))
        write(output/'ARCHIVE_RECEIPT.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
            dataset='tolokers-2', archive_path=str(path), sha256=sha(path),
            bytes=path.stat().st_size, verified_provider_md5=True,
            source_metadata=request['provider_metadata'], inventory=inventory,
            arrays_or_labels_decoded=False, extracted=False, trained=False))
        write(output/'TERMINAL.json', dict(completed=True, seconds=time.monotonic()-started,
            arrays_or_labels_decoded=False, final_labels_read=False, trained=False,
            scientific_result=False, automatic_retry=False))
    except Exception as error:
        write(output/'FAILED.json', dict(completed=False, seconds=time.monotonic()-started,
            error_type=type(error).__name__, error=str(error), automatic_retry=False))
        raise

if __name__ == '__main__':
    main()
