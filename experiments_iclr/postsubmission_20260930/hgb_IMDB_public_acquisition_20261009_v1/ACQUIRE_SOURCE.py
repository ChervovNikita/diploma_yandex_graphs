"""Acquire the specified public HGB archive; extract development inputs only."""
from pathlib import Path, PurePosixPath
import datetime
import hashlib
import json
import resource
import socket
import stat
import subprocess
import time
import urllib.request
import zipfile

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
Q = P / 'hgb_IMDB_public_acquisition_20261009_v1'
URL = 'https://drive.google.com/uc?export=download&id=18qXmmwKJBrEJxVQaYwKTL3Ny3fPqJeJ2'
ALLOW = {'node.dat', 'link.dat', 'label.dat', 'info.dat', 'meta.dat'}

assert socket.gethostname() == 'anogena-2-0'
assert subprocess.check_output(
    ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True
).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert Q.resolve().is_relative_to(R) and not Q.is_symlink()
Q.mkdir(parents=True, exist_ok=True)
assert not (Q / 'ACQUISITION.json').exists()
started = time.perf_counter()
record = dict(
    schema='hgb-public-archive-development-extraction-v1',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    canonical_HGB_commit='ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe',
    public_Drive_file_id='18qXmmwKJBrEJxVQaYwKTL3Ny3fPqJeJ2',
    public_Drive_folder_id='10-pf2ADCjq_kpJKFHHLHxr_czNNCJ3aX',
    source_url=URL, reported_archive_bytes=2102389,
    scientific_fit_started=False, TEST_truth_opened=False,
    official_TEST_membership_loaded=False, allowed_basenames=sorted(ALLOW),
    extracted=[], archive_members=[], status='started',
)
partial = Q / 'IMDB.zip.partial'
try:
    request = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=30) as response, partial.open('xb') as output:
        record['resolved_download_url'] = response.geturl()
        record['response_content_type'] = response.headers.get('Content-Type')
        total = 0
        while True:
            chunk = response.read(256 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > 32 * 1024 * 1024:
                raise ValueError('Public payload exceeded the declared acquisition size bound')
            output.write(chunk)
    record['downloaded_bytes'] = partial.stat().st_size
    record['archive_sha256'] = hashlib.sha256(partial.read_bytes()).hexdigest()
    if not zipfile.is_zipfile(partial):
        raise ValueError('Canonical public download did not return a ZIP archive')
    archive = Q / 'IMDB.zip'
    assert not archive.exists()
    partial.rename(archive)
    extracted = Q / 'development_inputs'
    extracted.mkdir(exist_ok=False)
    names = set()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            rel = PurePosixPath(member.filename)
            assert not rel.is_absolute() and '..' not in rel.parts
            assert not stat.S_ISLNK(member.external_attr >> 16)
            record['archive_members'].append(dict(
                name=member.filename, compressed_bytes=member.compress_size,
                bytes=member.file_size, extracted=False,
            ))
            if member.is_dir() or rel.name not in ALLOW:
                continue
            assert rel.name not in names and member.file_size <= 128 * 1024 * 1024
            names.add(rel.name)
            # label.dat.test and all other TEST members are never opened.
            data = source.read(member)
            target = extracted / rel.name
            with target.open('xb') as output:
                output.write(data)
            record['extracted'].append(dict(
                member=member.filename, path=str(target.relative_to(P)),
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
            ))
            record['archive_members'][-1]['extracted'] = True
    assert {'node.dat', 'link.dat', 'label.dat'} <= names
    record.update(status='complete', archive_path=str(archive.relative_to(P)),
                  untouched_TEST_members=[x['name'] for x in record['archive_members']
                                          if 'test' in x['name'].lower()])
except Exception as error:
    record.update(status='failed', failure_type=type(error).__name__, failure=str(error))
    raise
finally:
    record['inclusive_wall_seconds'] = time.perf_counter() - started
    record['CPU_seconds'] = resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime
    record['peak_RSS_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with (Q / 'ACQUISITION.json').open('x') as output:
        json.dump(record, output, indent=2)
        output.write('\n')
    print(json.dumps(record))
