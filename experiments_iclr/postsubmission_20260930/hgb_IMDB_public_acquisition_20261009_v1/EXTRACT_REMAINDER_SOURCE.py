"""Finish streaming development inputs from the already acquired exact ZIP."""
from pathlib import Path, PurePosixPath
import datetime
import hashlib
import json
import resource
import socket
import stat
import subprocess
import time
import zipfile

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
Q = P / 'hgb_IMDB_public_acquisition_20261009_v1'
assert socket.gethostname() == 'anogena-2-0'
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
archive = Q / 'IMDB.zip'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == 'dc98438c28f738ab1e7ba07aaeca4fc2e035d837b7bb3955fdd4f61f14dd81e1'
original = json.loads((Q / 'ACQUISITION.json').read_text())
assert original['status'] == 'failed' and original['failure_type'] == 'AssertionError'
assert original['downloaded_bytes'] == 2102389 and not original['TEST_truth_opened']
assert not (Q / 'EXTRACTION_COMPLETE.json').exists()
started = time.perf_counter()
allow = {'node.dat', 'link.dat', 'label.dat', 'info.dat', 'meta.dat'}
known = {PurePosixPath(row['member']).name: row for row in original['extracted']}
result = dict(
    schema='hgb-exact-archive-streaming-extraction-continuation-v1',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    archive_sha256=original['archive_sha256'], network_downloads=0,
    original_acquisition_receipt_sha256=hashlib.sha256((Q / 'ACQUISITION.json').read_bytes()).hexdigest(),
    amendment='Input member limit256MiB after observed185065301byte node.dat; streaming extraction',
    TEST_truth_opened=False, official_TEST_membership_loaded=False,
    scientific_fit_started=False, status='started', extracted=[], sealed_members=[],
)
try:
    names = set()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            rel = PurePosixPath(member.filename)
            assert not rel.is_absolute() and '..' not in rel.parts
            assert not stat.S_ISLNK(member.external_attr >> 16)
            if member.is_dir():
                continue
            if rel.name not in allow:
                result['sealed_members'].append(dict(name=member.filename, bytes=member.file_size))
                continue
            assert rel.name not in names and member.file_size <= 256 * 1024 * 1024
            names.add(rel.name)
            target = Q / 'development_inputs' / rel.name
            if rel.name in known:
                binding = known[rel.name]
                assert target.stat().st_size == binding['bytes']
                assert hashlib.sha256(target.read_bytes()).hexdigest() == binding['sha256']
                result['extracted'].append(dict(binding, reused_from_original=True))
                continue
            digest = hashlib.sha256()
            size = 0
            with source.open(member) as input_stream, target.open('xb') as output_stream:
                for block in iter(lambda: input_stream.read(1024 * 1024), b''):
                    output_stream.write(block)
                    digest.update(block)
                    size += len(block)
            assert size == member.file_size
            result['extracted'].append(dict(member=member.filename, path=str(target.relative_to(P)),
                                           bytes=size, sha256=digest.hexdigest(), reused_from_original=False))
    assert {'node.dat', 'link.dat', 'label.dat'} <= names
    result['status'] = 'complete'
except Exception as error:
    result.update(status='failed', failure_type=type(error).__name__, failure=str(error))
    raise
finally:
    result['inclusive_wall_seconds'] = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result['CPU_seconds'] = usage.ru_utime + usage.ru_stime
    result['peak_RSS_KiB'] = usage.ru_maxrss
    with (Q / 'EXTRACTION_COMPLETE.json').open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result))
