"""Remove two explicitly reviewed redundant files after preservation checks."""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
AUDIT = ROOT / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/planning/allocation_cleanup_20261027_v1'
TAR = 'experiments_iclr/postsubmission_20260930/final_sync_master_v29_20261013_v1/V29_EXACT_SOURCE_TRANSFER.tar'
TEMP = 'experiments_iclr/postsubmission_20260930/typed_context_allocation_qualification_execution_20261010_v1/untied_local_mul4/seed1/SELECTED_STATE.pt.tmp'
assert ROOT.resolve() == ROOT and AUDIT.resolve().is_relative_to(ROOT)

def metadata(path):
    value = path.lstat()
    assert stat.S_ISREG(value.st_mode) and path.resolve() == path
    return {'device': value.st_dev, 'inode': value.st_ino, 'bytes': value.st_size,
            'mtime_ns': value.st_mtime_ns, 'allocated_bytes': value.st_blocks * 512}

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()

preserved = json.loads((AUDIT / 'TRANSFER_PRESERVATION.json').read_text())
assert preserved['original_archive'] == TAR and preserved['all_historical_member_bytes_preserved']
assert metadata(ROOT / TAR) == preserved['original_metadata']
for row in preserved['members']:
    if row['disposition'] == 'identical_file_retained':
        path = ROOT / row['retained_path']
        assert path.resolve().is_relative_to(ROOT)
        assert metadata(path) == row['retained_metadata']
delta = ROOT / preserved['retained_difference_archive']
assert delta.resolve().is_relative_to(ROOT)
assert digest(delta) == preserved['difference_archive_sha256']

authority = json.loads((AUDIT / 'TEMP_REMOVAL_AUTHORITY.json').read_text())
record = authority['temporary_file']
assert record['path'] == TEMP and record['bytes'] == 4081471326
expected_temp = {key: record[key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'allocated_bytes')}
assert metadata(ROOT / TEMP) == expected_temp
terminal = authority['failure_evidence'][0]['parsed_observation']['qualifier_metadata']['TERMINAL.json']
assert terminal['exit_code'] == -15
temp_sha = digest(ROOT / TEMP)
assert metadata(ROOT / TEMP) == expected_temp

targets = {str(ROOT / TAR), str(ROOT / TEMP)}
open_handles = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdecimal():
        continue
    try:
        if proc.stat().st_uid != os.getuid():
            continue
        for fd in (proc / 'fd').iterdir():
            try:
                target = os.readlink(fd)
                if target in targets:
                    open_handles.append({'pid': proc.name, 'path': target})
            except OSError:
                pass
    except OSError:
        pass
assert not open_handles, open_handles

actions = [
    {'path': TAR, 'before': preserved['original_metadata'], 'sha256': preserved['original_sha256'],
     'reason': 'Every historical member retained identically or in verified historical-difference archive'},
    {'path': TEMP, 'before': expected_temp, 'sha256': temp_sha,
     'reason': authority['rationale']},
]
with (AUDIT / 'REDUNDANT_FILE_DELETION.jsonl').open('x') as log:
    for row in actions:
        path = ROOT / row['path']
        assert path.resolve().is_relative_to(ROOT) and metadata(path) == row['before']
        log.write(json.dumps({'event': 'verified_before_unlink', **row}) + '\n')
        log.flush()
        os.fsync(log.fileno())
        path.unlink()
        log.write(json.dumps({'event': 'deleted', 'path': row['path'], 'unix': time.time()}) + '\n')
        log.flush()
        os.fsync(log.fileno())
        print('Removed verified redundant file:', row['path'], flush=True)
result = {'deleted_files': actions, 'allocated_bytes_removed': sum(r['before']['allocated_bytes'] for r in actions),
          'original_transfer_history_preserved': True, 'metrics_and_ideas_deleted': False,
          'live_campaign_files_deleted': False, 'outside_repository_changes': False}
with (AUDIT / 'REDUNDANT_FILE_DELETION_RECEIPT.json').open('x') as out:
    json.dump(result, out, indent=2)
    out.write('\n')
print(json.dumps(result, indent=2), flush=True)
