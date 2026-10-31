"""Preserve historical differences before deleting a redundant transfer container.

This program never deletes files, extracts source for execution, or loads tensors.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile
import time

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = ROOT / 'experiments_iclr/postsubmission_20260930'
AUDIT = PHASE / 'benchmark_expansion_20261015_v1/planning/allocation_cleanup_20261027_v1'
SOURCE = PHASE / 'final_sync_master_v29_20261013_v1/V29_EXACT_SOURCE_TRANSFER.tar'
DELTA = AUDIT / 'transfer_historical_differences.tar.gz'
assert ROOT.resolve() == ROOT and AUDIT.resolve().is_relative_to(ROOT)
assert SOURCE.resolve() == SOURCE
assert not DELTA.exists()

def digest(stream):
    value = hashlib.sha256()
    while True:
        block = stream.read(4 * 1024**2)
        if not block:
            return value.hexdigest()
        value.update(block)

def metadata(path):
    value = path.lstat()
    return {'device': value.st_dev, 'inode': value.st_ino,
            'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns,
            'allocated_bytes': value.st_blocks * 512}

before = metadata(SOURCE)
receipt = json.loads(SOURCE.with_name('V29_EXACT_SOURCE_TRANSFER_RECEIPT.json').read_text())
with SOURCE.open('rb') as stream:
    source_sha = digest(stream)
assert source_sha == receipt['sha256'] and before['bytes'] == receipt['bytes']
print('Original transfer archive checksum verified.', flush=True)
rows, preserve = [], []
with tarfile.open(SOURCE, 'r:') as archive:
    for number, member in enumerate(archive):
        name = PurePosixPath(member.name)
        assert not name.is_absolute() and '..' not in name.parts
        row = {'name': member.name, 'bytes': member.size, 'type': member.type.decode('ascii'),
               'linkname': member.linkname, 'mode': member.mode, 'mtime': member.mtime,
               'uid': member.uid, 'gid': member.gid, 'uname': member.uname,
               'gname': member.gname, 'pax_headers': member.pax_headers}
        if member.isfile():
            with archive.extractfile(member) as stream:
                row['sha256'] = digest(stream)
            existing = PHASE / str(name)
            unchanged = False
            if existing.is_file() and not existing.is_symlink() and existing.resolve().is_relative_to(ROOT):
                original = metadata(existing)
                if original['bytes'] == member.size:
                    with existing.open('rb') as stream:
                        unchanged = digest(stream) == row['sha256']
                    unchanged = unchanged and metadata(existing) == original
                if unchanged:
                    row['retained_path'] = str(existing.relative_to(ROOT))
                    row['retained_metadata'] = original
            row['disposition'] = 'identical_file_retained' if unchanged else 'preserved_in_difference_archive'
            if not unchanged:
                preserve.append(member.name)
        else:
            row['disposition'] = 'preserved_in_difference_archive'
            preserve.append(member.name)
        rows.append(row)
        if number and number % 1000 == 0:
            print('Checked archive members:', number, flush=True)
    with tarfile.open(DELTA, 'x:gz', compresslevel=6) as output:
        for name in preserve:
            member = archive.getmember(name)
            if member.isfile():
                with archive.extractfile(member) as stream:
                    output.addfile(member, stream)
            else:
                output.addfile(member)

expected = {row['name']: row for row in rows if row['disposition'] == 'preserved_in_difference_archive'}
verified = []
with tarfile.open(DELTA, 'r:gz') as archive:
    for member in archive:
        row = expected[member.name]
        assert member.size == row['bytes'] and member.type.decode('ascii') == row['type']
        assert member.linkname == row['linkname']
        if member.isfile():
            with archive.extractfile(member) as stream:
                assert digest(stream) == row['sha256']
        verified.append(member.name)
assert len(verified) == len(expected) and set(verified) == set(expected)
assert metadata(SOURCE) == before
with DELTA.open('rb') as stream:
    delta_sha = digest(stream)
result = {'original_archive': str(SOURCE.relative_to(ROOT)), 'original_metadata': before,
          'original_sha256': source_sha, 'original_member_count': len(rows),
          'retained_difference_archive': str(DELTA.relative_to(ROOT)),
          'difference_archive_metadata': metadata(DELTA), 'difference_archive_sha256': delta_sha,
          'difference_archive_members': len(preserve), 'members': rows,
          'all_historical_member_bytes_preserved': True,
          'original_archive_deleted': False, 'finished_unix': time.time()}
with (AUDIT / 'TRANSFER_PRESERVATION.json').open('x') as output:
    json.dump(result, output, indent=2)
    output.write('\n')
print(json.dumps({k: v for k, v in result.items() if k != 'members'}, indent=2), flush=True)
