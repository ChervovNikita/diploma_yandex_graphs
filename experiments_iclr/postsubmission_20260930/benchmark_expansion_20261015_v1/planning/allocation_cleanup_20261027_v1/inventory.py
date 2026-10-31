"""Read-only storage inventory of the explicitly authorized project repository."""
import json
import os
from pathlib import Path
import stat
import time
from collections import Counter

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
assert ROOT.resolve() == ROOT
started = time.time()
seen = set()
sizes = Counter()
extensions = Counter()
large = []
links = []
errors = []
file_count = 0

def error(exc):
    errors.append({'path': exc.filename, 'error': type(exc).__name__})

for directory, subdirs, files in os.walk(ROOT, followlinks=False, onerror=error):
    current = Path(directory)
    subdirs.sort()
    for name in list(subdirs):
        path = current / name
        if path.is_symlink():
            links.append({'path': str(path.relative_to(ROOT)), 'target': os.readlink(path)})
            subdirs.remove(name)
    for name in files:
        path = current / name
        try:
            info = path.lstat()
        except OSError as exc:
            error(exc)
            continue
        relative = path.relative_to(ROOT)
        if stat.S_ISLNK(info.st_mode):
            links.append({'path': str(relative), 'target': os.readlink(path)})
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        file_count += 1
        inode = (info.st_dev, info.st_ino)
        blocks = info.st_blocks * 512 if inode not in seen else 0
        seen.add(inode)
        extensions[path.suffix or '<none>'] += blocks
        sizes['.'] += blocks
        for length in range(1, min(len(relative.parts), 5)):
            sizes['/'.join(relative.parts[:length])] += blocks
        if info.st_size >= 16 * 1024**2:
            large.append({'path': str(relative), 'bytes': info.st_size,
                          'allocated_bytes': info.st_blocks * 512,
                          'counted_allocated_bytes': blocks, 'links': info.st_nlink,
                          'device': info.st_dev, 'inode': info.st_ino,
                          'mtime_ns': info.st_mtime_ns})

print(json.dumps({'root': str(ROOT), 'started_unix': started, 'finished_unix': time.time(),
                  'regular_files': file_count, 'unique_regular_inodes': len(seen),
                  'allocated_bytes_excludes_directory_entries': sizes['.'],
                  'sizes_to_depth_four': dict(sizes), 'extensions': dict(extensions),
                  'large_files': sorted(large, key=lambda x: x['allocated_bytes'], reverse=True),
                  'symlinks_not_followed': links, 'errors': errors,
                  'file_contents_or_model_tensors_loaded': False}, sort_keys=True))
