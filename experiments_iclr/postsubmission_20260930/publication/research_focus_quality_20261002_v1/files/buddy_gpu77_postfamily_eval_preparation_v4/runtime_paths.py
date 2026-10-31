"""Explicit repository-local runtime paths; not a kernel sandbox claim."""
import os
from pathlib import Path

PATHS = {
    'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp',
    'XDG_CACHE_HOME': 'cache/xdg', 'XDG_CONFIG_HOME': 'cache/xdg-config', 'XDG_DATA_HOME': 'cache/xdg-data',
    'TORCH_HOME': 'cache/torch', 'TORCH_EXTENSIONS_DIR': 'cache/torch-extensions',
    'TORCHINDUCTOR_CACHE_DIR': 'cache/torchinductor', 'TRITON_CACHE_DIR': 'cache/triton',
    'CUDA_CACHE_PATH': 'cache/cuda', 'MPLCONFIGDIR': 'cache/matplotlib',
    'NUMBA_CACHE_DIR': 'cache/numba', 'PYTHONPYCACHEPREFIX': 'cache/pycache',
}


def canonical_inside(path, root):
    path, root = Path(path).absolute(), Path(root).resolve()
    if path != path.resolve() or not path.is_relative_to(root) or path == root:
        raise RuntimeError('Runtime/output path must be canonical inside the repository')
    return path


def path_environment(repo, runtime_root):
    root = canonical_inside(runtime_root, repo)
    values = {name: str(canonical_inside(root / relative, repo)) for name, relative in PATHS.items()}
    values.update(PYTHONDONTWRITEBYTECODE='1', CUDA_CACHE_DISABLE='1', OUTDATED_IGNORE='1')
    return values


def create_paths(repo, runtime_root):
    values = path_environment(repo, runtime_root)
    for relative in sorted(set(PATHS.values())):
        # A prior CPU/data-free boundary qualification may prepare these paths.
        # Scientific family outputs have a separate exclusive namespace.
        path = canonical_inside(Path(runtime_root) / relative, repo)
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise RuntimeError('Runtime path must be a directory')
    return values


def writable_regular_fd_gate(records, repo):
    for record in records:
        if record['regular_file'] and record['writable']:
            canonical_inside(record['target'], repo)


def inherited_fd_audit(repo):
    # Metadata only; never open/read/write an outside regular-file payload.
    import fcntl
    import stat
    folder = Path('/proc/self/fd')
    if not folder.is_dir():
        raise RuntimeError('Linux inherited-FD metadata audit unavailable')
    rows = []
    for entry in folder.iterdir():
        try:
            fd = int(entry.name)
            status = os.fstat(fd)
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            target = os.readlink(entry)
        except (OSError, ValueError):
            continue  # the directory iterator's own transient descriptor may close
        rows.append(dict(fd=fd, regular_file=stat.S_ISREG(status.st_mode),
                         writable=(flags & os.O_ACCMODE) != os.O_RDONLY, target=target))
    writable_regular_fd_gate(rows, repo)
    return dict(outside_writable_regular_file_FDs_observed=False,
                writable_regular_file_FDs=[row for row in rows if row['regular_file'] and row['writable']],
                scope='Inherited regular-file descriptors only; no hardware write-enforcement claim')
