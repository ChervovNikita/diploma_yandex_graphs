"""Stdlib output-path guard shared by scientific and runtime entry points."""
import os
from pathlib import Path


def require_project_output(value, packet):
    """Require a new absolute run path in this project, outside sealed trees.

    Validate before model imports/data reads. The caller creates the directory
    with exist_ok=False; this check itself writes no files.
    """
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('Output must be an explicit absolute project path.')
    packet = Path(packet).resolve()
    project = packet.parent
    path = path.resolve()
    if path == project or not path.is_relative_to(project):
        raise ValueError('Output must be a new run directory inside the project.')
    if path == packet or path.is_relative_to(packet):
        raise ValueError('Output cannot be inside the sealed source packet.')
    for ancestor in (path, *path.parents):
        if ancestor == project.parent:
            break
        if (ancestor / 'SEAL.json').exists():
            raise ValueError(f'Output cannot be inside sealed source: {ancestor}')
    if path.exists():
        raise ValueError('Output directory already exists; use a new explicit path.')
    parent = path.parent
    while not parent.exists():
        parent = parent.parent
    if (not parent.is_dir() or not parent.stat().st_mode & 0o222
            or not os.access(parent, os.W_OK | os.X_OK)):
        raise ValueError('Output parent must be an existing writable directory.')
    return path
