#!/usr/bin/env python3
"""Future selected-path metadata step only; no imports of numerical packages."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'anogena-2-0' or Path.cwd().resolve() != REPO:
        raise ValueError('Use the authorized one-GPU repository')
    devices = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True, timeout=30)
    if devices.stdout.split() != ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Authorized singleton physical device differs')
    if platform.python_version() != '3.11.14':
        raise ValueError('Use the observed one-GPU interpreter')
    output = args.output_directory.resolve()
    if not output.is_relative_to(PHASE) or output.exists():
        raise ValueError('Fresh resolver output directory must stay in the authorized phase')
    primary = {}
    for line in (HERE / 'requirements.in').read_text().splitlines():
        name, version = line.split('==')
        primary[canonical(name)] = version
    installed = {}
    # Match importlib.metadata/PYTHONPATH precedence instead of selecting an
    # arbitrary duplicate distribution when pip/venv metadata overlap.
    names = {canonical(d.metadata['Name']) for d in metadata.distributions() if d.metadata.get('Name')}
    for name in sorted(names):
        d = metadata.distribution(name)
        installed[name] = dict(version=d.version, root=str(d.locate_file('')),
                               Requires_Python=d.metadata.get('Requires-Python'),
                               Requires_Dist=d.requires)
    bound = json.loads((HERE / 'INPUT_BINDINGS.json').read_text())
    for name, version in bound['protected_selected_versions'].items():
        if installed.get(canonical(name), {}).get('version') != version:
            raise ValueError('Observed protected version changed: ' + name)
    if any(name in installed for name in primary):
        raise ValueError('The seven-package missing-state changed; retain a fresh successor receipt')
    constraints = {name: row['version'] for name, row in installed.items()}
    constraints.update(primary)
    preferences = json.loads((HERE / 'retained_missing_transitive_preferences.json').read_text())
    for name, version in preferences.items():
        constraints.setdefault(canonical(name), version)
    output.mkdir()
    (output / 'RESOLVER_CONSTRAINTS.txt').write_text(''.join(
        name + '==' + version + '\n' for name, version in sorted(constraints.items())))
    receipt = dict(status='STDLIB_METADATA_CONSTRAINTS_ONLY', host=socket.gethostname(),
                   python=platform.python_version(), executable=sys.executable,
                   PYTHONPATH=os.environ.get('PYTHONPATH'), installed=installed,
                   primary=primary, numerical_imports=False, datasets_opened=False,
                   packages_installed=False,
                   protected_constraints_sha256=hashlib.sha256((HERE / 'protected_constraints.txt').read_bytes()).hexdigest())
    (output / 'BEFORE_METADATA.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], existing_packages=len(installed),
                         output_directory=str(output)), sort_keys=True))


if __name__ == '__main__':
    main()
