"""Stdlib prospective contract and pin verification for the DDI pilot."""
import hashlib
import json
from pathlib import Path


ARMS = ('native_m1', 'target_only', 'joint', 'separate')
SEEDS = (0, 1, 2)
EPOCHS = 100


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_pins(packet):
    binding = json.loads((packet / 'INPUT_BINDINGS.json').read_text())
    for row in binding['source_pins'] + binding['TRAIN_cost_evidence_pins']:
        path = packet.parent / row['path']
        if path.stat().st_size != row['bytes'] or file_sha256(path) != row['sha256']:
            raise ValueError(f'Pinned input changed: {row["path"]}')
    return binding


def scientific_contract(config):
    return {key: value for key, value in config.items() if key not in ('release_enabled', 'root_admission')}


def read_release(path, packet):
    config = json.loads(Path(path).read_text())
    canonical = json.loads((packet / 'config.json').read_text())
    if (json.dumps(scientific_contract(config),sort_keys=True,allow_nan=False)
            != json.dumps(scientific_contract(canonical),sort_keys=True,allow_nan=False)):
        raise ValueError('The prereleased pilot scientific contract must remain exact.')
    if config['release_enabled'] is not True:
        raise SystemExit('Pilot release disabled: no numerical imports, artifact load or training started.')
    admission = config['root_admission']
    if admission['approved'] is not True or not admission['supervisor'] or not admission['capacity_observation']:
        raise ValueError('Explicit root admission, supervisor and capacity evidence are required.')
    fraction = admission['cuda_allocator_fraction']
    if type(fraction) not in (int, float) or not 0 < fraction <= 1:
        raise ValueError('Root must declare a CUDA allocator fraction in (0, 1].')
    for field in ('host_cap_GiB', 'per_cell_wall_cap_seconds'):
        if type(admission[field]) is not int or admission[field] <= 0:
            raise ValueError(f'Root must declare a positive integer {field}.')
    return config
