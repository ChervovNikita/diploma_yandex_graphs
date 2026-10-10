"""Stdlib-only admission for original-resident diagnostic archives; no owner."""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PHASE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
METHODS = ('shared_own_only', 'native_pool_credit', 'source_view_supervision', 'uncoupled_source_contrast',
           'COMMON_cycle', 'assigned_source_supply', 'plain_native', 'untied_same_six_factors')
EDGES = (0.0, .02, .10, .25, .50)


class InputUnavailable(RuntimeError):
    def __init__(self, fields):
        super().__init__('required_original_probability_or_mask_field_absent')
        self.fields = tuple(fields)


def require(value, code):
    if not value:
        raise RuntimeError(code)


@dataclass(frozen=True)
class Caps:
    source_bound: bool = False
    archive_read: bool = False
    runtime: bool = False
    root_review_sha256: str = ''

    def require(self):
        require(self.source_bound is self.archive_read is self.runtime is True
                and re.fullmatch('[0-9a-f]{64}', self.root_review_sha256) is not None,
                'disabled_original_archive_reader')


CLOSED = Caps()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def inside(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve().is_relative_to(PHASE), 'outside_authorized_phase')
    return path


def bound_json(row):
    path = inside(row['path'])
    require(path.suffix == '.json' and path.is_file() and not path.is_symlink(), 'not_bound_JSON')
    raw = path.read_bytes()
    require(len(raw) == row['bytes'] and digest(raw) == row['sha256'], 'changed_bound_JSON')
    return json.loads(raw)


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def admitted(release_path, caps=CLOSED):
    caps.require()
    require(socket.gethostname() == 'anogena-2-0' and HERE.parent == PHASE, 'wrong_literal_allocation')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=5)
    require(inventory.strip().splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'wrong_sole_GPU')
    release_path = inside(release_path)
    raw = release_path.read_bytes()
    release = json.loads(raw)
    require(release['schema'] == 'root-IMDB24-stored-confidence-release-v1'
            and release['enabled'] is True and release['root_source_review_approved'] is True
            and release['root_stored_confidence_readout_approved'] is True
            and release['purpose'] == 'all24_original_observed_event_margin_bins'
            and release['automatic_retry'] is False and release['re_inference_allowed'] is False,
            'disabled_or_wrong_scope')
    manifest_raw, seal_raw = (HERE/'MANIFEST.json').read_bytes(), (HERE/'SEAL.json').read_bytes()
    seal = json.loads(seal_raw)
    require(digest(manifest_raw) == release['source_manifest_sha256'] == seal['manifest_sha256']
            and digest(seal_raw) == release['source_seal_sha256']
            and seal['source_only'] is True and seal['execution_enabled'] is False, 'unbound_source')
    for row in json.loads(manifest_raw)['files']:
        require(Path(row['path']).name == row['path'], 'nonlocal_source_payload')
        data = (HERE/row['path']).read_bytes()
        require(len(data) == row['bytes'] and digest(data) == row['sha256'], 'changed_source_payload')
    require(digest((HERE/'PROTOCOL.json').read_bytes()) == release['protocol_sha256']
            and digest((HERE/'INPUT_BINDINGS.json').read_bytes()) == release['input_bindings_sha256'], 'changed_protocol_or_inputs')
    require(release['root_review']['sha256'] == caps.root_review_sha256, 'unbound_review')
    review = bound_json(release['root_review'])
    require(review['root_source_review_approved'] is True and all(review[k] == release[k] for k in
            ('source_manifest_sha256', 'source_seal_sha256', 'protocol_sha256', 'input_bindings_sha256')), 'wrong_root_review')
    require(release['finite_external_owner_bound'] is True
            and type(release['wall_budget_seconds']) in (int, float)
            and math.isfinite(release['wall_budget_seconds']) and release['wall_budget_seconds'] > 0,
            'finite_root_owner_required')
    require(release['runtime_executable'] == str(Path(sys.executable).resolve())
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'wrong_CPU_runtime')
    bindings = json.loads((HERE/'INPUT_BINDINGS.json').read_text())
    require(len(bindings['banks']) == 24 and {(r['pair'], r['method']) for r in bindings['banks']}
            == {(r, m) for r in (1, 2, 3) for m in METHODS}, 'incomplete24_roster')
    output = inside(release['output_directory'])
    require(not output.exists() and not output.is_relative_to(HERE), 'output_not_fresh')
    return release, bindings, output, digest(raw)
