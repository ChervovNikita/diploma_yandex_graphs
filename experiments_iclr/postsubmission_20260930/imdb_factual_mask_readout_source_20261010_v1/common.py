"""Standard-library gates for one disabled-by-default resident mask readout."""
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
HOST = 'anogena-2-0'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
METHODS = ('shared_own_only', 'native_pool_credit', 'source_view_supervision',
           'uncoupled_source_contrast', 'COMMON_cycle', 'assigned_source_supply',
           'plain_native', 'untied_same_six_factors')
MASKS = ('member_correct', 'full_pool_correct', 'common_wrong_events', 'any_correct_member_events')
PAIRED_REFERENCES = ('shared_own_only', 'plain_native', 'untied_same_six_factors')


def require(condition, code):
    if not condition:
        raise RuntimeError(code)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def route_check():
    require(socket.gethostname() == HOST, 'wrong_host')
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                               check=True, capture_output=True, text=True, timeout=5)
    require(inventory.stdout.strip().splitlines() == [GPU], 'wrong_GPU_inventory')


def inside_phase(path):
    value = Path(path)
    require(value.is_absolute() and value.resolve().is_relative_to(PHASE.resolve()), 'path_outside_phase')
    return value


def bound_bytes(descriptor):
    path = inside_phase(descriptor['path'])
    require(path.is_file() and not path.is_symlink(), 'missing_or_symlinked_bound_file')
    require(type(descriptor['bytes']) is int and descriptor['bytes'] >= 0, 'invalid_bound_length')
    require(re.fullmatch('[0-9a-f]{64}', descriptor['sha256']) is not None, 'invalid_bound_digest')
    require(path.stat().st_size == descriptor['bytes'], 'bound_length_changed')
    raw = path.read_bytes()
    require(len(raw) == descriptor['bytes'] and digest(raw) == descriptor['sha256'], 'bound_bytes_changed')
    return raw


def write_json(path, value):
    require(not path.exists(), 'output_already_exists')
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def admitted(release_path, release_sha256):
    # Called before resident metadata/archive reads and before the Torch import.
    route_check()
    release_path = inside_phase(release_path)
    release_raw = release_path.read_bytes()
    require(digest(release_raw) == release_sha256, 'release_digest_changed')
    release = json.loads(release_raw)
    require(release.get('schema') == 'root-finite-IMDB-factual-mask-release-v1', 'wrong_release_schema')
    require(release.get('enabled') is True and release.get('root_source_review_approved') is True
            and release.get('root_mask_readout_approved') is True, 'disabled_readout')
    require(release.get('purpose') == 'all24_resident_factual_masks_only', 'wrong_purpose')
    require(release.get('runtime_executable') == str(Path(sys.executable).resolve()), 'runtime_not_bound')
    cap = release.get('wall_budget_seconds')
    require(type(cap) in (int, float) and math.isfinite(cap) and cap > 0, 'missing_finite_external_budget')
    manifest_raw = (HERE / 'MANIFEST.json').read_bytes()
    require(digest(manifest_raw) == release.get('source_manifest_sha256'), 'source_manifest_changed')
    manifest = json.loads(manifest_raw)
    for record in manifest['files']:
        require(Path(record['path']).name == record['path'], 'nonlocal_manifest_file')
        raw = (HERE / record['path']).read_bytes()
        require(len(raw) == record['bytes'] and digest(raw) == record['sha256'], 'source_file_changed')
    bindings = json.loads((HERE / 'INPUT_BINDINGS.json').read_text())
    require(release.get('input_bindings_sha256') == digest((HERE / 'INPUT_BINDINGS.json').read_bytes()), 'bindings_changed')
    require(release.get('study_binding') == bindings['study_binding'], 'wrong_study')
    require(release.get('all24_roster_required') is True and release.get('automatic_retry') is False, 'wrong_readout_scope')
    review = json.loads(bound_bytes(release['source_review_receipt']))
    require(review.get('root_source_review_approved') is True
            and review.get('source_manifest_sha256') == release['source_manifest_sha256'], 'review_not_bound_to_source')
    output = inside_phase(release['output_directory'])
    require(not output.is_relative_to(HERE), 'output_inside_immutable_source')
    return release, bindings, output


def verify_custody(bindings):
    require(bindings['study_binding'] == 'd0b30e8297bbedc6063a0bf0c553869157165759d8cd2a41e93e4fe4324d0ea4', 'study_changed')
    require({(row['pair'], row['method']) for row in bindings['banks']}
            == {(pair, method) for pair in (1, 2, 3) for method in METHODS}
            and len(bindings['banks']) == 24, 'incomplete_or_duplicate_roster')
    records = {}
    for descriptor in bindings['required_resident_metadata']:
        require(Path(descriptor['path']).suffix == '.json', 'nonmetadata_custody_input')
        records[descriptor['path']] = json.loads(bound_bytes(descriptor))
    for row in bindings['banks']:
        result = records[row['result_binding']['path']]
        require(result.get('complete') is True and result.get('status') == 'complete'
                and result.get('TEST_file_access') is False, 'original_bank_not_complete')
        require(result.get('role_seed') == row['pair'] and result.get('base_seed') == row['pair'], 'original_role_changed')
        require(result.get('checkpoint_sha256') == row['selected_checkpoint_sha256'], 'selected_origin_changed')
    return [{'path': d['path'], 'bytes': d['bytes'], 'sha256': d['sha256']}
            for d in bindings['required_resident_metadata']]
