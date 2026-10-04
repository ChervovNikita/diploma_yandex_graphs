"""Authenticate original all25 custody metadata and hash bytes without deserialization.

CONFIG is the pinned transport input. No training, checkpoint loading, model
construction, scorer call, TEST read or GPU observation occurs in this script.
"""
from datetime import datetime, timezone
from hashlib import sha256
import base64
import json
from pathlib import Path
import sys
import zlib

REPO = Path(CONFIG['repo'])
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
OUT = PHASE / 'ncnc_v4_scientific_audit_release_preparation_20261004_v1'
SOURCE = PHASE / 'ncnc_selected_checkpoint_metric_audit_v4_source_20261004'
FAMILY = PHASE / 'graph_ncNC_predictive_family_execution_root_20261003_v1'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def verify(row):
    path = Path(row['path'])
    require(path.is_absolute() and path.resolve() == path and path.is_file(), 'Borrowed custody path')
    require(('bytes' not in row or path.stat().st_size == row['bytes']) and sha(path) == row['sha256'], 'Byte custody differs: ' + str(path))
    return path


def local(path):
    return {**pin(path), 'path': path.name}


def save(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


require(Path.cwd().resolve() == REPO and not OUT.exists(), 'Wrong repository or reused preparation')
packets = []
for row in CONFIG['packets']:
    root = PHASE / row['root']
    verify({'path': str(root / 'MANIFEST.json'), 'sha256': row['manifest_sha256']})
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    require(len({r['path'] for r in manifest['files']}) == len(manifest['files']), 'Duplicate source rows')
    for receipt in manifest['files']:
        path = root / receipt['path']
        require(path.is_relative_to(root), 'Source path escaped')
        verify({'path': str(path), 'bytes': receipt.get('bytes', receipt.get('size')), 'sha256': receipt['sha256']})
    packets.append({'root': str(root), 'manifest': pin(root / 'MANIFEST.json'), 'payload_count': len(manifest['files'])})
bindings = json.loads((SOURCE / 'SOURCE_BINDINGS.json').read_text())
lock_pin = bindings['original_lock_provenance']['remote']
lock = json.loads(verify(lock_pin).read_text())
require(lock['identity'] == CONFIG['identity'] and lock['status'] == 'COMPLETE_FAMILY_LOCKED', 'Original frozen family differs')
require(len(lock['inputs']) == 20 and len(lock['cells']) == 25 and lock['unique_fits_completed'] == 35, 'Full20/35/25 denominator required')
physical = {}
for gpu in (0, 1):
    queue = FAMILY / ('queue_GPU' + str(gpu))
    release = json.loads((FAMILY / ('ROOT_FIT_RELEASE_GPU' + str(gpu) + '.json')).read_text())
    for index, invocation in enumerate(release['authorized_invocations']):
        path = queue / ('CHILD_' + str(index) + '_TERMINAL.json')
        terminal = json.loads(path.read_text())
        require((terminal['unit'], terminal['base_seed'], terminal['output_directory']) == (invocation['unit'], invocation['base_seed'], invocation['output_directory']) and type(terminal['exit_code']) is int and terminal['exit_code'] == 0, 'Original physical fit terminal differs')
        key = (terminal['unit'], terminal['base_seed'])
        require(key not in physical, 'Duplicate physical custody')
        physical[key] = pin(path)
unit_pins = []
for binding in lock['inputs']:
    root = Path(binding['output_directory'])
    require(root.resolve() == root and root.is_relative_to(FAMILY / 'units') and binding.get('disposition', 'COMPLETE') == 'COMPLETE', 'Borrowed original unit')
    require(sha(root / 'COMPLETE.json') == binding['complete_sha256'], 'Original complete terminal changed')
    unit_pins.append({'unit': binding['unit'], 'base_seed': binding['base_seed'], 'output_directory': str(root), 'disposition': 'COMPLETE', 'terminal': local(root / 'COMPLETE.json'), 'physical_terminal': physical[(binding['unit'], binding['base_seed'])], 'attempts': local(root / 'ATTEMPTS.json'), 'journal': local(root / 'JOURNAL.json'), 'closure': local(root / 'FAMILY_CLOSURE.json')})
require(len(unit_pins) == len(physical) == 20, 'Original custody denominator differs')
sys.path.insert(0, str(SOURCE))
from replay_gate import family_gate
from audit_contract import PROFILE_CONTRACT, sum_case_work
context = {'identity': CONFIG['identity'], 'release': {'family_lock': lock_pin, 'unit_custody': unit_pins}, 'output': OUT / 'audit/run01'}
gate = family_gate(context)
require(gate['full_success'] and gate['unique_fits_completed'] == 35 and gate['complete_served_cells'] == 25, 'Original metadata/byte gate failed')
runtime_pin = CONFIG['runtime_authority']
runtime = json.loads(verify(runtime_pin).read_text())
data_pin = CONFIG['data_authority']
authority = json.loads(verify(data_pin).read_text())
runtime_files = [{'path': runtime['interpreter_path'], 'sha256': runtime['interpreter_sha256']}, *runtime['runtime_source_pins'], *runtime['runtime_binary_files'], runtime['negative_sampler'], authority['ogb_evaluator']]
for row in runtime_files:
    verify(row)
review_pin = CONFIG['source_review']
review = json.loads(verify(review_pin).read_text())
require(review['status'] == 'PASS' and review['sidecar_manifest_sha256'] == CONFIG['source_sha256'] and review['identity'] == CONFIG['identity'] and review['execution_authorized'] is False, 'Exact independent review differs')
SYNTH = PHASE / 'ncnc_v4_fixture_release_preparation_20261004_v1'
synthetic_release = SYNTH / 'ROOT_SYNTHETIC_RELEASE_ENABLED.json'
require(sha(synthetic_release) == CONFIG['synthetic_release_sha256'], 'Wrong owned qualification release')
qualification_path = SYNTH / 'qualification/run01/QUALIFICATION.json'
qualification = json.loads(qualification_path.read_text())
plan = json.loads((SOURCE / 'FABRICATED_QUALIFICATION_PLAN.json').read_text())
qualification_pass = qualification.get('status') == 'PASS'
if qualification_pass:
    expected = set(plan['full_core_cases']) | set(plan['shared_contract_cases'])
    require(qualification['schema'] == 'ncnc-selected-checkpoint-metric-audit-synthetic-qualification-v4' and qualification['sidecar_manifest_sha256'] == CONFIG['source_sha256'] and qualification['identity'] == CONFIG['identity'], 'Qualification identity differs')
    require(qualification['source_entry'] == pin(SOURCE / 'replay_synthetic.py') and qualification['runtime_authority'] == runtime_pin and qualification['profile_contract'] == PROFILE_CONTRACT, 'Qualification source/runtime/profile differs')
    require(qualification['case_count'] == len(qualification['cases']) == len(expected) == 22 and {r['case'] for r in qualification['cases']} == expected and all(r['status'] == 'PASS' for r in qualification['cases']), 'Complete22-case PASS required')
    require(qualification['actual_work'] == sum_case_work(qualification['cases']) and qualification['actual_work']['planned'] == 1920, 'Qualification literal summed work differs')
    require(qualification['fabricated_inputs_only'] is True and qualification['study_lock_data_outcome_checkpoint_accessed'] is False and qualification['TEST_opened'] is False and qualification['scientific_fit_updates'] == qualification['new_training_updates'] == 0, 'Qualification scope differs')
OUT.mkdir(mode=0o700)
invocation = {'stage': 'selected_checkpoint_metric_audit', 'output_directory': str(OUT / 'audit/run01'), 'cuda_visible_devices': 'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998'}
resource = {'schema': 'ncnc-selected-checkpoint-metric-audit-runtime-resource-admission-v4', 'status': 'PENDING_ROOT_RESOURCE_ADMISSION', 'sidecar_manifest_sha256': CONFIG['source_sha256'], 'identity': CONFIG['identity'], 'family_lock_sha256': lock_pin['sha256'], 'invocation': invocation, 'dispatch_recheck_required': True, 'minimum_GPU_free_MiB': 16384, 'minimum_host_MemAvailable_bytes': 17179869184, 'GPU_observation_performed': False, 'execution_authorized': False, 'runtime_authority': runtime_pin, 'source_and_runtime_byte_custody': 'PASS_METADATA_ONLY', 'data_authority': data_pin, 'ordinary_runtime_synthetic_qualification': pin(qualification_path) if qualification_pass else None}
save('RUNTIME_RESOURCE_ADMISSION_DISABLED_CANDIDATE.json', resource)
candidate = {'schema': 'ncnc-selected-checkpoint-metric-audit-root-release-v4', 'execution_enabled': False, 'root_authorization_reference': None, 'authorized_stages': [], 'authorized_invocations': [], 'candidate_invocation_not_authorized': invocation, 'sidecar_manifest_sha256': CONFIG['source_sha256'], 'cuda_visible_devices': invocation['cuda_visible_devices'], 'family_id': CONFIG['identity']['family_id'], 'family_lock_output_directory': CONFIG['identity']['family_lock_output_directory'], 'family_lock': lock_pin, 'unit_custody': unit_pins, 'data_authority': data_pin, 'data_authority_sha256': data_pin['sha256'], 'runtime_authority': runtime_pin, 'runtime_authority_sha256': runtime_pin['sha256'], 'profile_contract': PROFILE_CONTRACT, 'independent_source_review': review_pin, 'ordinary_runtime_synthetic_qualification': pin(qualification_path) if qualification_pass else None, 'runtime_resource_admission': None, 'candidate_resource_admission_not_PASS': pin(OUT / 'RUNTIME_RESOURCE_ADMISSION_DISABLED_CANDIDATE.json')}
for role, row in bindings['original_manifests'].items():
    candidate[role + '_root'] = str(PHASE / row['local_root'])
    candidate[role + '_manifest_sha256'] = row['manifest_sha256']
candidate['unresolved_before_root_enabled_release'] = ([] if qualification_pass else ['Complete exact22-case qualification PASS and bound final receipt']) + ['Root review and separately authored PASS runtime/resource admission binding original identity/lock and exact invocation', 'Separately enabled root ordinary audit release with exact byte hash', 'Dispatch resource recheck and fresh audit/run01 plus supervision_run01']
save('ROOT_ORDINARY_AUDIT_RELEASE_DISABLED_CANDIDATE.json', candidate)
save('ORIGINAL_FAMILY_CUSTODY_AUTHENTICATION.json', {'schema': 'ncnc-v4-original-frozen-family-custody-authentication-v1', 'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_STDLIB_METADATA_AND_BINARY_HASHES_NO_DESERIALIZATION', 'identity': CONFIG['identity'], 'family_lock': lock_pin, 'unit_custody': unit_pins, 'complete_units': 20, 'unique_fits': 35, 'served_cells': 25, 'original_optimizer_steps_metadata': 59500, 'family_gate': 'PASS', 'source_packets': packets, 'runtime_authority': runtime_pin, 'runtime_files': runtime_files, 'data_authority': data_pin, 'actual_TRAIN_raw_VALID_bytes_opened': False, 'checkpoint_tensors_loaded': False, 'TEST_opened': False, 'new_fits': 0, 'scorer_calls': 0, 'per_seed_scores_used_for_decisions': False})
save('QUALIFICATION_STATUS_AND_BINDING.json', {'UTC': datetime.now(timezone.utc).isoformat(), 'status': qualification['status'], 'complete_case_count_PASS': sum(r['status'] == 'PASS' for r in qualification['cases']), 'not_attempted': sum(r['status'] == 'NOT_ATTEMPTED' for r in qualification['cases']), 'actual_work': qualification.get('actual_work'), 'qualification_pin': pin(qualification_path) if qualification_pass else None, 'exact_original_runtime_22case_PASS_authenticated': qualification_pass, 'synthetic_root_release': pin(synthetic_release)})
if qualification_pass:
    with (OUT / 'QUALIFICATION_PASS_COPY.json').open('xb') as stream:
        stream.write(qualification_path.read_bytes())
physical_path = SYNTH / 'supervision_run01/PHYSICAL_TERMINAL.json'
if physical_path.exists():
    with (OUT / 'SYNTHETIC_PHYSICAL_TERMINAL_COPY.json').open('xb') as stream:
        stream.write(physical_path.read_bytes())
names = [p.name for p in sorted(OUT.iterdir()) if p.is_file()]
files = {}
for name in names:
    data = (OUT / name).read_bytes()
    files[name] = {'bytes': len(data), 'sha256': sha256(data).hexdigest(), 'data': base64.b64encode(data).decode()}
print(json.dumps({'compressed_receipts_base64': base64.b64encode(zlib.compress(json.dumps({'status': 'SCIENTIFIC_AUDIT_METADATA_PREPARED_DISABLED', 'files': files}).encode())).decode()}))
