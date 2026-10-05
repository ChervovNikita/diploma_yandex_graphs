"""Independent static source/metadata audit; no prepared source imports/execution."""
import ast
import difflib
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PACKET = BASE / 'shared_private_transfer_allocation_replication_preparation_20261005_v1'
MAPPING = BASE / 'shared_private_transfer_allocation_replication_metadata_review_20261005_v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(name, value):
    with (HERE / name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


assert sha(PACKET / 'MANIFEST.json') == '5a667680152ff17b0e19aff2beb367f84b064013b5bc48d2b55d01318f59f4e9'
assert sha(PACKET / 'SEAL.json') == '4fa4418626f02670cd9e09472673252445b60babadf47f6970f899a377b8e8b6'
manifest = read(PACKET / 'MANIFEST.json')
inputs = [PACKET / 'MANIFEST.json', PACKET / 'SEAL.json', MAPPING / 'FROZEN_26_CELLS.json', MAPPING / 'MANIFEST.json']
python_count = 0
for entry in manifest['files']:
    path = PACKET / entry['path']
    assert path.resolve().is_relative_to(PACKET.resolve())
    assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
    inputs.append(path)
    if path.suffix == '.py':
        ast.parse(path.read_text(), filename=str(path))
        python_count += 1
assert read(PACKET / 'SEAL.json')['manifest_sha256'] == sha(PACKET / 'MANIFEST.json')
inheritance = []
for family, old in [('original30', BASE / 'shared_private_transfer_paired_pilot_preparation_20261005_v2'), ('companion9', BASE / 'shared_private_transfer_row0_companion_launch_activation_20261005_v1/singleton')]:
    for name in ['pilot_common.py', 'run_queue.py']:
        before_path, after_path = old / name, PACKET / family / name
        inputs.append(before_path)
        before_text, after_text = before_path.read_text(), after_path.read_text()
        before = {q.name: ast.dump(q, include_attributes=False) for q in ast.parse(before_text).body if isinstance(q, ast.FunctionDef)}
        after = {q.name: ast.dump(q, include_attributes=False) for q in ast.parse(after_text).body if isinstance(q, ast.FunctionDef)}
        assert set(before) == set(after)
        changed = [q for q in before if before[q] != after[q]]
        assert changed == (['verify_packet'] if name == 'pilot_common.py' else ['main'])
        patch = ''.join(difflib.unified_diff(before_text.splitlines(True), after_text.splitlines(True), fromfile=str(before_path.relative_to(BASE)), tofile=str(after_path.relative_to(BASE))))
        assert patch == (PACKET / (family + '_' + name + '.patch')).read_text()
        inheritance.append({'family': family, 'file': name, 'changed_functions': changed, 'all_other_functions_AST_identical': True, 'patch_matches_actual_bytes': True})

a = read(PACKET / 'PROSPECTIVE_AMENDMENT.json')
mapping = read(MAPPING / 'FROZEN_26_CELLS.json')
mapped = {q['cell_id']: q for q in mapping['cells']}
assert [q['cell_id'] for q in a['new_selected_attempts']] == mapping['fixed26_order']
assert [cell for q in a['queues'] for cell in q['ordered_cell_ids']] == mapping['fixed26_order']
assert len(set(q['physical_attempt_id'] for q in a['new_selected_attempts'])) == 26
counts = {'original30': 0, 'companion9': 0}
for row in a['new_selected_attempts']:
    frozen = mapped[row['cell_id']]
    family = row['family']
    assert family == ('companion9' if frozen['family'] == 'row0_companion9' else 'original30')
    assert row['scientific_configuration'] == frozen['scientific_configuration'] and row['fit_bounds_seconds'] == frozen['fit_bounds_seconds']
    job = read(PACKET / 'disabled_jobs' / (row['cell_id'] + '.json'))
    base = read(PACKET / ('BASE_JOB_' + family + '.json'))
    assert all(job[k] == v for k, v in row['scientific_configuration'].items())
    assert job['physical_attempt_id'] == row['physical_attempt_id']
    assert all(job[k] is False for k in ['fits_authorized', 'VALID_values_access', 'retry', 'TEST_access', 'automatic_retry_within_attempt'])
    assert job['soft_seconds'] == row['fit_bounds_seconds']['soft_seconds']
    assert job['source_manifest_sha256'] == frozen['allocation_source_manifest_sha256']
    assert job['program_sha256'] == frozen['allocation_program_sha256'] and job['runtime_versions'] == frozen['runtime_versions']
    for k, v in frozen['qualification_and_source_authorities'].items():
        assert job[k] == v
    for k in ['cpu_threads', 'cpu_interop_threads', 'available_manifest_relative', 'available_manifest_sha256', 'constant_adjacency_recursive_adjoint_authorized', 'expected_hostname', 'schema', 'purpose', 'cohort_plan_relative', 'cohort_plan_sha256']:
        assert job[k] == base[k]
    if 'physical_gpu_uuid' in base:
        assert job['physical_gpu_uuid'] == base['physical_gpu_uuid']
    descriptor = next(q for q in a['queues'] if q['queue_id'] == row['queue_id'])
    expected_output = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/' + descriptor['execution_directory_relative'] + '/runs/' + row['cell_id']
    assert job['output_directory'] == expected_output
    counts[family] += 1
assert counts == {'original30': 20, 'companion9': 6}
plans = {}
for family in counts:
    ref = a['family_plans'][family]
    path = BASE / ref.get('local_metadata_locator', ref)['path']
    inputs.append(path)
    assert sha(path) == ref['sha256']
    plan = read(path)
    assert plan['cells'] == [q['scientific_configuration'] for q in a['logical_cells39'] if q['family'] == family]
    assert plan['root_adopted_after_TRAIN_cost'] is True and plan['selection_budget_fairness_approved'] is True
    assert plan['selection'] == 'first_maximum_complete_VALID_MRR_rounded4' and plan['TEST_closed'] is True
    for key in ['complete_cycle_cost_evidence', 'root_numeric_decision', 'fit_bounds_by_cell']:
        assert plan[key] == a['family_metadata'][family][key]
    plans[family] = plan
assert a['selected39_order'] == plans['original30']['execution_order'] + plans['companion9']['execution_order']
assert len(set(a['selected39_order'])) == 39
assert a['chosen_b0_donors'] == {'original30': 'shared_private_transfer_paired_pilot_execution_root_20261005_v2', 'companion9': 'shared_private_transfer_row0_companion_b0_execution_root_20261005_v1'}
for q in a['queues']:
    assert q['ordered_cell_ids'] == [cell for cell in plans[q['family']]['execution_order'] if cell.startswith(q['block'] + '_')]
    fit_seconds = sum(a['family_metadata'][q['family']]['fit_bounds_by_cell'][cell.removeprefix(q['block'] + '_')]['hard_seconds'] for cell in q['ordered_cell_ids'])
    assert q['queue_hard_seconds'] == fit_seconds + len(q['ordered_cell_ids']) * a['resource_limits']['resource_wait_seconds'] + q['queue_overhead_seconds']
assert sum(q['queue_hard_seconds'] for q in a['queues']) == a['total_queue_hard_allowance_seconds'] == 877200
assert a['resource_limits'] == plans['original30']['resource_limits'] == plans['companion9']['resource_limits']
assert a['original_physical30_cap_exception_required'] is True and a['original_no_same_cell_retry_exception_required'] is True

static = read(PACKET / 'STATIC_VERIFICATION.json')
for proof in static['numerical_source_proof']:
    path = BASE / proof['path']
    inputs.append(path)
    assert sha(path) == proof['sha256'] and path.stat().st_size == proof['bytes']
for ref in read(PACKET / 'REUSED_CUSTODY_HELPERS.json').values():
    path = BASE / ref['path']
    inputs.append(path)
    assert sha(path) == ref['sha256']
    if path.suffix == '.py':
        ast.parse(path.read_text(), filename=str(path))
h = read(PACKET / 'ATTEMPT_HISTORY.json')
assert h['old77_all20_registered_cell_ids'] == mapping['fixed26_order'][:20]
for old in h['old77_attempt_registration_and_observation']:
    assert old['latest_terminal_status'] == 'UNKNOWN_AFTER_WITHDRAWAL' and old['old_scores_read'] is False
    for key in ['queue', 'launch_receipt', 'actual_original_donor_registration', 'actual_original_provider_admission', 'original_root_release']:
        path = BASE / old[key]['path']
        inputs.append(path)
        assert sha(path) == old[key]['sha256']
    queue = read(BASE / old['queue']['path'])
    assert [q['cell_id'] for q in queue['entries']] == [cell for cell in h['old77_all20_registered_cell_ids'] if cell.startswith(old['block'] + '_')]
assert h['old77_final_fit_count'] is None and h['old77_final_cost_seconds'] is None
assert h['physical_fits_lower_bound_including_known_old77_starts'] == 45
assert h['old_new_outcome_selection'] is False
disabled = {'ROOT_RELEASE_DISABLED.json': 'execution_authorized', 'ROOT_JOB_REVIEW_DISABLED.json': 'all26_generated_jobs_reviewed', 'B0_CUSTODY_DISABLED.json': 'root_authenticated_all13_terminal_and_artifact_bytes', 'COLLECTION_RELEASE_DISABLED.json': 'root_collection_approved', 'HISTORY_INVENTORY_RELEASE_DISABLED.json': 'root_history_inventory_approved'}
for name, flag in disabled.items():
    assert read(PACKET / name)[flag] is False
collector = (PACKET / 'collect39_metadata.py').read_text().splitlines()
assert "intent = c.read_metadata(c.binding(launch['launch_intent']))" in collector[72]
assert "before = c.read_metadata(c.binding(predecessor['block_freeze']))" in collector[110]
findings = [{'id': 'R1', 'priority': 2, 'title': 'Resolve donor-local nested evidence through authenticated replicas', 'file': str((PACKET / 'collect39_metadata.py').relative_to(BASE)), 'sites': [{'line': 73, 'reference': "launch['launch_intent']", 'actual_donor': 'current new donor'}, {'line': 111, 'reference': "predecessor['block_freeze']", 'actual_donor': 'the corresponding fixed predecessor donor from release.donors'}], 'problem': 'Both source-emitted references contain original donor-directory paths. Raw c.binding resolves them in the collection phase without applying replica_directory_relative. A valid authenticated replica under a different directory is rejected, or the code observes a separate original-path file rather than that replica.', 'required_repair': 'Use c.donor_binding for the current donor launch intent; find the matching predetermined predecessor family/block donor and use c.donor_binding for its BLOCK_FREEZE reference. Preserve source-fixed original path and SHA joins with each selected donor.', 'source_author_acknowledged': True, 'effect': 'Blocks reliable full39 metadata collection; scientific configuration and disabled state are unaffected.', 'status': 'unresolved in sealed V1; fresh V2 repair required'}]
checks = {'package_members_authenticated': len(manifest['files']), 'new_python_files_AST_parsed': python_count, 'inherited_operational_functions': inheritance, 'all26_exact_disabled_job_science': True, 'counts': counts, 'fixed39_order_unique': True, 'fixed_b0_donors': a['chosen_b0_donors'], 'unchanged_numerical_source_byte_bindings': static['numerical_source_proof'], 'four_queue_hard_allowance_seconds': 877200, 'old20_original_attempt_registration_bytes_authenticated': True, 'old77_terminal_and_final_cost_status': 'unknown; preserved explicitly', 'all_execution_collection_inventory_templates_disabled': True, 'full39_gate_static_review': 'authenticate39 checks all39 source/terminal/artifact records before registry emission; history wrapper calls authenticate39 again before first history-byte hash; donor-local path error prevents PASS for this implementation', 'current_allocation_and_b0_actual_custody_not_established': True, 'concrete_findings': findings}
inputs = list(dict.fromkeys(inputs))
bindings = [{'path': str(q.relative_to(BASE)), 'bytes': q.stat().st_size, 'sha256': sha(q)} for q in inputs]
emit('CHECKS.json', checks)
emit('FINDINGS.json', {'status': 'REQUEST_CHANGES_DISABLED', 'findings': findings})
emit('SOURCE_BINDINGS.json', {'inputs': bindings, 'no_numeric_results_histories_models_tensors_or_labels_opened': True, 'prepared_source_imported_or_executed': False})
emit('REVIEW.json', {'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'REQUEST_CHANGES_DISABLED', 'source_manifest_sha256': sha(PACKET / 'MANIFEST.json'), 'source_seal_sha256': sha(PACKET / 'SEAL.json'), 'concrete_issue_count': 1, 'affected_sites': 2, 'root_execution_approval': False, 'manuscript_certification': False, 'scientific_execution': False, 'SSH_remote_or_current_route_checks': False, 'canonical_edits': False, 'next_step': 'Preserve V1 and review a fresh V2 repair of the two donor-local nested custody joins; no launch.'})
report = '''# Independent operational allocation replication review — V1

**REQUEST CHANGES; keep disabled.** One concrete P2 custody-path defect affects two joins in `collect39_metadata.py`. No numerical or scientific configuration change is requested. This review does not admit execution or certify a manuscript.

## Concrete finding

`new_prefit` reads `launch['launch_intent']` at line73 and each `predecessor['block_freeze']` at line111 using raw `c.binding`. The reviewed launcher emits these paths inside the corresponding original donor directories. The helper's actual replica interface maps that donor directory to `replica_directory_relative` through `c.donor_binding`.

When an authentic replica is stored under a different directory, the current code resolves the original path instead of the authenticated replica. Collection fails if that original-path file is absent; if it exists independently, the wrong location is observed. The source author confirms canonical-path mirroring is not a collection requirement.

Repair the launch-intent join through the current donor. For predecessor BLOCK_FREEZE evidence, find the corresponding predetermined predecessor donor by family/block in the full release and resolve through that donor. Retain the emitter-fixed path/SHA checks and exact selected-donor joins. Use a new sealed source V2; leave V1 untouched.

## Verified unchanged work

- The V1 manifest authenticates all52 members; all new Python files parse statically. No prepared source was imported or executed.
- All26 real disabled jobs exactly preserve the sealed configurations, bounds, order, source/gate/runtime/feature/negative authority and query/schedule mapping:20 main replicas plus6 first F1 attempts. The original rich capable-single and all F1 controls remain present.
- The canonical main30 and adopted F1 nine-cell plan bytes and full39 order agree. Fixed b0 donors are the original10 main plus3 F1. No old-versus-new score selection or method-level substitution is added.
- Only `pilot_common.verify_packet` and `run_queue.main` change within the inherited family sources. All other functions, including `run_fit`, `await_resources`, output accounting, physical host, environment, training-source checks and supervisor loading, are AST-identical. Patch records match the actual byte differences.
- Eight numerical source-file hashes match their recorded unchanged qualified-source proof. Existing numerical/source/runtime reviews are reused; no numerical gate is rerun and no numerical result is decoded.
- Original physical30/no-same-cell-retry exceptions are explicit prospective authority. All new attempt IDs are unique; per-attempt automatic retry remains false. Original20 registrations, admissions, releases, queues and launch bindings are preserved. Their current terminal status and final counts/costs remain unknown; saved elapsed samples are kept separately. At least45 physical scientific starts are disclosed for the39 chosen fits.
- Existing resource limits and complete fixed queues are preserved. Conservative four-queue hard allowance877200 seconds reconciles. The dispatcher requires all earlier queues terminal and their actual processes absent; an exclusive launch intent prevents a second launch of that queue.

## Full39 custody and scope

The collector authenticates six predetermined whole-block donors and all39 exact jobs, sources, external dependencies, actual queue/child terminal receipts, fixed input identities and selected artifact bytes before emitting a complete registry. Genuine original b0 evidence uses the existing reviewed original/companion helpers; new attempts use fresh replication registrations. It does not parse source FREEZE, CONFIG, histories or prediction/checkpoint payloads. The separately disabled inventory wrapper reauthenticates all39 before its first history-byte hash. A separately reviewed amended D2 semantic/scoring adapter remains required.

The two incorrect replica joins prevent an operational PASS despite that intended global ordering. Actual current provider/GPU/resource status, b0 all13 terminal/artifact custody and remote runtime/data/source availability remain root duties before any separately enabled release. This static review establishes none of those physical facts.

All root execution, job-review, b0-custody, collection and history-inventory templates remain disabled. No SSH, transport, launch, numerical compute, score/history/result/tensor/label read, current route check, cleanup, canonical change or manuscript edit was performed.

## Files

- FINDINGS.json: the P2 defect and its two exact sites.
- CHECKS.json: preserved contracts, static comparisons and limits.
- SOURCE_BINDINGS.json: exact reviewed safe source/metadata hashes.
- REVIEW.json: bounded request-changes status.
- audit.py: independent stdlib AST/hash/metadata comparison, without inspected-source import/execution.
'''
with (HERE / 'REPORT.md').open('x') as handle:
    handle.write(report)
assert all(sha(BASE / q['path']) == q['sha256'] for q in bindings)
emit('MANIFEST.json', {'files': [{'path': q.name, 'bytes': q.stat().st_size, 'sha256': sha(q)} for q in sorted(HERE.iterdir()) if q.is_file()], 'status': 'REQUEST_CHANGES_DISABLED', 'scientific_execution': False, 'source_manifest_sha256': sha(PACKET / 'MANIFEST.json')})
emit('SEAL.json', {'manifest_sha256': sha(HERE / 'MANIFEST.json'), 'status': 'SEALED_INDEPENDENT_OPERATIONAL_REVIEW_REQUEST_CHANGES'})
for q in HERE.iterdir():
    os.chmod(q, 0o444)
print(json.dumps({'status': 'REQUEST_CHANGES_DISABLED', 'manifest_sha256': sha(HERE / 'MANIFEST.json'), 'seal_sha256': sha(HERE / 'SEAL.json'), 'issue_count': 1, 'affected_sites': 2}))
