"""Bounded stdlib mapping audit; no execution payload/score reads or launches."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MAPPING = BASE / 'shared_private_transfer_allocation_replication_metadata_review_20261005_v1'
PLAN = BASE / 'shared_private_transfer_paired_pilot_execution_root_20261005_v2/COHORT_PLAN.json'
F1 = BASE / 'shared_private_transfer_row0_single_companion_preparation_20261005_v1/COMPANION_PLAN_DISABLED.json'
MAIN_RELEASE = BASE / 'shared_private_transfer_paired_pilot_execution_root_20261005_v2/ROOT_RELEASE.json'
F1_RELEASE = BASE / 'shared_private_transfer_row0_companion_launch_activation_20261005_v1/ROOT_RELEASE_B0.json'
F1_JOB = BASE / 'shared_private_transfer_row0_companion_launch_activation_20261005_v1/control_singleton/authenticated_remote/shared_private_transfer_row0_companion_b0_execution_root_20261005_v1/jobs/b0_F1_end_live.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def emit(name, value):
    with (HERE / name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


inputs = list(sorted(MAPPING.iterdir())) + [PLAN, F1, MAIN_RELEASE, F1_RELEASE, F1_JOB]
bindings = [{'path': str(p.relative_to(BASE)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in inputs]
assert sha(MAPPING / 'MANIFEST.json') == '234ee8df0e690f06a5d51cad894a39641ea9eefb6c116041aa60fd90e10d8c92'
assert read(MAPPING / 'SEAL.json')['manifest']['sha256'] == sha(MAPPING / 'MANIFEST.json')
for entry in read(MAPPING / 'MANIFEST.json')['files']:
    path = BASE / entry['path']
    assert path.parent == MAPPING
    assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
x = read(MAPPING / 'FROZEN_26_CELLS.json')
p = read(PLAN)
f = read(F1)
main_release = read(MAIN_RELEASE)
f1_release = read(F1_RELEASE)
job = read(F1_JOB)
main_cells = {c['cell_id']: c for c in p['cells']}
f1_cells = {c['cell_id']: c for c in f['cells']}
expected_order = [k for k in p['execution_order'] if k.startswith(('b1_', 'b2_'))] + [k for k in f['execution_order'] if k.startswith(('b1_', 'b2_'))]
assert expected_order == x['fixed26_order'] == [c['cell_id'] for c in x['cells']]
assert len(set(expected_order)) == 26
counts = {'original30': 0, 'row0_companion9': 0}
for c in x['cells']:
    original = main_cells if c['family'] == 'original30' else f1_cells
    bounds = p['fit_bounds_by_cell'] if c['family'] == 'original30' else f['fit_bounds_by_cell']
    assert c['scientific_configuration'] == original[c['cell_id']]
    assert c['fit_bounds_seconds'] == bounds[c['scientific_configuration']['cell']]
    assert c['runtime_versions'] == job['runtime_versions']
    counts[c['family']] += 1
assert counts == {'original30': 20, 'row0_companion9': 6}
assert len(main_cells) == 30 and len(f1_cells) == 9
assert sum(k.startswith('b0_') for k in main_cells) == 10
assert sum(k.startswith('b0_') for k in f1_cells) == 3
assert p['selection'] == f['selection'] == 'first_maximum_complete_VALID_MRR_rounded4'
assert all(c['scientific_configuration']['schedule'] == {'eval_every_cycles': 5, 'max_cycles': 60, 'validation_miss_limit': 11} for c in x['cells'])
assert all((c['scientific_configuration']['outer_size'], c['scientific_configuration']['inner_size']) == (64, 256) for c in x['cells'])
assert f['initial_parity_target'] == 'F4_route0_not_pooled_logits'
assert f['inner_operation'] == 'four_fixed_stream_forwards_mean_normalized_own_loss_one_private_commit'
assert p['TEST_closed'] is True and f['TEST_closed'] is True
assert x['scores_accessed'] is False and x['execution_authorized_here'] is False
budget = read(MAPPING / 'REPORT.json')['budgets']
assert sum(v['hard_seconds'] for v in p['fit_bounds_by_cell'].values()) == budget['original10_fit_hard_seconds_per_block'] == 306000
assert sum(v['hard_seconds'] for v in f['fit_bounds_by_cell'].values()) == budget['F1_3_fit_hard_seconds_per_block'] == 84000
assert main_release['queue_hard_seconds'] == budget['original10_queue_hard_seconds_per_block'] == 342900
assert f1_release['queue_hard_seconds'] == budget['F1_3_queue_hard_seconds_per_block'] == 95700
assert 2 * (342900 + 95700) == budget['four_queue_total_hard_seconds'] == 877200
e = read(MAPPING / 'LOCAL_OBSERVATION_EVIDENCE.json')
assert (e['b0_original30']['completed'], e['b0_companion9']['completed']) == (10, 3)
assert all(row['scientific_queue_launched'] is True and row['latest_terminal_status'] == 'UNKNOWN_AFTER_WITHDRAWAL' and row['last_saved_status_is_not_current_terminal_evidence'] is True for row in e['old77'])
known_started = sum(row['last_saved_completed_count'] + bool(row['last_saved_active_cell_id']) for row in e['old77'])
requirements = read(MAPPING / 'REPLICATION_REQUIREMENTS.json')
assert known_started == requirements['known_old77_started_physical_fits_lower_bound'] == 6
assert requirements['total_physical_fits_lower_bound_including_known_old77'] == 39 + known_started == 45
assert requirements['original30_physical_fit_cap_and_no_retry_need_explicit_replication_exception'] is True
assert requirements['setup_fallback_condition_not_met'] == 'b1/b2 scientific fits already started'
assert x['same_physical_GPU'] == p['resource_assignment']['b0']
assert all(sha(BASE / entry['path']) == entry['sha256'] for entry in bindings)

checks = {'status': 'PASS_MAPPING_ONLY', 'mapping_seal_authenticated': True, 'exact_26_configurations_bounds_and_order': True, 'replica_cell_counts': counts, 'predetermined_b0_donor_counts': {'original': 10, 'F1': 3}, 'logical_required_cells': 39, 'known_old77_started_fit_lower_bound': known_started, 'prospective_physical_fit_lower_bound': 45, 'four_queue_hard_seconds': 877200, 'schedule': {'max_cycles': 60, 'eval_every_cycles': 5, 'validation_miss_limit': 11}, 'query_sizes': {'outer': 64, 'inner': 256}, 'selector': p['selection'], 'exact_scientific_source_identifiers_by_family': {family: {'program_sha256': sorted({c['allocation_program_sha256'] for c in x['cells'] if c['family'] == family}), 'source_manifest_sha256': sorted({c['allocation_source_manifest_sha256'] for c in x['cells'] if c['family'] == family})} for family in counts}, 'old77_current_status': 'UNKNOWN_AFTER_WITHDRAWAL', 'no_current_GPU_or_donor_artifact_custody_claim': True, 'adapter_source_not_yet_reviewed': True, 'all_inputs_unchanged': True, 'concrete_findings': []}
emit('CHECKS.json', checks)
emit('SOURCE_BINDINGS.json', {'inputs': bindings, 'numerical_authority_result_descriptors_followed': False, 'execution_payload_descriptors_followed': False})
emit('REVIEW.json', {'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_MAPPING_ONLY_NO_EXECUTION_ADMISSION', 'concrete_findings': [], 'conclusion': 'The metadata mapping correctly specifies a prospective availability-driven allocation replication amendment with all26 fixed cells, predetermined13 b0 donors and full39 scoring requirement. Operational adapters and actual root custody/admission remain separate requirements.', 'scientific_scores_read': False, 'prepared_source_imported': False, 'remote_contact': False, 'compute_or_launch': False, 'canonical_or_manuscript_edit': False, 'adapter_review_pending': True, 'root_execution_approval_required': True})
report = '''# Independent allocation replication mapping review

**PASS for the sealed metadata mapping only.** No concrete mapping flaw was found. This report does not approve execution or certify the manuscript. Disabled operational adapters will receive a distinct bounded review after their source is sealed.

## Exact mapping

The 26 cell configurations, fit bounds and order exactly reproduce the immutable original30 plan and companion9 spec: original b1 ten, original b2 ten, F1 b1 three, then F1 b2 three. All six F1 live/detached/ordinary companions are present. The original capable-single live/detached/ordinary controls remain present in both blocks. The predetermined completed b0 donors are ten original cells plus three F1 cells; full scientific completion remains 30+9=39.

All mapped cells retain seed, factor seed, support/stream block, rule, geometry, outer64/inner256 queries, 60-cycle horizon, five-cycle VALID cadence, eleven-miss limit and first-max rounded4 selector. Family-specific program and source-manifest identifiers remain distinct: the original singleton architecture serves20 original cells and the qualified row0 singleton serves6 F1 cells. The F1 spec preserves four explicit inner streams and F4 route0 initial parity, rather than pooled-output parity. Existing numerical source/runtime and methodological reviews are reused, not requalified here.

The resource-budget arithmetic reconciles: original ten-cell fit hard sum306000 seconds and queue342900; F1 three-cell fit84000 and queue95700; four queues877200 seconds. These are hard bounds, not predicted durations, and do not alter the common scientific horizon.

## Prospective authority and preservation

The original b1/b2 scientific queues already started. The setup-fallback condition therefore does not apply. The mapping explicitly requires a prospective replication amendment and an exception to the original physical-fit/no-retry release; it does not label this as setup fallback, resumption or a first attempt. The20 main reruns are fresh scientific attempts; the6 F1 companions were previously unreleased.

The mapping preserves all original77 attempt registrations and their unknown current status, both old and new histories and costs, fresh attempt identities and prospective whole-block donor choices. It forbids selecting old-versus-new scores or borrowing favorable old cells. Six prior old77 fits are known to have started from saved metadata, so39 selected fits imply at least45 physical fits; the final old77 count and cost remain unknown.

The full39 authentication requirement precedes mechanism scoring or history parsing. No old score, outcome file, model/tensor, numerical authority result or history was opened in this review. No77/MacLink contact, monitoring, copy, signal or cleanup occurred.

## Remaining operational duties

The metadata report expressly leaves current authorized GPU custody and actual b0 terminal/selected-artifact custody to root. It also identifies the old fallback-only/b0-only adapters as insufficient operational authority. The prospective adapters must enforce all fixed mappings, real source/runtime/data/gate identities, explicit replication/cost preservation, fresh owned finite supervision, stop-on-failure/no automatic retry and full39 collection before scoring. Root must approve execution only after the adapter review and current custody/admission checks.

This review authenticates the sealed mapping and compares local frozen metadata. It does not claim current provider availability, bitwise cross-provider equivalence, donor-artifact completeness, model correctness, a scientific outcome, novelty or manuscript acceptance.

## Files

- CHECKS.json: exact mapping/count/budget checks and preserved requirements.
- SOURCE_BINDINGS.json: reviewed metadata bytes and hashes.
- REVIEW.json: bounded status and scope.
- review.py: standard-library mapping comparison; no source imports or launch calls.
'''
with (HERE / 'REPORT.md').open('x') as handle:
    handle.write(report)
emit('MANIFEST.json', {'files': [{'path': q.name, 'bytes': q.stat().st_size, 'sha256': sha(q)} for q in sorted(HERE.iterdir()) if q.is_file()], 'scientific_execution': False, 'status': 'PASS_MAPPING_ONLY_NO_EXECUTION_ADMISSION'})
emit('SEAL.json', {'manifest_sha256': sha(HERE / 'MANIFEST.json'), 'status': 'SEALED_INDEPENDENT_MAPPING_REVIEW'})
for q in HERE.iterdir():
    os.chmod(q, 0o444)
print(json.dumps({'status': 'PASS_MAPPING_ONLY_NO_EXECUTION_ADMISSION', 'manifest_sha256': sha(HERE / 'MANIFEST.json'), 'seal_sha256': sha(HERE / 'SEAL.json')}))
