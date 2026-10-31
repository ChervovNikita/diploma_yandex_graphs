"""Stdlib decision draft and per-phase metadata generator; never runs science."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from continuation_support import (HERE, PHASE, REMOTE_PHASE, PHASES, FIRST_CELL, require, read,
    write, descriptor, object_hash, source_descriptors, verify_sources, registry_guard, finite_plan,
    row_for, completed_phase, decision_guard, scan_state, expected_admission, own_sources, remote, mirror,
    R17_REL, GPU_UUID, LOGIN, deployment_ancillary)

REGISTRY_PATH = str(REMOTE_PHASE/'graph_init_precision_execution_root_v2/study_v2/GRAPH_INIT_ATTEMPT_REGISTRY.json')
RUN_ROOT = str(REMOTE_PHASE/'graph_init_precision_execution_root_v2/coordinator_run_v2')


def protected():
    from build_admissions import protected_metadata
    records = protected_metadata()+deployment_ancillary()
    for name in ('build_admissions.py', 'build_continuation.py', 'continuation_support.py',
                 'continuation_entry.py', 'finite_coordinator.py', 'lineage_support.py', 'launch_registration.py'):
        records.append(descriptor(HERE/name))
    unique = {r['path']: r for r in records}
    return [unique[key] for key in sorted(unique)]

def template():
    null = {'path': None, 'sha256': None}
    decision = {'schema': 'graph-init-finite-continuation-root-decision-v1', 'approved': False,
        'approved_by': '', 'approved_utc': None, 'scientific_phases_authorized': False,
        'allowed_phases': list(PHASES), 'compare_authorized': False, 'report_authorized': False,
        'final_labels_authorized': False, 'automatic_retry_authorized': False,
        'fixed_native_protocol_unchanged': True, 'source_seals': source_descriptors(),
        'continuation_source_manifest': {'path': remote(HERE/'MANIFEST.json'), 'sha256': None},
        'continuation_source_seal': {'path': remote(HERE/'SEAL.json'), 'sha256': None},
        'independent_R17_source_audit_accepted': False,
        'independent_R17_source_audit': descriptor(PHASE/'graph_init_source_audit_v1/round17_v3_precision_recheck/REPORT.json'),
        'continuation_source_review_accepted': False, 'continuation_source_review': null.copy(),
        'attempt_registry': {'path': REGISTRY_PATH, 'sha256': None}, 'study_id': None,
        'anchor_directory': str(Path(REGISTRY_PATH).parent), 'contexts_sha256': None,
        'finite_plan': [], 'finite_plan_sha256': None,
        'registration_launch_request': {'path': str(REMOTE_PHASE/HERE.name/'admitted_v1/ROOT_REGISTER_REQUEST.json'), 'sha256': None},
        'registration_whole_supervision_terminal': {'path': str(REMOTE_PHASE/HERE.name/'supervision/register_v2_outer/TERMINAL.json'), 'sha256': None},
        'registration_root_terminal': {'path': str(REMOTE_PHASE/HERE.name/'root_receipts/register_v2/TERMINAL.json'), 'sha256': None},
        'coordinator_run_root': RUN_ROOT, 'protected_files': protected(),
        'remaining_attempt_counts': {'qualify': 6, 'warm': 6, 'initialize': 30, 'fit': 30},
        'schedule_rule': 'All six fresh cold qualifiers first; then all warm; then all five initialized arms per context; then all same-arm fits. Every cap equals its registered resource forecast.',
        'dependency_rule': 'Current-registry canonical successful claim+terminal+FREEZE only; no copied checkpoint or cross-registry binding.',
        'failure_rule': 'Any failed, unresolved or pre-launch coordinator/whole-process attempt stops this finite run; retain all evidence and do not retry.',
        'scope': 'Outcome-aware exploratory cfg0 matched-arm study. No new recipes/tolerances/seeds/configs. Warm uses qualified cold context; every initialize checks actual-warm AD/Adam/logit/installation/RNG. No final labels or report.'}
    if mirror(REGISTRY_PATH).is_file():
        record = descriptor(REGISTRY_PATH)
        registry = registry_guard(record)
        plan = finite_plan(registry)
        decision.update({'attempt_registry': record, 'study_id': registry['study_id'],
            'anchor_directory': registry['anchor_directory'], 'contexts_sha256': object_hash(registry['contexts']),
            'finite_plan': plan, 'finite_plan_sha256': object_hash(plan)})
        for name in ('registration_launch_request', 'registration_whole_supervision_terminal', 'registration_root_terminal'):
            if mirror(decision[name]['path']).is_file():
                decision[name] = descriptor(decision[name]['path'])
    return decision


def prepare_decision(registry_path, output):
    verify_sources()
    decision = template()
    manifests = own_sources()
    decision['continuation_source_manifest'], decision['continuation_source_seal'] = manifests
    record = descriptor(registry_path)
    registry = registry_guard(record)
    completed = scan_state(record, registry)
    require(completed == {}, 'Fresh precision registry must contain zero previous phase attempts')
    plan = finite_plan(registry)
    decision.update({'attempt_registry': record, 'study_id': registry['study_id'],
        'anchor_directory': registry['anchor_directory'], 'contexts_sha256': object_hash(registry['contexts']),
        'finite_plan': plan, 'finite_plan_sha256': object_hash(plan)})
    for name in ('registration_launch_request', 'registration_whole_supervision_terminal', 'registration_root_terminal'):
        decision[name] = descriptor(decision[name]['path'])
    whole = read(decision['registration_whole_supervision_terminal']['path'])
    root = read(decision['registration_root_terminal']['path'])
    require(whole['complete'] is True and whole['within_whole_cap'] is True and
            whole['root_request_unchanged'] is True and root['completed'] is True,
            'Successful new registration whole/root supervision required')
    write(mirror(output), decision)
    return output


def next_row(decision, registry, plan):
    completed = scan_state(decision['attempt_registry'], registry)
    keys = [r['key'] for r in plan]
    done = [key for key in keys if key in completed]
    require(done == keys[:len(done)] and set(completed) == set(done),
            'Completed attempts must be an exact prefix of the signed finite order')
    return None if len(done) == len(plan) else plan[len(done)]


def make_admission(decision_path, key, output):
    decision, registry, plan = decision_guard(decision_path)
    row = next_row(decision, registry, plan)
    require(row is not None and row['key'] == key, 'Only next exact signed attempt may be admitted')
    out = mirror(output)
    expected = Path(decision['coordinator_run_root'])/'requests'/key
    require(remote(out) == str(expected), 'Admission must use fixed coordinator request identity')
    out.mkdir(parents=True, exist_ok=False)
    admission = expected_admission(decision, registry, row)
    write(out/'ADMISSION.json', admission)
    run_root = Path(decision['coordinator_run_root'])
    request = {'schema': 'graph-init-continuation-launch-request-v1', 'root_admitted': True,
        'action': row['phase'], 'attempt': row, 'root_decision': descriptor(decision_path),
        'phase_admission': descriptor(out/'ADMISSION.json'), 'output': row['output'],
        'entry_script': remote(HERE/'continuation_entry.py'),
        'driver_script': str(REMOTE_PHASE/R17_REL/'prototype/graph_init_driver.py'),
        'source_packet': str(REMOTE_PHASE/R17_REL), 'source_seals': source_descriptors(),
        'whole_cap_seconds': row['whole_cap_seconds'], 'full_fit_admitted': row['phase'] == 'fit',
        'heldout_scoring_admitted': False, 'compare_admitted': False,
        'receipt_directory': str(run_root/'root_receipts'/key),
        'supervisor_directory': str(run_root/'supervision'/(key+'_inner')),
        'outer_supervisor_directory': str(run_root/'supervision'/(key+'_outer')),
        'local_launch_receipt': str(run_root/'supervision'/(key+'_LAUNCH.json')),
        'allocation_route': descriptor(PHASE/'protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json'),
        'automatic_retry': False, 'final_labels_accessible': False}
    write(out/'REQUEST.json', request)
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare-decision')
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('make-admission')
    p.add_argument('--decision', type=Path, required=True)
    p.add_argument('--key', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare-decision':
        prepare_decision(args.registry, args.output)
    else:
        make_admission(args.decision, args.key, args.output)
    print(str(args.output))


if __name__ == '__main__':
    main()
