"""Stdlib decision draft and per-phase metadata generator; never runs science."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from continuation_support import (HERE, PHASE, REMOTE_PHASE, PHASES, FIRST_CELL, require, read,
    write, descriptor, object_hash, source_descriptors, verify_sources, registry_guard, finite_plan,
    row_for, completed_phase, decision_guard, scan_state, expected_admission, own_sources, remote, mirror,
    R17_REL, GPU_UUID, LOGIN, deployment_ancillary)

REGISTRY_PATH = str(REMOTE_PHASE/'graph_init_execution_root_v1/study_v1/GRAPH_INIT_ATTEMPT_REGISTRY.json')
RUN_ROOT = str(REMOTE_PHASE/'graph_init_execution_continuation_v1/coordinator_run_v1')


def protected():
    # Exact code/control files, plus all existing initial packet payload/seal bindings.
    records = []
    for path in ('protocols/launch_modern_root_v1.py', 'protocols/bounded_run_v1.py',
                 'protocols/run_authorized_v2.py', 'protocols/run_logged.py', 'protocols/repo_env.sh',
                 'protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json',
                 'graph_init_execution_root_v1/MANIFEST.json', 'graph_init_execution_root_v1/SEAL.json',
                 'graph_init_execution_continuation_v1/continuation_support.py',
                 'graph_init_execution_continuation_v1/build_continuation.py',
                 'graph_init_execution_continuation_v1/continuation_entry.py',
                 'graph_init_execution_continuation_v1/finite_coordinator.py'):
        records.append(descriptor(PHASE/path))
    initial = read(PHASE/'graph_init_execution_root_v1/prepared_v1/ROOT_REGISTER_REQUEST_DRAFT.json')
    records += initial['protected_files']
    records += deployment_ancillary()
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
        'independent_R17_source_audit_accepted': False, 'independent_R17_source_audit': null.copy(),
        'continuation_source_review_accepted': False, 'continuation_source_review': null.copy(),
        'attempt_registry': {'path': REGISTRY_PATH, 'sha256': None}, 'study_id': None,
        'anchor_directory': str(Path(REGISTRY_PATH).parent), 'contexts_sha256': None,
        'initial_qualification_freeze': {'path': str(Path(REGISTRY_PATH).parent/'qualify/Squirrel_seed17/FREEZE.json'), 'sha256': None},
        'initial_qualification_terminal': null.copy(), 'finite_plan': [], 'finite_plan_sha256': None,
        'initial_whole_supervision_terminal': {'path': str(REMOTE_PHASE/'graph_init_execution_root_v1/supervision/qualify_Squirrel_seed17_v1_outer/TERMINAL.json'), 'sha256': None},
        'initial_root_terminal': {'path': str(REMOTE_PHASE/'graph_init_execution_root_v1/root_receipts/qualify_Squirrel_seed17_v1/TERMINAL.json'), 'sha256': None},
        'coordinator_run_root': RUN_ROOT, 'protected_files': protected(),
        'remaining_attempt_counts': {'qualify': 5, 'warm': 6, 'initialize': 30, 'fit': 30},
        'schedule_rule': 'All remaining cold qualifiers first; then all warm; then all five initialized arms per context; then all same-arm fits. Every cap equals its registered resource forecast.',
        'dependency_rule': 'Current-registry canonical successful claim+terminal+FREEZE only; no copied checkpoint or cross-registry binding.',
        'failure_rule': 'Any failed, unresolved or pre-launch coordinator/whole-process attempt stops this finite run; retain all evidence and do not retry.',
        'scope': 'Outcome-aware exploratory cfg0 matched-arm study. No new recipes/tolerances/seeds/configs. Warm uses qualified cold context; every initialize checks actual-warm AD/Adam/logit/installation/RNG. No final labels or report.'}
    if mirror(REGISTRY_PATH).is_file():
        record = descriptor(REGISTRY_PATH)
        registry = registry_guard(record)
        plan = finite_plan(registry)
        first = row_for(registry, [c for c in registry['contexts'] if (c['graph'], c['seed']) == FIRST_CELL][0], 'qualify')
        decision.update({'attempt_registry': record, 'study_id': registry['study_id'],
            'anchor_directory': registry['anchor_directory'], 'contexts_sha256': object_hash(registry['contexts']),
            'initial_qualification_terminal': {'path': str(Path(registry['anchor_directory'])/'terminals'/(first['key']+'.json')), 'sha256': None},
            'finite_plan': plan, 'finite_plan_sha256': object_hash(plan)})
        claim = mirror(Path(registry['anchor_directory'])/'claims'/(first['key']+'.json'))
        terminal_path = mirror(Path(registry['anchor_directory'])/'terminals'/(first['key']+'.json'))
        if claim.is_file() and terminal_path.is_file():
            freeze, terminal, _ = completed_phase(record, registry, first)
            decision['initial_qualification_freeze'] = freeze
            decision['initial_qualification_terminal'] = terminal
            for name in ('initial_whole_supervision_terminal', 'initial_root_terminal'):
                path = decision[name]['path']
                if mirror(path).is_file():
                    decision[name] = descriptor(path)
    return decision


def prepare_decision(registry_path, output):
    verify_sources()
    decision = template()
    manifests = own_sources()
    decision['continuation_source_manifest'], decision['continuation_source_seal'] = manifests
    record = descriptor(registry_path)
    registry = registry_guard(record)
    completed = scan_state(record, registry)
    first = row_for(registry, [c for c in registry['contexts'] if (c['graph'], c['seed']) == FIRST_CELL][0], 'qualify')
    require(set(completed) == {first['key']}, 'Decision preparation requires only successful initial qualifier; no prior scientific run')
    freeze, terminal, _ = completed_phase(record, registry, first)
    plan = finite_plan(registry)
    decision.update({'attempt_registry': record, 'study_id': registry['study_id'],
        'anchor_directory': registry['anchor_directory'], 'contexts_sha256': object_hash(registry['contexts']),
        'initial_qualification_freeze': freeze, 'initial_qualification_terminal': terminal,
        'finite_plan': plan, 'finite_plan_sha256': object_hash(plan)})
    whole = read(decision['initial_whole_supervision_terminal']['path'])
    root = read(decision['initial_root_terminal']['path'])
    require(whole['complete'] is True and whole['within_whole_cap'] is True and
            whole['root_request_unchanged'] is True and root['completed'] is True,
            'Successful initial whole/root supervision is required')
    write(mirror(output), decision)
    return output


def next_row(decision, registry, plan):
    completed = scan_state(decision['attempt_registry'], registry)
    first = row_for(registry, [c for c in registry['contexts'] if (c['graph'], c['seed']) == FIRST_CELL][0], 'qualify')
    keys = [r['key'] for r in plan]
    done = [key for key in keys if key in completed]
    require(done == keys[:len(done)] and set(completed) == {first['key']} | set(done),
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
