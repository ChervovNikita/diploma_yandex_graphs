"""Real SQLite registry and launcher policy; fabricated contracts, no workers."""
import argparse
from collections import Counter
import copy
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

OUT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--controller', required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.controller).resolve()))
imports = []


class NoNumerical(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'scipy', 'pandas', 'ogb', 'torch_geometric'}:
            imports.append(fullname)
            raise AssertionError('Numerical imports forbidden')


sys.meta_path.insert(0, NoNumerical())
import hpo_frontier as frontier
import launcher
import registry as registry_module


def reject(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except (ValueError, RuntimeError):
        return
    raise AssertionError('Expected contract/frontier rejection')


def deny(*args, **kwargs):
    raise AssertionError('No worker, scientific source or transport may execute')


with tempfile.TemporaryDirectory(dir=OUT / 'tests') as tmp:
    root = Path(tmp)
    runs, placement, dependencies, studies = {}, {}, {}, {}
    for worker in ('allocation/gpu0', 'server77/gpu0', 'server77/gpu1'):
        name = worker.replace('/', '-')
        spec = {'study_id': name, 'runs': [], 'release': {'checkpoint_retention_policy': {'unused_search_binary_release': {'audited_metadata_selectors': {}, 'unused_search_fits': {}}}}}
        retention = spec['release']['checkpoint_retention_policy']['unused_search_binary_release']
        for number in range(3):
            sid = 'select' + str(number)
            parents = ['c' + str(number) + 'fit' + str(index) for index in range(4)]
            for identifier in parents:
                run = {'run_id': identifier, 'kind': 'model_fit', 'method': 'fabricated', 'split': 0, 'seed': 0, 'body': 0, 'parents': [], 'config': {'stage': 'search', 'reservation_mib': 10, 'estimated_seconds': 100000 if number == 2 else 100}}
                spec['runs'].append(run)
                retention['unused_search_fits'][identifier] = {'direct_metadata_selection_ids': [sid]}
            selector = {'run_id': sid, 'kind': 'selection', 'method': 'fabricated', 'split': 0, 'seed': 0, 'body': 0, 'parents': parents, 'config': {'stage': 'search_selection', 'reservation_mib': 1, 'estimated_seconds': 1}}
            spec['runs'].append(selector)
            retention['audited_metadata_selectors'][sid] = {'fit_cohort_ids': sorted(parents), 'binding': {'factory': 'fabricated_selector', 'source': {'sha256': 'fabricated'}}}
        studies[name] = {'spec': spec}
        for run in spec['runs']:
            key = name + '/' + run['run_id']
            runs[key] = {'study': name, 'run': run}
            placement[key] = {'worker': worker}
            dependencies[key] = [name + '/' + parent for parent in run['parents']]
    policy = frontier.declare(runs, placement, {name: value['spec'] for name, value in studies.items()})
    assert len(policy['cohorts']) == 9 and all(x == 8 for x in policy['maximum_two_cohort_fit_parent_records_per_worker'].values())
    plan = {'hpo_cohort_admission': policy}
    frontier.validate(plan, runs, placement, studies)
    changed = copy.deepcopy(plan)
    changed['hpo_cohort_admission']['maximum_open_cohorts_per_GPU'] = 3
    reject(frontier.validate, changed, runs, placement, studies)
    changed = copy.deepcopy(placement)
    changed['allocation-gpu0/c0fit0']['worker'] = 'server77/gpu0'
    reject(frontier.declare, runs, changed, {name: value['spec'] for name, value in studies.items()})
    changed = copy.deepcopy(studies)
    del changed['allocation-gpu0']['spec']['release']['checkpoint_retention_policy']['unused_search_binary_release']['unused_search_fits']['c0fit0']
    reject(frontier.declare, runs, placement, {name: value['spec'] for name, value in changed.items()})
    # Shared fit membership opens all cohorts atomically, without overrunning cap.
    shared_runs, shared_studies = copy.deepcopy(runs), copy.deepcopy(studies)
    shared_selector = shared_runs['allocation-gpu0/select1']['run']
    shared_selector['parents'].append('c0fit0')
    shared_spec = shared_studies['allocation-gpu0']['spec']
    shared_spec['runs'] = [shared_runs['allocation-gpu0/' + run['run_id']]['run'] for run in shared_spec['runs']]
    shared_retention = shared_spec['release']['checkpoint_retention_policy']['unused_search_binary_release']
    shared_retention['audited_metadata_selectors']['select1']['fit_cohort_ids'] = sorted(shared_selector['parents'])
    shared_retention['unused_search_fits']['c0fit0']['direct_metadata_selection_ids'] = ['select0', 'select1']
    shared_policy = frontier.declare(shared_runs, placement, {name: value['spec'] for name, value in shared_studies.items()})
    assert len(shared_policy['fit_memberships']['allocation-gpu0/c0fit0']) == 2
    opened = {'allocation/gpu0': {'allocation-gpu0/select2'}}
    assert not frontier.may_admit('allocation-gpu0/c0fit0', 'allocation/gpu0', shared_policy, opened)
    opened = {}
    assert frontier.may_admit('allocation-gpu0/c0fit0', 'allocation/gpu0', shared_policy, opened)
    frontier.reserve_admission('allocation-gpu0/c0fit0', 'allocation/gpu0', shared_policy, opened)
    assert len(opened['allocation/gpu0']) == 2
    # Exact paid-source anchors are separately retained, never treated as unused HPO states.
    protected_runs, protected_studies, protected_placement = copy.deepcopy(runs), copy.deepcopy(studies), copy.deepcopy(placement)
    source = {'run_id': 'search-anchor-source', 'kind': 'model_fit', 'method': 'fabricated_source', 'adapter': 'fabricated.py:make_adapter',
        'parents': [], 'config': {'stage': 'search', 'pt_stage': 'source'}}
    paid = {'run_id': 'paid', 'kind': 'model_fit', 'method': 'fabricated_paid', 'parents': ['search-anchor-source'], 'config': {'stage': 'comparative'}}
    for run in (source, paid):
        key = 'allocation-gpu0/' + run['run_id']
        protected_runs[key] = {'study': 'allocation-gpu0', 'run': run}
        protected_placement[key] = {'worker': 'allocation/gpu0'}
        protected_studies['allocation-gpu0']['spec']['runs'].append(run)
    protected_studies['allocation-gpu0']['spec']['release']['checkpoint_retention_policy']['unused_search_binary_release']['protected_search_source_fits'] = {
        source['run_id']: {'factory': source['adapter'], 'source': {'path': 'fabricated.py', 'bytes': 1, 'sha256': 'fabricated'},
            'contract_sha256': frontier.fingerprint(source), 'direct_paid_fit_ids': ['paid'], 'full_state_required': True, 'release_under_unused_search_policy': False}}
    protected_policy = frontier.declare(protected_runs, protected_placement, {name: value['spec'] for name, value in protected_studies.items()})
    assert protected_policy['fixed_protected_source_fit_records_per_worker'] == {'allocation/gpu0': 1}
    assert frontier.may_admit('allocation-gpu0/search-anchor-source', 'allocation/gpu0', protected_policy, {'allocation/gpu0': {'allocation-gpu0/select0', 'allocation-gpu0/select1'}})
    mutated = copy.deepcopy(protected_runs)
    mutated['allocation-gpu0/search-anchor-source']['run']['config']['pt_stage'] = 'changed'
    reject(frontier.declare, mutated, protected_placement, {name: value['spec'] for name, value in protected_studies.items()})
    db = registry_module.Registry(root / 'fabricated.sqlite')
    with patch.object(registry_module, 'validate', lambda spec: None):
        for value in studies.values():
            db.register(value['spec'])

    def complete(key, success=True):
        study, identifier = key.split('/')
        db.claim(study, identifier, {'pid': 999999, 'hostname': 'fabricated', 'started': 0, 'start_ticks': 1, 'pgid': 999999})
        db.finish(study, identifier, {'status': 'complete' if success else 'failed'}, success=success)

    for worker in ('allocation/gpu0', 'server77/gpu0', 'server77/gpu1'):
        study = worker.replace('/', '-')
        for number in (0, 1):
            complete(study + '/c' + str(number) + 'fit0')
    released, closure_calls = set(), []
    retention_module = types.ModuleType('unused_search_retention')

    def release_closed(root, study, registry, selector_id):
        key = study['study_id'] + '/' + selector_id
        closure_calls.append(key)
        return {'closed': key in released, 'blocked': [] if key in released else ['fabricated_tombstone_pending'], 'tombstone_receipts': [{'path': 'fabricated', 'bytes': 1, 'sha256': 'fabricated'}] if key in released else []}

    retention_module.cohort_release_closed = release_closed
    snapshots = {}
    with patch.dict(sys.modules, {'unused_search_retention': retention_module}), patch.object(launcher.subprocess, 'Popen', deny):
        for host in ('allocation', 'server77'):
            snapshots[host] = frontier.inspect(root, plan, runs, db.states(), placement, studies, db, host, [], {})
            for opened in snapshots[host]['open_cohorts_by_worker'].values():
                assert len(opened) == 2
            capacities = {worker: 100 for worker in policy['largest_cohort_fit_parent_records_per_worker'] if worker.startswith(host + '/')}
            admitted = launcher.schedule_ready(runs, db.states(), placement, dependencies, [], capacities, host, cohort_policy=policy, cohort_snapshot=snapshots[host])
            expected = {'allocation/gpu0': 2} if host == 'allocation' else {'server77/gpu0': 2, 'server77/gpu1': 2}
            assert Counter(placement[key]['worker'] for key in admitted) == expected
            assert all('/c2fit' not in key for key in admitted)  # Longest new cohort is blocked.
        # Every parent terminal makes short selectors first without waiting for the global search gate.
        for number in (0, 1):
            for index in (1, 2, 3):
                complete('allocation-gpu0/c' + str(number) + 'fit' + str(index))
        snap = frontier.inspect(root, plan, runs, db.states(), placement, studies, db, 'allocation', [], {})
        admitted = launcher.schedule_ready(runs, db.states(), placement, dependencies, [], {'allocation/gpu0': 100}, 'allocation', cohort_policy=policy, cohort_snapshot=snap)
        assert admitted == ['allocation-gpu0/select0', 'allocation-gpu0/select1']
        for number in (0, 1):
            complete('allocation-gpu0/select' + str(number))
        snap = frontier.inspect(root, plan, runs, db.states(), placement, studies, db, 'allocation', [], {})
        assert len(snap['open_cohorts_by_worker']['allocation/gpu0']) == 2
        admitted = launcher.schedule_ready(runs, db.states(), placement, dependencies, [], {'allocation/gpu0': 100}, 'allocation', cohort_policy=policy, cohort_snapshot=snap)
        assert admitted == [] and frontier.attention(snap, db.states(), runs, placement, 'allocation')
        released.update({'allocation-gpu0/select0', 'allocation-gpu0/select1'})
        cache = {}
        snap = frontier.inspect(root, plan, runs, db.states(), placement, studies, db, 'allocation', [], cache)
        assert snap['validated_closed_cohorts'] == 2 and not snap['open_cohorts_by_worker'] and len(cache) == 2
        admitted = launcher.schedule_ready(runs, db.states(), placement, dependencies, [], {'allocation/gpu0': 100}, 'allocation', cohort_policy=policy, cohort_snapshot=snap)
        assert len(admitted) == 2 and all('/c2fit' in key for key in admitted)
        previous = len(closure_calls)
        frontier.inspect(root, plan, runs, db.states(), placement, studies, db, 'allocation', [], cache)
        assert len(closure_calls) == previous
        snap = frontier.inspect(root, plan, runs, db.states(), placement, studies, db, 'allocation', [], {})
        assert snap['validated_closed_cohorts'] == 2 and len(closure_calls) == previous + 2  # Restart reaudits.
        snap = frontier.inspect(root, plan, runs, db.states(), placement, studies, db, 'allocation', [{'key': admitted[0]}], {})
        assert snap['open_cohorts_by_worker']['allocation/gpu0'] == ['allocation-gpu0/select2']  # Preclaim admission is reserved.
        # Failures remain open even if selector metadata later completes.
        complete('allocation-gpu0/c2fit0', success=False)
        for index in (1, 2, 3):
            complete('allocation-gpu0/c2fit' + str(index))
        complete('allocation-gpu0/select2')
        released.add('allocation-gpu0/select2')
        snap = frontier.inspect(root, plan, runs, db.states(), placement, studies, db, 'allocation', [], {})
        assert snap['open_cohorts_by_worker']['allocation/gpu0'] == ['allocation-gpu0/select2']
        assert 'failed_parent_has_no_closed_binary_release_authority' in snap['held_cohort_reasons']['allocation-gpu0/select2']
        # A registry that already violates the cap is refused before admission.
        complete('server77-gpu0/c2fit0')
        reject(frontier.inspect, root, plan, runs, db.states(), placement, studies, db, 'server77', [], {})
        reject(launcher.schedule_ready, runs, db.states(), placement, dependencies, [], {'allocation/gpu0': 100}, 'allocation', cohort_policy=policy)
        # Existing gate function still permits search_selection and blocks comparative.
        gates = {'global_gates': {'whole_search': {'path': 'whole_search.json', 'required_for_stages': ['comparative', 'author_cli'], 'required_hosts': []}}}
        assert launcher.gate_ready(root, gates, 'fabricated', 'search_selection') is True
        assert launcher.gate_ready(root, gates, 'fabricated', 'comparative') is False
        (root / 'whole_search.json').write_text(json.dumps({'status': 'complete', 'plan_hash': 'fabricated', 'closed_phase': 'whole_search', 'host_receipts': {}}))
        assert launcher.gate_ready(root, gates, 'fabricated', 'comparative') is True
    db.close()

assert not imports
print(json.dumps({'schema': 'bounded-HPO-real-registry-launcher-synthetic-QA-v1', 'fabricated_studies': 3,
    'fabricated_cohorts': 9, 'synthetic_SQLite_registries': 1, 'max_open_cohorts_per_GPU': 2,
    'three_worker_independent_two_slot_cap_checks': True, 'longest_new_cohort_blocked': True,
    'ready_metadata_selectors_prioritized': True, 'completed_unreleased_and_failed_cohorts_keep_reservations': True,
    'validated_release_reopens_admission': True, 'restart_reaudits_exact_release': True,
    'preclaim_active_reservation': True, 'changed_policy_worker_and_unqualified_fit_rejected': True,
    'overlapping_cohort_memberships_reserve_atomically': True,
    'exact_paid_source_anchors_separately_retained_and_contract_mutants_rejected': True,
    'preexisting_overcap_registry_rejected': True, 'missing_snapshot_rejected': True,
    'original_global_gate_function_unchanged_and_exercised': True, 'numerical_imports': 0,
    'actual_input_arrays': 0, 'models': 0, 'optimizer_updates': 0, 'scientific_children': 0,
    'server77_access': False, 'scientific_fits': 0, 'GO': False}, sort_keys=True))
