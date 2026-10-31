"""Frozen two-cohort HPO admission with registry/tombstone reconstruction.

No tensors, fit callbacks, approval writers or deletion operations occur here.
Completed-but-unreleased and failed cohorts keep their admission reservation.
"""
from __future__ import annotations
from collections import defaultdict
from spec import fingerprint

SCHEMA = 'gnnm-prospective-two-open-HPO-cohorts-per-GPU-v1'


def declare(runs, placement, studies):
    cohorts = {}
    members = defaultdict(set)
    protected = {}
    direct_children = defaultdict(set)
    for key, row in runs.items():
        for parent in row['run']['parents']:
            direct_children[row['study'] + '/' + parent].add(key)
    search = {key for key, row in runs.items() if row['run']['kind'] == 'model_fit' and row['run']['config'].get('stage') == 'search'}
    for study_id, study in sorted(studies.items()):
        policy = study['release'].get('checkpoint_retention_policy', {}).get('unused_search_binary_release', {})
        for identifier, bound in sorted(policy.get('protected_search_source_fits', {}).items()):
            key = study_id + '/' + identifier
            if key not in search or bound.get('full_state_required') is not True or bound.get('release_under_unused_search_policy') is not False:
                raise ValueError('Only exact protected paid-source search fits may bypass releasable cohorts')
            run = runs[key]['run']
            if bound['factory'] != run['adapter'] or bound['contract_sha256'] != fingerprint(run):
                raise ValueError('Protected search source factory/full contract changed')
            direct = direct_children[key]
            if not direct or any(runs[child]['run']['kind'] != 'model_fit' for child in direct):
                raise ValueError('Protected search source must feed exact paid fits')
            if sorted(study_id + '/' + child for child in bound['direct_paid_fit_ids']) != sorted(direct):
                raise ValueError('Protected search paid-consumer roster changed')
            protected[key] = {**bound, 'worker': placement[key]['worker'], 'study': study_id}
    for key, row in sorted(runs.items()):
        selector = row['run']
        if selector['kind'] != 'selection' or selector['config'].get('stage') != 'search_selection':
            continue
        study_id = row['study']
        study = studies[study_id]
        policy = study['release'].get('checkpoint_retention_policy', {}).get('unused_search_binary_release', {})
        bound = policy.get('audited_metadata_selectors', {}).get(selector['run_id'])
        if not bound:
            raise ValueError('HPO cohort lacks source-pinned prospective unused-search retention')
        parents = sorted(study_id + '/' + name for name in selector['parents'])
        if not parents or any(parent not in search for parent in parents):
            raise ValueError('HPO selector must bind only actual search model fits')
        if bound['fit_cohort_ids'] != sorted(selector['parents']):
            raise ValueError('Frozen selector cohort differs from actual dependencies')
        worker = placement[key]['worker']
        if any(placement[parent]['worker'] != worker for parent in parents):
            raise ValueError('A bounded HPO cohort must stay on one assigned GPU')
        for parent in parents:
            identifier = runs[parent]['run']['run_id']
            entry = policy.get('unused_search_fits', {}).get(identifier)
            if not entry or selector['run_id'] not in entry['direct_metadata_selection_ids']:
                raise ValueError('HPO fit is outside audited metadata-only retention')
            members[parent].add(key)
        cohorts[key] = {'worker': worker, 'study': study_id, 'selector_id': selector['run_id'],
                        'fit_keys': parents, 'metadata_selector_binding': bound['binding']}
    if set(members) & set(protected) or set(members) | set(protected) != search:
        raise ValueError('Every search fit requires exact releasable cohort membership or a source-pinned protected paid-source authority')
    for parent, memberships in members.items():
        direct = direct_children[parent]
        if direct != memberships:
            raise ValueError('A search state has a direct consumer outside audited cohorts')
        if len(memberships) > 2:
            raise ValueError('A fit cannot atomically open more cohorts than the two-cohort cap')
    largest = defaultdict(int)
    for value in cohorts.values():
        largest[value['worker']] = max(largest[value['worker']], len(value['fit_keys']))
    return {'schema': SCHEMA, 'maximum_open_cohorts_per_GPU': 2, 'cohorts': cohorts,
        'protected_search_source_fits': protected,
        'fixed_protected_source_fit_records_per_worker': {worker: sum(value['worker'] == worker for value in protected.values()) for worker in sorted({value['worker'] for value in protected.values()})},
        'fit_memberships': {key: sorted(value) for key, value in sorted(members.items())},
        'largest_cohort_fit_parent_records_per_worker': dict(sorted(largest.items())),
        'maximum_two_cohort_fit_parent_records_per_worker': {worker: 2 * count for worker, count in sorted(largest.items())},
        'closure_requires': 'complete metadata selector, exact terminal owners, validated unused-search tombstones and absent binaries for every completed fit; failures remain reserved',
        'ready_search_selection_priority': True, 'job_configs_assignments_global_gates_unchanged': True}


def validate(plan, runs, placement, studies):
    policy = plan.get('hpo_cohort_admission')
    if policy is None:
        return  # Legacy packet preflight remains supported; new compiler always declares it.
    if policy != declare(runs, placement, {name: value['spec'] for name, value in studies.items()}):
        raise ValueError('Frozen HPO cohort admission differs from actual study DAG/retention/assignment')


def inspect(root, plan, runs, states, placement, studies, registry, host, active, closed_cache):
    policy = plan.get('hpo_cohort_admission')
    if policy is None:
        return None
    if policy.get('schema') != SCHEMA or policy.get('maximum_open_cohorts_per_GPU') != 2:
        raise ValueError('Exact two-cohort prospective admission policy required')
    active_keys = {item['key'] for item in active}
    open_by_worker = defaultdict(list)
    blocked = {}
    closed_count = 0
    for key, cohort in policy['cohorts'].items():
        worker = cohort['worker']
        if worker.split('/')[0] != host:
            continue
        parents = cohort['fit_keys']
        started = states[key] != 'registered' or any(states[parent] != 'registered' or parent in active_keys for parent in parents) or key in active_keys
        if not started:
            continue
        if key in closed_cache:
            if states[key] != 'complete' or any(states[parent] != 'complete' for parent in parents) or active_keys & {key, *parents}:
                raise ValueError('A validated closed HPO cohort changed custody or registry state')
            closed_count += 1
            continue
        reasons = []
        if key in active_keys or active_keys.intersection(parents):
            reasons.append('owned_child_still_active')
        if states[key] != 'complete':
            reasons.append('metadata_selector_' + states[key])
        if any(states[parent] == 'failed' for parent in parents):
            reasons.append('failed_parent_has_no_closed_binary_release_authority')
        if any(states[parent] not in {'complete', 'failed'} for parent in parents):
            reasons.append('fit_cohort_not_terminal')
        if not reasons:
            from unused_search_retention import cohort_release_closed
            study = studies[cohort['study']]['spec']
            selection = registry.run(cohort['study'], cohort['selector_id'])
            if selection['contract'] != runs[key]['run'] or selection['state'] != 'complete' or (selection.get('result') or {}).get('status') != 'complete':
                raise ValueError('Selector registry evidence differs from successful frozen metadata contract')
            for parent in parents:
                run = runs[parent]['run']
                actual = registry.run(cohort['study'], run['run_id'])
                if actual['contract'] != run or actual['state'] != states[parent]:
                    raise ValueError('Search fit registry contract/state changed')
            audit = cohort_release_closed(root, study, registry, cohort['selector_id'])
            if audit['closed']:
                closed_cache[key] = fingerprint(audit)
                closed_count += 1
                continue
            reasons.extend(audit['blocked'])
        open_by_worker[worker].append(key)
        blocked[key] = sorted(set(reasons))
    cap = policy['maximum_open_cohorts_per_GPU']
    if any(len(value) > cap for value in open_by_worker.values()):
        raise ValueError('Registry reconstructs more open HPO cohorts than the frozen GPU cap; inspect retained state before launch')
    counts = {worker: sum(len(policy['cohorts'][key]['fit_keys']) for key in value) for worker, value in open_by_worker.items()}
    protected_started = defaultdict(int)
    for key, value in policy['protected_search_source_fits'].items():
        if value['worker'].split('/')[0] == host and (states[key] != 'registered' or key in active_keys):
            protected_started[value['worker']] += 1
    return {'schema': SCHEMA, 'host': host, 'maximum_open_cohorts_per_GPU': cap,
        'open_cohorts_by_worker': {worker: sorted(value) for worker, value in sorted(open_by_worker.items())},
        'current_maximum_registered_fit_parent_records_per_worker': dict(sorted(counts.items())),
        'validated_closed_cohorts': closed_count, 'held_cohort_reasons': dict(sorted(blocked.items())),
        'fixed_protected_source_fit_records_per_worker': policy['fixed_protected_source_fit_records_per_worker'],
        'started_protected_source_fit_records_per_worker': dict(sorted(protected_started.items())),
        'validated_closed_cohort_release_digests': {key: value for key, value in sorted(closed_cache.items()) if policy['cohorts'][key]['worker'].split('/')[0] == host},
        'registry_and_exact_release_tombstones_audited': True}


def priority(key, runs):
    run = runs[key]['run']
    selector = run['kind'] == 'selection' and run['config'].get('stage') == 'search_selection'
    return (0 if selector else 1, -run['config'].get('estimated_seconds', 0), key)


def admission_sets(policy, snapshot):
    if policy is None:
        return None
    if snapshot is None or snapshot.get('schema') != SCHEMA:
        raise ValueError('Registry-audited HPO frontier is required before admission')
    return {worker: set(value) for worker, value in snapshot['open_cohorts_by_worker'].items()}


def may_admit(key, worker, policy, opened):
    if policy is None or key not in policy['fit_memberships']:
        return True
    memberships = set(policy['fit_memberships'][key])
    if any(policy['cohorts'][cohort]['worker'] != worker for cohort in memberships):
        raise ValueError('Cohort admission worker changed')
    return len(opened.get(worker, set()) | memberships) <= policy['maximum_open_cohorts_per_GPU']


def reserve_admission(key, worker, policy, opened):
    if policy is not None and key in policy['fit_memberships']:
        opened.setdefault(worker, set()).update(policy['fit_memberships'][key])


def attention(snapshot, states, runs, placement, host):
    if snapshot is None:
        return None
    pending_search = [key for key in runs if placement[key]['worker'].startswith(host + '/') and states[key] == 'registered' and runs[key]['run']['config'].get('stage') in {'search', 'search_selection'}]
    if pending_search and snapshot['held_cohort_reasons']:
        return {'pending_search_records': len(pending_search), 'held_cohort_reasons': snapshot['held_cohort_reasons'],
                'action': 'Inspect failed owners, selector evidence or exact unused-search release; no new cohort or fit retry is authorized'}
    return None
