"""Narrow launcher admission and compiler binding amendment on a fresh copy."""
import ast
import hashlib
import json
from pathlib import Path
import sys

runtime = Path(sys.argv[1]).resolve()
assert (runtime / 'hpo_frontier.py').is_file()
proof = {}


def edit(name, changes, expected_changed):
    path = runtime / name
    before = path.read_bytes()
    source = before.decode()
    for old, new in changes:
        assert source.count(old) == 1, (name, old, source.count(old))
        source = source.replace(old, new)
    after = source.encode()
    def functions(raw):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
    left, right = functions(before), functions(after)
    assert set(left) == set(right)
    changed = sorted(name for name in left if left[name] != right[name])
    assert changed == sorted(expected_changed), changed
    path.chmod(path.stat().st_mode | 0o200)
    path.write_bytes(after)
    proof[name] = {'before_sha256': hashlib.sha256(before).hexdigest(), 'after_sha256': hashlib.sha256(after).hexdigest(), 'only_changed_functions': changed}


edit('compile_master_plan.py', [
    ('STAGE_KIND = ', "CONTROLLER_FILES = CONTROLLER_FILES + ('hpo_frontier.py',)\nSTAGE_KIND = "),
    ("    plan['group_granularity'] = granularity", "    from hpo_frontier import declare as declare_hpo_frontier\n    plan['hpo_cohort_admission'] = declare_hpo_frontier(records, {key: {'worker': worker} for key, worker in assigned.items()}, {study['study_id']: study for study in studies})\n    plan['group_granularity'] = granularity"),
    ("    if 'phase_rendezvous_capacity' in plan:\n        summary['phase_rendezvous_capacity']", "    summary['hpo_cohort_admission'] = copy.deepcopy(plan['hpo_cohort_admission'])\n    if 'phase_rendezvous_capacity' in plan:\n        summary['phase_rendezvous_capacity']")
], ['assemble'])
edit('launcher.py', [
    ('    return studies, runs, placement, dependencies', '    from hpo_frontier import validate as validate_hpo_frontier\n    validate_hpo_frontier(plan, runs, placement, studies)\n    return studies, runs, placement, dependencies'),
    ('*, slots=2):', '*, slots=2, cohort_policy=None, cohort_snapshot=None):'),
    ('    keys = sorted(runs, key=lambda key: (-runs[key]["run"]["config"].get("estimated_seconds", 0), key))', '    from hpo_frontier import priority, admission_sets, may_admit, reserve_admission\n    opened = admission_sets(cohort_policy, cohort_snapshot)\n    keys = sorted(runs, key=lambda key: priority(key, runs))'),
    ('        reserve = runs[key]["run"]["config"].get("reservation_mib")', '        if not may_admit(key, worker, cohort_policy, opened):\n            continue\n        reserve = runs[key]["run"]["config"].get("reservation_mib")'),
    ('        result.append(key)\n        occupied[worker]', '        result.append(key)\n        reserve_admission(key, worker, cohort_policy, opened)\n        occupied[worker]'),
    ('    active, stop, closed_local_gates = {}, {"requested": False}, set()', '    active, stop, closed_local_gates = {}, {"requested": False}, set()\n    closed_hpo_cohorts, previous_frontier_digest = {}, None\n    from hpo_frontier import inspect as inspect_hpo_frontier, attention as hpo_attention'),
    ('            states = {key: all_states[key] for key in runs}', '            states = {key: all_states[key] for key in runs}\n            frontier = inspect_hpo_frontier(root, plan, runs, states, placement, studies, registry, host, list(active.values()), closed_hpo_cohorts)\n            if frontier is not None:\n                frontier_digest = fingerprint(frontier)\n                if frontier_digest != previous_frontier_digest:\n                    event("hpo_registry_frontier_audit", {"plan_hash": digest, "frontier": frontier, "frontier_digest": frontier_digest})\n                    previous_frontier_digest = frontier_digest'),
    ('                                       list(active.values()), capacities, host)', '                                       list(active.values()), capacities, host,\n                                       cohort_policy=plan.get("hpo_cohort_admission"), cohort_snapshot=frontier)'),
    ('                for key in ready:', '                if not active and not ready:\n                    held = hpo_attention(frontier, states, runs, placement, host)\n                    if held is not None:\n                        event("hpo_retention_attention_required", held)\n                        raise RuntimeError("HPO frontier cannot advance under its frozen retention cap; inspect exact closure evidence")\n                for key in ready:')
], ['validate_plan', 'schedule_ready', 'run_supervisor'])
print(json.dumps({'schema': 'bounded-HPO-frontier-narrow-controller-amendment-v1', 'files': proof,
    'source_learning_fit_budgets_selectors_jobs_assignments_phase_gates_unchanged': True,
    'prospective_signed_plan_policy_required': True, 'max_open_cohorts_per_GPU': 2,
    'models': 0, 'scientific_fits': 0, 'GO': False}, sort_keys=True))
