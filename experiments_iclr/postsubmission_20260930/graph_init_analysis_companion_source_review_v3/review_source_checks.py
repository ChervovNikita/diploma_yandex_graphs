"""Independent v3 recheck using source, registry and synthetic in-memory metadata."""
import ast
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'graph_init_analysis_companion_v3/analyze_complete_report.py'
SOURCE_SHA = '3b5119830c3640bb9cb777cb1c813e973b770e972ffc2fe87c38f1fe29b9b8b8'
REGISTRY = PHASE / 'graph_init_precision_execution_root_v2/study_v2/GRAPH_INIT_ATTEMPT_REGISTRY.json'
REGISTRY_SHA = '715c361c82b543d4c436a565ccaa86470a9204433998d410954f2f268300307f'


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main():
    source = SOURCE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA
    namespace = {'__file__': str(SOURCE), '__name__': 'synthetic_recheck_only'}
    exec(compile(ast.parse(source), str(SOURCE), 'exec'), namespace)
    summary = namespace['summaries']
    lo, hi = 0., 20.
    for _ in range(100):
        mid = (lo + hi) / 2.
        if .5 + mid / (2. * math.sqrt(mid * mid + 2.)) < .975:
            lo = mid
        else:
            hi = mid
    t_ref = (lo + hi) / 2.
    sample = summary([-3., -2., -1.])
    assert abs(sample['paired_t95_low'] - (-2. - t_ref / math.sqrt(3.))) < 1e-12
    assert abs(sample['paired_t95_high'] - (-2. + t_ref / math.sqrt(3.))) < 1e-12
    for values in itertools.product((-1., 0., 1.), repeat=3):
        nonzero = [v for v in values if v]
        if nonzero:
            observed = abs(sum(1 if v > 0 else -1 for v in nonzero))
            signs = list(itertools.product((-1, 1), repeat=len(nonzero)))
            p_ref = sum(abs(sum(s)) >= observed for s in signs) / len(signs)
        else:
            p_ref = 1.
        assert summary(values)['sign_reference_p'] == p_ref
    main_node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    start = next(i for i, n in enumerate(main_node.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'primary' for t in n.targets))
    holm = ast.fix_missing_locations(ast.Module(body=main_node.body[start:start + 3], type_ignores=[]))
    tests = [{'metric': 'NLL', 'sign_reference_p': p} for p in [.25, .25, .25, .5, .5, 1., 1., 1.]]
    exec(compile(holm, 'synthetic_Holm', 'exec'), {'statistics_rows': tests})
    assert all(t['sign_reference_holm_p'] == 1. and t['sign_reference_holm_reject_at_05'] is False for t in tests)
    registry_bytes = REGISTRY.read_bytes()
    assert hashlib.sha256(registry_bytes).hexdigest() == REGISTRY_SHA
    registry = json.loads(registry_bytes)
    contexts = {object_hash(c): c for c in registry['contexts']}
    assert len(contexts) == 6
    for a in registry['attempts']:
        assert a['key'] == object_hash({k: a[k] for k in ('context_sha256', 'phase', 'arm')})
        assert a['context_sha256'] in contexts
    assert len({a['key'] for a in registry['attempts']}) == 72
    for h in contexts:
        assert Counter(a['phase'] for a in registry['attempts'] if a['context_sha256'] == h) == {
            'qualify': 1, 'warm': 1, 'initialize': 5, 'fit': 5}

    fixture = HERE / 'in_memory_fixture_not_written'
    assert not fixture.exists()
    run_path = fixture / 'coordinator_run'
    registry_path = fixture / 'study/GRAPH_INIT_ATTEMPT_REGISTRY.json'
    remote_root = namespace['REMOTE']
    documents = {}
    def put(path, value):
        documents[path.resolve()] = value
    def remote(path):
        return str(remote_root / path.relative_to(PHASE))
    def descriptor(path):
        return {'path': remote(path), 'sha256': object_hash(documents[path.resolve()])}
    put(run_path / 'COMPLETED.json', {'schema': 'graph-init-finite-coordinator-terminal-v1',
        'completed': True, 'registered_successful_terminals': 72,
        'compare_report_or_final_labels_executed': False})
    synthetic_registry = {'contexts': registry['contexts'], 'attempts': []}
    receipts, whole_paths = [], {}
    for original in registry['attempts']:
        a = dict(original)
        key = a['key']
        output = fixture / 'outputs' / key
        a['output'] = remote(output)
        synthetic_registry['attempts'].append(a)
        cap = contexts[a['context_sha256']]['resource_forecast']['forecast']['wall_seconds_by_phase'][a['phase']]
        launch_attempt = dict(a, whole_cap_seconds=cap)
        phase_claim = registry_path.parent / 'claims' / (key + '.json')
        put(phase_claim, {'synthetic': True, 'key': key})
        freeze = output / 'FREEZE.json'
        put(freeze, {'phase': a['phase'], 'synthetic': True})
        terminal = registry_path.parent / 'terminals' / (key + '.json')
        put(terminal, {'completed': True, 'final_labels_read': False,
            'claim': descriptor(phase_claim), 'freeze': descriptor(freeze)})
        outer = run_path / 'supervision' / (key + '_outer')
        inner = run_path / 'supervision' / (key + '_inner')
        request = run_path / 'requests' / key / 'REQUEST.json'
        put(request, {'schema': 'graph-init-continuation-launch-request-v1', 'root_admitted': True,
            'attempt': launch_attempt, 'action': a['phase'], 'output': a['output'],
            'outer_supervisor_directory': remote(outer), 'supervisor_directory': remote(inner),
            'whole_cap_seconds': cap})
        put(run_path / (key + '_LAUNCH_CLAIM.json'), {'attempt': launch_attempt, 'request': descriptor(request)})
        start_path = outer / 'START.json'
        put(start_path, {'schema': 'gnnm-whole-process-bound-start-v1',
            'ssh_destination': 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
            'root_request': {'path': str(request.relative_to(PHASE)), 'sha256': object_hash(documents[request.resolve()])},
            'outer_output': remote(outer), 'inner_supervisor_directory': remote(inner), 'whole_cap_seconds': cap})
        whole = outer / 'TERMINAL.json'
        put(whole, {'schema': 'gnnm-whole-process-bound-terminal-v1', 'complete': True,
            'within_whole_cap': True, 'root_request_unchanged': True, 'timed_out': False,
            'child_exit_code': 0, 'whole_supervised_seconds': 7., 'START_sha256': object_hash(documents[start_path.resolve()])})
        whole_paths[key] = whole
        receipt = run_path / (key + '_COMPLETED.json')
        put(receipt, {'phase_terminal': descriptor(terminal), 'whole_terminal': descriptor(whole),
            'whole_supervised_seconds': 7., 'whole_process_costs_charged': True, 'whole_cap_seconds': cap})
        receipts.append(receipt)
    put(registry_path, synthetic_registry)
    reads = []
    def fake_read(path):
        path = Path(path).resolve()
        reads.append(str(path))
        return documents[path]
    def fake_sha(path):
        return object_hash(documents[Path(path).resolve()])
    def fake_bound(record):
        path = namespace['mirror'](record['path'])
        assert fake_sha(path) == record['sha256']
        return path
    class SyntheticRun:
        def __fspath__(self):
            return str(run_path)
        def __truediv__(self, value):
            return run_path / value
        def glob(self, pattern):
            assert pattern == '*_COMPLETED.json'
            return receipts
    namespace.update(REGISTRY=registry_path, REGISTRY_SHA=object_hash(synthetic_registry),
        COORDINATOR=run_path, read=fake_read, sha=fake_sha, bound=fake_bound)
    _, rows, bindings = namespace['closed_cohort'](SyntheticRun())
    assert len(rows) == len(bindings) == 72 and len({b['whole_terminal']['path'] for b in bindings}) == 72
    assert sum(r['outer_supervised_seconds'] for r in rows) == 504.
    ordered = sorted(receipts)
    first_key = ordered[0].name.removesuffix('_COMPLETED.json')
    other_key = ordered[1].name.removesuffix('_COMPLETED.json')
    record = documents[ordered[0].resolve()]
    old_outer = record['whole_terminal']
    record['whole_terminal'] = descriptor(whole_paths[other_key])
    try:
        namespace['closed_cohort'](SyntheticRun())
    except ValueError as error:
        reuse_message = str(error)
        assert reuse_message == 'Cost terminal is not the canonical per-attempt outer terminal'
    else:
        raise AssertionError('v2 counterexample remains accepted')
    record['whole_terminal'] = old_outer
    launch = documents[(run_path / (first_key + '_LAUNCH_CLAIM.json')).resolve()]
    whole_path = whole_paths[first_key]
    whole = documents[whole_path.resolve()]
    correct_start_hash = whole['START_sha256']
    whole['START_sha256'] = 'synthetic-wrong-start'
    record['whole_terminal'] = descriptor(whole_path)
    try:
        namespace['verified_whole_cost'](run_path, first_key, launch, record)
    except ValueError as error:
        assert str(error) == 'Whole terminal changed START identity'
    else:
        raise AssertionError('wrong START hash accepted')
    whole['START_sha256'] = correct_start_hash
    record['whole_terminal'] = descriptor(whole_path)
    correct_cap = record['whole_cap_seconds']
    record['whole_cap_seconds'] = correct_cap + 1
    try:
        namespace['verified_whole_cost'](run_path, first_key, launch, record)
    except ValueError as error:
        assert str(error) == 'Whole-process cap binding differs'
    else:
        raise AssertionError('inconsistent cap accepted')
    record['whole_cap_seconds'] = correct_cap

    # A numeric consistency gap remains: equality/cap flag is trusted without seconds <= cap.
    whole['whole_supervised_seconds'] = correct_cap + 1
    record['whole_supervised_seconds'] = correct_cap + 1
    record['whole_terminal'] = descriptor(whole_path)
    seconds, _, _ = namespace['verified_whole_cost'](run_path, first_key, launch, record)
    assert seconds == correct_cap + 1
    excess_accepted = {'cap': correct_cap, 'seconds': seconds, 'within_whole_cap_flag': True}
    whole['whole_supervised_seconds'] = record['whole_supervised_seconds'] = 7.
    record['whole_terminal'] = descriptor(whole_path)

    report_reads = []
    def pending(run):
        raise ValueError('synthetic pending closure')
    def forbidden_report(*args):
        report_reads.append(True)
        raise AssertionError('report read before closure')
    namespace.update(closed_cohort=pending, validated_report=forbidden_report)
    saved_argv = sys.argv
    try:
        sys.argv = [str(SOURCE), '--report', str(fixture / 'FINAL_REPORT.json'),
            '--coordinator-run', str(run_path), '--output', str(fixture / 'analysis')]
        try:
            namespace['main']()
        except ValueError as error:
            assert str(error) == 'synthetic pending closure'
        else:
            raise AssertionError('pending cohort accepted')
    finally:
        sys.argv = saved_argv
    assert not report_reads and not fixture.exists()
    assert not any('FINAL_REPORT' in path for path in reads)
    print(json.dumps({'source_sha256': SOURCE_SHA, 'registry_sha256': REGISTRY_SHA,
        'df2_interval_correct': True, 'sign_reference_cases_correct': 27,
        'Holm_formula_correct': True, 'actual_registry_mapping_correct': True,
        'valid_synthetic_72_phase_cost_fixture_passed': True,
        'original_v2_outer_reuse_counterexample_rejected': True, 'reuse_rejection': reuse_message,
        'changed_START_hash_rejected': True, 'inconsistent_cap_rejected': True,
        'over_cap_seconds_with_true_flag_still_accepted': excess_accepted,
        'closure_precedes_report_reader': True,
        'scientific_imports': False, 'scientific_execution': False,
        'real_reports_labels_logits_checkpoints_terminals_or_freezes_opened': False,
        'synthetic_fixture_files_written': False}, indent=2))


if __name__ == '__main__':
    main()
