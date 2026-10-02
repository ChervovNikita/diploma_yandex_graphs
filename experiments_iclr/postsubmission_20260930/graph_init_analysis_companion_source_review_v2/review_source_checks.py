"""Independent synthetic source checks; never open experiment outcomes."""
import ast
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'graph_init_analysis_companion_v2/analyze_complete_report.py'
SOURCE_SHA = '8e8816cc01e3ef595fb54785d81c2a72f473f9da2e8cd9ab4ae609138eeac770'
REGISTRY = PHASE / 'graph_init_precision_execution_root_v2/study_v2/GRAPH_INIT_ATTEMPT_REGISTRY.json'
REGISTRY_SHA = '715c361c82b543d4c436a565ccaa86470a9204433998d410954f2f268300307f'


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main():
    source = SOURCE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA
    namespace = {'__file__': str(SOURCE), '__name__': 'synthetic_source_review_only'}
    exec(compile(ast.parse(source), str(SOURCE), 'exec'), namespace)
    summary = namespace['summaries']
    # Independently invert the exact df=2 CDF using bisection, not the source formula.
    low, high = 0., 20.
    for _ in range(100):
        middle = (low + high) / 2.
        cdf = .5 + middle / (2. * math.sqrt(middle * middle + 2.))
        if cdf < .975:
            low = middle
        else:
            high = middle
    t_ref = (low + high) / 2.
    result = summary([-3., -2., -1.])
    assert abs(result['paired_t95_low'] - (-2. - t_ref / math.sqrt(3.))) < 1e-12
    assert abs(result['paired_t95_high'] - (-2. + t_ref / math.sqrt(3.))) < 1e-12
    sign_cases = []
    for values in itertools.product((-1., 0., 1.), repeat=3):
        record = summary(values)
        nonzero = [v for v in values if v]
        if not nonzero:
            p_ref = 1.
        else:
            observed = abs(sum(1 if v > 0 else -1 for v in nonzero))
            signs = list(itertools.product((-1, 1), repeat=len(nonzero)))
            p_ref = sum(abs(sum(s)) >= observed for s in signs) / len(signs)
        assert record['sign_reference_p'] == p_ref
        sign_cases.append(record['sign_reference_p'])
    # Extract and execute the actual Holm block on synthetic primary contrasts.
    tree = ast.parse(source)
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    holm_nodes = [n for n in main_node.body if 207 <= n.lineno <= 212]
    holm_tree = ast.fix_missing_locations(ast.Module(body=holm_nodes, type_ignores=[]))
    synthetic_primary = [{'metric': 'NLL', 'sign_reference_p': x}
                         for x in [.25, .25, .25, .5, .5, 1., 1., 1.]]
    holm_namespace = {'statistics_rows': synthetic_primary}
    exec(compile(holm_tree, 'synthetic_Holm_block', 'exec'), holm_namespace)
    assert all(row['sign_reference_holm_p'] == 1. and row['sign_reference_holm_reject_at_05'] is False
               for row in synthetic_primary)
    registry_bytes = REGISTRY.read_bytes()
    assert hashlib.sha256(registry_bytes).hexdigest() == REGISTRY_SHA
    registry = json.loads(registry_bytes)
    context_map = {object_hash(c): c for c in registry['contexts']}
    expected_pairs = {(g, s) for g in ('Squirrel', 'Photo') for s in (17, 29, 43)}
    assert {(c['graph'], c['seed']) for c in context_map.values()} == expected_pairs
    for a in registry['attempts']:
        assert a['context_sha256'] in context_map
        assert a['key'] == object_hash({k: a[k] for k in ('context_sha256', 'phase', 'arm')})
    assert len({a['key'] for a in registry['attempts']}) == 72
    for h in context_map:
        a = [a for a in registry['attempts'] if a['context_sha256'] == h]
        assert Counter(a['phase'] for a in a) == {'qualify': 1, 'warm': 1, 'initialize': 5, 'fit': 5}

    # A entirely synthetic, in-memory 72-phase cohort. No physical fixture exists.
    fixture = HERE / 'in_memory_fixture_not_written'
    assert not fixture.exists()
    run_path = fixture / 'coordinator_run'
    registry_path = fixture / 'study/GRAPH_INIT_ATTEMPT_REGISTRY.json'
    synthetic_registry = {'contexts': registry['contexts'], 'attempts': []}
    documents = {}
    remote_root = namespace['REMOTE']
    to_remote = lambda path: str(remote_root / path.relative_to(PHASE))
    put = lambda path, value: documents.__setitem__(path.resolve(), value)
    def descriptor(path):
        return {'path': to_remote(path), 'sha256': object_hash(documents[path.resolve()])}
    put(run_path / 'COMPLETED.json', {'schema': 'graph-init-finite-coordinator-terminal-v1',
        'completed': True, 'registered_successful_terminals': 72,
        'compare_report_or_final_labels_executed': False})
    shared_outer = run_path / 'supervision/ONE_REUSED_OUTER/TERMINAL.json'
    put(shared_outer, {'schema': 'gnnm-whole-process-bound-terminal-v1', 'complete': True,
        'within_whole_cap': True, 'root_request_unchanged': True, 'timed_out': False,
        'child_exit_code': 0, 'whole_supervised_seconds': 7., 'START_sha256': 'synthetic-unchecked-start'})
    receipt_paths = []
    for original in registry['attempts']:
        a = dict(original)
        output = fixture / 'outputs' / a['key']
        a['output'] = to_remote(output)
        synthetic_registry['attempts'].append(a)
        key = a['key']
        phase_claim = registry_path.parent / 'claims' / (key + '.json')
        put(phase_claim, {'synthetic': True, 'key': key})
        freeze = output / 'FREEZE.json'
        put(freeze, {'phase': a['phase'], 'synthetic': True})
        terminal = registry_path.parent / 'terminals' / (key + '.json')
        put(terminal, {'completed': True, 'final_labels_read': False,
            'claim': descriptor(phase_claim), 'freeze': descriptor(freeze)})
        put(run_path / (key + '_LAUNCH_CLAIM.json'), {'attempt': a})
        receipt = run_path / (key + '_COMPLETED.json')
        put(receipt, {'phase_terminal': descriptor(terminal), 'whole_terminal': descriptor(shared_outer),
            'whole_supervised_seconds': 7., 'whole_process_costs_charged': True})
        receipt_paths.append(receipt)
    put(registry_path, synthetic_registry)
    read_paths = []
    def synthetic_read(path):
        path = Path(path).resolve()
        read_paths.append(str(path))
        return documents[path]
    def synthetic_sha(path):
        return object_hash(documents[Path(path).resolve()])
    def synthetic_bound(record):
        path = namespace['mirror'](record['path'])
        assert synthetic_sha(path) == record['sha256']
        return path
    class SyntheticRun:
        def __fspath__(self):
            return str(run_path)
        def __truediv__(self, value):
            return run_path / value
        def glob(self, pattern):
            assert pattern == '*_COMPLETED.json'
            return receipt_paths
    namespace.update(REGISTRY=registry_path, REGISTRY_SHA=object_hash(synthetic_registry),
        COORDINATOR=run_path, read=synthetic_read, sha=synthetic_sha, bound=synthetic_bound)
    _, cost_rows, bindings = namespace['closed_cohort'](SyntheticRun())
    assert len(cost_rows) == 72
    assert len({b['whole_terminal']['path'] for b in bindings}) == 1
    assert sum(r['outer_supervised_seconds'] for r in cost_rows) == 504.
    assert not any('FINAL_REPORT' in path for path in read_paths)
    # Test main's sequencing with a simulated not-closed cohort; no report read occurs.
    report_opened = []
    def pending_closure(run):
        raise ValueError('synthetic pending closure')
    def forbidden_report(*args):
        report_opened.append(True)
        raise AssertionError('report opened before closure')
    namespace.update(closed_cohort=pending_closure, validated_report=forbidden_report)
    import sys
    old_argv = sys.argv
    try:
        sys.argv = [str(SOURCE), '--report', str(fixture / 'FINAL_REPORT.json'),
            '--coordinator-run', str(run_path), '--output', str(fixture / 'analysis')]
        try:
            namespace['main']()
        except ValueError as exc:
            assert str(exc) == 'synthetic pending closure'
        else:
            raise AssertionError('unclosed synthetic cohort accepted')
    finally:
        sys.argv = old_argv
    assert not report_opened and not fixture.exists()
    print(json.dumps({'source_sha256': SOURCE_SHA, 'registry_sha256': REGISTRY_SHA,
        'df2_t975_independent_CDF_reference': t_ref, 'df2_interval_correct': True,
        'sign_reference_cases_checked': len(sign_cases), 'sign_reference_correct': True,
        'eight_contrast_Holm_correct': True, 'native_registry_key_mapping_correct': True,
        'closure_precedes_report_access': True,
        'outer_identity_counterexample': {'all_72_completions_accepted': True,
            'unique_outer_terminals': 1, 'one_terminal_seconds': 7.,
            'incorrectly_accepted_cohort_seconds': 504.},
        'counterexample_scope': 'synthetic in-memory metadata only; no physical experiment files read',
        'real_report_labels_logits_checkpoints_or_terminals_opened': False,
        'scientific_imports': False, 'scientific_execution': False}, indent=2))


if __name__ == '__main__':
    main()
