"""Source text/JSON/AST custody checks; no target or numerical execution."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import copy
import hashlib
import json

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
CANDIDATE = PHASE / 'graph_count_conditioned_pattern_cpu_supervisor_preparation_20261004_v3'
PREVIOUS = PHASE / 'graph_count_conditioned_pattern_cpu_supervisor_preparation_20261004_v2'
CORE = PHASE / 'graph_count_conditioned_pattern_loss_prototype_preparation_20261004_v1'
EXPECTED = 'e71e869ac04bd898e7b02bb8c767cee5f61422be2759865574f8e4a06d1f982a'
CORE_EXPECTED = '9198a01dcc11c635bc47d57551b271ad0515009c41adc6b5fc7757e38aae657e'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(root, rows):
    result = []
    for row in rows:
        path = root / row['path']
        assert path.suffix not in ('.pt', '.npy', '.npz', '.pkl'), path
        assert path.is_file() and not path.is_symlink(), path
        actual = {'path': str(path.relative_to(PHASE)), 'sha256': sha(path), 'bytes': path.stat().st_size}
        assert actual['sha256'] == row['sha256'] and actual['bytes'] == row['bytes'], path
        result.append({**actual, 'match': True})
    return result


def main():
    assert sha(CANDIDATE / 'MANIFEST.json') == EXPECTED
    assert sha(CORE / 'MANIFEST.json') == CORE_EXPECTED
    manifest, binding = read(CANDIDATE / 'MANIFEST.json'), read(CANDIDATE / 'SOURCE_BINDING.json')
    candidate_rows = verify(CANDIDATE, manifest['files'])
    external_rows = verify(PHASE, binding['external_input_pins'])
    core_rows = verify(CORE, read(CORE / 'MANIFEST.json')['files'])
    core_external_rows = verify(PHASE, read(CORE / 'SOURCE_BINDING.json')['external_source_pins'])
    predecessor_rows = verify(PREVIOUS, read(PREVIOUS / 'MANIFEST.json')['files'])
    previous_review = PHASE / 'graph_count_conditioned_pattern_cpu_supervisor_fresh_review_20261004_v2'
    previous_review_rows = verify(previous_review, read(previous_review / 'MANIFEST.json')['files'])
    core_review = PHASE / 'graph_count_conditioned_pattern_loss_independent_source_review_20261004_v1'
    core_review_rows = verify(core_review, read(core_review / 'MANIFEST.json')['files'])
    current_receipt, current_checks = read(CANDIDATE / 'INPUT_HASH_RECEIPT.json'), read(CANDIDATE / 'STATIC_CHECKS.json')
    assert current_receipt['inputs'] == binding['external_input_pins']
    assert current_receipt['verified_pins'] == current_checks['external_pins_checked'] == len(external_rows) == 40
    assert current_receipt['scope'] == 'CURRENT_V3_SOURCE_BINDING_EXTERNAL_INPUT_PINS_ONLY'
    old = ast.parse((PREVIOUS / 'supervise_cpu.py').read_text())
    new = ast.parse((CANDIDATE / 'supervise_cpu.py').read_text())
    trimmed = copy.deepcopy(new)
    held = next(node for node in trimmed.body if isinstance(node, ast.FunctionDef) and node.name == 'held_child_signal')
    guard = ast.unparse(held.body.pop(0))
    assert guard == "require(type(child_pid) is int and child_pid > 0, 'Only a positive held direct-child PID may be signaled')"
    main_function = next(node for node in trimmed.body if isinstance(node, ast.FunctionDef) and node.name == 'supervise')
    handler = next(node for node in main_function.body if isinstance(node, ast.FunctionDef) and node.name == 'interrupted')
    child_safe = ast.unparse(handler.body.pop(0))
    assert child_safe == "if os.getpid() != terminal['supervisor_pid']:\n    os._exit(128 + signum)"
    assert ast.dump(old, include_attributes=False) == ast.dump(trimmed, include_attributes=False)
    changed = [row['path'] for row in read(PREVIOUS / 'MANIFEST.json')['files']
               if (CANDIDATE / row['path']).read_bytes() != (PREVIOUS / row['path']).read_bytes()]
    assert set(changed) == {'INPUT_HASH_RECEIPT.json', 'README.md', 'SOURCE_BINDING.json', 'STATIC_CHECKS.json', 'supervise_cpu.py'}
    unchanged = {name: (CANDIDATE / name).read_bytes() == (PREVIOUS / name).read_bytes()
                 for name in ('PLAN.json', 'LIMITS.json', 'ROOT_RELEASE_TEMPLATE.json', 'PREPARATION_NOTE.json', 'V1_TO_V2.diff', 'V1_TO_V2_PROVENANCE.json')}
    assert all(unchanged.values())
    calls = [node for node in ast.walk(new) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    fork_sites = [node.lineno for node in calls if isinstance(node.func.value, ast.Name) and node.func.value.id == 'os' and node.func.attr == 'fork']
    assert len(fork_sites) == 1
    for name in ('conditional_loss.py', 'conditional_single.py', 'oracles.py', 'qualify_cpu.py'):
        ast.parse((CORE / name).read_text())
    original = read(core_review / 'REVIEW.json')
    assert [row['id'] for row in original['blocking_findings']] == ['F01']
    f02 = next(row for row in original['nonblocking_findings'] if row['id'] == 'F02')
    assert f02['blocks_declared_valid_fixture_CPU_oracles'] is False
    receipt = {'schema': 'graph-count-conditioned-pattern-supervisor-v3-review-static-checks-v1',
               'UTC': datetime.now(timezone.utc).isoformat(), 'candidate_manifest_sha256': EXPECTED,
               'core_manifest_sha256': CORE_EXPECTED, 'candidate_payloads': candidate_rows,
               'external_input_pins': external_rows, 'core_payloads': core_rows,
               'core_external_source_pins': core_external_rows, 'v2_predecessor_payloads': predecessor_rows,
               'v2_review_payloads': previous_review_rows, 'original_core_review_payloads': core_review_rows,
               'current_40_row_receipt_and_STATIC_CHECKS_exact': True, 'changed_v2_payloads': changed,
               'positive_PID_guard': guard, 'inherited_child_safe_handler': child_safe,
               'whole_AST_except_two_insertions_exact_v2': True, 'unchanged_payloads': unchanged,
               'numerical_child_fork_sites': fork_sites, 'core_and_old_findings_unchanged': True,
               'target_import_compile_execution': False, 'numerical_imports_or_oracles': False,
               'data_state_score_array_reads': False, 'runtime_binary_or_package_state_read': False,
               'actual_process_signal_RSS_or_supervisor_probe': False, 'remote_access_or_dispatch': False}
    (HERE / 'STATIC_CHECKS.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'wrapper_payloads': len(candidate_rows), 'wrapper_bytes': sum(row['bytes'] for row in candidate_rows),
                      'external_pins': len(external_rows), 'core_payloads': len(core_rows),
                      'core_bytes': sum(row['bytes'] for row in core_rows), 'core_external_pins': len(core_external_rows),
                      'status': 'SOURCE_ONLY_DELTA_CHECKS_PASS'}))


if __name__ == '__main__':
    main()
