"""Fabricated clock test of the exact sealed final classification AST, no child launch."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'pubmed_heart_native_numerical_qualification_source_20261004_v2'
EXPECTED_MANIFEST = 'a45c8b4394618dba8a20b4e8116810ef2e4bfb5410b90d2a41c52f24b7c1ccd6'


def main():
    manifest_bytes = (SOURCE / 'MANIFEST.json').read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == EXPECTED_MANIFEST
    for row in json.loads(manifest_bytes)['files']:
        data = (SOURCE / row['path']).read_bytes()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    data = (SOURCE / 'supervise.py').read_bytes()
    tree = ast.parse(data)
    body = next(node.body for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'qualification' for t in node.targets))
    end = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'terminal' for t in node.targets))
    selected = body[start:end + 1]
    code = compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE / 'supervise.py'), 'exec')
    cases = [
        ('below_cap', .90, .01, None, 0, True, 'COMPLETE'),
        ('at_cap', 1.0, 0, None, 0, True, 'COMPLETE'),
        ('over_cap', 1.01, 0, None, 0, True, 'FAILED'),
        ('descriptor_crosses_cap', .98, .03, None, 0, True, 'FAILED'),
        ('existing_violation_retained', .90, .01, {'kind': 'host_RSS'}, 0, True, 'FAILED'),
        ('failed_child', .90, .01, None, 1, True, 'FAILED'),
        ('missing_custody', .90, .01, None, 0, False, 'FAILED'),
        ('failed_child_after_cap', .98, .03, None, 1, True, 'FAILED'),
    ]
    results = []
    for name, wait_elapsed, hash_work, prior_violation, exit_code, has_custody, expected in cases:
        clock = {'elapsed': wait_elapsed, 'monotonic_calls': 0, 'hash_calls': 0}
        def monotonic():
            clock['monotonic_calls'] += 1
            return clock['elapsed']
        def fake_sha(path):
            clock['hash_calls'] += 1
            clock['elapsed'] += hash_work
            return 'fabricated_digest'
        class FakePath:
            def __init__(self, exists): self.present = exists
            def exists(self): return self.present
            def stat(self): return SimpleNamespace(st_size=1)
        namespace = dict(result_path=FakePath(True), custody_path=FakePath(has_custody), sha=fake_sha,
            time=SimpleNamespace(monotonic=monotonic), started=0.0, cap_violation=prior_violation,
            caps={'wall_seconds': 1.0}, exit_code=exit_code, result={'status': 'PASS'},
            peak={'CUDA_observed': False}, args=SimpleNamespace(stage='cpu_bookkeeping', release_sha256='fabricated_release'),
            supervisor={}, identity={}, observed_peak=0, kernel_peak=0,
            usage=SimpleNamespace(ru_utime=0, ru_stime=0), release={'source_manifest_sha256': EXPECTED_MANIFEST})
        exec(code, namespace)
        terminal = namespace['terminal']
        assert terminal['status'] == expected, (name, terminal)
        assert clock['monotonic_calls'] == 1
        assert terminal['inclusive_supervisor_wall_seconds'] == clock['elapsed']
        if clock['elapsed'] > 1 and prior_violation is None:
            assert terminal['cap_violation']['kind'] == 'wall'
        if prior_violation is not None:
            assert terminal['cap_violation'] == prior_violation
        results.append(dict(case=name, wait_elapsed=wait_elapsed, final_elapsed=clock['elapsed'],
                            descriptor_hash_calls=clock['hash_calls'], expected=expected,
                            actual=terminal['status'], terminal=terminal))
    for filename in ('supervise.py', 'qualify.py', 'native_bodies.py'):
        compile((SOURCE / filename).read_bytes(), str(SOURCE / filename), 'exec')
    receipt = dict(status='PASS', source_manifest_sha256=EXPECTED_MANIFEST,
        exact_supervisor_sha256=hashlib.sha256(data).hexdigest(), cases=results,
        source_ast_lines=[selected[0].lineno, selected[-1].end_lineno],
        controlled_clock_not_physical_time=True, entire_supervisor_executed=False,
        source_environment_gate_verified=False, child_launches=0, scientific_imports=0,
        GPU_access=False, TEST_access=False, syntax_compiled_not_imported=['supervise.py', 'qualify.py', 'native_bodies.py'],
        limitation='Exact final classification AST only; complete physical execution and resources remain unverified.')
    destination = HERE / 'FINAL_BOUNDARY_QUALIFICATION.json'
    assert not destination.exists()
    destination.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'cases': len(results), 'entire_supervisor_executed': False}))


if __name__ == '__main__':
    main()
