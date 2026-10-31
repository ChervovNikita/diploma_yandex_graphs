"""Standard-library source and synthetic exception checks; no remote execution."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'industrial_cpu_native_import_qualification_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def syntax_tree(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    compile(tree, str(path), 'exec')
    return tree


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def run_checks():
    python = {p.name: dict(sha256=sha(p), AST_and_compile='passed')
              for p in sorted(HERE.glob('*.py')) if syntax_tree(p)}
    assert (HERE/'proven_cpu_policy.py').read_bytes() == (V1/'proven_cpu_policy.py').read_bytes()
    old = syntax_tree(V1/'native_import_worker.py')
    new = syntax_tree(HERE/'native_import_worker.py')
    assert ast.dump(function(old,'restrict_native')) == ast.dump(function(new,'restrict_native'))
    # All capability/FD/Landlock/seccomp operation calls and arguments stay exact.
    def enforcement_calls(tree):
        calls=[]
        for node in ast.walk(function(tree,'main')):
            if isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Name) and node.func.id == 'restrict_native' or
                isinstance(node.func, ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id == 'policy'):
                calls.append(ast.dump(node))
        return calls
    assert enforcement_calls(old) == enforcement_calls(new)
    for name in ('root_outer.py','root_run_native_imports.py'):
        normalized = (HERE/name).read_text().replace('qualification-v2','qualification-v1').replace('launch-v2','launch-v1')
        assert normalized == (V1/name).read_text()
    original_contract = json.loads((V1/'IMPORT_CONTRACT.json').read_text())
    contract = json.loads((HERE/'IMPORT_CONTRACT.json').read_text())
    assert contract['run_container'] == 'industrial_cpu_native_import_qualification_v2/root_runs'
    normalized = dict(contract,run_container=original_contract['run_container'])
    assert normalized == original_contract
    snapshot = json.loads((HERE/'V1_PRESERVED_HASHES.json').read_text())
    for item in snapshot['files']:
        path=V1/item['path']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    # Execute ONLY the two pure stdlib reporting functions extracted from AST.
    helpers=ast.Module(body=[function(new,n) for n in ('diagnostic_text','exception_diagnostics')],type_ignores=[])
    namespace={'json':json}
    exec(compile(ast.fix_missing_locations(helpers),'<isolated-exception-reporting-fixture>','exec'),namespace)
    diagnostics=namespace['exception_diagnostics']
    try:
        try:
            raise OSError(13,'SYNTHETIC loader denial','/SYNTHETIC/no-file-opened.so')
        except OSError as cause:
            raise ImportError('SYNTHETIC outer NumPy wrapper') from cause
    except ImportError as outer:
        explicit=diagnostics(outer)
    assert len(explicit['nodes']) == 2
    assert explicit['nodes'][0]['cause_id'] == explicit['nodes'][0]['context_id'] == 1
    assert explicit['nodes'][1]['attributes']['errno'] == 13
    assert explicit['nodes'][1]['attributes']['filename'] == '/SYNTHETIC/no-file-opened.so'
    assert explicit['nodes'][0]['frames_outer_to_inner'] and explicit['nodes'][1]['frames_outer_to_inner']
    assert 'SYNTHETIC loader denial' in explicit['traceback_text']
    try:
        try:
            raise ModuleNotFoundError('SYNTHETIC hidden context')
        except ModuleNotFoundError:
            raise ImportError('SYNTHETIC suppressed outer') from None
    except ImportError as outer:
        suppressed=diagnostics(outer)
    assert suppressed['nodes'][0]['cause_id'] is None
    assert suppressed['nodes'][0]['context_id'] == 1 and suppressed['nodes'][0]['suppress_context']
    assert suppressed['nodes'][1]['type'] == 'ModuleNotFoundError'
    left,right=RuntimeError('SYNTHETIC left'),RuntimeError('SYNTHETIC right')
    left.__cause__=right;right.__context__=left
    cycle=diagnostics(left)
    assert len(cycle['nodes']) == 2 and cycle['nodes'][1]['context_id'] == 0
    long_error=RuntimeError('X'*20000)
    bounded=diagnostics(long_error)
    assert bounded['nodes'][0]['message_truncated'] and len(bounded['nodes'][0]['message']) == 16384
    chain=RuntimeError('SYNTHETIC 0');cursor=chain
    for i in range(40):
        cursor.__cause__=RuntimeError('SYNTHETIC '+str(i+1));cursor=cursor.__cause__
    limited=diagnostics(chain)
    assert len(limited['nodes']) == 32 and limited['chain_truncated']
    fixtures={'explicit_cause':explicit,'suppressed_context':suppressed,'cycle':cycle,
              'long_message':dict(message_length=len(bounded['nodes'][0]['message']),truncated=True),
              'chain_limit':dict(nodes=len(limited['nodes']),truncated=True)}
    (HERE/'SYNTHETIC_EXCEPTION_FIXTURES.json').write_text(json.dumps(fixtures,indent=2)+'\n')
    return dict(schema='CPU-native-import-v2-source-checks-v1',status='PASSED_SOURCE_ONLY',
        python=python,proven_policy_byte_identical=True,Landlock_rule_function_AST_identical=True,
        capability_FD_Landlock_seccomp_calls_AST_identical=True,
        contract_identical_except_fresh_run_container=True,
        outer_and_launcher_identical_except_receipt_schema_identity=True,
        preserved_v1_files=len(snapshot['files']),exception_synthetic_fixture_checks=5,
        scientific_imports=False,scientific_or_remote_execution=False,
        model_or_dataset_access=False,GPU_execution=False,
        restriction_policy_or_library_public_grants_widened=False)


if __name__ == '__main__':
    result=run_checks()
    (HERE/'SOURCE_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'policy_byte_identical':True,
                      'contract_grants_unchanged':True,'synthetic_fixture_checks':5}))
