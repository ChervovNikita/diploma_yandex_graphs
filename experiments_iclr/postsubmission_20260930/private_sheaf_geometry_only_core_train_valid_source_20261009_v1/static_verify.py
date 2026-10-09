"""AST and source hash verification only, never numeric/model/data execution."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
STDLIB_LOCAL = {'argparse', 'ast', 'bank', 'copy', 'gc', 'hashlib', 'importlib', 'independent', 'json', 'pathlib',
                'resource', 'shared_fit', 'support', 'sys', 'time'}


def read(path): return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def check(value, name):
    if not value: raise AssertionError(name)
    return name


def parsed(path): return ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))


def function(tree, name): return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def verify():
    checks, pins, protocol = [], read(HERE / 'SOURCE_BINDINGS.json'), read(HERE / 'PROTOCOL.json')
    for file in ('support.py', 'bank.py', 'shared_fit.py', 'independent.py', 'runner.py', 'static_verify.py'):
        tree = parsed(HERE / file)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
                check(all(name.split('.')[0] in STDLIB_LOCAL for name in names), 'Deferred numerical imports: ' + file)
        checks.append('AST parse and stdlib/local top-level imports: ' + file)
    bank = parsed(HERE / 'bank.py')
    calls = [node.func.attr for node in ast.walk(bank) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    checks.append(check({'_shared_copy_memo', '_module_at', '_replace', 'SharedBELinear'}.issubset(calls)
        and 'SharedNativeBank' not in calls, 'Reuse actual adapter helpers, not legacy all-Linear private bank'))
    constants = [node.value for node in ast.walk(bank) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    checks.append(check('sheaf_learners.' in constants and '.linear1' in constants, 'Replacement paths restricted to ordered incidence Linear'))
    fit = function(parsed(HERE / 'shared_fit.py'), 'fit')
    epoch_loop = next(node for node in ast.walk(fit) if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'epoch')
    member_loop = next(node for node in epoch_loop.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'member')
    backwards = [node for node in ast.walk(member_loop) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'backward']
    checks.append(check(len(backwards) == 1 and isinstance(backwards[0].func.value, ast.BinOp)
        and isinstance(backwards[0].func.value.op, ast.Div) and isinstance(backwards[0].func.value.right, ast.Constant)
        and backwards[0].func.value.right.value == 4, 'Streamed own NLL/4 backward'))
    check(not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'step' for node in ast.walk(member_loop)), 'No optimizer step between members')
    step_statement = next(index for index, node in enumerate(epoch_loop.body) if any(isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute) and value.func.attr == 'step' for value in ast.walk(node)))
    checks.append(check(step_statement > epoch_loop.body.index(member_loop), 'One Adam step only after every old-parameter member backward'))
    check(sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'step' for node in ast.walk(fit)) == 1, 'One shared Adam step expression')
    independent = parsed(HERE / 'independent.py')
    native_fits = [node for node in ast.walk(independent) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'fit_one']
    checks.append(check(len(native_fits) == 1, 'Direct original V2 complete independent fit_one'))
    protocol_arg = native_fits[0].args[4]
    checks.append(check(isinstance(protocol_arg, ast.Call) and {node.arg for node in protocol_arg.keywords} == {'optimizer', 'checkpoint_rule'},
        'Complete exact V2 fit_one protocol argument contract'))
    checks.append(check(protocol['members'] == 4 and protocol['seeds'] == [1103,2207,3301]
        and protocol['context_regularizer'] == 0 and protocol['configuration_search'] is False,
        'Four members, three paired seeds, regularizer off, no grid'))
    checks.append(check(protocol['architecture'] == 'root_frozen_native15_winner' and protocol['independent_seed_rule'] == 'base+1000003*member',
        'Unfrozen architecture and exact independent offset seed rule'))
    checks.append(check(read(HERE / 'RELEASE_TEMPLATE_DISABLED.json')['enabled'] is False
        and read(HERE / 'FROZEN_ARCHITECTURE_RECEIPT_TEMPLATE_DISABLED.json')['frozen'] is False, 'Inactive release and unfrozen root receipt'))
    original_support = parsed(PHASE / pins['BSNN_wrapper_directory'] / 'support.py')
    own_selector = function(original_support, 'selector').body[-1].value
    v2 = parsed(PHASE / pins['NSD_V2_directory'] / 'baseline_runner.py')
    checks.append(check(any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'key' for target in node.targets)
        and ast.dump(node.value, include_attributes=False) == ast.dump(own_selector, include_attributes=False) for node in ast.walk(v2)), 'Borrowed selector AST equals actual V2'))
    for row in pins['files']:
        path = (PHASE / row['path']).resolve(strict=True)
        check(path.is_relative_to(PHASE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Pinned source: ' + row['path'])
    checks.append('All existing NSD/native adapter/V2 role-placement-fit/scorer/Apache source bytes unchanged')
    preserved = []
    for row in pins['immutable_packets']:
        root = PHASE / row['directory']
        check(sha(root / 'MANIFEST.json') == row['manifest_sha256'], 'Immutable manifest: ' + row['directory'])
        payloads = read(root / 'MANIFEST.json')['files']
        for payload in payloads:
            path = (root / payload['path']).resolve(strict=True)
            check(path.is_relative_to(root) and path.stat().st_size == payload['bytes'] and sha(path) == payload['sha256'], 'Immutable payload: ' + row['directory'] + '/' + payload['path'])
        preserved.append(dict(directory=row['directory'], manifest_sha256=row['manifest_sha256'], verified_payloads=len(payloads)))
    checks.append('All preceding sealed packets preserved; running screen source untouched')
    if (HERE / 'MANIFEST.json').exists():
        for row in read(HERE / 'MANIFEST.json')['files']:
            path = (HERE / row['path']).resolve(strict=True)
            check(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Own payload: ' + row['path'])
        seal = read(HERE / 'SEAL.json')
        checks.append(check(seal['source_only'] is True and seal['execution_enabled'] is False
            and seal['manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Own inactive manifest/seal'))
    return dict(status='passed', verification='AST and source byte hashes only', checks=checks, preserved_packets=preserved,
        architecture_frozen=False, model_or_runner_or_numeric_provider_import=False, numerical_execution=False,
        data_array_outcome_or_checkpoint_access=False, server_or_installation_actions=False, execution_enabled=False)


if __name__ == '__main__': print(json.dumps(verify(), indent=2) + '\n')
