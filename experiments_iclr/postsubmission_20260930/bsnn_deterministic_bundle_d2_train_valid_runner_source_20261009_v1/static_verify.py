"""AST/hash only; never import runner, original models/providers or role data."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
STDLIB = {'argparse', 'ast', 'hashlib', 'importlib', 'json', 'pathlib', 'resource', 'sys', 'time'}


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


def verify():
    checks, pins, protocol = [], read(HERE / 'SOURCE_BINDINGS.json'), read(HERE / 'PROTOCOL.json')
    for file in ('runner.py', 'static_verify.py'):
        tree = parsed(HERE / file)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
                check(all(name.split('.')[0] in STDLIB for name in names), 'Stdlib top-level imports: ' + file)
        checks.append('AST parse and deferred numerical imports: ' + file)
    bsnn = read(PHASE / pins['BSNN_wrapper_directory'] / 'PROTOCOL.json')
    v2 = read(PHASE / pins['actual_NSD_V2_directory'] / 'RUNNER_PROTOCOL.json')
    checks.append(check(len(protocol['configs']) == 1 and protocol['configs'][0]['native_args'] == bsnn['native_args'], 'One exact declared BSNN starting configuration'))
    checks.append(check(protocol['seeds'] == bsnn['seeds'] == v2['seeds'] == [1103, 2207, 3301], 'Exact three paired seeds'))
    checks.append(check(all(protocol['optimizer'][key] == value for key, value in bsnn['optimizer'].items())
        and protocol['optimizer']['max_epochs'] == bsnn['max_epochs'] == 500
        and protocol['optimizer']['patience'] == bsnn['patience'] == 200, 'Same starting Adam/grouping/bounds'))
    tree = parsed(HERE / 'runner.py')
    fit_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'fit_one']
    checks.append(check(len(fit_calls) == 1 and isinstance(fit_calls[0].func.value, ast.Name)
        and fit_calls[0].func.value.id == 'helpers', 'Direct original V2 fit_one reuse'))
    checks.append(check(not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in
        {'forward', 'evaluate', 'optimizer', 'selector', 'metrics', 'fit', 'fit_one'} for node in ast.walk(tree)), 'No copied/adapted model/fit/scorer/selector implementation'))
    native = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'native_class')
    checks.append(check(any(isinstance(node, ast.Constant) and node.value == 'models.disc_models' for node in ast.walk(native))
        and any(isinstance(node, ast.Attribute) and node.attr == 'DiscreteBundleSheafDiffusion' for node in ast.walk(native)), 'Authentic deterministic bundle class import'))
    adapter = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'adapter')
    checks.append(check(any(isinstance(node, ast.Attribute) and node.attr == 'weight_learners' for node in ast.walk(adapter))
        and any(isinstance(node, ast.Attribute) and node.attr == 'full_left_right_idx' for node in ast.walk(adapter)), 'Explicit original per-layer edge-index placement extension'))
    baseline = parsed(PHASE / pins['actual_NSD_V2_directory'] / 'baseline_runner.py')
    selector = ast.parse('(scores["valid"]["auroc"], -scores["valid"]["nll"], -epoch)', mode='eval').body
    checks.append(check(any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'key' for target in node.targets)
        and ast.dump(node.value, include_attributes=False) == ast.dump(selector, include_attributes=False) for node in ast.walk(baseline)), 'Actual V2 unchanged selector AST'))
    for row in pins['files']:
        path = (PHASE / row['path']).resolve(strict=True)
        check(path.is_relative_to(PHASE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'External source: ' + row['path'])
    checks.append('All original deterministic author/Apache/V2/BSNN source bytes pinned and unchanged')
    preserved = []
    for row in pins['immutable_packets']:
        root = PHASE / row['directory']
        check(sha(root / 'MANIFEST.json') == row['manifest_sha256'], 'Immutable manifest: ' + row['directory'])
        payloads = read(root / 'MANIFEST.json')['files']
        for payload in payloads:
            path = (root / payload['path']).resolve(strict=True)
            check(path.is_relative_to(root) and path.stat().st_size == payload['bytes'] and sha(path) == payload['sha256'], 'Immutable payload: ' + row['directory'] + '/' + payload['path'])
        preserved.append(dict(directory=row['directory'], manifest_sha256=row['manifest_sha256'], verified_payloads=len(payloads)))
    checks.append('All five preceding sealed source packets preserved')
    checks.append(check(read(HERE / 'RELEASE_TEMPLATE_DISABLED.json')['enabled'] is False, 'Disabled release template'))
    checks.append(check(protocol['configuration_search'] is False and protocol['pure_sampling_or_KL_ablation'] is False
        and protocol['evaluation_full_calls'] == protocol['serving_full_calls'] == 1, 'Required fixed fairness control, disclosed native work and family differences'))
    if (HERE / 'MANIFEST.json').exists():
        for row in read(HERE / 'MANIFEST.json')['files']:
            path = (HERE / row['path']).resolve(strict=True)
            check(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Own payload: ' + row['path'])
        seal = read(HERE / 'SEAL.json')
        checks.append(check(seal['source_only'] is True and seal['execution_enabled'] is False
            and seal['manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Own manifest and inactive seal'))
    return dict(status='passed', verification='AST and source byte hashes only', checks=checks, preserved_packets=preserved,
        model_runner_or_numeric_provider_import=False, numeric_execution=False, data_array_or_outcome_access=False,
        server_or_installation_actions=False, general_map_screen_modified=False, execution_enabled=False)


if __name__ == '__main__': print(json.dumps(verify(), indent=2) + '\n')
