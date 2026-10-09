"""AST/hash verification only; never import wrapper, author, numeric providers or data."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
STDLIB_IMPORTS = {'argparse', 'ast', 'gc', 'hashlib', 'importlib', 'json', 'pathlib', 'resource', 'time', 'sys'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def check(value, label):
    if not value:
        raise AssertionError(label)
    return label


def tree(path):
    return ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))


def function(parsed, name):
    return next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == name)


def assigned(parsed, name):
    return [node.value for node in ast.walk(parsed) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]


def same(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def verify():
    checks = []
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    protocol = read(HERE / 'PROTOCOL.json')
    baseline = tree(PHASE / pins['nsd_runner_directory'] / 'baseline_runner.py')
    support = tree(HERE / 'support.py')
    runner = tree(HERE / 'runner.py')
    for filename in ('support.py', 'runner.py', 'static_verify.py'):
        parsed = tree(HERE / filename)
        for node in parsed.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
                check(all(name.split('.')[0] in STDLIB_IMPORTS or name == 'support' for name in names), 'Deferred numeric imports: ' + filename)
        checks.append('AST parse and stdlib/local top-level imports: ' + filename)
    own_selector = function(support, 'selector').body[-1].value
    checks.append(check(any(same(own_selector, value) for value in assigned(baseline, 'key')), 'Selector AST equals actual V2 inline selector'))
    reference_protocol = read(PHASE / pins['nsd_runner_directory'] / 'RUNNER_PROTOCOL.json')
    checks.append(check(protocol['seeds'] == reference_protocol['seeds'] == [1103, 2207, 3301], 'All three exact paired seeds'))
    proposal = read(PHASE / pins['feasibility_directory'] / 'PROPOSED_STARTING_CONFIG.json')
    checks.append(check(protocol['native_args'] == proposal['native_args'] and protocol['optimizer'] == proposal['optimizer']
        and protocol['max_epochs'] == proposal['epochs'] == 500 and protocol['patience'] == proposal['patience'] == 200,
        'Single exact starting config and optimizer proposal'))
    fit = function(runner, 'fit')
    beta = assigned(fit, 'beta')[0]
    author_driver = tree(PHASE / pins['author_directory'] / 'exp/run.py')
    author_beta = assigned(function(author_driver, 'train'), 'beta')[0]

    class OffsetEpoch(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'epoch':
                return ast.BinOp(left=ast.Name(id='epoch', ctx=ast.Load()), op=ast.Sub(), right=ast.Constant(value=1))
            return node

    checks.append(check(same(beta, OffsetEpoch().visit(author_beta)), 'Exact native annealing AST with one-based log offset'))
    evaluation = function(runner, 'evaluate')
    four_draw_loops = [node for node in ast.walk(evaluation) if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range'
        and len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.Constant) and node.iter.args[0].value == 4]
    checks.append(check(len(four_draw_loops) == 1, 'Exactly four full-model evaluation draws'))
    checks.append(check(same(assigned(evaluation, 'pooled')[0], ast.parse('torch.log(torch.mean(torch.stack(probabilities), 0))', mode='eval').body), 'Probability-mean evaluation pooling AST'))
    native_train_calls = [node for node in ast.walk(fit) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == 'model']
    checks.append(check(len(native_train_calls) == 1, 'One native full forward expression per training epoch'))
    release = read(HERE / 'RELEASE_TEMPLATE_DISABLED.json')
    checks.append(check(release['enabled'] is False and release['runtime_qualification_passed'] is False
        and release['native_bsnn_work_qualification_passed'] is False and release['baseline_configuration_reviewed'] is False,
        'Root release template remains disabled/unqualified/unreviewed'))
    for row in pins['files']:
        path = (PHASE / row['path']).resolve(strict=True)
        check(path.is_relative_to(PHASE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'External source binding: ' + row['path'])
    checks.append('Every pinned original-author/V2 helper/source/Apache-license byte hash matches')
    checks.append(check(sha(PHASE / pins['nsd_runner_directory'] / 'SOURCE_SEAL.json') == 'ac6a6bf2946dab206c0c5fd2384f9050fff4d7c0d8924ec1e7ed3121547d4f6e', 'Actual V2 source seal file hash'))
    preserved = []
    for row in pins['immutable_prior_packets']:
        root = PHASE / row['directory']
        check(sha(root / 'MANIFEST.json') == row['manifest_sha256'], 'Preserved prior manifest: ' + row['directory'])
        manifest = read(root / 'MANIFEST.json')
        for payload in manifest['files']:
            path = (root / payload['path']).resolve(strict=True)
            check(path.is_relative_to(root) and path.stat().st_size == payload['bytes'] and sha(path) == payload['sha256'], 'Preserved prior payload: ' + row['directory'] + '/' + payload['path'])
        preserved.append(dict(directory=row['directory'], manifest_sha256=row['manifest_sha256'], verified_payload_files=len(manifest['files'])))
    checks.append('All three immutable prior manifests and payloads unchanged')
    if (HERE / 'MANIFEST.json').exists():
        for row in read(HERE / 'MANIFEST.json')['files']:
            path = (HERE / row['path']).resolve(strict=True)
            check(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Own sealed payload: ' + row['path'])
        seal = read(HERE / 'SEAL.json')
        checks.append(check(seal['execution_enabled'] is False and seal['source_only'] is True
            and seal['manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Own source manifest and disabled seal'))
    return dict(status='passed', verification='AST and file byte hashes only', checks=checks,
                preserved_prior_packets=preserved, model_or_wrapper_import=False, numeric_execution=False,
                role_or_array_or_outcome_access=False, research_host_actions=False,
                runtime_qualification_passed=False, native_bsnn_work_qualification_passed=False,
                execution_enabled=False)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2) + '\n')
