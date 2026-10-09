"""Parse/hash only; no qualifier/wrapper/model/provider/data import or execution."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
STDLIB = {'argparse', 'ast', 'contextlib', 'gc', 'hashlib', 'importlib', 'json', 'pathlib', 'resource', 'sys', 'time'}


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


def same(left, right): return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def verify():
    checks, pins, protocol = [], read(HERE / 'SOURCE_BINDINGS.json'), read(HERE / 'PROTOCOL.json')
    for file in ('qualifier.py', 'static_verify.py'):
        tree = parsed(HERE / file)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [value.name for value in node.names] if isinstance(node, ast.Import) else [node.module]
                check(all(name.split('.')[0] in STDLIB for name in names), 'Top-level stdlib imports: ' + file)
        checks.append('AST parse and stdlib-only top-level imports: ' + file)
    tree = parsed(HERE / 'qualifier.py')
    run, draws = function(tree, 'run'), function(tree, 'four_draws')
    checks.append(check(not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in {'metrics', 'roc_auc_score', 'evaluate', 'fit', 'fit_one'} for node in ast.walk(tree)), 'No scorer, VALID evaluation, baseline fit or selector call'))
    checks.append(check(not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(run)), 'No training epoch/seed/config loop'))
    checks.append(check(sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'backward' for node in ast.walk(run)) == 1
        and sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'step' for node in ast.walk(run)) == 1, 'One backward and one Adam step expression'))
    full_train_calls = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'model']
    checks.append(check(len(full_train_calls) == 1, 'One full native TRAIN call expression'))
    four = [node for node in ast.walk(draws) if isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range' and len(node.iter.args) == 1
        and isinstance(node.iter.args[0], ast.Constant) and node.iter.args[0].value == 4]
    checks.append(check(len(four) == 1 and sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == 'four_draws' for node in ast.walk(run)) == 2, 'Four genuine eval calls and four reconstruction calls'))
    pooled = next(node.value for node in ast.walk(draws) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'pooled' for target in node.targets))
    checks.append(check(same(pooled, ast.parse('torch.log(torch.mean(torch.stack(probabilities), 0))', mode='eval').body), 'Exact probability-mean pooling expression'))
    beta = next(node.value for node in ast.walk(run) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'beta' for target in node.targets))
    author = parsed(PHASE / pins['author_directory'] / 'exp/run.py')
    original_beta = next(node.value for node in ast.walk(function(author, 'train')) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'beta' for target in node.targets))
    class SoleEpoch(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'epoch': return ast.BinOp(left=ast.Constant(value=1), op=ast.Sub(), right=ast.Constant(value=1))
            return node
    checks.append(check(same(beta, SoleEpoch().visit(original_beta)), 'Exact native epoch0 annealing expression'))
    checks.append(check(protocol['seed'] == 1103 and protocol['TRAIN_updates'] == 1 and protocol['evaluation_draws'] == 4
        and protocol['reconstruction_draws'] == 4 and protocol['gross_max_abs_logp'] == protocol['gross_max_abs_probability'] == 0.001,
        'Fixed one-update/eight-eval-call protocol and practical gross limits'))
    wrapper_protocol = read(PHASE / pins['wrapper_directory'] / 'PROTOCOL.json')
    checks.append(check(protocol['native_args'] == wrapper_protocol['native_args'] and protocol['optimizer'] == wrapper_protocol['optimizer'], 'Exact sealed original starting configuration and optimizer'))
    for row in pins['files']:
        path = (PHASE / row['path']).resolve(strict=True)
        check(path.is_relative_to(PHASE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'External pinned source: ' + row['path'])
    checks.append('All sealed wrapper/V2/original-author/source-license bytes match')
    preserved = []
    for row in pins['immutable_packets']:
        root = PHASE / row['directory']
        check(sha(root / 'MANIFEST.json') == row['manifest_sha256'], 'Immutable manifest: ' + row['directory'])
        payloads = read(root / 'MANIFEST.json')['files']
        for payload in payloads:
            path = (root / payload['path']).resolve(strict=True)
            check(path.is_relative_to(root) and path.stat().st_size == payload['bytes'] and sha(path) == payload['sha256'], 'Immutable payload: ' + row['directory'] + '/' + payload['path'])
        preserved.append(dict(directory=row['directory'], manifest_sha256=row['manifest_sha256'], verified_payloads=len(payloads)))
    checks.append('Authentic wrapper and all three preceding packets unchanged')
    template = read(HERE / 'RELEASE_TEMPLATE_DISABLED.json')
    checks.append(check(template['enabled'] is False, 'Root release remains disabled'))
    if (HERE / 'MANIFEST.json').exists():
        for row in read(HERE / 'MANIFEST.json')['files']:
            path = (HERE / row['path']).resolve(strict=True)
            check(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Own sealed payload: ' + row['path'])
        seal = read(HERE / 'SEAL.json')
        checks.append(check(seal['source_only'] is True and seal['execution_enabled'] is False
            and seal['manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Own disabled manifest/seal'))
    return dict(status='passed', verification='AST and source byte hashes only', checks=checks, preserved_packets=preserved,
        qualifier_wrapper_model_or_provider_import=False, numerical_execution=False, role_array_or_outcome_access=False,
        server_or_installation_actions=False, engineering_qualification_passed=False, execution_enabled=False)


if __name__ == '__main__': print(json.dumps(verify(), indent=2) + '\n')
