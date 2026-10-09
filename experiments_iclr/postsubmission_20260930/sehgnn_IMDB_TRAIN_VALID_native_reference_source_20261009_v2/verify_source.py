"""Stdlib-only AST, hash, recipe and inactive-template checks; no data/model execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


def same_function(name, local, origin):
    require(ast.dump(functions(local)[name], include_attributes=False)
            == ast.dump(functions(origin)[name], include_attributes=False), 'Exact author/helper function AST: ' + name)


def check():
    sources = read(HERE / 'SOURCE_BINDINGS.json')
    for row in sources['files']:
        path = HERE.parent / row['path']
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Pinned origin ' + row['path'])
    seam = HERE.parent / sources['seam_directory']
    require((HERE / 'model.py').read_bytes() == (seam / 'public_sources/sehgnn_hgb_model.txt').read_bytes(), 'Byte-exact pinned native model')
    for name in sources['native_helper_functions']:
        same_function(name, HERE / 'native_helpers.py', seam / 'public_sources/sehgnn_hgb_utils.txt')
    require(set(functions(HERE / 'native_helpers.py')) == set(sources['native_helper_functions']), 'Only allowed author helper functions')
    for name in ('capture_rng', 'restore_rng', 'synchronize', 'parameter_counts'):
        same_function(name, HERE / 'state_helpers.py', HERE.parent / 'private_sheaf_train_valid_runner_20261009_v2/baseline_runner.py')
    for name in ('cpu_tree', 'exact'):
        same_function(name, HERE / 'state_helpers.py', HERE.parent / 'bsnn_cayley_d2_train_valid_runner_source_20261009_v1/support.py')
    trees = {}
    for path in sorted(HERE.glob('*.py')):
        trees[path.name] = ast.parse(path.read_text())
    for path in sorted(HERE.glob('*.json')):
        read(path)
    numerical = {'torch', 'numpy', 'dgl', 'torch_sparse', 'sklearn'}
    for filename in ('runner.py', 'engine.py', 'state_helpers.py', 'verify_source.py'):
        for node in trees[filename].body:
            if isinstance(node, ast.Import):
                require(not any(alias.name.split('.')[0] in numerical for alias in node.names), 'No top-level numerical provider import in ' + filename)
            elif isinstance(node, ast.ImportFrom):
                require((node.module or '').split('.')[0] not in numerical, 'No top-level numerical provider import in ' + filename)
    forbidden = {'load_dataset', 'data_loader', 'check_acc', 'evaluate_test', 'gen_file_for_evaluate'}
    for filename in ('runner.py', 'engine.py', 'native_helpers.py'):
        for node in ast.walk(trees[filename]):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
                require(name not in forbidden, 'No stock input or TEST scorer call ' + name)
    engine = trees['engine.py']
    loaders = [node for node in ast.walk(engine) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == 'DataLoader']
    require(len(loaders) == 1, 'Native TRAIN DataLoader only; serving is a prebuilt list')
    kwargs = {item.arg: ast.literal_eval(item.value) for item in loaders[0].keywords}
    require(kwargs == {'batch_size': 10000, 'shuffle': True, 'drop_last': False}, 'Native loader defaults including generator=None')
    model_calls = [node for node in ast.walk(engine) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Subscript) and isinstance(node.func.slice, ast.Constant)
                   and node.func.slice.value == 'model_class']
    require(len(model_calls) == 1 and [ast.literal_eval(model_calls[0].args[i]) for i in (0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13)]
            == ['IMDB', 512, 512, 5, 'M', .5, 0., 0., 2, 4, 'none', False], 'Exact native model constructor recipe')
    text = (HERE / 'engine.py').read_text()
    require('range(1 if qualify else 200)' in text and "scores['VALID']['BCE'] < best_loss" in text
            and 'epoch - best_epoch > 50' in text and 'weights_only=True' in text
            and "torch.eye(count)" in text and "rt['remove_diag'](value) @ label_source" in text,
            'Literal epoch, selector, patience, custody, identity and label-operator seams')
    protocol = read(HERE / 'PROTOCOL.json')
    budget = read(seam / 'SYMBOLIC_BACKBONE_BUDGET.json')
    require(protocol['feature_paths'] == budget['feature_paths'] and protocol['label_paths'] == budget['label_paths']
            and protocol['predicted_native_model_parameters'] == 83659532
            and protocol['reference_seeds'] == [1, 2, 3, 4, 5] and protocol['qualification_seeds'] == [1], 'Full native channels and literal author seed recipe')
    for filename, action, seeds in (('QUALIFICATION_RELEASE.disabled.json', 'qualify_one_full_native_TRAIN_update', [1]),
                                     ('REFERENCE_RELEASE.disabled.json', 'fit_literal_five_seed_TRAIN_VALID_reference', [1, 2, 3, 4, 5])):
        release = read(HERE / filename)
        require(release['enabled'] is False and release['root_source_review_approved'] is False
                and release['action'] == action and release['seeds'] == seeds
                and release['source_seal_sha256'] is None and release['expected_runtime_versions']['dgl'] is None,
                'Inactive release with unverified fields explicit ' + filename)
    return {'status': 'passed', 'source_only': True, 'model_bytes_exact': True, 'author_helper_ASTs_exact': True,
            'existing_state_helper_ASTs_exact': True, 'all_Python_ASTs_valid': True,
            'all_JSON_valid': True, 'inactive_release_templates': True, 'recipe_and_no_TEST_call_checks': True,
            'real_data_provider_model_or_numerical_execution': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check()
    if args.output:
        require(args.output.resolve().parent == HERE and not args.output.exists(), 'Fresh local source-check receipt only')
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
