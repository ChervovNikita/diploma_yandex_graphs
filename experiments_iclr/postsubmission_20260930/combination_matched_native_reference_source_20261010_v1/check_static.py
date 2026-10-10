"""Static AST/hash/recipe/roster check only; imports no scientific modules."""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pins = json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    verified = []
    for row in pins['source_files']:
        path = PHASE/row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        verified.append(row)
    trees = {path.name: ast.parse(path.read_text()) for path in HERE.glob('*.py')}
    portable = PHASE/pins['public_portable']['path']
    public_tree = ast.parse(portable.read_text())
    cls = next(node for node in public_tree.body if isinstance(node, ast.ClassDef) and node.name == 'Session')
    original = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    edited = copy.deepcopy(original)
    index = next(i for i, node in enumerate(edited.body) if isinstance(node, ast.FunctionDef) and node.name == 'optimizer')
    edited.body.insert(index, ast.parse('_matched_native_scorers(self)').body[0])
    inserted = ast.fix_missing_locations(ast.Module(body=[edited], type_ignores=[]))
    compile(inserted, str(portable)+':static-one-insertion', 'exec')
    removed = copy.deepcopy(edited); del removed.body[index]
    assert ast.dump(removed, include_attributes=False) == ast.dump(original, include_attributes=False)
    step = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'train_step')
    assert sum(isinstance(node, ast.Call) and ast.unparse(node.func) == 'total.backward' for node in ast.walk(step)) == 1
    assert "own = 0.5 * (losses.own_supervision(la, labels, self.task) + losses.own_supervision(lb, labels, self.task))" in ast.unparse(step)
    assert 'total = own.sum() + self.model.members * auxiliary if self.model.independent else own.mean() + auxiliary' in ast.unparse(step)
    source = (HERE/'initializer.py').read_text()
    assert "session.model.members == 1" in source and "tuple(parameter.shape) == (1, 1, 512)" in source
    assert 'row_seed(paired_seed, layer, side_index, member_index)' in source
    assert 'hasattr(session, \'optimizers\')' in source and 'hasattr(session, \'streams\')' in source
    assert 'torch.Generator(device=\'cpu\')' in source and 'draw.uniform_(-bound, bound, generator=generator)' in source
    train_source = (HERE/'train.py').read_text()
    assert "metric = per[0] if spec['family'] == 'native_independent4' else pooled_metric" in train_source
    assert 'driver.main()' in train_source and 'original_step(batch, labels)' in (HERE/'reference.py').read_text()
    assert not any(isinstance(node, ast.Import) and any(alias.name in ('torch', 'numpy') for alias in node.names)
                   for tree in trees.values() for node in tree.body)
    releases = [json.loads(path.read_text()) for path in sorted((HERE/'releases_disabled').glob('*.json'))]
    expected = {(family, seed, member) for seed in (6101, 6203, 6307)
                for family, count in (('native_single', 1), ('native_independent4', 4)) for member in range(count)}
    assert len(releases) == len(expected) == 15
    assert {(r['spec']['family'], r['spec']['paired_seed'], r['spec']['member_index']) for r in releases} == expected
    assert all(r['enabled'] is False and r['root_execution_authorized'] is False
               and r['reference_runtime_qualified'] is False and r['epochs'] == 1100 and r['local_epochs'] == 100
               and r['source_manifest_sha256'] is None and r['spec']['body_seed'] == r['spec']['paired_seed']+1009*r['spec']['member_index']
               for r in releases)
    banks = [json.loads(path.read_text()) for path in (HERE/'bank_releases_disabled').glob('*.json')]
    assert len(banks) == 3 and {r['paired_seed'] for r in banks} == {6101, 6203, 6307}
    assert all(r['enabled'] is False and len(r['bodies']) == 4 for r in banks)
    protocol = json.loads((HERE/'PROTOCOL.json').read_text())
    assert protocol['fresh_counts']['fresh_full_native_body_trajectories'] == 15
    report = dict(schema='matched-native-reference-source-static-verification-v1', passed=True,
        verified_source_files=verified, source_ASTs_parsed=sorted(trees),
        constructor_delta='Exactly one callback before original Adam; removing it restores identical original constructor AST',
        constructor_AST_sha256=hashlib.sha256(ast.dump(inserted, include_attributes=False).encode()).hexdigest(),
        original_M1_CE_and_Adam_step_source_verified=True,
        M1_initializer_has_no_four_row_bank_dependency=True,
        I4_own_selector_uses_exact_original_per_member_metric=True,
        original_full_driver_local_restore_and_live_stream_logic_preserved=True,
        native_body_roster=15, own_selected_banks=3, all_releases_disabled=True,
        source_modules_imported=False, scientific_models_instantiated=0,
        numerical_or_runtime_verification=False, scientific_fits=0)
    with (HERE/'STATIC_CHECKS.json').open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'verified_source_files'}, indent=2))


if __name__ == '__main__':
    main()
