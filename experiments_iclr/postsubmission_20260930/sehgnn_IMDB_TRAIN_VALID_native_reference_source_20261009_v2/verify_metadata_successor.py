"""Exact one-line metadata successor verification; stdlib only."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior = (OLD/'runner.py').read_text()
    current = (HERE/'runner.py').read_text()
    before = 'versions = {name: value.__version__ for name, value in providers.items()}'
    after = 'versions = {name: str(value.__version__) for name, value in providers.items()}'
    assert prior.count(before) == current.count(after) == 1
    assert current.replace(after, before) == prior
    unchanged = json.loads((HERE/'SUCCESSOR_DELTA.json').read_text())['byte_identical_payload']
    for name in unchanged:
        assert (HERE/name).read_bytes() == (OLD/name).read_bytes()
    engine = (HERE/'engine.py').read_text()
    assert "torch.load(checkpoint, map_location='cpu', weights_only=True)" in engine
    assert 'add_safe_globals' not in current+engine and 'weights_only=False' not in current+engine
    tree = ast.parse(current)
    runtime = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'runtime')
    assignment = next(node for node in runtime.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'versions' for target in node.targets))
    expression = ast.Expression(assignment.value)
    ast.fix_missing_locations(expression)
    class VersionText(str):
        pass
    descriptors = {'torch': SimpleNamespace(__version__=VersionText('2.1.2+cu118')),
                   'numpy': SimpleNamespace(__version__='1.26.4')}
    result = eval(compile(expression, '<isolated-metadata-expression>', 'eval'), {'providers': descriptors, 'str': str})
    assert result == {'torch': '2.1.2+cu118', 'numpy': '1.26.4'}
    assert all(type(value) is str for value in result.values())
    assert not {'torch','numpy','dgl','torch_sparse','sklearn'} & set(sys.modules)
    record = dict(schema='native-SeHGNN-metadata-successor-static-verification-v2', status='passed',
                  sole_executable_change='literal provider-version strings before runtime checkpoint identity',
                  old_runner_sha256=sha(OLD/'runner.py'), new_runner_sha256=sha(HERE/'runner.py'),
                  byte_identical_payload=unchanged, weights_only_true_preserved=True,
                  isolated_metadata_str_subclass_witness='passed_without_provider_import',
                  numerical_model_data_checkpoint_or_server_execution=False,
                  numerical_or_floating_tolerance_tests=False, native_v2_qualification=False)
    (HERE/'METADATA_SUCCESSOR_STATIC.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status='passed', runner_sha256=record['new_runner_sha256'], numerical_model_data_checkpoint_or_server_execution=False)))


if __name__ == '__main__':
    main()
