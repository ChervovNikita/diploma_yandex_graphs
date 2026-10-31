"""Structural and literal combinatorial checks. No numerical imports/execution."""
import ast
import json
from pathlib import Path
import random
from types import SimpleNamespace
from plan import CONDITIONS, SEEDS, STAGE1, condition, plan, resource_forecast
from native import HERE, sha


def extract(function, environment):
    tree = ast.parse((HERE/'method.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<literal combinatorial fixture>', 'exec'), environment)
    return environment[function]


def main():
    parsed = {}
    for file in HERE.glob('*.py'):
        parsed[file.name] = ast.parse(file.read_text(), filename=str(file))
    assert len(CONDITIONS) == 9 and len(STAGE1) == 3 and len(SEEDS) == 3
    assert plan()['stage1_records'] == 9 and plan()['stage2_records'] == 18
    assert not plan()['scientific_runner_enabled']
    assert condition('single_four_view_core')['factual_views'] == 1
    assert condition('single_four_view_core')['masked_views'] == 4
    assert condition('untied4_shared_decoder_core')['genuine_independent'] is False
    assert condition('independent4_native')['shared_decoder'] is False
    env = {'hashlib':__import__('hashlib')}
    partition = extract('balanced_hash_partition', env)
    ids = tuple(range(100))
    quarters = partition(ids)
    assert sorted(x for row in quarters for x in row) == list(ids)
    assert max(map(len, quarters))-min(map(len, quarters)) <= 1
    assert partition(tuple(reversed(ids))) == quarters
    # The fake tensor constructor is only an identity for returned integer lists.
    # No Torch import, model, labels or feature data are involved.
    env = {'torch':SimpleNamespace(tensor=lambda x, dtype: x, long='literal')}
    negatives = extract('negative_indices', env)
    for size in (33, 34, 79):
        values = negatives(size, random.Random(17))
        assert len(values) == size
        for anchor, row in enumerate(values):
            assert len(row) == len(set(row)) == 32
            assert anchor not in row and all(0 <= k < size for k in row)
        assert values == negatives(size, random.Random(17))
    try: negatives(32, random.Random(17))
    except ValueError: pass
    else: raise AssertionError('Insufficient anchors silently changed loss')
    source = (HERE/'method.py').read_text()
    assert 'self._view(masked_x, aux)' in source
    assert 'z[indices[start:end]]' in source and 'x[indices' not in source
    assert 'total.backward()' in source and 'parameter.grad =' not in source
    core_node = next(n for n in parsed['method.py'].body if isinstance(n, ast.FunctionDef) and n.name == 'core_loss')
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'detach' for n in ast.walk(core_node))
    native_source = (HERE/'native.py').read_text()
    assert 'ast.ClassDef' in native_source and 'exec_module(value)' in native_source  # only pinned factor helper
    assert 'torch.manual_seed' not in native_source
    result = dict(complete=True, kind='stdlib source/combinatorial checks only',
                  parsed_python_files=sorted(parsed), literal_partition_pass=True,
                  exact_other32_negative_id_pass=True, all_view_single_exposure_pass=True,
                  science_enabled=False, model_data_checkpoint_execution=False,
                  numerical_gradient_parity_established=False, competence_established=False,
                  forecast_parameters=resource_forecast()['native_parameters'])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__': main()
