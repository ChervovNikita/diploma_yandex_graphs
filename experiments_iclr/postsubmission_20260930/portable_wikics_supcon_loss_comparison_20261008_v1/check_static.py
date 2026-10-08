"""Stdlib-only source/recipe qualification; never imports Torch or datasets."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    for name in ('train.py', 'losses.py', 'recompute.py', 'test_semantics.py', 'check_static.py'):
        ast.parse((HERE/name).read_text(), filename=name)
    assert 'torch' not in sys.modules
    driver = load('_static_source_driver', HERE/'train.py')
    public = PHASE/'portable_internal_be_public_interface_20261007_v2'
    assert driver.public_dependency(public) == public.resolve()
    assert driver.CONDITIONS == {'class_full_cross_view': (.05, 0.), 'supcon_eq2': (.05, 0.)}
    assert driver.SEEDS == (6101, 6203, 6307)
    assert driver.RECOMPUTE_SHA == sha(HERE/'recompute.py')
    original = PHASE/'portable_wikics_unit_attribution_20261007_v1'
    author = PHASE/'internal_BE_WikiCS_unit_mechanism_ablation_source_20261007_v2'
    assert (HERE/'recompute.py').read_bytes() == (original/'recompute.py').read_bytes() == (author/'recompute.py').read_bytes()
    src = (HERE/'train.py').read_text()
    assert 'return self.original.alignment_loss(a, b, labels, task, temperature, identities)' in src
    assert "return a.sum() * 0  # Same differentiable zero as original alignment_only." in src
    replay = load('_static_source_replay', HERE/'recompute.py')
    for condition in driver.CONDITIONS:
        spec = driver.identity(condition, replay)
        assert (spec['alignment_weight'], spec['residual_weight'], spec['temperature']) == (.05, 0., .2)
        assert (spec['members'], spec['epochs'], spec['local_epochs'], spec['own_views'], spec['max_objects']) == (4, 1100, 100, 2, 512)
        assert spec['underlying_session_arm'] == 'be_unit_contrastive'
        assert spec['projector'] is False and spec['cross_member_contrast'] is False
    for filename in ('train.py', 'test_semantics.py'):
        result = subprocess.run([sys.executable, '-B', str(HERE/filename), '--help'],
                                text=True, capture_output=True)
        assert result.returncode == 0
    assert 'torch' not in sys.modules
    print(json.dumps(dict(schema='WikiCS-SupCon-stdlib-source-check-v1', complete=True,
        AST=True, public_dependency_all_hashes=True, original_replay_exact_bytes=True,
        original_class_full_loss_delegated=True, residual_zero=True,
        locked_conditions_seeds_recipe=True, stdlib_CLI_help=True,
        torch_imported=False, models_or_data_imported=False,
        scientific_execution=False, numerical_fixture_run=False), sort_keys=True))


if __name__ == '__main__':
    main()
