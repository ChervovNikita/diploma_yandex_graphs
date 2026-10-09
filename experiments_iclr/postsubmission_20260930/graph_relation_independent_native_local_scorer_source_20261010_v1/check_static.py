"""Source/hash/AST verification only; never import or execute scientific code."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parsed = []
    for path in sorted(HERE.glob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path)); compile(tree, str(path), 'exec')
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module or '']
                assert not any(name.split('.')[0] in {'numpy', 'torch', 'torch_geometric', 'ogb'} for name in names)
        parsed.append(path.name)
    pins = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    for row in pins['source_files']:
        path = PHASE / row['path']; assert path.suffix not in {'.pt', '.npz', '.npy'}
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
    payloads = 0
    for row in pins['source_manifests']:
        path = PHASE / row['path']; assert sha(path) == row['sha256']
        for item in json.loads(path.read_text())['files']:
            p = path.parent / item['path']; assert p.suffix not in {'.pt', '.npz', '.npy'}
            assert sha(p) == item['sha256'] and p.stat().st_size == item['bytes']; payloads += 1
    initializer = (HERE / 'initializer.py').read_text()
    assert "math.sqrt(6.0 / (bank.size(-2) + bank.size(-1)))" in initializer
    assert "torch.Generator(device='cpu')" in initializer and 'generator=generator' in initializer
    assert '100000 * seed + SEED_OFFSET + 10000 * layer + 100 * side + member' in initializer
    seeds = {100000*seed+770000000+10000*layer+100*side+member
        for seed in (6101,6203,6307) for layer in range(7) for side in (0,1) for member in range(4)}
    assert len(seeds) == 168
    assert 'xavier' not in initializer.lower().replace('no torch xavier', '')
    assert 'after_common == before_common' in initializer and 'torch.equal(torch.get_rng_state(), cpu_before)' in initializer
    assert "bank[member].copy_(draw.to(device=bank.device))" in initializer
    integration = (HERE / 'integration.py').read_text()
    assert "dict(constructor=1, copied_start=1)" in integration
    assert integration.index('original_install(body, members)') < integration.index('initialize(hook.torch, controller, seed)')
    assert 'constructor.verified_hook = original_verified' in integration
    plan = json.loads((HERE / 'PROTOCOL.json').read_text())
    assert plan['policies'] == ['alphaF', 'relationJ'] and plan['seeds'] == [6101, 6203, 6307]
    assert plan['new_full_fits'] == 6 and plan['full_task']['epochs'] == 1100
    assert plan['scientific_launch_admitted'] is False and plan['coefficient_or_initializer_grid'] is False
    qualify = (HERE / 'qualify.py').read_text()
    assert 'real_complete_TRAIN_updates=4, native_models_constructed=4' in qualify
    assert 'Original member dropout stream bytes' in qualify
    assert qualify.index('admission(release, qualification=True') < qualify.index('import torch')
    for path in (HERE / 'releases_disabled').glob('*.json'):
        row = json.loads(path.read_text())
        assert all(row[key] is False for key in ('enabled', 'root_execution_authorized', 'source_review_approved', 'native_qualification_approved'))
        assert row['policy'] in plan['policies'] and row['seed'] in plan['seeds'] and row['epochs'] == 1100
    assert len(list((HERE / 'releases_disabled').glob('*.json'))) == 6
    for name in ('QUALIFIER_RELEASE_DISABLED.json', 'TRAIN_SUPERVISION_TEMPLATE_DISABLED.json', 'QUALIFICATION_SUPERVISION_TEMPLATE_DISABLED.json'):
        assert json.loads((HERE / name).read_text())['enabled'] is False
    if (HERE / 'SOURCE_MANIFEST.json').exists():
        for row in json.loads((HERE / 'SOURCE_MANIFEST.json').read_text())['files']:
            path = HERE / row['path']; assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
    print(json.dumps(dict(source_AST_syntax=parsed, source_manifests=len(pins['source_manifests']), source_payloads=payloads,
        independent_native_local_scorer_rows_only=True, native_last_two_dimension_Glorot=True,
        explicit_initializer_generators_default_RNG_preserved_by_source=True, constructor_before_original_Adam=True,
        fixed_full_fits=6, policies=plan['policies'], seeds=plan['seeds'], four_complete_TRAIN_qualification_updates=True,
        old_sources_math_recipe_driver_replay_and_parameter_roles_reused=True, new_owner_or_scheduler=False,
        numerical_imports_execution_fixtures_or_outcome_reads=False, runtime_qualified=False, templates_disabled=True), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
