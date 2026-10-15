"""Narrow metadata regression for child host-config environment binding.

Does not execute either CLI, host guard, supervisor, model or parent suite.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    here = Path(__file__).resolve().parent
    expansion = here.parent
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(expansion / 'portable_dispatcher_v3'))
    from environment import negative_cli_environment
    from host_binding import config_path, load_config, pin

    bindings = json.loads((here / 'SOURCE_BINDINGS.json').read_text())
    for row in bindings['files']:
        path = (expansion / row['path_relative_to_expansion']).resolve()
        assert path.is_relative_to(expansion)
        raw = path.read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']

    allocation = pin(config_path('allocation/gpu0'))
    parent = {'BENCHMARK_HOST_CONFIG': allocation['path'],
              'BENCHMARK_HOST_CONFIG_SHA256': allocation['sha256'],
              'CUDA_VISIBLE_DEVICES': 'GPU-44039938-fd82-41d2-fefd-de71514e2fac',
              'PRESERVED_EXAMPLE': 'unchanged'}
    before = dict(parent)
    # Reproduce the original bootstrap failure by loading only bundled JSON.
    with patch.dict(os.environ, parent, clear=True):
        try:
            load_config(config_path('server77/gpu0'))
        except ValueError as error:
            assert str(error) == 'Host configuration hash changed'
        else:
            raise AssertionError('Original inherited-hash failure must reproduce')
    loaded = []
    for worker in ('allocation/gpu0', 'server77/gpu0'):
        local = pin(config_path(worker))
        child = negative_cli_environment(parent, local)
        assert parent == before and child is not parent
        assert set(child) == set(parent)
        for name in parent:
            if name not in ('BENCHMARK_HOST_CONFIG', 'BENCHMARK_HOST_CONFIG_SHA256'):
                assert child[name] == parent[name]
        assert child['BENCHMARK_HOST_CONFIG'] == local['path']
        assert child['BENCHMARK_HOST_CONFIG_SHA256'] == local['sha256']
        with patch.dict(os.environ, child, clear=True):
            config, receipt = load_config(config_path(worker))
        assert receipt == local and config['worker'] == worker
        assert config['host_enabled'] is (worker == 'allocation/gpu0')
        loaded.append(worker)

    source = (here / 'qualify_orchestration.py').read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not item.name.startswith(('torch', 'torch_geometric')) for item in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or '').startswith(('torch', 'torch_geometric'))
    assert "('server77/gpu0', 'Host disabled before probes')" in source
    assert "child_config_pin = pin(config_path(worker), root)" in source
    assert "child_env = negative_cli_environment(os.environ, child_config_pin)" in source
    assert 'cwd=root, env=child_env, capture_output=True' in source
    assert "require(len(result['cases']) == 8" in source
    assert "== 3, 'Exactly three metadata-only supervisors'" in source
    assert 'torch' not in sys.modules
    original = json.loads((here / 'PARENT_FAILURE.json').read_text())
    assert original['run']['result_receipt']['sha256'] == 'ccbe0af6d37d85cc7e66c7e222c5e6ed01365a9df8529ab9c746a12f20a02e11'
    assert original['new_execution_performed'] is False
    print(json.dumps({'schema': 'negative-cli-environment-source-check-v2', 'status': 'PASS',
        'inherited_hash_failure_reproduced_metadata_only': True,
        'corrected_worker_configs_loaded': loaded, 'parent_environment_unchanged': True,
        'CUDA_mask_unchanged': True, 'strict_disabled_before_probes_assertion_retained': True,
        'unchanged_source_pins_verified': len(bindings['files']),
        'engineering_suite_executed': False, 'negative_CLI_invocations': 0,
        'host_probes': 0, 'supervisors_started': 0, 'scientific_fits': 0,
        'models': 0, 'optimizer_updates': 0, 'TEST_accessed': False, 'remote_calls': 0,
        'native_runtime_readiness': 'pending', 'full_campaign_GO': False}, indent=2))


if __name__ == '__main__':
    main()
