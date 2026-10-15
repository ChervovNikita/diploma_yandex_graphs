"""Regression for the actual inherited-allocation-hash metadata-loop failure only."""
import ast
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import prepare_projection as successor


def main():
    parent = successor.HERE
    package = successor.pinned(parent / 'MANIFEST.json', successor.PARENT_MANIFEST_SHA256)
    host = parent / 'hosts/allocation_gpu0.json'
    environment = {'BENCHMARK_HOST_CONFIG': str(host),
                   'BENCHMARK_HOST_CONFIG_SHA256': successor.ALLOCATION_BOOTSTRAP_SHA256}
    with patch.dict(os.environ, environment), \
         patch('socket.gethostname', side_effect=AssertionError('No hostname probe permitted')), \
         patch('subprocess.check_output', side_effect=AssertionError('No process/hardware probe permitted')):
        configs = successor.metadata_host_configs(package)
        assert set(configs) == set(successor.WORKERS)
        assert configs['server77/gpu0']['host_enabled'] is False
        assert configs['server77/gpu1']['host_enabled'] is False
        assert os.environ['BENCHMARK_HOST_CONFIG_SHA256'] == successor.ALLOCATION_BOOTSTRAP_SHA256
        try:
            successor.load_config(successor.config_path('server77/gpu0'))
        except ValueError as exc:
            assert str(exc) == 'Host configuration hash changed'
        else:
            raise AssertionError('Execution/bootstrap environment pin was weakened')
    def functions(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
    old, new = functions(parent / 'prepare_projection.py'), functions(Path(successor.__file__))
    unchanged = ['relocate_path', 'relocate_pin', 'project_task', 'build_ownership']
    assert all(old[name] == new[name] for name in unchanged)
    assert hashlib.sha256((parent / 'MANIFEST.json').read_bytes()).hexdigest() == successor.PARENT_MANIFEST_SHA256
    compile(Path(successor.__file__).read_text(), successor.__file__, 'exec')
    result = {'schema': 'portable-preparer-v3-metadata-loop-regression', 'status': 'PASS',
              'checks': ['all_three_configs_read_with_own_manifest_pin_under_allocation_environment',
                         'allocation_environment_pin_and_disabled77_execution_semantics_preserved',
                         'four_projection_functions_AST_identical_and_published_v2_manifest_unchanged'],
              'metadata_generation_executed': False, 'native_jobs_started': 0, 'hardware_or_network_probes': 0,
              'server77_accessed': False, 'published_v2_modified': False}
    path = Path(__file__).with_name('CHECKS.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
