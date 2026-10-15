"""Bind the actual 27 population contracts to the existing reporting registry.

Pure metadata preparation; constructs no model or campaign authorization.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import resource
import socket
import subprocess
import sys
import time


ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
E = ROOT / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'


def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                   text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert Path(__file__).resolve().parent == E / 'final_roster_population_bind_v1'
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    started = time.monotonic()
    folder = Path(__file__).parent / 'generated_v1'
    assert not folder.exists()
    folder.mkdir()
    try:
        module = E / 'final_selected_scoring_v2/contracts.py'
        assert hashlib.sha256(module.read_bytes()).hexdigest() == '86b57392e75f00a9b64589396f7c659baba0b257fdc7374006845c1c594c5c51'
        spec = importlib.util.spec_from_file_location('_original_final_contracts', module)
        c = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(c)
        def read_pin(path, sha):
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == sha
            return {'path': str(path), 'bytes': len(raw), 'sha256': sha}, json.loads(raw)
        registry_pin, registry = read_pin(
            E / 'benchmark_reporting_portable_rebind_v1/generated_v1/INDEX.json',
            '7df83c231a2a7d7e24f745988549a872784bf81fd7e006d708b492a6d69d66e9')
        population_pin, population = read_pin(
            E / 'final_role_population_preparation_v1/allocation_population_v3/ROLE_CONTRACT_INDEX.json',
            '032e029665bb518d123353eb77fac5c651dd160d2cb0040893723b79f05e4f6a')
        profiles = {}
        for pin in registry['preset_files']:
            value = c.verify(pin, ROOT, json_metadata=True)
            key = value['registry']['presets'][0]['id']
            assert key not in profiles
            profiles[key] = value
        roles = {key: c.verify(pin, ROOT, json_metadata=True) for key, pin in population['contracts'].items()}
        roster = c.compile_roster(registry_pin, profiles, population['contracts'], roles,
                                  E / 'campaign_final_selected_outputs_v1')
        assert roster['launch_permission'] is False
        def write(name, value):
            path = folder / name
            raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
            with path.open('xb') as f:
                f.write(raw)
            return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        roster_pin = write('ROSTER.json', roster)
        result = {'status': 'complete', 'roster': roster_pin, 'registry': registry_pin,
                  'population_index': population_pin, 'counts': roster['counts'], 'tasks': len(roster['tasks']),
                  'elapsed_seconds': time.monotonic() - started, 'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  'new_training_fits': 0, 'models_constructed': 0, 'TEST_payload_accessed': False,
                  'TEST_scored': False, 'full_campaign_GO': False, 'native_inference_qualified': False}
        write('RESULT.json', result)
        print(json.dumps(result))
    except BaseException as error:
        import traceback
        with (folder / 'FAILURE.json').open('x') as f:
            json.dump({'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc(),
                       'elapsed_seconds': time.monotonic() - started, 'full_campaign_GO': False}, f, indent=2)
        raise


if __name__ == '__main__':
    main()
