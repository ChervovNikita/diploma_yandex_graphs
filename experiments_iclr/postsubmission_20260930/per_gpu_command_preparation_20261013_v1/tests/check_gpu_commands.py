"""Fabricated metadata only; exercise the actual launcher without children."""
import argparse
from collections import Counter
import fcntl
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

OUT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--controller', required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.controller).resolve()))
blocked_imports = []


class NumericalDenied(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'scipy', 'pandas', 'ogb', 'torch_geometric'}:
            blocked_imports.append(fullname)
            raise AssertionError('Numerical import denied: ' + fullname)


sys.meta_path.insert(0, NumericalDenied())
import launcher
spec = importlib.util.spec_from_file_location('gpu_command', OUT / 'runtime/gpu_command.py')
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


def denied(*args, **kwargs):
    raise AssertionError('Registry/child/transport is denied in metadata fixtures')


def rejected(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, RuntimeError, BlockingIOError):
        return
    raise AssertionError('Expected metadata rejection')


with tempfile.TemporaryDirectory(dir=OUT / 'tests') as tmp:
    root = Path(tmp)
    output = root / 'output'
    output.mkdir()
    observed = [{'index': index, 'uuid': 'fabricated-' + str(index), 'total_mib': 80, 'free_mib': 70} for index in (0, 1)]
    plan = {'slots_per_gpu': 2, 'output_root': 'output', 'global_gates': {'search': {}, 'comparative': {}, 'final': {}}}
    digest = 'fabricated-plan-hash'
    runs, placement, dependencies = {}, {}, {}
    for host, indices in [('allocation', (0,)), ('server77', (0, 1))]:
        for index in indices:
            worker = host + '/gpu' + str(index)
            for number in range(4):
                key = worker + '/run' + str(number)
                runs[key] = {'study': 'fabricated', 'run': {'kind': 'model_fit', 'config': {'stage': 'search', 'reservation_mib': 20, 'estimated_seconds': 10 - number}}}
                placement[key] = {'worker': worker, 'group': worker}
                dependencies[key] = []
    fixture = (plan, digest, {}, runs, placement, dependencies)
    calls = []

    def host_check(host, candidate):
        calls.append(('host_check', host, bool(candidate)))
        return observed[:1] if host == 'allocation' else observed

    def load_plan(*args):
        calls.append(('load_plan', len(args)))
        return fixture

    with patch.object(launcher, 'Registry', denied), patch.object(launcher.subprocess, 'Popen', denied), patch.object(launcher, 'host_check', host_check), patch.object(launcher, 'load_plan', load_plan), patch.object(launcher, 'runtime_preflight', lambda *args: {'status': 'fabricated_no_runtime_import'}), patch.object(launcher, 'receipt_authentication', lambda *args: (root, {'remote_name': 'fabricated'})), patch.object(launcher, 'local_terminal_roster', lambda *args: None):
        cases = []
        for host, index in [('allocation', 0), ('server77', 0), ('server77', 1)]:
            calls.clear()
            value = gpu.preflight(root, root / 'PLAN.json', host, Path(sys.executable), index)
            assert value['selected_run_records'] == 4 and value['slots_per_gpu'] == 2
            assert value['global_gates'] == ['comparative', 'final', 'search'] and value['GO'] is False
            if host == 'allocation':
                assert calls[0] == ('host_check', 'allocation', False) and calls[1][0] == 'load_plan'
            cases.append(value['worker'])
        rejected(gpu.selected_worker, plan, placement, observed, 'allocation', 1)
        rejected(gpu.selected_worker, plan, placement, observed, 'server77', 2)
        rejected(gpu.selected_worker, {**plan, 'slots_per_gpu': 1}, placement, observed, 'server77', 0)
        rejected(gpu.selected_worker, plan, {}, observed, 'server77', 0)
        rejected(gpu.start, root, root / 'PLAN.json', None, 'server77', Path(sys.executable), 0)
        owners = {'supervisor': {'pid': 101, 'start_ticks': 1001, 'pgid': 101}, 'rendezvous': {'pid': 102, 'start_ticks': 1002, 'pgid': 102}}
        owner = {'host': 'server77', 'plan_hash': digest, **owners}
        (output / 'server77.startup-owner.json').write_text(json.dumps(owner))
        with patch.object(launcher, 'process_identity', lambda pid: owners['supervisor'] if pid == 101 else owners['rendezvous']):
            first = gpu.start(root, root / 'PLAN.json', root / 'FABRICATED_APPROVAL_REFERENCE', 'server77', Path(sys.executable), 0)
            assert first['status'] == 'already_running' and not first['scientific_children_restarted']
            assert launcher.enabled_workers(output, 'server77', digest, observed) == {'server77/gpu0'}
            second = gpu.start(root, root / 'PLAN.json', root / 'FABRICATED_APPROVAL_REFERENCE', 'server77', Path(sys.executable), 1)
            assert second['status'] == 'already_running' and not second['scientific_children_restarted']
            assert launcher.enabled_workers(output, 'server77', digest, observed) == {'server77/gpu0', 'server77/gpu1'}
            with (output / 'server77.startup.lock').open('a+') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                rejected(gpu.start, root, root / 'PLAN.json', root / 'FABRICATED_APPROVAL_REFERENCE', 'server77', Path(sys.executable), 1)
            assert json.loads((output / 'server77.startup-owner.json').read_text()) == owner
        # Actual scheduling obeys independent two-slot quotas, reservations and parents.
        states = {key: 'registered' for key in runs}
        capacities = {'server77/gpu0': 100, 'server77/gpu1': 100}
        admitted = launcher.schedule_ready(runs, states, placement, dependencies, [], capacities, 'server77')
        assert Counter(placement[key]['worker'] for key in admitted) == {'server77/gpu0': 2, 'server77/gpu1': 2}
        disabled = launcher.schedule_ready(runs, states, placement, dependencies, [], {'server77/gpu0': 100}, 'server77')
        assert len(disabled) == 2 and all(placement[key]['worker'] == 'server77/gpu0' for key in disabled)
        tight = launcher.schedule_ready(runs, states, placement, dependencies, [], {'server77/gpu0': 20}, 'server77')
        assert len(tight) == 1
        active = [{'key': 'server77/gpu0/run0', 'worker': 'server77/gpu0', 'reserve_mib': 20}]
        pending = launcher.schedule_ready(runs, states, placement, dependencies, active, {'server77/gpu0': 100}, 'server77')
        assert len(pending) == 1
        dependencies['server77/gpu0/run0'] = ['server77/gpu0/run3']
        pending = launcher.schedule_ready(runs, states, placement, dependencies, [], {'server77/gpu0': 100}, 'server77')
        assert 'server77/gpu0/run0' not in pending
        states['server77/gpu0/run3'] = 'complete'
        pending = launcher.schedule_ready(runs, states, placement, dependencies, [], {'server77/gpu0': 100}, 'server77')
        assert 'server77/gpu0/run0' in pending
        # Real supervisor reads enabled workers before applying capacities.
        tree = __import__('ast').parse(Path(launcher.__file__).read_text())
        supervisor = next(n for n in tree.body if isinstance(n, __import__('ast').FunctionDef) and n.name == 'run_supervisor')
        source = __import__('ast').get_source_segment(Path(launcher.__file__).read_text(), supervisor)
        assert "enabled = enabled_workers(output, host, digest, current)" in source
        assert "if worker in enabled" in source and 'schedule_ready(' in source

assert not blocked_imports
print(json.dumps({'schema': 'gpu-companion-metadata-and-real-launcher-synthetic-QA-v1',
    'preflight_workers': cases, 'slots_per_worker': 2,
    'actual_detach_existing_owner_calls': 2, 'duplicate_supervisors_started': 0,
    'gpu_requests_accumulate': True, 'concurrent_startup_lock_rejection_preserved': True,
    'actual_scheduler_quota_disabled_GPU_reservation_active_slot_dependency_checks': 6,
    'metadata_rejections': 5, 'original_owner_preserved': True,
    'numerical_imports': 0, 'registry_constructions': 0, 'child_processes': 0,
    'server77_access': False, 'actual_input_arrays_decoded': 0, 'models': 0,
    'optimizer_updates': 0, 'scientific_fits': 0, 'GO': False}, sort_keys=True))
