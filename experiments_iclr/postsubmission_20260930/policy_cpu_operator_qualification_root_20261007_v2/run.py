"""Explicit CPU-only operator qualification; no scientific data or training family."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import resource
import socket
import subprocess
import sys
import time
import traceback

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
SOURCE = PHASE / 'cognn_polynormer_external_message_inactive_prototype_20261007_v1'
HERE = Path(__file__).resolve().parent
PINS = {
    'prototype.py': 'd0827ad71fe94409b09df9c696bfa387849053ee4de73e5e5fc58323d16d1bea',
    'runtime_fixture.py': '19b581879139af8e2dff219c2e0825df8ef696b68a3c61e44add9686d3052d83',
    'MANIFEST.json': 'a0ab255f606453f94ffbe990ccb6e3fbc3d7ac5afa49ac5bd4b3128749ca75a9',
}
NATIVE_PIN = '9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert Path.cwd() == REPO and HERE.is_relative_to(PHASE)
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True
    ).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    for filename, expected in PINS.items():
        assert digest(SOURCE / filename) == expected
    native = HERE / 'pinned_native_model.py'
    assert digest(native) == NATIVE_PIN
    factors = PHASE / 'portable_internal_be_public_interface_20261007_v2/core/factors.py'
    destination = HERE / 'RESULT.json'
    assert not destination.exists()
    started = time.monotonic()
    record = {
        'schema': 'graph-policy-CPU-operator-qualification-v1',
        'start_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_pins': PINS, 'native_sha256': NATIVE_PIN,
        'driver_sha256': digest(__file__), 'interpreter': sys.executable,
        'CUDA_VISIBLE_DEVICES': '', 'roles_or_checkpoints_accessed': False,
        'scientific_fits': 0, 'scientific_adoption': False,
        'scope': 'Artificial operator/gradient/Adam fixtures, not representative quality evidence.',
    }
    prototype = None
    try:
        sys.path.insert(0, str(SOURCE))
        import prototype
        assert prototype.RUNTIME_ENABLED is False
        # Root-authorized activation changes only this process's module variable.
        # The frozen source bytes, native parameters, live jobs and host settings stay intact.
        prototype.RUNTIME_ENABLED = True
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        import torch_geometric
        import runtime_fixture
        record['providers'] = {'torch': torch.__version__, 'PyG': torch_geometric.__version__}
        record['checks'] = runtime_fixture.run(native, factors)
        record['status'] = 'PASS'
        record['fixture_Adam_transitions'] = 4
    except Exception as error:
        record.update(status='FAIL', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        if prototype is not None:
            prototype.RUNTIME_ENABLED = False
        record['elapsed_seconds'] = time.monotonic() - started
        record['terminal_UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record['CPU_seconds'] = usage.ru_utime + usage.ru_stime
        record['peak_RSS_KiB_linux'] = usage.ru_maxrss
        record['frozen_source_bytes_unchanged'] = all(
            digest(SOURCE / filename) == expected for filename, expected in PINS.items())
        with destination.open('x') as handle:
            json.dump(record, handle, indent=2)
            handle.write('\n')
    print(json.dumps(record))
    if record['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
