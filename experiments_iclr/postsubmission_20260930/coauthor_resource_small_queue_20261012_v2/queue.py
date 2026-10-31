"""Finite source-bound engineering queue; optimizer updates are intercepted."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
MASTER = 'c58fd5b0b590d581a2cbc5e12f7da3090c57ee781f2e1434ab835874c6874c08'


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(pid):
    path = Path('/proc') / str(pid) / 'stat'
    if not path.exists():
        return None
    fields = path.read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'pgid': int(fields[2]), 'start_ticks': int(fields[19]), 'state': fields[0]}


def receipt(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def guard():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                   text=True, timeout=10).splitlines() == [GPU]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    guard()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = Path(config['project_root']).resolve()
    assert config_path.is_relative_to(root)
    output = root / config['output']
    assert output.resolve().is_relative_to(root) and not output.exists()
    output.mkdir()
    helper = root / config['helper']['path']
    manifest = root / config['manifest']['path']
    assert receipt(helper) == {k: config['helper'][k] for k in ('bytes', 'sha256')}
    assert receipt(manifest) == {k: config['manifest'][k] for k in ('bytes', 'sha256')}
    contract = json.loads(manifest.read_text())
    assert contract['master_hash'] == MASTER and len(config['case_ids']) == 10
    assert len(set(config['case_ids'])) == 10 and set(config['case_ids']) <= {c['case_id'] for c in contract['cases']}
    runtime = root / config['runtime']
    assert runtime.resolve().is_relative_to(root) and runtime.is_dir()
    for case in contract['cases']:
        if case['case_id'] not in config['case_ids']:
            continue
        assert case['run']['adapter'].startswith(config['runtime'] + '/')
        study_path = root / case['study_binding']['path']
        assert receipt(study_path) == {k: case['study_binding'][k] for k in ('bytes', 'sha256')}
        study = json.loads(gzip.decompress(study_path.read_bytes()))
        for bound in study['source']['files']:
            if bound['path'].startswith(config['runtime'] + '/'):
                assert receipt(root / bound['path']) == {k: bound[k] for k in ('bytes', 'sha256')}
    started = time.monotonic()
    save(output / 'OWNER_START.json', {'owner': identity(os.getpid()), 'config': receipt(config_path),
                                     'scientific_fits': 0, 'GO_created': False})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONPATH=str(runtime))
    preflight = subprocess.run([config['python'], '-B', '-c', 'import prepare_node_data,run_study,engine'],
                               cwd=runtime, env=dict(env, CUDA_VISIBLE_DEVICES=''),
                               capture_output=True, text=True, timeout=60)
    save(output / 'CPU_IMPORT_PREFLIGHT.json', {'exit_code': preflight.returncode,
        'stdout': preflight.stdout, 'stderr': preflight.stderr, 'scientific_fits': 0, 'CUDA_executions': 0})
    assert preflight.returncode == 0, 'Preserved CPU source import failure; no engineering children launched'
    rows = []
    for index, identifier in enumerate(config['case_ids'], 1):
        guard()
        destination = output / ('case_%02d_%s' % (index, identifier))
        command = [config['python'], '-B', str(helper), '--project-root', str(root),
                   '--output', str(destination.relative_to(root)), '--mode', 'execute-node-case',
                   '--manifest', str(manifest.relative_to(root)), '--case-id', identifier,
                   '--reviewed-helper-sha256', config['helper']['sha256'],
                   '--warm-iterations', '3', '--case-timeout-seconds', '240']
        log_path = output / ('case_%02d.log' % index)
        with log_path.open('xb') as log:
            child = subprocess.Popen(command, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            owned = identity(child.pid)
            save(output / ('CASE_%02d_START.json' % index),
                 {'case_id': identifier, 'child': owned, 'command': command})
            timeout = False
            try:
                exit_code = child.wait(timeout=270)
            except subprocess.TimeoutExpired:
                observed = identity(child.pid)
                assert observed is not None and all(observed[k] == owned[k] for k in ('pid', 'pgid', 'start_ticks'))
                os.killpg(owned['pgid'], signal.SIGTERM)
                try:
                    exit_code = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    observed = identity(child.pid)
                    assert observed is not None and all(observed[k] == owned[k] for k in ('pid', 'pgid', 'start_ticks'))
                    os.killpg(owned['pgid'], signal.SIGKILL)
                    exit_code = child.wait()
                timeout = True
        result_path = destination / 'RESULT.json'
        if not result_path.exists() and (destination / 'TERMINAL_FAILURE.json').exists():
            result_path = destination / 'TERMINAL_FAILURE.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        row = {'case_id': identifier, 'exit_code': exit_code, 'timeout': timeout,
               'result': str(result_path.relative_to(root)) if result else None,
               'result_receipt': receipt(result_path) if result else None,
               'status': result.get('status') if result else 'MISSING_TERMINAL_RESOURCE_RECEIPT'}
        if result and result.get('status') == 'TRAIN_PROXY_COMPLETE':
            assert all(result[k] == 0 for k in ('actual_optimizer_updates', 'actual_optimizer_kernel_calls',
                                               'heldout_original_file_accesses', 'scores_emitted'))
        rows.append(row)
        save(output / 'PROGRESS.json', {'completed': len(rows), 'expected': 10, 'cases': rows})
    save(output / 'CLOSURE.json', {'status': 'CLOSED', 'owner': identity(os.getpid()), 'cases': rows,
                                 'passed': sum(r['status'] == 'TRAIN_PROXY_COMPLETE' and r['exit_code'] == 0 for r in rows),
                                 'wall_seconds': time.monotonic() - started,
                                 'scientific_fits': 0, 'GO_created': False, 'automatic_retries': 0,
                                 'scope': 'one_body_random_TRAIN_callback_only;I4_export_and_full_study_concurrency_unqualified'})


if __name__ == '__main__':
    main()
