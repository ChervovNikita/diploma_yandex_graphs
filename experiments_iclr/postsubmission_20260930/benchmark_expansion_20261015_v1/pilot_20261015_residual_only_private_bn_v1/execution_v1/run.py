"""Three exact frozen fits through the existing native runner, serially."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
E = ROOT / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
PILOT = E / 'pilot_20261015_residual_only_private_bn_v1'
OUT = E / 'results/residual_only_private_bn_v1'


def guard():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    peers = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert set(peers) <= {'640428'}, 'Unexpected GPU peer; root review needed'
    if peers:
        assert int(Path('/proc/640428/stat').read_text().rsplit(')', 1)[1].split()[19]) == 6055039165


def pin(path, expected):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)


def write(name, value):
    path = OUT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main():
    guard()
    assert not OUT.exists(), 'Preserve earlier execution; no automatic restart'
    pin(PILOT / 'SPEC.json', 'a69e41b00fe751106c3584134e86ebab40ec714aa47742155763c5b97086496c')
    spec = json.loads((PILOT / 'SPEC.json').read_text())
    admission = json.loads((PILOT / 'execution_v1/ROOT_ADMISSION.json').read_text())
    assert admission['approved'] and admission['new_fits'] == 3 and not admission['full_campaign_GO']
    assert admission['publication_commit'] == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    qualification = E / 'qualification/residual_only_private_bn_cpu_v1_attempt01/RESULT.json'
    pin(qualification, admission['CPU_qualification_sha256'])
    assert json.loads(qualification.read_text())['status'] == 'passed'
    for p in spec['source_pins'] + spec['data_role_pins']:
        pin(p['path'], p['sha256'])
    for name in spec['absent_role_overrides']:
        assert not Path(name).exists()
    for row in spec['reused_wrapper_controls']:
        pin(row['authority']['path'], row['authority']['sha256'])
        value = json.loads(Path(row['authority']['path']).read_text())
        assert value['status'] == 'complete' and value['job_id'] == row['job_id']
    assert [j['seed'] for j in spec['jobs']] == [101, 202, 303]
    for job in spec['jobs']:
        pin(PILOT / job['file'], job['sha256'])
    OUT.mkdir(parents=True)
    write('START.json', {'started_unix': time.time(), 'owner_pid': os.getpid(),
                        'admission': admission, 'new_fits': 3, 'automatic_retries': 0,
                        'maximum_new_GPU_workers': 1, 'per_fit_wall_limit_seconds': 14400})
    outcomes = []
    child = None
    try:
        for job in spec['jobs']:
            guard()
            command = job['existing_runner_argv_after_root_admission']
            row = {'seed': job['seed'], 'id': job['id'], 'started_unix': time.time(), 'command': command}
            outcomes.append(row)
            environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1')
            with (OUT / ('seed' + str(job['seed']) + '.log')).open('xb') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row['worker_pid'] = child.pid
                row['worker_start_ticks'] = int(Path('/proc/' + str(child.pid) + '/stat').read_text().rsplit(')', 1)[1].split()[19])
                write('OUTCOMES.json', outcomes)
                row['returncode'] = child.wait(timeout=14400)
            row['finished_unix'] = time.time()
            result = json.loads((Path(command[-1]) / 'RESULT.json').read_text())
            row.update(status=result['status'], completed_epochs=result.get('completed_epochs'),
                       selected_epoch=result.get('selected_epoch'), stop_reason=result.get('stop_reason'))
            write('OUTCOMES.json', outcomes)
            if row['returncode'] or result['status'] != 'complete':
                raise RuntimeError('Fit failed; later seeds remain unstarted')
        write('END.json', {'status': 'complete', 'finished_unix': time.time(), 'outcomes': outcomes})
        return 0
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        write('END.json', {'status': 'failed', 'finished_unix': time.time(), 'outcomes': outcomes,
                           'failure': {'type': type(error).__name__, 'message': str(error),
                                       'traceback': traceback.format_exc(limit=8)}})
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
