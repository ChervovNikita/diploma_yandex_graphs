"""One fixed TRAIN-only component qualification on the authorized CUDA route."""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


REMOTE = r'''
from datetime import datetime, timezone
from pathlib import Path
import base64, hashlib, json, os, subprocess, sys, time
repo = Path(REPO); phase = repo / 'experiments_iclr/postsubmission_20260930'
assert Path.cwd() == repo
assert subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
    capture_output=True, text=True, check=True).stdout.splitlines() == [UUID]
payload = json.loads(sys.stdin.read())
root = phase / OUTPUT_NAME; root.mkdir(exist_ok=True)
for row in payload:
    path = phase / row['path']; raw = base64.b64decode(row['data'], validate=True)
    assert path.resolve().is_relative_to(root) and not path.is_symlink()
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    if path.exists(): assert path.read_bytes() == raw
    else:
        with path.open('xb') as stream: stream.write(raw)
cases = [(family, condition, stage) for family, conditions in (
    ('bank', ('tied_persistent', 'untied_persistent', 'tied_shuffled', 'tied_random_null')),
    ('reference', ('native_member', 'view_augmented_single')))
    for condition in conditions for stage in ('local', 'global')]
env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(root))
results = []
started = time.perf_counter()
for family, condition, stage in cases:
    capacity = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout.strip()
    if int(capacity) < 35000:
        results.append(dict(family=family, condition=condition, stage=stage,
            status='SKIPPED_INSUFFICIENT_CO_RESIDENT_FREE_MEMORY', free_MiB=int(capacity)))
        continue
    command = [str(repo / '.venv/bin/python'), '-B', str(root / 'check_one.py'),
               '--family', family, '--condition', condition, '--stage', stage]
    cell_start = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=180)
        row = dict(family=family, condition=condition, stage=stage, exit_code=result.returncode,
                   wall_seconds=time.perf_counter() - cell_start, stdout=result.stdout, stderr=result.stderr)
        if result.returncode == 0: row['result'] = json.loads(result.stdout)
    except subprocess.TimeoutExpired as error:
        row = dict(family=family, condition=condition, stage=stage, status='OWNED_CHECK_TIMEOUT',
                   wall_seconds=time.perf_counter() - cell_start,
                   stdout=(error.stdout or b'').decode(errors='replace'),
                   stderr=(error.stderr or b'').decode(errors='replace'))
    with (root / (family + '_' + condition + '_' + stage + '_PHYSICAL.json')).open('x') as stream:
        json.dump(row, stream, indent=2); stream.write('\n')
    results.append(row)
    # Failures are retained. Stop remaining costly cases at the first real failure.
    if row.get('exit_code') != 0: break
value = dict(UTC=datetime.now(timezone.utc).isoformat(), cases=results, expected_cases=len(cases),
    wall_seconds=time.perf_counter() - started, route=dict(login=LOGIN, repository=str(repo), GPU_UUID=UUID),
    all_component_checks_complete=len(results) == len(cases) and all(r.get('exit_code') == 0 for r in results),
    predictive_values_read=False, VALIDATION_or_TEST_labels_read=False, scientific_training_updates=0,
    checkpoint_or_logits_fetched=False, automatic_retry=False, co_resident_with_original_Amazon_queue=True,
    accepted_devices='explicit CUDA only', output_paths='absolute project-contained only')
with (root / 'QUALIFICATION.json').open('x') as stream:
    json.dump(value, stream, indent=2); stream.write('\n')
print(json.dumps(value))
'''


def main():
    census = PHASE / 'graph_view_train_coverage_census_execution_root_20261004_v1/INPUT_BINDINGS.json'
    with (HERE / 'INPUT_BINDINGS.json').open('xb') as stream: stream.write(census.read_bytes())
    files = [HERE / 'check_one.py', HERE / 'INPUT_BINDINGS.json']
    payload = []
    for path in files:
        raw = path.read_bytes()
        if path.suffix == '.py': compile(raw, str(path), 'exec')
        payload.append(dict(path=str(path.relative_to(PHASE)), bytes=len(raw),
                            sha256=hashlib.sha256(raw).hexdigest(), data=base64.b64encode(raw).decode()))
    save('STAGE_INVENTORY.json', [{k: v for k, v in row.items() if k != 'data'} for row in payload])
    constants = dict(REPO=REPO, LOGIN=LOGIN, UUID=UUID, OUTPUT_NAME=HERE.name)
    code = ''.join(key + '=' + repr(value) + '\n' for key, value in constants.items()) + REMOTE
    compile(code, '<full-shape-component-qualification>', 'exec')
    (HERE / 'REMOTE_SOURCE.py.txt').write_text(code)
    command = 'cd ' + shlex.quote(REPO) + ' && ' + shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', code])
    ssh = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
           '-o', 'ServerAliveCountMax=3', LOGIN, command]
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(ssh, input=json.dumps(payload), capture_output=True, text=True, timeout=2300)
    save('TRANSPORT.json', dict(start_UTC=started, terminal_UTC=datetime.now(timezone.utc).isoformat(),
        exit_code=result.returncode, stderr=result.stderr, remote_source_sha256=hashlib.sha256(code.encode()).hexdigest(),
        stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(), private_key_contents_read=False))
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout); save('QUALIFICATION.json', value)
    print(json.dumps(dict(UTC=value['UTC'], all_component_checks_complete=value['all_component_checks_complete'],
        cases=[dict(family=r['family'], condition=r['condition'], stage=r['stage'],
          exit_code=r.get('exit_code'), status=r.get('status'),
          result={k: r['result'][k] for k in ('step_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes',
              'function_replay', 'next_update_replay')} if 'result' in r else dict(stderr=r.get('stderr')))
          for r in value['cases']], predictive_values_read=False), indent=2))


if __name__ == '__main__': main()
