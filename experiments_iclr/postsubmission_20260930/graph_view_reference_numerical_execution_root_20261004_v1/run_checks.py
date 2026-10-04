"""Run synthetic reference implementation checks on the authorized server CPU.

No real inputs, model checkpoints, predictive values or scientific fitting.
"""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json
import shlex
import subprocess
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'accuracy_first_graph_view_reference_source_preparation_20261004_v1'
MANIFEST = '54695f4ed086ed843835a7b040e8ac82a7e211cf17e39ae06b226a23c60b5c05'
PROTOCOL = '56c96715bc84eb23a9da8cb982f23a1147db18e16787df5690b7e7163b464708'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


REMOTE = r'''
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, json, os, subprocess, sys, time, zlib
repo = Path(REPO); phase = repo / 'experiments_iclr/postsubmission_20260930'
assert Path.cwd() == repo
assert subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
    capture_output=True, text=True, check=True).stdout.splitlines() == [UUID]
payload = json.loads(zlib.decompress(base64.b64decode(sys.stdin.read(), validate=True)))
for row in payload:
    path = phase / row['path']
    assert path.resolve().is_relative_to(phase) and not path.is_symlink()
    raw = base64.b64decode(row['data'], validate=True)
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    if path.exists():
        assert path.read_bytes() == raw
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(raw)
root = phase / OUTPUT_NAME; root.mkdir(exist_ok=True)
source = phase / SOURCE_NAME
command = [str(repo / '.venv/bin/python'), '-B', str(source / 'test_numerical.py'),
           '--execute', '--manifest-sha256', MANIFEST, '--protocol-sha256', PROTOCOL]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(root))
started = time.perf_counter()
result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=120)
value = dict(UTC=datetime.now(timezone.utc).isoformat(), route=dict(login=LOGIN,
    repository=str(repo), GPU_UUID=UUID), command=command, source_manifest_sha256=MANIFEST,
    protocol_sha256=PROTOCOL, exit_code=result.returncode, stdout=result.stdout,
    stderr=result.stderr, wall_seconds=time.perf_counter() - started, runtime_device='cpu',
    real_data_read=False, predictive_values_read=False, scientific_training_updates=0,
    synthetic_implementation_checks_only=True, deliberate_fixture_root=str(root))
with (root / 'NUMERICAL_CHECKS.json').open('x') as stream:
    json.dump(value, stream, indent=2); stream.write('\n')
print(json.dumps(value))
'''


def main():
    assert sha((SOURCE / 'MANIFEST.json').read_bytes()) == MANIFEST
    assert sha((SOURCE / 'PROTOCOL.json').read_bytes()) == PROTOCOL
    manifest = json.loads((SOURCE / 'MANIFEST.json').read_text())
    files = []
    for row in manifest['payload']:
        path = SOURCE / row['path']; raw = path.read_bytes()
        assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
        files.append(path)
    files += [SOURCE / 'MANIFEST.json', SOURCE / 'SEAL.json']
    bindings = json.loads((SOURCE / 'SOURCE_BINDINGS.json').read_text())
    for row in bindings['files'].values():
        path = PHASE / row['path']; raw = path.read_bytes()
        assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
        files.append(path)
    payload = []
    for path in dict.fromkeys(files):
        assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
        raw = path.read_bytes()
        payload.append(dict(path=str(path.relative_to(PHASE)), bytes=len(raw), sha256=sha(raw),
                            data=base64.b64encode(raw).decode()))
    save('STAGE_INVENTORY.json', [{k: v for k, v in row.items() if k != 'data'} for row in payload])
    constants = dict(REPO=REPO, LOGIN=LOGIN, UUID=UUID, MANIFEST=MANIFEST, PROTOCOL=PROTOCOL,
                     OUTPUT_NAME=HERE.name, SOURCE_NAME=SOURCE.name)
    code = ''.join(key + '=' + repr(value) + '\n' for key, value in constants.items()) + REMOTE
    compile(code, '<reference-cpu-checks>', 'exec')
    (HERE / 'REMOTE_SOURCE.py.txt').write_text(code)
    command = 'cd ' + shlex.quote(REPO) + ' && ' + shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', code])
    ssh = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15', LOGIN, command]
    result = subprocess.run(ssh, input=base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode(),
                            capture_output=True, text=True, timeout=150)
    save('TRANSPORT.json', dict(UTC=datetime.now(timezone.utc).isoformat(), exit_code=result.returncode,
         stderr=result.stderr, stdout_sha256=sha(result.stdout.encode()), remote_source_sha256=sha(code.encode()),
         private_key_contents_read=False))
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout); save('NUMERICAL_CHECKS.json', value)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
