"""Deploy the separately sealed reader and start its finite owner once."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'SAGE_context_positive_exclusion_complete_reader_source_20261011_v1'
REVIEW = PHASE / 'SAGE_context_positive_exclusion_complete_reader_independent_review_20261011_v1'
REMOTE = r'''
import base64,hashlib,json,socket,subprocess,time,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=P/'SAGE_context_positive_exclusion_pilot_root_20261011_v1'
assert not (H/'READER_JOB_START_V1.json').exists()
assert not (H/'READER_JOB_END_V1.json').exists()
assert not (H/'complete_analysis_v1').exists()
for row in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))):
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 q=P/rel;assert q.resolve().is_relative_to(P)
 data=base64.b64decode(row['data']);assert hashlib.sha256(data).hexdigest()==row['sha256']
 q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():assert q.read_bytes()==data
 else:
  with q.open('xb') as stream:stream.write(data)
with (H/'reader_watch.stdout.log').open('x') as out,(H/'reader_watch.stderr.log').open('x') as err:
 proc=subprocess.Popen([str(R/'.venv/bin/python'),'-B',str(H/'reader_after_closure_v1.py')],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
for _ in range(50):
 if (H/'READER_JOB_START_V1.json').exists() or proc.poll() is not None:break
 time.sleep(.1)
assert (H/'READER_JOB_START_V1.json').exists(),dict(code=proc.poll(),stderr=(H/'reader_watch.stderr.log').read_text()[-1800:])
value=dict(watcher_PID=proc.pid,start=json.loads((H/'READER_JOB_START_V1.json').read_text()),TEST_access=False)
with (H/'READER_WATCH_LAUNCH_V1.json').open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
print('CONTEXT_READER_LAUNCH='+json.dumps(value))
'''


def main():
    assert not (HERE / 'READER_WATCH_TRANSPORT_V1.json').exists()
    admission = json.loads((HERE / 'READER_ADMISSION_V1.json').read_text())
    assert admission['review_passed'] is True and admission['new_quality_opened'] is False
    files = [HERE / 'reader_after_closure_v1.py', HERE / 'READER_ADMISSION_V1.json']
    files += sorted(q for q in SOURCE.iterdir() if q.is_file())
    files += [PHASE / row['path'] for row in admission['bound_files']
              if row['path'].startswith(REVIEW.name + '/')]
    files = list(dict.fromkeys(files))
    payload = []
    for path in files:
        assert path.resolve().is_relative_to(PHASE.resolve())
        assert path.stat().st_size < 1000000
        data = path.read_bytes()
        payload.append(dict(path=str(path.relative_to(PHASE)),
                            sha256=hashlib.sha256(data).hexdigest(),
                            data=base64.b64encode(data).decode()))
    packed = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    remote = REMOTE.replace('PAYLOAD', repr(packed))
    args = ['ssh', '-tt', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
            '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no',
            'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
            'python3 -B -c ' + shlex.quote(remote)]
    result = subprocess.run(args, text=True, capture_output=True, timeout=55)
    with (HERE / 'READER_WATCH_TRANSPORT_V1.json').open('x') as stream:
        stream.write(json.dumps(dict(exit_code=result.returncode,
                                    stdout=result.stdout, stderr=result.stderr), indent=2) + '\n')
    assert result.returncode == 0, result.stderr[-1800:]
    line = next(x.split('=', 1)[1] for x in result.stdout.splitlines()
                if x.startswith('CONTEXT_READER_LAUNCH='))
    value = json.loads(line)
    with (HERE / 'ACTUAL_READER_WATCH_LAUNCH_V1.json').open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
