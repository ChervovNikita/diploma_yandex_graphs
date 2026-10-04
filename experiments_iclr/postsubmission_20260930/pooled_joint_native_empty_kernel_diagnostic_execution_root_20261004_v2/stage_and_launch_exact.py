"""Release only the synchronized exact native diagnostic on fabricated inputs."""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json
import shlex
import subprocess
import sys
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
RELAY = PHASE / 'gpu77_connection_recovery_v1'
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
REMOTE = REMOTE_PHASE / HERE.name
SOURCE = PHASE / 'pooled_joint_native_empty_kernel_diagnostic_preparation_20261004_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def run(identity, code):
    command = 'cd ' + shlex.quote(str(REPO)) + " && /usr/bin/python3 -I -S -B - <<'DIAGROOTPY'\n" + code + '\nDIAGROOTPY\n'
    assert len(command.encode()) < 100000
    path = RELAY / (identity + '.txt')
    with path.open('x') as stream:
        stream.write(command)
    result = subprocess.run([sys.executable, '-B', str(RELAY / 'run_gpu77_v3.py'), '--id', identity,
                             '--command-file', str(path)], capture_output=True, text=True)
    save(HERE / (identity + '_LOCAL_TRANSPORT.json'), dict(exit_code=result.returncode, stdout=result.stdout,
          stderr=result.stderr, command_sha256=hashlib.sha256(command.encode()).hexdigest()))
    assert result.returncode == 0, result.stderr + result.stdout[:1200]
    outer = json.loads(result.stdout)
    assert outer['exit_code'] == 0, outer
    return json.loads(outer['stdout'])


manifest = json.loads((SOURCE / 'MANIFEST.json').read_text())
assert sha(SOURCE / 'MANIFEST.json') == '0e75b301e548704aee01cf42a3bfb8f2af5fe944bbe69ea6b50738d47f74da83'
for row in manifest['files']:
    local = SOURCE / row['path']
    assert sha(local) == row['sha256'] and local.stat().st_size == row['bytes']
files = [SOURCE / row['path'] for row in manifest['files']] + [SOURCE / 'MANIFEST.json']
bindings = json.loads((SOURCE / 'SOURCE_BINDINGS.json').read_text())
for row in bindings['files']:
    local = PHASE / row['relative_path']
    assert sha(local) == row['sha256'] and local.stat().st_size == row['bytes']
    if True:  # Every bound source/failure descriptor is a small metadata/code file.
        files.append(local)
payload = []
for local in files:
    raw = local.read_bytes()
    payload.append(dict(path=str(local.relative_to(PHASE)), bytes=len(raw), sha256=sha(local),
                        data=base64.b64encode(raw).decode()))
save(HERE / 'SOURCE_STAGE_INVENTORY.json', [{k: v for k, v in row.items() if k != 'data'} for row in payload])
compressed = zlib.compress(json.dumps(payload).encode(), 9)
encoded = base64.b64encode(compressed).decode()
chunks = [encoded[start:start + 40000] for start in range(0, len(encoded), 40000)]
for number, chunk in enumerate(chunks, 1):
    code = 'from pathlib import Path\nimport hashlib,json,os\n'
    code += 'repo=Path(' + repr(str(REPO)) + ');root=Path(' + repr(str(REMOTE / 'source_chunks')) + ')\n'
    code += 'assert Path.cwd()==repo and os.uname().nodename=="peptide";root.mkdir(parents=True,exist_ok=True)\n'
    code += 'raw=' + repr(chunk) + '.encode()\nwith (root/' + repr('part%03d.b64' % number) + ').open("xb") as stream:stream.write(raw)\n'
    code += 'print(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))\n'
    result = run('pooled_native_exact_diag_v2_source_chunk%03d_20261004' % number, code)
    assert result['sha256'] == hashlib.sha256(chunk.encode()).hexdigest()
code = 'from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,zlib\n'
code += 'repo=Path(' + repr(str(REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ');root=Path(' + repr(str(REMOTE / 'source_chunks')) + ')\n'
code += 'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
code += 'compressed=base64.b64decode(b"".join((root/("part%03d.b64"%n)).read_bytes() for n in range(1,' + str(len(chunks) + 1) + ')),validate=True)\n'
code += 'assert hashlib.sha256(compressed).hexdigest()==' + repr(hashlib.sha256(compressed).hexdigest()) + '\nrows=json.loads(zlib.decompress(compressed))\n'
code += 'for row in rows:\n path=phase/row["path"];assert path.resolve().is_relative_to(phase);raw=base64.b64decode(row["data"],validate=True);assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]\n if path.exists():assert not path.is_symlink() and path.read_bytes()==raw\n else:\n  path.parent.mkdir(parents=True,exist_ok=True)\n  with path.open("xb") as stream:stream.write(raw)\n'
code += 'print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status="SEALED_DIAGNOSTIC_AND_PRIOR_FAILURE_STAGED",files=len(rows),bytes=sum(r["bytes"] for r in rows),numerical_execution=False)))\n'
save(HERE / 'SOURCE_STAGE_RECEIPT.json', run('pooled_native_exact_diag_v2_source_join_20261004', code))
release = json.loads((SOURCE / 'ROOT_RELEASE.example.json').read_text())
release.update(execution_enabled=True, root_authorization_reference=str(REMOTE / 'ROOT_EXACT_DIAGNOSTIC_ADMISSION.json'),
               preparation_manifest_sha256=sha(SOURCE / 'MANIFEST.json'))
release['authorized_invocations'] = [row for row in release['authorized_invocations'] if row['case'] == 'exact']
admission = dict(UTC=datetime.now(timezone.utc).isoformat(), status='APPROVED_EXACT_DIAGNOSTIC_ONLY',
                 source_manifest_sha256=sha(SOURCE / 'MANIFEST.json'),
                 root_review='Read unchanged direct-native delegation, synchronized same-overload observers, failure/source custody and owned bounded child.',
                 preserved_prior_failure=True, metadata_only_nonfinite_float_tag_repair=True, prior_observer_failure_is_not_kernel_attribution=True, fits=0, data_VALID_TEST_old_checkpoint_access=False,
                 native_qualification_pass_allowed=False, automatic_retry=False)
save(HERE / 'ROOT_EXACT_DIAGNOSTIC_ADMISSION.json', admission)
save(HERE / 'ROOT_RELEASE_exact.json', release)
payload = []
for local in (HERE / 'ROOT_EXACT_DIAGNOSTIC_ADMISSION.json', HERE / 'ROOT_RELEASE_exact.json'):
    payload.append(dict(path=str(REMOTE / local.name), sha256=sha(local), data=base64.b64encode(local.read_bytes()).decode()))
runtime = json.loads((PHASE / 'graph_ncNC_predictive_runtime_authority_root_20261003_v1/RUNTIME_AUTHORITY.json').read_text())
environment = dict(CUDA_VISIBLE_DEVICES=release['cuda_visible_devices'], OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(REPO / '.gnnm_runtime/buddy_extra_v1/site'))
cmd = [runtime['interpreter_path'], '-B', str(REMOTE_PHASE / SOURCE.name / 'diagnose.py'), '--case', 'exact',
       '--release', str(REMOTE / 'ROOT_RELEASE_exact.json'), '--output', release['authorized_invocations'][0]['output_directory']]
code = 'from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,subprocess\n'
code += 'repo=Path(' + repr(str(REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ');root=Path(' + repr(str(REMOTE)) + ')\n'
code += 'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
code += 'query=subprocess.run(["nvidia-smi","--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=15)\n'
code += 'gpu_rows=[[v.strip() for v in line.split(",")] for line in query.stdout.splitlines()];assert int(next(row[1] for row in gpu_rows if row[0]==' + repr(release['cuda_visible_devices']) + '))>=4096\n'
code += 'rows=' + repr(payload) + '\nfor row in rows:\n raw=base64.b64decode(row["data"],validate=True);assert hashlib.sha256(raw).hexdigest()==row["sha256"]\n with Path(row["path"]).open("xb") as stream:stream.write(raw)\n'
code += 'environment=' + repr(environment) + ';command=' + repr(cmd) + '\n'
code += 'with (root/"DETACHED.stdout").open("xb") as out,(root/"DETACHED.stderr").open("xb") as err:\n child=subprocess.Popen(command,cwd=repo,env=dict(os.environ,**environment),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\n'
code += 'raw=(Path("/proc")/str(child.pid)/"stat").read_text();fields=raw[raw.rfind(")")+2:].split()\n'
code += 'result=dict(UTC=datetime.now(timezone.utc).isoformat(),status="EXACT_FABRICATED_DIAGNOSTIC_LAUNCHED",supervisor_PID=child.pid,supervisor_start_ticks=int(fields[19]),command=command,environment=environment,GPU_rows=gpu_rows,fits=0,TEST_access=False)\n'
code += 'with (root/"DETACHED_LAUNCH.json").open("x") as stream:json.dump(result,stream,indent=2);stream.write("\\n")\nprint(json.dumps(result))\n'
save(HERE / 'DETACHED_LAUNCH.json', run('pooled_native_exact_diag_v2_launch_20261004', code))
print('Exact diagnostic launched; native qualification remains unpassed.')
