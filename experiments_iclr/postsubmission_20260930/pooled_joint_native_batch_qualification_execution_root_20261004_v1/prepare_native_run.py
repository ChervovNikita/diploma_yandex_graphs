"""Stage and release the reviewed fabricated-native check in the project only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import sys
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'pooled_joint_native_batch_qualification_preparation_20261004_v1'
RELAY = PHASE / 'gpu77_connection_recovery_v1'
WRAPPER = RELAY / 'run_gpu77_v3.py'
REMOTE_REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REMOTE_REPO / 'experiments_iclr/postsubmission_20260930'
REMOTE = REMOTE_PHASE / HERE.name
GPU = 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'
INTERPRETER = '/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def run(identity, code):
    command = 'cd ' + shlex.quote(str(REMOTE_REPO)) + ' && /usr/bin/python3 -I -S -B - <<\'ROOTPY\'\n' + code + '\nROOTPY\n'
    assert len(command.encode()) < 100000
    destination = RELAY / (identity + '.txt')
    with destination.open('x') as f:
        f.write(command)
    result = subprocess.run([sys.executable, '-B', str(WRAPPER), '--id', identity,
                             '--command-file', str(destination)], capture_output=True, text=True)
    write(HERE / (identity + '_LOCAL_TRANSPORT.json'), dict(exit_code=result.returncode,
            stdout=result.stdout, stderr=result.stderr, command_sha256=hashlib.sha256(command.encode()).hexdigest()))
    assert result.returncode == 0, result.stderr + result.stdout[:1000]
    wrapper = json.loads(result.stdout)
    assert wrapper['exit_code'] == 0
    return json.loads(wrapper['stdout'])


def stage():
    review = json.loads((HERE / 'ROOT_SOURCE_REVIEW.json').read_text())
    assert review['status'] == 'PASS_SOURCE_FOR_NATIVE_ONLY'
    assert digest(SOURCE / 'MANIFEST.json') == review['preparation_manifest_sha256']
    paths = [SOURCE / 'MANIFEST.json', HERE / 'ROOT_SOURCE_REVIEW.json']
    paths += [SOURCE / row['path'] for row in json.loads((SOURCE / 'MANIFEST.json').read_text())['files']]
    inventory = json.loads((HERE / 'SOURCE_STAGE_INVENTORY.json').read_text())
    payload = []
    for p in paths:
        b = p.read_bytes()
        payload.append(dict(path=str(p.relative_to(PHASE)), bytes=len(b),
                            sha256=hashlib.sha256(b).hexdigest(), data=base64.b64encode(b).decode()))
    packed = base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()
    code = "from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,subprocess,zlib\n"
    code += 'repo=Path(' + repr(str(REMOTE_REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ')\n'
    code += 'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
    code += 'payload=json.loads(zlib.decompress(base64.b64decode(' + repr(packed) + ')))\n'
    code += "for row in payload:\n p=(phase/row['path']).resolve();assert p.is_relative_to(phase);b=base64.b64decode(row['data'],validate=True);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']\n if p.exists():assert not p.is_symlink() and p.read_bytes()==b\n else:\n  p.parent.mkdir(parents=True,exist_ok=True)\n  with p.open('xb') as f:f.write(b)\n"
    code += 'inventory=' + repr(inventory) + '\n'
    code += "for row in inventory:\n p=(phase/row['path']).resolve();assert p.is_relative_to(phase) and p.is_file() and not p.is_symlink();b=p.read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],str(p)\n"
    code += 'interpreter=Path(' + repr(INTERPRETER) + ')\n'
    code += "assert hashlib.sha256(interpreter.read_bytes()).hexdigest()=='14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'\n"
    code += "g=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);assert g.returncode==0\n"
    code += "rows=[[x.strip() for x in line.split(',')] for line in g.stdout.splitlines()];assert {x[0] for x in rows}=={'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}\n"
    code += 'selected=next(x for x in rows if x[0]==' + repr(GPU) + ');assert int(selected[1])>=8192\n'
    code += "available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'));assert available>=16*1024**3\n"
    code += "print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status='PASS_SOURCE_AND_RESOURCE_METADATA',source_files_verified=len(inventory),staged_files=len(payload),GPU_rows=rows,host_MemAvailable_bytes=available,scientific_data_or_state_read=False)))\n"
    result = run('pooled_native_source_stage_20261004_v2', code)
    write(HERE / 'SOURCE_RESOURCE_STAGE_RECEIPT.json', result)
    print(json.dumps(result))


def release():
    observation = json.loads((HERE / 'SOURCE_RESOURCE_STAGE_RECEIPT.json').read_text())
    assert observation['status'] == 'PASS_SOURCE_AND_RESOURCE_METADATA'
    admission = dict(schema='root-pooled-fabricated-native-admission-v1', UTC=datetime.now(timezone.utc).isoformat(),
                     status='PASS', observation=observation, minimum_GPU_free_MiB=8192,
                     minimum_host_MemAvailable_bytes=16*1024**3, dispatch_recheck_required=True,
                     root_source_review_sha256=digest(HERE / 'ROOT_SOURCE_REVIEW.json'),
                     preparation_manifest_sha256=digest(SOURCE / 'MANIFEST.json'),
                     scope='Fabricated native engineering only; no data, VALID, TEST, fit or donor state.')
    write(HERE / 'ROOT_NATIVE_ADMISSION.json', admission)
    value = json.loads((SOURCE / 'ROOT_RELEASE_NATIVE.example.json').read_text())
    value.update(execution_enabled=True, root_authorization_reference=str(REMOTE / 'ROOT_NATIVE_ADMISSION.json'),
                 preparation_manifest_sha256=digest(SOURCE / 'MANIFEST.json'), cuda_visible_devices=GPU)
    write(HERE / 'ROOT_NATIVE_RELEASE.json', value)
    rows = []
    for name in ('ROOT_NATIVE_ADMISSION.json', 'ROOT_NATIVE_RELEASE.json'):
        p = HERE / name; b = p.read_bytes()
        rows.append(dict(path=str((REMOTE / name)), bytes=len(b), sha256=digest(p), data=base64.b64encode(b).decode()))
    invocation = value['authorized_invocations'][0]
    output = Path(invocation['output_directory'])
    env = dict(CUDA_VISIBLE_DEVICES=GPU, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
               PYTHONPATH=str(REMOTE_REPO / '.gnnm_runtime/buddy_extra_v1/site'))
    command = [INTERPRETER, '-B', str(REMOTE_PHASE / SOURCE.name / 'qualify.py'), '--stage', 'native',
               '--release', str(REMOTE / 'ROOT_NATIVE_RELEASE.json'), '--output', str(output)]
    code = "from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,subprocess,time\n"
    code += 'repo=Path(' + repr(str(REMOTE_REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ');root=Path(' + repr(str(REMOTE)) + ')\n'
    code += 'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
    code += 'rows=' + repr(rows) + '\n'
    code += "for row in rows:\n p=Path(row['path']);assert p.resolve().is_relative_to(phase);b=base64.b64decode(row['data'],validate=True);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']\n with p.open('xb') as f:f.write(b)\n"
    code += "g=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);assert g.returncode==0\n"
    code += 'gpu=' + repr(GPU) + ';gpu_rows=[[v.strip() for v in l.split(",")] for l in g.stdout.splitlines()];assert int(next(r[1] for r in gpu_rows if r[0]==gpu))>=8192\n'
    code += "available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'));assert available>=16*1024**3\n"
    code += 'output=Path(' + repr(str(output)) + ');assert not output.exists();output.parent.mkdir(parents=True,exist_ok=True)\n'
    code += 'command=' + repr(command) + ';env=dict(os.environ,**' + repr(env) + ')\n'
    code += "with (root/'NATIVE_SUPERVISOR_STDOUT.txt').open('xb') as out,(root/'NATIVE_SUPERVISOR_STDERR.txt').open('xb') as err:\n proc=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\n"
    code += "raw=(Path('/proc')/str(proc.pid)/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()\n"
    code += "result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='NATIVE_QUALIFICATION_LAUNCHED',supervisor_PID=proc.pid,supervisor_start_ticks=int(fields[19]),command=command,environment={k:env[k] for k in " + repr(list(env)) + "},GPU_rows=gpu_rows,host_MemAvailable_bytes=available,output=str(output),fits=0,VALID_TEST_access=False)\n"
    code += "with (root/'NATIVE_DETACHED_LAUNCH.json').open('x') as f:json.dump(result,f,indent=2);f.write('\\n')\nprint(json.dumps(result))\n"
    result = run('pooled_native_qualification_launch_20261004_v1', code)
    write(HERE / 'NATIVE_DETACHED_LAUNCH.json', result)
    print(json.dumps({k: result[k] for k in ('status','supervisor_PID','supervisor_start_ticks','output','fits','VALID_TEST_access')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('stage','release'))
    args = parser.parse_args()
    (stage if args.stage == 'stage' else release)()
