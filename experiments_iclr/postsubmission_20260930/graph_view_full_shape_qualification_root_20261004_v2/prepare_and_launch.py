"""Preserve failed replay; test a fixed deterministic runtime in fresh processes.

No scientific fitting, predictive decisions, existing-job changes or retries.
"""
from datetime import datetime, timezone
from pathlib import Path
import base64
import difflib
import hashlib
import importlib.util
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
OLD = PHASE / 'graph_view_full_shape_qualification_root_20261004_v1'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def main():
    old_result = json.loads((OLD / 'QUALIFICATION.json').read_text())
    assert old_result['all_component_checks_complete']
    assert all(r['result']['function_replay']['close'] for r in old_result['cases'])
    policy = dict(UTC=datetime.now(timezone.utc).isoformat(),
        torch_use_deterministic_algorithms=True, warn_only=False,
        CUBLAS_WORKSPACE_CONFIG=':4096:8', cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False,
        scope='Fresh component QA only. Future scientific protocol must explicitly adopt and charge this policy for every arm.',
        existing_scientific_families_changed=False, source_and_loss_algorithms_changed=False,
        relaxed_replay_thresholds=False, predictive_values_read=False,
        previous_transport_reset=True, previous_checks_completed_without_retry=True,
        previous_qualification_sha256=hashlib.sha256((OLD/'QUALIFICATION.json').read_bytes()).hexdigest())
    save('DETERMINISTIC_RUNTIME_PROSPECTIVE_POLICY.json', policy)
    original = (OLD / 'check_one.py').read_text()
    needle = "    assert torch.cuda.device_count() == 1\n"
    assert original.count(needle) == 1
    successor = original.replace(needle, needle +
        "    torch.use_deterministic_algorithms(True, warn_only=False)\n"
        "    torch.backends.cuda.matmul.allow_tf32 = False\n"
        "    torch.backends.cudnn.allow_tf32 = False\n")
    successor = successor.replace("        parameters=sum(p.numel() for p in model.parameters()),",
        "        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),\n"
        "        parameters=sum(p.numel() for p in model.parameters()),")
    compile(successor, str(HERE/'check_one.py'), 'exec')
    with (HERE/'check_one.py').open('x') as stream: stream.write(successor)
    with (HERE/'DIFF_V1_V2.patch').open('x') as stream:
        stream.writelines(difflib.unified_diff(original.splitlines(keepends=True), successor.splitlines(keepends=True),
            fromfile='v1/check_one.py', tofile='v2/check_one.py'))
    spec = importlib.util.spec_from_file_location('retained_full_shape_qa_client_v1', OLD/'run_checks.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    body = old.REMOTE.replace('payload = json.loads(sys.stdin.read())', 'payload = []')
    body = body.replace("    PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(root))",
                        "    PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(root), CUBLAS_WORKSPACE_CONFIG=':4096:8')")
    assert "CUBLAS_WORKSPACE_CONFIG=':4096:8'" in body
    constants = dict(REPO=REPO, LOGIN=LOGIN, UUID=UUID, OUTPUT_NAME=HERE.name)
    supervisor = ''.join(key+'='+repr(value)+'\n' for key,value in constants.items()) + body
    compile(supervisor, str(HERE/'supervisor.py'), 'exec')
    with (HERE/'supervisor.py').open('x') as stream: stream.write(supervisor)
    with (HERE/'INPUT_BINDINGS.json').open('xb') as stream: stream.write((OLD/'INPUT_BINDINGS.json').read_bytes())
    payload = []
    for name in ('check_one.py','supervisor.py','INPUT_BINDINGS.json','DETERMINISTIC_RUNTIME_PROSPECTIVE_POLICY.json'):
        raw=(HERE/name).read_bytes()
        payload.append(dict(name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                            data=base64.b64encode(raw).decode()))
    save('STAGE_INVENTORY.json', [{k:v for k,v in row.items() if k!='data'} for row in payload])
    code = ''.join(key+'='+repr(value)+'\n' for key,value in constants.items()) + r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess,sys
repo=Path(REPO);phase=repo/'experiments_iclr/postsubmission_20260930';root=phase/OUTPUT_NAME
assert Path.cwd()==repo
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
    capture_output=True,text=True,check=True).stdout.splitlines()==[UUID]
rows=json.loads(sys.stdin.read());root.mkdir(exist_ok=True)
for row in rows:
    path=root/row['name'];assert path.parent==root and not path.is_symlink()
    raw=base64.b64decode(row['data'],validate=True)
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    if path.exists():assert path.read_bytes()==raw
    else:
        with path.open('xb') as stream:stream.write(raw)
assert not (root/'DETACHED_LAUNCH.json').exists() and not (root/'QUALIFICATION.json').exists()
command=['/usr/bin/python3','-I','-S','-B',str(root/'supervisor.py')]
with (root/'SUPERVISOR.stdout').open('xb') as out,(root/'SUPERVISOR.stderr').open('xb') as err:
    child=subprocess.Popen(command,cwd=repo,env=os.environ.copy(),stdin=subprocess.DEVNULL,
        stdout=out,stderr=err,start_new_session=True)
raw=(Path('/proc')/str(child.pid)/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
value=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,start_ticks=int(fields[19]),
    group=int(fields[2]),session=int(fields[3]),command=command,
    scientific_training_updates=0,predictive_values_read=False,automatic_retry=False)
with (root/'DETACHED_LAUNCH.json').open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(value))
'''
    compile(code, '<deterministic-qa-launch>', 'exec')
    with (HERE/'REMOTE_STAGE_AND_LAUNCH.py.txt').open('x') as stream: stream.write(code)
    command='cd '+shlex.quote(REPO)+' && '+shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code])
    ssh=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
         '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no',
         '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',LOGIN,command]
    result=subprocess.run(ssh,input=json.dumps(payload),capture_output=True,text=True,timeout=30)
    save('LAUNCH_TRANSPORT.json',dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,
         stderr=result.stderr,stdout=result.stdout,private_key_contents_read=False))
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout);save('DETACHED_LAUNCH.json',value)
    print(json.dumps(value))


if __name__=='__main__':main()
