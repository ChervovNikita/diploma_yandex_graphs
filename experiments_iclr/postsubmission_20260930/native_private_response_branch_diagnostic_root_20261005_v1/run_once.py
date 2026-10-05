"""Run one frozen synthetic functional-branch diagnosis."""
from pathlib import Path
from datetime import datetime,timezone
import base64
import hashlib
import importlib.util
import json
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
PREP=PHASE/'native_private_response_branch_diagnostic_preparation_20261005_v1'
HELPER=PHASE/'learnability_responsibility_native_synthetic_execution_root_20261005_v1/run_once.py'
MANIFEST_SHA='4b98ee570b61b0af56e0daee9824c36d96dd9bc58fae02e8fef36e07ac57e864'
WORKER_SHA='e985bb5b6d572889f6e7859a93b93aee7c900f9436173c32bfa427a5e3cb420b'

def main():
    assert not (HERE/'EXECUTION_RELEASE.json').exists()
    assert hashlib.sha256((PREP/'MANIFEST.json').read_bytes()).hexdigest()==MANIFEST_SHA
    manifest=json.loads((PREP/'MANIFEST.json').read_text())
    for row in manifest['files']:
        data=(PREP/row['path']).read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    spec=importlib.util.spec_from_file_location('root_synthetic_transport',HELPER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    remote=helper.REMOTE
    old="worker=phase/'learnability_responsibility_native_numerical_worker_preparation_20261005_v1/qualify.py'"
    new="worker=phase/'native_private_response_branch_diagnostic_preparation_20261005_v1/diagnose.py'"
    assert remote.count(old)==1;remote=remote.replace(old,new)
    old="argv=[str(runtime),'-B',str(worker),'--execute-authorized','--mode','synthetic','--source-root',str(phase),'--output',str(out/'output')]"
    new="argv=[str(runtime),'-B',str(worker),'--execute-authorized','--output',str(out/'output')]"
    assert remote.count(old)==1;remote=remote.replace(old,new)
    release={'UTC':datetime.now(timezone.utc).isoformat(),'purpose':'One fixed synthetic branch diagnosis, not qualification',
             'worker_sha256':WORKER_SHA,'manifest_sha256':MANIFEST_SHA,
             'transport_helper_sha256':hashlib.sha256(HELPER.read_bytes()).hexdigest(),
             'remote_wrapper_sha256':hashlib.sha256(remote.encode()).hexdigest(),
             'original_failed_qualification_preserved':True,'original_tolerances_unchanged':True,
             'scientific_fit_or_predictive_or_qualification_pass_authorized':False}
    (HERE/'EXECUTION_RELEASE.json').write_text(json.dumps(release,indent=2)+'\n')
    files=[]
    for path in sorted(PREP.iterdir()):
        assert path.is_file() and not path.is_symlink()
        b=path.read_bytes()
        files.append({'path':str(path.relative_to(PHASE)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()})
    request={'files':files,'packet_manifests':{PREP.name:MANIFEST_SHA},'worker_sha256':WORKER_SHA,'run_name':HERE.name,'release':release}
    command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
             '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
             '-o','UpdateHostKeys=no','-o','ConnectTimeout=20',helper.LOGIN,
             'cd '+shlex.quote(helper.REPO)+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(remote)]
    child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    child.stdin.write(json.dumps(request));child.stdin.close();transcript=[]
    for line in child.stdout:
        transcript.append(line);item=json.loads(line)
        if item['stage']=='launched':
            (HERE/'LAUNCH_RECEIPT.json').write_text(json.dumps(item['owned_process'],indent=2)+'\n')
            print(json.dumps({'stage':'launched','PID':item['owned_process']['PID'],'start_ticks':item['owned_process']['start_ticks']}),flush=True)
        else:
            (HERE/'TERMINAL.json').write_text(json.dumps(item['terminal'],indent=2)+'\n')
            if item['result'] is not None:(HERE/'RESULT.json').write_text(json.dumps(item['result'],indent=2)+'\n')
            print(json.dumps({'stage':'terminal','exit_code':item['terminal']['exit_code'],'status':item['result'].get('status') if item['result'] else None}),flush=True)
    stderr=child.stderr.read();exit_code=child.wait()
    (HERE/'TRANSPORT_RECEIPT.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':exit_code,'stdout_bytes':len(''.join(transcript).encode()),'stdout_sha256':hashlib.sha256(''.join(transcript).encode()).hexdigest(),'stderr':stderr},indent=2)+'\n')
    raise SystemExit(exit_code)

if __name__=='__main__':main()
