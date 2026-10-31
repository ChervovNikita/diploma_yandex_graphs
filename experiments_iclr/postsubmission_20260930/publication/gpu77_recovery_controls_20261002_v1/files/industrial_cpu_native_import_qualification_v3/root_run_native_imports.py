"""UNEXECUTED fixed-route source-only CPU native-import qualification wrapper."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid

HERE = Path(__file__).resolve().parent
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
SSH = ['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no',
       '-o','StrictHostKeyChecking=yes',LOGIN]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', help='Fresh single-use qualification identity.')
    args = parser.parse_args()
    run_id = args.run_id or ('CPU_imports_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:12])
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}',run_id):
        raise ValueError('Simple fresh identity required')
    manifest = json.loads((HERE/'MANIFEST.json').read_text())
    seal = json.loads((HERE/'SEAL.json').read_text())
    manifest_sha = hashlib.sha256((HERE/'MANIFEST.json').read_bytes()).hexdigest()
    if seal['manifest']['sha256'] != manifest_sha:
        raise RuntimeError('Source manifest differs from seal')
    for item in manifest['payload']:
        path = HERE/Path(item['path']).name
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Reviewed source preparation changed: '+path.name)
    files = {name:(HERE/name).read_text() for name in ('proven_cpu_policy.py','native_import_worker.py','root_outer.py','IMPORT_CONTRACT.json')}
    pins = {name:hashlib.sha256(value.encode()).hexdigest() for name,value in files.items()}
    wrapper_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bundle = {'run_id':run_id,'files':files,'pins':pins,'wrapper_sha256':wrapper_sha}
    contract = json.loads(files['IMPORT_CONTRACT.json'])
    output = HERE/'root_runs'/run_id
    if output.parent.is_symlink():
        raise RuntimeError('Actual local receipt directory required')
    output.parent.mkdir(parents=True,exist_ok=True); output.mkdir(mode=0o700)
    command = shlex.join(['/usr/bin/python3','-I','-S','-B','-c',files['root_outer.py'],json.dumps(bundle)])
    launch = {'schema':'local-root-CPU-native-import-launch-v3','run_id':run_id,'ssh_destination':LOGIN,
              'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_pins':pins,
              'wrapper_sha256':wrapper_sha,'manifest_sha256':manifest_sha,'automatic_retry':False,
              'dataset_load_model_construction_or_fit_requested':False,'GPU_compute_requested':False,
              'public_GPU_inventory_preflight':True,'transport_cap_seconds':contract['transport_cap_seconds']}
    started = time.monotonic(); proc = None
    try:
        proc = subprocess.run([*SSH,command],stdin=subprocess.DEVNULL,text=True,capture_output=True,
                              timeout=contract['transport_cap_seconds'],check=False)
        (output/'stdout.txt').write_text(proc.stdout); (output/'stderr.txt').write_text(proc.stderr)
        try: receipt = json.loads(proc.stdout)
        except ValueError: receipt = None
        if receipt is not None:
            (output/'NATIVE_IMPORT_CAPABILITY.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        launch.update(exit_code=proc.returncode,status='COMPLETED_CPU_IMPORT_DIAGNOSTIC_ONLY' if proc.returncode==0 else 'REFUSED_OR_FAILED_DIAGNOSTIC')
    except BaseException as error:
        launch.update(status='LOCAL_TRANSPORT_FAILED',error_type=type(error).__name__,error=str(error))
        if isinstance(error,subprocess.TimeoutExpired):
            for name,value in [('stdout.txt',error.stdout),('stderr.txt',error.stderr)]:
                if value is not None:
                    (output/name).write_bytes(value if isinstance(value,bytes) else value.encode())
    launch['seconds'] = time.monotonic()-started
    (output/'ROOT_LAUNCH.json').write_text(json.dumps(launch,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'receipt':str(output/'ROOT_LAUNCH.json'),'status':launch['status']},sort_keys=True))
    return 0 if proc is not None and proc.returncode == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
