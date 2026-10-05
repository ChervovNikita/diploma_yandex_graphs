#!/usr/bin/env python3
"""Stage receipts and detach the single reviewed resource supervisor."""
from datetime import datetime, timezone
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
supervisor = (HERE / 'supervisor.py').read_text()
ast.parse(supervisor)
supervisor_sha = hashlib.sha256(supervisor.encode()).hexdigest()
remote = r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess,sys
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
OUTPUT=PHASE/'pencil_citeseer_zero_update_resource_execution_20261005_v1'
PLAN=PHASE/'pencil_citeseer_bounded_zero_update_resource_plan_20261005_v1'
DEPS=PHASE/'pencil_citeseer_missing_ancillary_resolver_execution_20261005_v1'
CPU=PHASE/'pencil_citeseer_cpu_import_qualification_20261005_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload=PAYLOAD_LITERAL
assert Path.cwd().resolve()==REPO and socket.gethostname()=='anogena-2-0'
uuids=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=30)
assert uuids.stdout.split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert not OUTPUT.exists()
assert sha(CPU/'RESULT.json')=='759ca8413dc8ec1e13dfcbef16cbe388e29b3f9cd6e472cdc9bf4459453c87ac'
assert sha(DEPS/'INSTALLED_FILE_INVENTORY.json')=='339dc97e7a1b5cb8280317c48e18b1749a4c7c065ce7d682043dd75b3dfd63bf'
assert sha(DEPS/'INSTALL_EXECUTION_RECEIPT.json')=='ae7147efcd438c1f710fa8a51cb7233a92321dd88c1de016f6266fbd88001335'
cpu=json.loads((CPU/'RESULT.json').read_text())
installed=json.loads((DEPS/'INSTALLED_FILE_INVENTORY.json').read_text())
metadata=json.loads((DEPS/'POSTINSTALL_SELECTED_METADATA.json').read_text())
assert cpu['status']=='PASS_EXACT_NATIVE_CPU_IMPORTS_ONLY' and cpu['original61_provider_metadata_unchanged'] is True
assert len(installed['files'])==7620 and len(metadata)==83
root=Path(installed['root'])
assert root==PHASE/'pencil_one_gpu_dependency_overlay_20261005_v1'
dependency=dict(schema='one_gpu_pencil_dependency_admission_v1',host='anogena-2-0',
 one_gpu_native_pencil_import_pass=True,
 qualification_scope='Exact CPU imports only; no GPU, constructor, fit or optimizer readiness follows',
 PYTHONPATH=cpu['child_environment']['PYTHONPATH'],installed_file_inventory_complete=True,
 runtime_distribution_versions={k:v['version'] for k,v in metadata.items()},
 dependency_files=[dict(path=str(root/r['path']),sha256=r['sha256']) for r in installed['files']],
 import_source_pins=cpu['added_package_import_source_pins'],
 CPU_import_RESULT_sha256=sha(CPU/'RESULT.json'),
 installed_inventory_sha256=sha(DEPS/'INSTALLED_FILE_INVENTORY.json'),
 postinstall_metadata_sha256=sha(DEPS/'POSTINSTALL_SELECTED_METADATA.json'),
 original61_provider_metadata_unchanged=True)
binding=PLAN/'PENCIL_ONE_GPU_DEPENDENCY_BINDING.json'
release_path=PLAN/'ROOT_RESOURCE_RELEASE.json'
assert not binding.exists() and not release_path.exists()
OUTPUT.mkdir()
binding.write_text(json.dumps(dependency,indent=2,sort_keys=True)+'\n')
release=dict(schema='pencil_citeseer_zero_update_resource_release_v1',
 root_source_review_approved=True,full_zero_update_resource_probe_authorized=True,
 preserve_active_ncn_cohort=True,
 source_manifest_sha256='f1a5cae24d746f32360bafe7dcb1eb7557356a05cbb53b28d41a28dcd93894a3',
 dependency_binding_phase_relative=str(binding.relative_to(PHASE)),dependency_binding_sha256=sha(binding),
 caps=dict(wall_seconds=3600,max_parent_plus_live_descendant_RSS_bytes=128*1024**3,
 max_cuda_allocated_bytes=24*1024**3,max_cuda_reserved_bytes=28*1024**3),
 authorization_scope='One complete native PENCIL epoch0 TRAIN backward plus complete VALID traversal; zero optimizer updates; no scores or TEST; after exact36-fit cohort freeze and immediate host/GPU/memory gate',
 source_bytes_unchanged=True,scientific_fit_authorized=False,no_retry=True)
release_path.write_text(json.dumps(release,indent=2,sort_keys=True)+'\n')
source=base64.b64decode(payload['supervisor_base64'])
assert hashlib.sha256(source).hexdigest()==payload['supervisor_sha256']
with (OUTPUT/'supervisor.py').open('xb') as f:f.write(source)
authorization=dict(root_authorized=True,UTC=datetime.now(timezone.utc).isoformat(),
 approved_source_sha256='36a4ccef3f5973ce0583ecbb060672371cb171513b99b48cebb3ef50177ecf51',
 supervisor_sha256=payload['supervisor_sha256'],scope=release['authorization_scope'],
 comparative_outcome_files_opened=False,other_jobs_preserved=True,automatic_retry=False)
(OUTPUT/'AUTHORIZATION.json').write_text(json.dumps(authorization,indent=2,sort_keys=True)+'\n')
env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1';env.pop('PYTHONPATH',None)
log=(OUTPUT/'supervisor_stdout_stderr.log').open('xb')
child=subprocess.Popen([sys.executable,str(OUTPUT/'supervisor.py')],cwd=REPO,env=env,
 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
log.close()
raw=(Path('/proc')/str(child.pid)/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
launch=dict(UTC=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),cwd=str(Path.cwd()),
 supervisor_pid=child.pid,supervisor_pgid=int(fields[2]),supervisor_sid=int(fields[3]),
 supervisor_start_ticks=int(fields[19]),supervisor_sha256=payload['supervisor_sha256'],
 release_sha256=sha(release_path),dependency_binding_sha256=sha(binding),
 dependency_binding_bytes=binding.stat().st_size,dependency_inventory_server_only=True,
 output_directory=str(OUTPUT),launch_count=1,scientific_fits=0,optimizer_updates=0,
 detached=True,other_jobs_signalled=False,comparative_outcomes_opened=False)
(OUTPUT/'LAUNCH_RECEIPT.json').write_text(json.dumps(launch,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(launch=launch,release=release,authorization=authorization),sort_keys=True))
'''
payload = dict(supervisor_base64=base64.b64encode(supervisor.encode()).decode(),
               supervisor_sha256=supervisor_sha)
remote = remote.replace('PAYLOAD_LITERAL', repr(payload))
ast.parse(remote)
(HERE / 'REMOTE_LAUNCH_SOURCE.py').write_text(remote)
start = dict(UTC=datetime.now(timezone.utc).isoformat(),
             supervisor_sha256=supervisor_sha, launch_count=1,
             remote_harness_sha256=hashlib.sha256(remote.encode()).hexdigest())
(HERE / 'LOCAL_START.json').write_text(json.dumps(start, indent=2) + '\n')
command = ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-i',
           '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-p', '2222',
           'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
           'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && python3 -']
result = subprocess.run(command, input=remote, capture_output=True, text=True, timeout=90)
transport = dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,
                 stdout=result.stdout,stderr=result.stderr)
(HERE / 'LAUNCH_TRANSPORT.json').write_text(json.dumps(transport, indent=2) + '\n')
if result.returncode == 0:
    data = json.loads(result.stdout)
    for key, name in (('launch', 'LAUNCH_RECEIPT.json'), ('release', 'ROOT_RESOURCE_RELEASE.json'),
                      ('authorization', 'AUTHORIZATION.json')):
        (HERE / name).write_text(json.dumps(data[key], indent=2, sort_keys=True) + '\n')
    print(json.dumps(data['launch'], sort_keys=True))
else:
    print(json.dumps(transport, sort_keys=True))
    sys.exit(result.returncode)
