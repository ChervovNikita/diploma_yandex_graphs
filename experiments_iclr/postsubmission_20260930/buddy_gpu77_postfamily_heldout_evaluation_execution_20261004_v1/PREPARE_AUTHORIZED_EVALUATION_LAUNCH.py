"""Prepare, without executing, the single admitted existing V4 heldout stage."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import importlib.util
import json
import shlex

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
REPO='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
template_path=PHASE/'buddy_gpu77_postfamily_lock_audit_execution_20261004_v1/PREPARE_AUTHORIZED_AUDIT_LAUNCH.py'
spec=importlib.util.spec_from_file_location('reviewed_audit_once_template',template_path)
template=importlib.util.module_from_spec(spec);spec.loader.exec_module(template)
transport_path=PHASE/'gpu77_connection_recovery_v1/commands/buddy_v4_heldout_evaluation_precheck_20261004_v2/RECEIPT.json'
transport=json.loads(transport_path.read_text());assert transport['exit_code']==0
precheck=json.loads(transport['stdout'].strip())
assert precheck['status']=='PASS_NO_EXISTING_EVALUATION_OR_TEST_ATTEMPT'
assert (precheck['family_cells'],precheck['optimizer_fits'],precheck['total_completed_cell_epochs'])==(15,24,1500)
assert precheck['admission_exact_byte_gate_passed'] is True and precheck['scoring_GPU_available_without_stopping_other_jobs'] is True
assert precheck['TEST_payload_access'] is False and precheck['remote_writes'] is False
with (HERE/'PRECHECK_RESULT.json').open('x') as h:json.dump(precheck,h,indent=2);h.write('\n')
admission=(HERE/'ROOT_EVALUATION_ADMISSION.json').read_bytes()
admission_sha=hashlib.sha256(admission).hexdigest()
assert admission_sha=='48c4eb06863688d73e8b4c68458b2d494db1b615c7612486794c46013510125c'
assert hashlib.sha256((HERE/'ROOT_INDEPENDENT_LOCK_REVIEW.json').read_bytes()).hexdigest()=='a9ec8de1b5e6a0df56c892ad7ce844af49baec5ff481a6c174d6160eacc81ddc'
runner=template.RUNNER.replace('__ADMISSION_SHA__',repr(admission_sha))
runner=runner.replace('root_lock_supervision_20261004_v1','root_eval_supervision_20261004_v1')
runner=runner.replace('ROOT_LOCK_AUDIT_ADMISSION','ROOT_EVALUATION_ADMISSION')
runner=runner.replace("HERE/'root_lock_v1'","HERE/'root_eval_v1'")
runner=runner.replace("'audit-lock'","'evaluate'")
runner=runner.replace('buddy77_owned_lock_audit','buddy77_owned_heldout_evaluation')
runner=runner.replace('audit_once=True','evaluation_once=True')
runner=runner.replace('TEST=False','official_TEST_access_authorized=True')
runner=runner.replace('overlapping_internal_production_lock_seconds_not_added','overlapping_internal_evaluation_intervals_not_added')
client=template.CLIENT
client=client.replace('import os,sys,json,base64,hashlib,subprocess','import os,sys,json,base64,hashlib,subprocess,time,resource\nclient_started=time.monotonic()')
client=client.replace('root_lock_supervision_20261004_v1','root_eval_supervision_20261004_v1')
client=client.replace('ROOT_LOCK_AUDIT_ADMISSION','ROOT_EVALUATION_ADMISSION')
client=client.replace("HERE/'root_lock_v1'","HERE/'root_eval_v1'")
client=client.replace('run_lock_audit_once.py','run_heldout_evaluation_once.py')
client=client.replace("'buddy77_authorized_once_lock_audit_detached_launch_v1'","'buddy77_authorized_once_heldout_evaluation_detached_launch_v1'")
client=client.replace('audit_once=True','evaluation_once=True').replace("stage='audit-lock'","stage='evaluate'")
client=client.replace('evaluate_executed=False','evaluate_launched=True').replace('TEST_payload_access=False','TEST_payload_access_delegated_to_admitted_stage=True')
anchor="normal=payload['normal_runtime_boundary']['normal_runtime_qualification'];assert sha(Path(normal['path']))==normal['sha256']"
assert client.count(anchor)==1
client=client.replace(anchor,anchor+r'''
assert git('rev-parse','HEAD')=='6dad58e56e40175168f6fa2050d0855a97c3ab44' and not git('diff','--name-only')
for item in [payload['lock'],payload['lock_audit'],payload['audit_physical_terminal'],*payload['dependency_records'],*payload['cache_checks']]:
 path=Path(item['path']);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
assert all(Path(payload['qualified_runtime_paths'][name]).is_dir() for name in ['TMPDIR','TMP','TEMP','XDG_CACHE_HOME','TORCH_HOME','PYTHONPYCACHEPREFIX','NUMBA_CACHE_DIR','MPLCONFIGDIR','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR'])
''')
anchor="admission=base64.b64decode(payload['admission_base64']);assert hashlib.sha256(admission).hexdigest()==payload['admission_sha256']"
assert client.count(anchor)==1
client=client.replace(anchor,r'''
gpu_text=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout
gpu_rows=[]
for line in gpu_text.splitlines():
 parts=[part.strip() for part in line.split(',')];assert len(parts)==6
 gpu_rows.append(dict(uuid=parts[0],name=parts[1],memory_total_MiB=int(parts[2]),memory_free_MiB=int(parts[3]),memory_used_MiB=int(parts[4]),utilization_percent=int(parts[5])))
assert len(gpu_rows)==2 and {row['uuid'] for row in gpu_rows}==set(pins['physical_GPU_UUIDs'])
apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout.strip()
target=next(row for row in gpu_rows if row['uuid']==pins['scoring_GPU_UUID'])
assert not any(line.split(',')[0].strip()==target['uuid'] for line in apps.splitlines())
assert target['memory_free_MiB']>=0.9*target['memory_total_MiB'] and target['utilization_percent']==0
'''+anchor)
anchor='normal_host_mode=True,other_jobs_stopped=False)'
assert client.count(anchor)==1
client=client.replace(anchor,r'''normal_host_mode=True,other_jobs_stopped=False,
 GPU_availability=gpu_rows,compute_apps_observation=apps,scoring_GPU_available_without_stopping_other_jobs=True,
 launch_client_wall_seconds_through_child_start=time.monotonic()-client_started,
 launch_client_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 launch_client_CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
 launch_client_CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
 launch_client_scope='Source/ledger/checkpoint/cache/dependency byte gates and admission staging; receipt write tail excluded. Runner begins at Popen and a small launch tail overlaps physical runner wall.',
 numerical_retry=False,training_executed=False)''')
payload={key:precheck[key] for key in ('identity','source_checks','ledgers','prior_custody','normal_runtime_boundary',
 'lock','lock_audit','audit_physical_terminal','dependency_records','cache_checks','qualified_runtime_paths')}
payload.update(admission_base64=base64.b64encode(admission).decode(),admission_sha256=admission_sha,
 runner_base64=base64.b64encode(runner.encode()).decode(),runner_sha256=hashlib.sha256(runner.encode()).hexdigest())
client=client.replace('__PAYLOAD__',repr(base64.b64encode(json.dumps(payload).encode()).decode()))
compile(runner,'heldout_evaluation_once_runner','exec');compile(client,'heldout_evaluation_once_client','exec')
with (HERE/'DETACHED_RUNNER_SOURCE.py').open('x') as h:h.write(runner)
with (HERE/'LAUNCH_REMOTE.py.txt').open('x') as h:h.write(client)
command=PHASE/'gpu77_connection_recovery_v1/buddy_v4_heldout_evaluation_launch_20261004_v1_command.txt'
with command.open('x') as h:h.write(shlex.join(['/usr/bin/python3','-I','-S','-B','-c','import os;os.chdir('+repr(REPO)+');\n'+client])+'\n')
receipt=dict(schema='buddy77_heldout_once_launch_preparation_v1',UTC=datetime.now(timezone.utc).isoformat(),
 admission_sha256=admission_sha,root_review_sha256='a9ec8de1b5e6a0df56c892ad7ce844af49baec5ff481a6c174d6160eacc81ddc',
 source_template_sha256=hashlib.sha256(template_path.read_bytes()).hexdigest(),
 runner_sha256=hashlib.sha256(runner.encode()).hexdigest(),runner_bytes=len(runner.encode()),
 client_sha256=hashlib.sha256(client.encode()).hexdigest(),client_bytes=len(client.encode()),
 command_sha256=hashlib.sha256(command.read_bytes()).hexdigest(),command_file=str(command.relative_to(PHASE)),
 precheck_transport_sha256=hashlib.sha256(transport_path.read_bytes()).hexdigest(),
 prior_once_runner_procedure_preserved=True,normal_host_mode=True,evaluation_launched=False,TEST_payload_access=False,
 changes=['evaluate/admission/output/TEST-authorization stage bindings','fresh lock/dependency/cache/GPU availability byte guards',
          'launch client engineering wall/CPU/RSS accounting; intervals are not added to overlapping runner wall'],
 root_source_review_required_before_launch=True)
with (HERE/'LAUNCH_PREPARATION.json').open('x') as h:json.dump(receipt,h,indent=2);h.write('\n')
print(json.dumps(receipt))
