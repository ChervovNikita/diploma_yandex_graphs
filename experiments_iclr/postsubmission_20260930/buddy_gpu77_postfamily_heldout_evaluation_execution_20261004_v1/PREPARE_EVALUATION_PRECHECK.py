"""Adapt the successful stdlib-only metadata precheck to the separately admitted evaluation."""
from pathlib import Path
import base64
import hashlib
import importlib.util
import json
import shlex

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
REPO='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
admission_path=HERE/'ROOT_EVALUATION_ADMISSION.json'
review_path=HERE/'ROOT_INDEPENDENT_LOCK_REVIEW.json'
admission=admission_path.read_bytes()
assert hashlib.sha256(admission).hexdigest()=='48c4eb06863688d73e8b4c68458b2d494db1b615c7612486794c46013510125c'
assert hashlib.sha256(review_path.read_bytes()).hexdigest()=='a9ec8de1b5e6a0df56c892ad7ce844af49baec5ff481a6c174d6160eacc81ddc'
original=PHASE/'buddy_gpu77_postfamily_lock_audit_execution_20261004_v1/PRECHECK_REMOTE_v2.py.txt'
code=original.read_text()
def replace(old,new):
    global code
    assert code.count(old)==1,old
    code=code.replace(old,new)
replace("assert not git('diff','--cached','--name-only')", "assert git('rev-parse','HEAD')=='6dad58e56e40175168f6fa2050d0855a97c3ab44'\nassert not git('diff','--cached','--name-only') and not git('diff','--name-only')")
replace("[here/'ROOT_LOCK_AUDIT_ADMISSION.json',here/'root_lock_v1',here/'root_lock_supervision_20261004_v1']", "[here/'ROOT_EVALUATION_ADMISSION.json',here/'root_eval_v1',here/'root_eval_supervision_20261004_v1']")
replace("[p/'FAMILY_LOCK.json',p/'LOCK_AUDIT.json',p/'PHYSICAL_TERMINAL.json',p/'RUNNER_STARTED.json']", "[p/'EVALUATION_CLAIM.json',p/'EVALUATION_RECEIPT.json',p/'PHYSICAL_TERMINAL.json',p/'RUNNER_STARTED.json']")
replace("summaries=[];ledger_pins=[];training=0.;validation=0.","summaries=[];ledger_pins=[];locked_rows=[];training=0.;validation=0.")
replace("summary=guards.read_json(folder/'completion.json');summaries.append(summary)","summary=guards.read_json(folder/'completion.json');summaries.append(summary);locked_rows.append(row)")
start=code.index("print(json.dumps(dict(schema='buddy77_lock_audit_precheck_v1'",code.index('costs=dict'))
code=code[:start]+r'''
payload=json.loads(base64.b64decode(__ADMISSION__))
assert hashlib.sha256(base64.b64decode(__ADMISSION__)).hexdigest()=='48c4eb06863688d73e8b4c68458b2d494db1b615c7612486794c46013510125c'
assert sha(e.LOCK)=='4d4041ad0d02a36c94bd9112f4e01029fb722eee75339a5743a0259435c33fd9'
assert sha(e.LOCK_AUDIT)=='2c9d8b13759c88cfb657782c3e6a22938388e91f171e8a67ab49616c9b392b38'
audit=read(e.LOCK_AUDIT)
e.matches(audit,dict(schema='buddy77-production-family-lock-audit-v1',**identity,
 status='all15_production_lock_and_selected_checkpoints_audited',family_cells=15,optimizer_fits=24,epochs_per_cell=100,
 test_payload_opened=False,all_selected_checkpoints_runtime_validated=True,family_lock_sha256=sha(e.LOCK),
 locked_runs=locked_rows,exit_code=0,argv=e.lock_command()),'Reviewed production lock audit')
lock,checkpoints=guards.verify_family_metadata(e.LOCK,context['cache_manifest_sha256'],guards.file_sha(e.SOURCE/'CONFIG.json'))
assert lock['runs']==locked_rows and len(checkpoints)==15
physical=here/'root_lock_supervision_20261004_v1/PHYSICAL_TERMINAL.json'
assert sha(physical)=='80cda0da174f6807e90bbff447ffc1349867ec36e5350ea849fd7da1f23655da' and read(physical)['exit_code']==0
contract=read(e.DATA/'CONTRACT.json')
evaluation_identity=dict(**identity,family_lock_sha256=sha(e.LOCK),lock_audit_sha256=sha(e.LOCK_AUDIT),
 archive_sha256=contract['archive_sha256'],archive_member=e.TEST_MEMBER,scoring_GPU_UUID='GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998')
e.evaluation_admission_gate(payload,evaluation_identity)
assert payload['root_independent_lock_review_sha256']=='a9ec8de1b5e6a0df56c892ad7ce844af49baec5ff481a6c174d6160eacc81ddc'
assert safe(e.ARCHIVE).is_file() and e.ARCHIVE.stat().st_size==contract['archive_bytes']
assert str(e.PYTHON)=='/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python' and Path(e.PYTHON).is_file()
setup=read(launcher.EXTRA.parent/'SETUP_RECEIPT.json');assert setup['exit_code']==0
dependency_records=[descriptor(launcher.EXTRA.parent/'SETUP_RECEIPT.json')]
for name,digest in setup['installed_RECORD_sha256'].items():
 path=launcher.confined(launcher.EXTRA/name);assert sha(path)==digest;dependency_records.append(descriptor(path))
cache=read(e.CACHE/'manifest.json')
assert cache['test_split_opened'] is False and cache['graph_policy']=='training_only_all_splits'
assert set(cache['files'])=={'common.pt','train.pt','valid.pt','official_split_loader.py.txt'}
cache_checks=[]
for name,digest in cache['files'].items():
 path=launcher.confined(e.CACHE/name);assert sha(path)==digest;cache_checks.append(descriptor(path))
runtime_paths=e.path_environment(repo,e.LAUNCHER/'root_runtime_v1')
assert all(Path(runtime_paths[name]).is_dir() for name in e.PATHS)
gpu_text=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout
gpu_rows=[]
for line in gpu_text.splitlines():
 parts=[part.strip() for part in line.split(',')];assert len(parts)==6
 gpu_rows.append(dict(uuid=parts[0],name=parts[1],memory_total_MiB=int(parts[2]),memory_free_MiB=int(parts[3]),memory_used_MiB=int(parts[4]),utilization_percent=int(parts[5])))
assert len(gpu_rows)==2 and {row['uuid'] for row in gpu_rows}==set(e.UUIDS)
apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout.strip()
target=next(row for row in gpu_rows if row['uuid']==evaluation_identity['scoring_GPU_UUID'])
assert not any(line.split(',')[0].strip()==target['uuid'] for line in apps.splitlines())
assert target['memory_free_MiB']>=0.9*target['memory_total_MiB'] and target['utilization_percent']==0
print(json.dumps(dict(schema='buddy77_heldout_evaluation_metadata_precheck_v1',UTC=datetime.now(timezone.utc).isoformat(),
 status='PASS_NO_EXISTING_EVALUATION_OR_TEST_ATTEMPT',actual_git_root=str(repo),HEAD=git('rev-parse','HEAD'),
 physical_GPU_UUIDs=uuids,source_checks=source_checks,identity=evaluation_identity,ledgers=ledger_pins,
 lock=descriptor(e.LOCK),lock_audit=descriptor(e.LOCK_AUDIT),audit_physical_terminal=descriptor(physical),
 locked_rows=locked_rows,family_cells=15,optimizer_fits=24,total_completed_cell_epochs=1500,
 prior_custody=prior,normal_runtime_boundary=family_boundary,training_costs=costs,
 current_runtime_interpreter=str(e.PYTHON),dependency_records=dependency_records,cache_checks=cache_checks,
 qualified_runtime_paths=runtime_paths,current_Torch_runtime_will_be_verified_once_by_existing_evaluation=True,
 GPU_availability=gpu_rows,compute_apps_observation=apps,scoring_GPU_available_without_stopping_other_jobs=True,
 admission_exact_byte_gate_passed=True,precheck_wall_seconds=time.monotonic()-began,
 precheck_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,checkpoint_envelopes_deserialized=False,
 TEST_payload_access=False,remote_writes=False,process_signals=False,training_executed=False)))
'''
code=code.replace('__ADMISSION__',repr(base64.b64encode(admission).decode()))
code=code.replace("schema='buddy77_lock_audit_precheck_v1'","schema='buddy77_heldout_evaluation_metadata_precheck_v1'")
compile(code,'buddy_evaluation_metadata_precheck','exec')
with (HERE/'PRECHECK_REMOTE_v1.py.txt').open('x') as h:h.write(code)
command=PHASE/'gpu77_connection_recovery_v1/buddy_v4_heldout_evaluation_precheck_20261004_v1_command.txt'
with command.open('x') as h:h.write(shlex.join(['/usr/bin/python3','-I','-S','-B','-c','import os;os.chdir('+repr(REPO)+');\n'+code])+'\n')
receipt=dict(schema='buddy77_heldout_precheck_preparation_v1',source_template=str(original.relative_to(PHASE)),
 source_template_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),remote_source_sha256=hashlib.sha256(code.encode()).hexdigest(),
 command_file=str(command.relative_to(PHASE)),command_sha256=hashlib.sha256(command.read_bytes()).hexdigest(),
 admission_sha256=hashlib.sha256(admission).hexdigest(),metadata_only=True,remote_executed=False)
with (HERE/'PRECHECK_PREPARATION.json').open('x') as h:json.dump(receipt,h,indent=2);h.write('\n')
print(json.dumps(receipt))
