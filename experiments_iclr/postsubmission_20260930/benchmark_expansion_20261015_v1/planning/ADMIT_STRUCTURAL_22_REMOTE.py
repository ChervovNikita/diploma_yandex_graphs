import json,hashlib,subprocess,sys,os,datetime
from pathlib import Path
root=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');e=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1');pilot=e/'pilot_20261015_structural_private_risk_v1'
commit='7ad17c23a3e6bd66d3bb5a8900b2cc565dce3c7f'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==commit
assert json.loads((e/'PUSH_RECEIPT_21.json').read_text())['remote_commit']==commit
buddy=e/'qualification/portablev3_BUDDY_queued_lifecycle_attempt01'
assert json.loads((buddy/'SUPERVISOR.json').read_text())['status']=='passed'
assert json.loads((buddy/'qualification/RESULT.json').read_text())['updates_completed']==3170
sys.path.insert(0,str(e/'qualification/portablev4_link_queued_parent_authority_repair_v1'))
import supervise
state=supervise.gpu_state()
assert len(state['processes'])<=1
peers=[]
for row in state['processes']:
 assert row['pid']==640428 and row['start_ticks']=='6055039165'
 peers.append({'pid':row['pid'],'start_ticks':row['start_ticks'],'peak_host_RSS_bytes':16*1024**3,'peak_GPU_bytes':32*1024**3})
# Reuse measured complete uniform-run resource receipt, with explicit larger
# reservations for the second reverse collection. No host/cache modification.
evidence=supervise.receipt(e/'RESIDUAL_ANALYSIS_OUTCOME_21.json')
a={'peers':peers,'device':'cuda:0','GPU_resource_evidence':evidence,'GPU_resource_evidence_applicable':True,'own_GPU_reservation_bytes':16*1024**3}
case={'minimum_host_headroom_after_peer_growth_bytes':16*1024**3,'disk_reservation_bytes':4*1024**3,'host_RSS_reservation_bytes':12*1024**3}
resources=supervise.preflight(a,case)
resources.update(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),method_budget_note='Exact GPU composition not previously measured. Reserve16GiB GPU and12GiB host versus complete uniform parent max0.381GB allocated; exact v2 CPU execution passed. Native shape and four forward graph budget retained, two reverse collections. Actual fit peaks will be retained.')
resource_path=e/'STRUCTURAL_RESOURCE_ADMISSION_22.json'
with resource_path.open('x') as f:json.dump(resources,f,indent=2);f.write('\n')
ad=json.loads((pilot/'ADMISSION_TEMPLATE.json').read_text())
ad.update(approved=True,publication_approved=True,publication_commit=commit,resource_admission_approved=True,
 pilot_manifest_sha256=hashlib.sha256((pilot/'MANIFEST.json').read_bytes()).hexdigest(),reserved_GPU_bytes=16*1024**3,reserved_host_bytes=12*1024**3,
 allowed_other_GPU_processes=[{'pid':r['pid'],'start_ticks':int(r['start_ticks'])} for r in peers],slot_lock_path='analysis_outputs/selected_VALID_collection_gpu_slot.lock',
 resource_admission_note={'resource_receipt':supervise.receipt(resource_path),'no_third_worker':True,'BUDDY_terminal':supervise.receipt(buddy/'SUPERVISOR.json'),'model_scope':'six predeclared fits only; TEST remains closed; no large campaign GO'})
adpath=e/'STRUCTURAL_TRAINING_ADMISSION_22.json'
with adpath.open('x') as f:json.dump(ad,f,indent=2);f.write('\n')
sha=hashlib.sha256(adpath.read_bytes()).hexdigest()
command=[str(root/'.venv/bin/python'),'-B','-u',str(pilot/'run.py'),'--project-root',str(root),'--mode','launch','--admission',str(adpath),'--admission-sha256',sha]
log=e/'STRUCTURAL_OWNER_22.log'
with log.open('x') as f:
 child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0'))
result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'owner_pid':child.pid,'owner_start_ticks':supervise.process(child.pid)['start_ticks'],'command':command,'publication_commit':commit,'admission':supervise.receipt(adpath),'resources':supervise.receipt(resource_path),'new_fits':6,'new_concurrent_GPU_workers':1,'full_campaign_GO':False,'TEST_access':False,'launch_status':'spawned; first worker progress not yet verified'}
with (e/'STRUCTURAL_LAUNCH_22.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'launch':result,'admission':ad,'resources':resources}))
