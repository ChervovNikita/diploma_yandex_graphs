"""Supervise exactly eight reviewed native source callbacks once; no fit."""
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD,invoke
HERE=Path(__file__).resolve().parent/'native_qk_resource_results_root_20261012_v1'
REMOTE=GUARD+r'''
import signal
source=phase/'native_polynormer_qk_resource_source_20261012_v1'
helper=source/'qualify_native_qk_resources.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='d752f4994627dad2a03dd4e19a59032f124ba6afdabbdc66d2dcc3ee3b3d2c7d'
proposal=json.loads((source/'PROPOSAL.json').read_text())
cases=proposal['requested_cases'];assert len(cases)==8 and all(c['shape_config']['members']==4 for c in cases)
out=phase/'native_qk_resource_results_root_20261012_v1';assert not out.exists();out.mkdir()
decision={'exact_case_ids':[c['case_id'] for c in cases],'maximum_scientific_fits':0,'maximum_actual_optimizer_updates':0,
 'scope':'Eight exact native Polynormer local/global source callbacks for new-source shim admission, model checkpoint restore and source memory; no objective values or task scores.',
 'maximum_child_wall_seconds':630,'retry':False,'concurrent_children':1,'automatic_remaining_queue':False,'GO':False}
(out/'ROOT_ENGINEERING_DECISION.json').write_text(json.dumps(decision,indent=2)+'\n')
def ident(pid):
 f=Path('/proc')/str(pid)/'stat'
 if not f.exists():return None
 a=f.read_text().rsplit(')',1)[1].split();return {'pid':pid,'start_ticks':int(a[19]),'pgid':int(a[2])}
def scientific_custody():
 p=phase/'private_GRU_acquisition_root_20261011_v1/owner_v1'
 start=json.loads((p/'OWNER_START.json').read_text());observed={}
 for key in ('owner','child'):
  expected=start[key];now=ident(expected['pid']);observed[key]=now
  if now is not None:assert all(now[k]==expected[k] for k in ('pid','start_ticks','pgid'))
 if any(v is None for v in observed.values()):assert (p/'OWNER_END.json').exists()
 return observed
records=[]
for case in cases:
 custody=scientific_custody()
 result_relative='native_qk_resource_results_root_20261012_v1/case_'+case['case_id']
 command=[str(repo/'.venv/bin/python'),'-B',str(helper),'--project-root',str(phase),'--admission','native_qk_resource_admission_root_20261012_v1/ENGINEERING_ADMISSION.json','--case-id',case['case_id'],'--output',result_relative]
 with (out/('case_'+case['case_id']+'.log')).open('x') as log:
  child=subprocess.Popen(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  owner=ident(child.pid);assert owner is not None and owner['pgid']==child.pid
  row={'case_id':case['case_id'],'child':owner,'Physics_before':custody,'command':command}
  (out/('case_'+case['case_id']+'_OWNER.json')).write_text(json.dumps(row,indent=2)+'\n')
  try:code=child.wait(timeout=case['timeout_seconds']+30)
  except subprocess.TimeoutExpired:
   now=ident(child.pid);assert now==owner
   os.killpg(owner['pgid'],signal.SIGTERM)
   try:code=child.wait(timeout=10)
   except subprocess.TimeoutExpired:
    now=ident(child.pid);assert now==owner;os.killpg(owner['pgid'],signal.SIGKILL);code=child.wait(timeout=10)
   row['parent_timeout']=True
  row.update(exit_code=code,child_absent=ident(child.pid) is None,Physics_after=scientific_custody())
  result=phase/result_relative/'RESULT.json'
  if result.exists():
   v=json.loads(result.read_text());assert v['actual_optimizer_updates']==v['predictive_scores']==v['selected_metrics']==0 and v['actual_endpoint_targets_opened'] is False
   row['result']={'path':str(result.relative_to(phase)),'bytes':result.stat().st_size,'sha256':hashlib.sha256(result.read_bytes()).hexdigest()}
   row['resource']={k:v[k] for k in ('CUDA_peak_allocated_bytes','CUDA_peak_reserved_bytes','parameter_bytes','manual_moment_bytes','actual_optimizer_updates','predictive_scores')}
   row['resource']['warm_callback_median_seconds']=v['warm_median_seconds']
   row['resource']['random_engine_restore']=v['random_engine_restore']
   row['resource']['full_graph_serving_seconds']=v['full_graph_serving_seconds']
  else:row['failure_log_tail']=(out/('case_'+case['case_id']+'.log')).read_text()[-6000:]
  records.append(row);(out/'PROGRESS.json').write_text(json.dumps({'attempts':records,'automatic_remaining_queue':False,'GO':False},indent=2)+'\n')
  if code!=0:row['failure_preserved_no_retry']=True
closure={'attempts':records,'all_children_closed':all(r['child_absent'] for r in records),'fits':0,'GO':False,'automatic_remaining_queue':False}
(out/'CLOSURE.json').write_text(json.dumps(closure,indent=2)+'\n');print(json.dumps(closure))
'''
HERE.mkdir();v=invoke(REMOTE,[],HERE,'EXECUTION',5200)
assert v['exit_code']==0,'Parent transport failure retained; inspect exact children before further launch'
x=json.loads(v['stdout']);(HERE/'CLOSURE.json').write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps({'attempted_cases':len(x['attempts']),'passing_cases':sum(r['exit_code']==0 for r in x['attempts']),'all_children_closed':x['all_children_closed'],'fits':0,'GO':False}))
