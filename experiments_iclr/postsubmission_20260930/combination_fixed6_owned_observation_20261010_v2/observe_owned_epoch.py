"""One exact owned-handle observation; no quality fields, restart or signals."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
WRAPPER=PHASE/'gpu77_connection_recovery_v1/run_gpu77_v3.py'
ID='INDEPENDENT_SCORER_FIXED6_COMBINATION_CURRENT_OWNED_EPOCH_METADATA_20261010_v2'
REMOTE=r'''
import base64,hashlib,importlib.util,json,socket,subprocess
from datetime import datetime,timezone
from pathlib import Path
R=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git');P=R/'experiments_iclr/postsubmission_20260930';A=P/'graph_relation_independent_native_local_scorer_training_activation_root_20261010_v1'
assert socket.gethostname()=='peptide' and Path.cwd().resolve()==R
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
def binding(q):return dict(path=str(q.relative_to(P)),bytes=q.stat().st_size,sha256=sha(q))
plan_path=A/'OWNER_PLAN.json';assert sha(plan_path)=='88f10a0607fd416a11310a32338907d1656c5abfe0852175afed0b3d675929ef'
plan=json.loads(plan_path.read_text())
helper_path=P/plan['existing_ownership_helper']['path'];assert sha(helper_path)==plan['existing_ownership_helper']['sha256']
spec=importlib.util.spec_from_file_location('_scorer_current_exact_owned',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
files=[];lanes=[];errors=[]
def retain(q):
 data=q.read_bytes();row=dict(path=str(q.relative_to(P)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),base64=base64.b64encode(data).decode());files.append(row);return json.loads(data)
for suffix,expected_PID in (('8ced',4043992),('a998',4043993)):
 launch=json.loads((A/('LAUNCH_'+suffix+'.json')).read_text());assert launch['owner_PID']==expected_PID and launch['owner_identity']['start_ticks']==1774146656
 parent=helper.identity(expected_PID)
 assert parent is None or parent['start_ticks']==1774146656
 root=P/plan['lanes'][suffix]['owner_output'];records=[]
 for name in ('PARENT_OWNER.json','PROGRESS.json','FAILURE.json','COST.json','LANE_CLOSURE.json'):
  q=root/name
  if q.is_file():
   try:retain(q)
   except (OSError,ValueError) as error:errors.append(dict(path=str(q.relative_to(P)),type=type(error).__name__,error=str(error),transient=True))
 for cell in plan['lanes'][suffix]['cells']:
  cell_id=cell['cell_id'];child=None;metadata={}
  for tag in ('PREFLIGHT','CHILD_STARTED','CURRENT_PROCESS','CURRENT_RESOURCES','EXIT','PHYSICAL_TERMINAL'):
   q=root/'logs'/(cell_id+'.'+tag+'.json')
   if q.is_file():
    try:
     row=retain(q);metadata[tag]=binding(q)
     if tag=='CHILD_STARTED':child=row['raw_identity_observation']
    except (OSError,ValueError) as error:errors.append(dict(path=str(q.relative_to(P)),type=type(error).__name__,error=str(error),transient=True))
  release=json.loads((P/cell['path']).read_text());output=P/release['output'];epoch=None
  q=output/'PROGRESS.json'
  if q.is_file():
   try:
    data=q.read_bytes();row=json.loads(data)
    epoch=dict(epoch=row['epoch'],complete_epochs=row['complete_epochs'],steps=row['steps'],complete=row['complete'],source_bytes=len(data),source_sha256=hashlib.sha256(data).hexdigest(),source_path=str(q.relative_to(P)),quality_fields_omitted=True)
   except (OSError,ValueError,KeyError) as error:errors.append(dict(path=str(q.relative_to(P)),type=type(error).__name__,error=str(error),transient=True))
  current=helper.identity(child['PID']) if child else None
  assert current is None or current['start_ticks']==child['start_ticks']
  records.append(dict(cell_id=cell_id,physical_gpu_uuid=release['physical_gpu_uuid'],launched=child is not None,saved_child_identity=child,current_child_identity=current,epoch_progress=epoch,metadata=metadata))
 lanes.append(dict(lane=suffix,saved_owner_identity=launch['owner_identity'],current_owner_identity=parent,owned_owner_failure=(root/'FAILURE.json').exists(),lane_closure_present=(root/'LANE_CLOSURE.json').exists(),cells=records))
print(json.dumps(dict(schema='fixed6-owned-process-and-epoch-observation-v1',UTC=datetime.now(timezone.utc).isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),owner_plan=binding(plan_path),lanes=lanes,files=files,transient_monitoring_errors=errors,quality_metrics_returned=False,signals_sent=[],owner_restarts=0,other_jobs_inspected=False)))
'''

def main():
 assert hashlib.sha256(WRAPPER.read_bytes()).hexdigest()=='035b740ceb50eefcfa3cec2aef6dbd1294c769d133e2bc4cf88fb52b088421ff'
 command='cd '+shlex.quote('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')+' && /usr/bin/python3 -I -S -B -c '+shlex.quote(REMOTE)
 file=WRAPPER.parent/(ID+'_COMMAND.txt');file.write_text(command)
 result=subprocess.run(['python3','-B',str(WRAPPER),'--id',ID,'--command-file',str(file)],capture_output=True,text=True)
 with (HERE/'OWNED_OBSERVATION_TRANSPORT.json').open('x') as stream:json.dump(dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,owner_restarts=0,signals_sent=[]),stream,indent=2);stream.write('\n')
 assert result.returncode==0,'Retained monitoring failure; no owner restart or automatic observation retry'
 outer=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')][-1];assert outer['exit_code']==0
 report=next(json.loads(line) for line in outer['stdout'].splitlines() if line.startswith('{'))
 for row in report['files']:
  target=HERE/'owned_metadata'/row['path'];target.parent.mkdir(parents=True,exist_ok=True);data=base64.b64decode(row['base64']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  with target.open('xb') as stream:stream.write(data)
 report['files']=[{key:value for key,value in row.items() if key!='base64'} for row in report['files']]
 with (HERE/'OWNED_PROCESS_EPOCH_OBSERVATION.json').open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
 print(json.dumps(dict(UTC=report['UTC'],lanes=[dict(lane=row['lane'],owner_PID=(row['current_owner_identity'] or {}).get('PID'),owner_failure=row['owned_owner_failure'],cells=[dict(cell_id=cell['cell_id'],launched=cell['launched'],child_PID=(cell['current_child_identity'] or {}).get('PID'),epoch=cell['epoch_progress']) for cell in row['cells']]) for row in report['lanes']],transient_monitoring_errors=report['transient_monitoring_errors']),indent=2))

if __name__=='__main__':main()
