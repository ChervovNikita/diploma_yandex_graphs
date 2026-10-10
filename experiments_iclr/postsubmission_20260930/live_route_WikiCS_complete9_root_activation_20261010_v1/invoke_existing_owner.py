"""Fixed nine-record invocation; all child ownership is the existing run_fit."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,socket,subprocess,time
R=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
def save(q,v):
 t=q.with_name(q.name+'.partial');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(q)
def bind(row):
 q=(P/row['path']).resolve(strict=True);assert q.is_relative_to(P) and sha(q)==row['sha256'];return q
parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);a=parser.parse_args()
assert socket.gethostname()=='peptide' and Path.cwd()==R
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
plan_path=H/'OWNER_PLAN.json';assert sha(plan_path)==a.plan_sha256
plan=json.loads(plan_path.read_text());assert plan['enabled'] and not plan['automatic_retry']
assert [(r['kind'],r['seed']) for r in plan['records']]==[(k,s) for s in (6101,6203,6307) for k in ('baseline','exchange','separable')]
source=bind(plan['existing_run_fit_helper']);helper_path=bind(plan['existing_ownership_helper'])
loader=importlib.util.spec_from_file_location('_live9_existing_run_fit',source);owner=importlib.util.module_from_spec(loader);loader.loader.exec_module(owner)
assert owner.HELPER==helper_path
helper,context=owner.lane(plan['physical_gpu_uuid'])
context.SOURCE=bind(plan['entry_program']).parent;context.SOURCE_SHA=plan['adapter_manifest_sha256']
root=P/plan['supervision_output'];assert root.parent==P and not root.exists();root.mkdir();(root/'logs').mkdir()
started=time.monotonic();completed=[]
save(root/'PARENT_OWNER.json',dict(identity=helper.identity(os.getpid()),plan_sha256=a.plan_sha256,UTC=datetime.now(timezone.utc).isoformat(),numerical_work_by_parent=False))
try:
 for row in plan['records']:
  assert time.monotonic()-started+row['hard_seconds']+plan['resource_limits']['resource_wait_seconds']<plan['phase_hard_seconds']
  release=bind(row['release']);cfg=json.loads(release.read_text());output=P/cfg['output'];assert not output.exists()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['physical_gpu_uuid'],PYTHONPATH='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONHOME',None)
  save(root/'FAMILY_PROGRESS.json',dict(current_record=row['cell_id'],completed=completed,partial_quality_opened=False))
  entry=dict(cell_id=row['cell_id'],job_relative=str(release.relative_to(P)),job_sha256=row['release']['sha256'],hard_seconds=row['active_seconds'],argv=row['argv'])
  receipt=owner.run_fit(helper,root,entry,{'resource_limits':plan['resource_limits']},env,output,context)
  child=receipt['raw_identity_observation'];assert child is not None
  current=helper.identity(child['PID']);absent=current is None or current['start_ticks']!=child['start_ticks']
  rows=helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],5)
  no_cuda=not any(len(x)>=2 and x[1].strip()==str(child['PID']) for x in (r.split(',') for r in rows))
  save(root/'logs'/(row['cell_id']+'.PHYSICAL_TERMINAL.json'),dict(UTC=helper.now(),child_PID=child['PID'],child_start_ticks=child['start_ticks'],physical_gpu_uuid=plan['physical_gpu_uuid'],owned_PID_absent=absent,owned_PID_no_CUDA_rows=no_cuda,terminal_wait_observed=receipt['terminal_wait_observed']))
  assert receipt['exit_code']==0 and receipt['terminal_wait_observed'] and receipt['reason'] is None and not receipt['signals_sent'] and absent and no_cuda
  q=output/'COMPLETE.json';v=json.loads(q.read_text());assert v['complete'] and v['epochs']==v['steps']==1100 and v['root_cell_release_sha256']==row['release']['sha256'] and v['TEST_scoring'] is False
  completed.append(row['cell_id']);save(root/'FAMILY_PROGRESS.json',dict(current_record=None,completed=completed,partial_quality_opened=False))
 save(root/'FAMILY_COMPLETE.json',dict(complete=True,records=completed,all9_directly_waited=True,all9_owned_absence_verified=True,adapter_manifest_sha256=plan['adapter_manifest_sha256'],scientific_source_manifest_sha256=plan['scientific_source_manifest_sha256'],owner_seconds=time.monotonic()-started,TEST_access=False,quality_not_interpreted=True))
except BaseException as error:
 save(root/'FAMILY_FAILURE.json',dict(complete=False,completed=completed,error_type=type(error).__name__,error=str(error),owner_seconds=time.monotonic()-started,automatic_retry=False,no_partial_family_comparison=True));raise
