from pathlib import Path
from types import SimpleNamespace
import json,hashlib,importlib.util,socket,subprocess,os
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');PHASE=REPO/'experiments_iclr/postsubmission_20260930';ROOT=PHASE/'citeseer_gnnm_initialization_pilot_execution_root_20261006_v2'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def physical():
 assert Path.cwd()==REPO and socket.gethostname()=='anogena-2-0'
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[GPU]
def phase_file(rel):
 p=(PHASE/rel).resolve(strict=True);assert p.is_relative_to(PHASE.resolve()) and p.is_file();return p
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def load(name,p,expected):
 assert sha(p)==expected;spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
physical()
o=load('reviewed_run_loop',PHASE/'citeseer_known_ranking_control_gpu77_b0_owned_preparation_20261006_v1/owner.py','a8d36b95fd7faa767f44b3813c6e4f484d97c4dec72cc1c141cf4139ce4e462d')
h=load('owned_helper',PHASE/'shared_private_transfer_gpu77_qualification_preparation_20261005_v3/ownership_helpers.py','e71503c87865546319cddbbf7a4f9f15d13cdf9e4875e406d65de21e64e047fd');h.GPU=GPU
cfg=json.loads((ROOT/'WARM_OWNER_CONFIG.json').read_text());output=Path(cfg['argv'][6]);job=phase_file(cfg['job_relative']);assert sha(job)==cfg['job_sha256'] and not output.exists()
logs=ROOT/'warm_b0_owner';logs.mkdir();(logs/'logs').mkdir()
ident=h.identity(os.getpid());assert ident and ident['sid']==ident['pgid']==os.getpid();write(logs/'OWNER_STARTED.json',{'UTC':h.now(),'identity':ident,'TRAIN_only':True,'TEST_closed':True})
ctx=SimpleNamespace(REPO=REPO,SOURCE=PHASE/'citeseer_gnnm_initialization_pilot_preparation_20261006_v1',SOURCE_SHA=cfg['program_sha256'],GPU_UUID=GPU,GPU_UUIDS=(GPU,),physical_host=physical,phase_file=phase_file,sha=sha,write=write)
env=dict(os.environ,**cfg['environment']);env.pop('PYTHONHOME',None)
entry={'cell_id':'b0_warm','job_relative':cfg['job_relative'],'job_sha256':cfg['job_sha256'],'argv':cfg['argv'],'hard_seconds':5400}
receipt=o.run_fit(h,logs,entry,{'resource_limits':cfg['resource_limits']},env,output,ctx)
write(logs/'TERMINAL.json',receipt)
if receipt['exit_code']!=0 or receipt['reason'] is not None or receipt['signals_sent'] or not receipt['terminal_wait_observed']:raise RuntimeError('Warm failed; partial artifacts preserved')
freeze=json.loads((output/'FREEZE.json').read_text());assert freeze['phase']=='warm' and freeze['completed_cycles']==20 and freeze['VALID_TEST_access'] is False
write(logs/'WARM_COMPLETE.json',{'UTC':h.now(),'freeze_sha256':sha(output/'FREEZE.json'),'checkpoint_sha256':freeze['checkpoint_sha256'],'native_updates':freeze['counters']['native_Adam_updates'],'TEST_closed':True,'child_absent':h.identity(receipt['child_identity']['PID']) is None})
