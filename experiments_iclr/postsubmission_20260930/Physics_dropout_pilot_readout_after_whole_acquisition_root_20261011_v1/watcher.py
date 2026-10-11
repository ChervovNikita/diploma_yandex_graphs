"""Finite whole-family handoff; stop on failure, without partial quality views."""
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';H=Path(__file__).resolve().parent
S=P/'Physics_dropout_pilot_execution_source_20261011_v2';E=P/'Physics_dropout_pilot_acquisition_root_20261011_v1';Q=P/'Physics_dropout_pilot_full_input_qualification_root_20261011_v1'
os.chdir(R)
def receipt(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,a):
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(a,indent=2)+'\n');temp.replace(p)
def owner_ok(p):
 a=json.loads(p.read_text());assert a['status']=='COMPLETE_EXIT_0' and a['exit_code']==0 and a['direct_child_wait'] and a['child_pid_absent'] and a['owned_cuda_pid_absent'];return a
start=time.monotonic();stage='waiting_for_whole_acquisition';result=dict(complete=False,partial_quality_opened=False,no_model_retry=True)
try:
 assert receipt(S/'READOUT_MANIFEST.json')['sha256']=='71a70104d08dfca59120d5ff795882060dfeb889da1bcc87ce88f41ff2f05431'
 assert receipt(S/'ACQUISITION_MANIFEST.json')['sha256']=='adbdf299fd460801a7a184c6c8d924383f1a9710c1a9393c3580072cc8d5698f'
 root=json.loads((H/'ROOT_DECISION.json').read_text())
 write(H/'WATCH_START.json',dict(pid=os.getpid(),stage=stage,finite_wall_seconds=47000,root_decision=receipt(H/'ROOT_DECISION.json')))
 while not (E/'owner_v1/OWNER_END.json').exists():
  assert time.monotonic()-start<43500,'Acquisition owner closure did not arrive within finite window'
  time.sleep(30)
 owner_ok(E/'owner_v1/OWNER_END.json')
 panel=json.loads((E/'acquisition_v1/PANEL.json').read_text());assert panel['complete'] and panel['banks']==42 and panel['selected_states']==69 and panel['exports']==42
 base=dict(execution_enabled=True,root_decision=receipt(H/'ROOT_DECISION.json'),finite_external_owner=True,source_manifest=receipt(S/'READOUT_MANIFEST.json'),custody=receipt(Q/'fresh_roles_v1/CUSTODY.json'),panel=receipt(E/'acquisition_v1/PANEL.json'),acquisition_owner_end=receipt(E/'owner_v1/OWNER_END.json'))
 for stage,seconds in [('calibrate',1800),('score',900)]:
  config=dict(base,stage=stage,output=str(H/(stage+'_v1')),owner_output=str(H/(stage+'_owner_v1')),limits=dict(wall_seconds=seconds,cuda_bytes=48*1024**3,rss_bytes=64*1024**3))
  if stage=='score':
   owner_ok(H/'calibrate_owner_v1/OWNER_END.json');temps=json.loads((H/'calibrate_v1/TEMPERATURES.json').read_text());assert temps['complete'] and temps['endpoints']==42 and temps['actual_scalar_updates']==21000
   config.update(temperatures=receipt(H/'calibrate_v1/TEMPERATURES.json'),calibration_owner_end=receipt(H/'calibrate_owner_v1/OWNER_END.json'))
  path=H/(stage.upper()+'_CONFIG.json');write(path,config)
  proc=subprocess.run([str(R/'.venv/bin/python'),'-B','-u',str(S/'owner.py'),str(path)],cwd=R,capture_output=True,text=True,timeout=seconds+60)
  write(H/(stage+'_OWNER_TRANSPORT.json'),dict(exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
  assert proc.returncode==0,stage+' owner failed; see retained transport/child log'
  owner_ok(H/(stage+'_owner_v1/OWNER_END.json'))
 report=H/'score_v1/DEVELOPMENT_REPORT.json';assert json.loads(report.read_text())['complete']
 result.update(complete=True,report=receipt(report),all69states_before42CAL_before_onewholeDEV=True)
except BaseException as error:
 result['failure']=dict(type=type(error).__name__,message=str(error))
finally:
 result.update(last_stage=stage,seconds=time.monotonic()-start);write(H/'WATCH_END.json',result)
raise SystemExit(0 if result['complete'] else 1)
