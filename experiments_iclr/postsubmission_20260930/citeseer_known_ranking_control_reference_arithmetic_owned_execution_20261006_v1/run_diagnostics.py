"""Root-authorized two TRAIN-only diagnostic children using existing owner loop."""
from pathlib import Path
import hashlib,importlib.util,json,os,time
ROOT=Path(__file__).resolve().parent
owner_path=ROOT.parent/'citeseer_known_ranking_control_gpu77_b0_owned_preparation_20261006_v1/owner.py'
assert hashlib.sha256(owner_path.read_bytes()).hexdigest()=='a8d36b95fd7faa767f44b3813c6e4f484d97c4dec72cc1c141cf4139ce4e462d'
spec=importlib.util.spec_from_file_location('reviewed_rank77_owner',owner_path)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
h.physical_host();h.verify_source()
cfg=h.read(ROOT/'OWNER_CONFIG.json');owner_root=ROOT/'owner';assert not owner_root.exists();owner_root.mkdir();(owner_root/'logs').mkdir()
s,_=h.lane(h.GPU_UUIDS[0]);identity=s.identity(os.getpid());assert identity is not None and identity['pgid']==identity['sid']==os.getpid()
h.write(owner_root/'OWNER_STARTED.json',{'UTC':s.now(),'identity':identity,'fits':0,'qualification_claimed':False,'scope':'two_assigned_first_inner_TRAIN_arithmetic_diagnostics','retry':False})
started=time.monotonic()
try:
 receipts=h.parallel((0,1),cfg['commands'],cfg['jobs'],owner_root,cfg['resource_limits'],started,5100)
 results=[]
 for i in (0,1):
  output=Path(cfg['commands'][i]['argv'][6]);value=h.read(output/'RESULT.json')
  assert value['diagnostic_completed'] is True and value['qualification_claimed'] is False and value['fits']==0 and value['VALID_TEST_access'] is False
  results.append({'label':output.parent.name,'result_sha256':h.sha(output/'RESULT.json'),'result_bytes':(output/'RESULT.json').stat().st_size})
 h.write(owner_root/'DIAGNOSTICS_COMPLETE.json',{'UTC':s.now(),'receipts':receipts,'results':results,'fits':0,'qualification_claimed':False,'elapsed_seconds':time.monotonic()-started})
except BaseException as error:
 h.write(owner_root/'OWNER_FAILURE.json',{'UTC':s.now(),'error':type(error).__name__+': '+str(error),'partial_outputs_preserved':True,'fits':0,'qualification_claimed':False,'retry':False})
 raise
