"""One TRAIN-only CPU inspection of the reviewed pair-role source.

No model construction, TRAIN transition, VALID scoring or GPU context.
Runs the existing qualifier once; preserves any failure without retry.
"""
import argparse
import json
from pathlib import Path
from remote_control import call, REMOTE_EXPANSION, REMOTE_ROOT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest-sha256', required=True)
args = parser.parse_args()
E = Path(__file__).resolve().parent
code = r'''import hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
R=Path(@ROOT@);E=Path(@EXP@)
source=E/'research/molbace_pairwise_competence_v1'
assert hashlib.sha256((source/'MANIFEST.json').read_bytes()).hexdigest()==@MANIFEST@
sys.path.insert(0,str(E/'qualification/portablev4_link_queued_parent_authority_repair_v1'))
from supervise import preflight,process
out=E/'qualification/molbace_pair_TRAIN_support_25'
assert not out.exists()
out.mkdir()
report={'status':'preflight','scientific_fits':0,'TEST_accessed':False,'new_GPU_workers':0,'automatic_retries':0}
child=None;started=time.monotonic()
try:
 outcomes=json.loads((E/'results/structural_private_risk_v1/OUTCOMES.json').read_text())
 peers=[]
 old=process(640428)
 if old:
  assert old['start_ticks']=='6055039165'
  peers.append({'pid':640428,'start_ticks':6055039165,'peak_host_RSS_bytes':16*1024**3})
 for row in outcomes:
  if row.get('status')=='started' and process(row['worker_pid']):
   peers.append({'pid':row['worker_pid'],'start_ticks':row['worker_start_ticks'],'peak_host_RSS_bytes':12*1024**3})
 assert len(peers)<=2
 resources=preflight({'device':'cpu','peers':peers},{'minimum_host_headroom_after_peer_growth_bytes':8*1024**3,'host_RSS_reservation_bytes':8*1024**3,'disk_reservation_bytes':1024**3})
 admission={'schema':'root-reviewed-molbace-pair-CPU-v1','approved':True,'source_reviewed':True,'mode':'inspect','arm':'structured','manifest_sha256':@MANIFEST@,'new_scientific_fits':0,'TEST_access':False,'maximum_native_constructions':0,'maximum_optimizer_transitions':0,'maximum_full_VALID_calls':0,'resource_evidence':resources,'watchdog_seconds':300,'TERM_cleanup_seconds':10}
 ap=out/'ADMISSION.json';ap.write_text(json.dumps(admission,indent=2)+'\n')
 command=[str(R/'.venv/bin/python'),'-B','-u',str(source/'qualify.py'),'--project-root',str(R),'--mode','inspect','--arm','structured','--output',str(out/'native'),'--admission',str(ap),'--admission-sha256',hashlib.sha256(ap.read_bytes()).hexdigest()]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (out/'child.log').open('xb') as log:
  child=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 report.update(status='running',child=process(child.pid),command=command,admission=admission)
 (out/'SUPERVISOR.json').write_text(json.dumps(report,indent=2)+'\n')
 deadline=time.monotonic()+300
 while child.poll() is None:
  current=process(child.pid)
  if current and current['rss_bytes']>8*1024**3:raise RuntimeError('CPU inspection exceeded its host reservation')
  if time.monotonic()>=deadline:raise TimeoutError('TRAIN inspection reached the 300-second bound')
  try:child.wait(timeout=min(2,max(.01,deadline-time.monotonic())))
  except subprocess.TimeoutExpired:pass
 report['exit_code']=child.returncode
 rp=out/'native/RESULT.json'
 report['result']=json.loads(rp.read_text()) if rp.exists() else None
 report['status']='passed' if child.returncode==0 and report['result'] and report['result']['status']=='passed' else 'failed'
except BaseException as error:
 report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=8)})
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
finally:
 report['wall_seconds']=time.monotonic()-started
 (out/'SUPERVISOR.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
'''
for key, value in {'ROOT': REMOTE_ROOT, 'EXP': REMOTE_EXPANSION,
                   'MANIFEST': args.manifest_sha256}.items():
    code = code.replace('@' + key + '@', repr(value))
result = json.loads(call(code, timeout=340))
with (E/'PAIR_TRAIN_SUPPORT_25.json').open('x') as handle:
    json.dump(result, handle, indent=2); handle.write('\n')
table = (result.get('result') or {}).get('TRAIN_support', {}).get('role_receipt', {}).get('role_table', {})
print(json.dumps({'status': result['status'], 'wall_seconds': result['wall_seconds'],
                 'failure': result.get('failure'), 'TRAIN': table.get('TRAIN_inputs'),
                 'positive': table.get('observed_positive_inputs'),
                 'negative': table.get('observed_negative_inputs'),
                 'all_pairs': table.get('all_input_pair_support'),
                 'opposite_pairs': table.get('observed_opposite_pair_support'),
                 'opposite_pair_route_mass': table.get('opposite_label_route_mass')}))
