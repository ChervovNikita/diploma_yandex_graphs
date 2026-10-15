"""One original-node CPU selected-VALID replay, with zero training."""
import json
from pathlib import Path
from remote_control import call, REMOTE_EXPANSION, REMOTE_ROOT

E = Path(__file__).resolve().parent
code = r'''import hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
R=Path(@ROOT@);E=Path(@EXP@)
package=E/'qualification/final_selected_node_NCNC_VALID_v2'
assert hashlib.sha256((package/'MANIFEST.json').read_bytes()).hexdigest()=='3e6adc2cbeb0fa420fad7a41c912481fb5b466fcd4c7540bf07bdfe677b56217'
sys.path.insert(0,str(E/'qualification/portablev4_link_queued_parent_authority_repair_v1'))
from supervise import preflight,process
output=E/'qualification/node_C4_selected_replay_26'
assert not output.exists();output.mkdir()
report={'status':'preflight','scientific_fits':0,'TRAIN_updates':0,'TEST_accessed':False,'new_GPU_workers':0,'automatic_retries':0}
child=None;started=time.monotonic()
def pin(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
try:
 outcomes=json.loads((E/'results/structural_private_risk_v1/OUTCOMES.json').read_text())
 peers=[];old=process(640428)
 if old:
  assert old['start_ticks']=='6055039165'
  peers.append({'pid':640428,'start_ticks':6055039165,'peak_host_RSS_bytes':16*1024**3})
 live=[r for r in outcomes if r.get('status')=='started' and process(r['worker_pid'])]
 assert len(live)==1,'Refresh CPU admission if existing scientific owner is between fits'
 peers.append({'pid':live[0]['worker_pid'],'start_ticks':live[0]['worker_start_ticks'],'peak_host_RSS_bytes':12*1024**3})
 resources=preflight({'device':'cpu','peers':peers},{'minimum_host_headroom_after_peer_growth_bytes':4*1024**3,'host_RSS_reservation_bytes':4*1024**3,'disk_reservation_bytes':1024**3})
 resource_path=output/'RESOURCES.json';resource_path.write_text(json.dumps(resources,indent=2)+'\n')
 request=json.loads((package/'ROOT_ADMISSION_TEMPLATE_node_C4.json').read_text())
 request.update(status='ADMITTED',harness_manifest_sha256=pin(package/'MANIFEST.json')['sha256'],INPUTS_sha256=pin(package/'INPUTS.json')['sha256'],python_executable=str(R/'.venv/bin/python'),qualified_python_review=pin(E/'ROOT_CPU_PYTHON_REVIEW_15.json'),peer_resource_review=pin(resource_path),output=str(output/'native'))
 admission=output/'ADMISSION.json';admission.write_text(json.dumps(request,indent=2)+'\n')
 command=[str(R/'.venv/bin/python'),'-B','-u',str(package/'replay.py'),'--root',str(R),'--admission',str(admission),'--case','node_C4']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (output/'child.log').open('xb') as log:child=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 report.update(status='running',child=process(child.pid),command=command,admission=pin(admission));peak=0;deadline=time.monotonic()+190
 (output/'SUPERVISOR.json').write_text(json.dumps(report,indent=2)+'\n')
 while child.poll() is None:
  current=process(child.pid)
  if current:peak=max(peak,current['rss_bytes'])
  if peak>4*1024**3:raise RuntimeError('Own CPU replay exceeded physical RSS reservation')
  if time.monotonic()>=deadline:raise TimeoutError('CPU replay exceeded outer 190-second bound')
  if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>1024**3:raise RuntimeError('Own CPU replay exceeded output disk reservation')
  try:child.wait(timeout=min(.5,max(.01,deadline-time.monotonic())))
  except subprocess.TimeoutExpired:pass
 rp=output/'native/RESULT.json';report.update(exit_code=child.returncode,peak_RSS_sampled_bytes=peak,result=json.loads(rp.read_text()) if rp.exists() else None)
 report['status']='passed' if child.returncode==0 and report['result'] and report['result']['status']=='passed' else 'failed'
except BaseException as error:
 report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=8)})
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
finally:
 report['wall_seconds']=time.monotonic()-started
 (output/'SUPERVISOR.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
'''
code = code.replace('@ROOT@', repr(REMOTE_ROOT)).replace('@EXP@', repr(REMOTE_EXPANSION))
result = json.loads(call(code, timeout=230))
with (E/'NODE_SELECTED_VALID_REPLAY_26.json').open('x') as handle:
    json.dump(result,handle,indent=2);handle.write('\n')
native = result.get('result') or {}
print(json.dumps({'status':result['status'],'seconds':result['wall_seconds'],
                 'failure':result.get('failure') or native.get('failure'),
                 'actual_work':native.get('actual_work'),'parity':native.get('parity')}))
