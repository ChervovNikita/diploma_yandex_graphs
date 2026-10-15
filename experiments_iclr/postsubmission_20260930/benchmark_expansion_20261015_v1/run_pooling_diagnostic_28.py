"""Execute the reviewed fixed-pooling CPU diagnostic once, without training."""
import json
from pathlib import Path
from remote_control import call, REMOTE_EXPANSION, REMOTE_ROOT

E = Path(__file__).resolve().parent
code = r'''import hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
R=Path(@ROOT@);E=Path(@EXP@)
package=E/'research/residual_positive_bce_pooling_v1'
manifest=@MANIFEST@
assert hashlib.sha256((package/'MANIFEST.json').read_bytes()).hexdigest()==manifest
sys.path.insert(0,str(E/'qualification/portablev4_link_queued_parent_authority_repair_v1'))
from supervise import preflight,process
output=E/'qualification/residual_positive_bce_pooling_28'
analysis_output=E/'analysis_outputs/residual_positive_bce_pooling_v1'
assert not output.exists();output.mkdir()
report={'status':'preflight','scientific_fits':0,'TRAIN_updates':0,'TEST_accessed':False,
        'new_GPU_workers':0,'automatic_retries':0,'reviewed_manifest_sha256':manifest}
child=None;started=time.monotonic()
def pin(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
try:
 peers=[];old=process(640428)
 if old:
  assert old['start_ticks']=='6055039165'
  peers.append({'pid':640428,'start_ticks':6055039165,'peak_host_RSS_bytes':16*1024**3})
 outcomes=json.loads((E/'results/structural_private_risk_v1/OUTCOMES.json').read_text())
 live=[r for r in outcomes if r.get('status')=='started' and process(r['worker_pid'])]
 assert len(live)==1,'Refresh CPU resource check if the scientific queue is between fits or terminal'
 peers.append({'pid':live[0]['worker_pid'],'start_ticks':live[0]['worker_start_ticks'],'peak_host_RSS_bytes':12*1024**3})
 resources=preflight({'device':'cpu','peers':peers},{'minimum_host_headroom_after_peer_growth_bytes':4*1024**3,
             'host_RSS_reservation_bytes':4*1024**3,'disk_reservation_bytes':128*1024**2})
 (output/'RESOURCES.json').write_text(json.dumps(resources,indent=2)+'\n')
 command=[str(R/'.venv/bin/python'),'-B','-u',str(package/'run.py'),'--manifest-sha256',manifest,'--output-dir',str(analysis_output)]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
          NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (output/'child.log').open('xb') as log:
  child=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 report.update(status='running',child=process(child.pid),command=command,resources=pin(output/'RESOURCES.json'))
 (output/'SUPERVISOR.json').write_text(json.dumps(report,indent=2)+'\n')
 peak=0;deadline=time.monotonic()+120
 while child.poll() is None:
  current=process(child.pid)
  if current:peak=max(peak,current['rss_bytes'])
  if peak>4*1024**3:raise RuntimeError('CPU diagnostic exceeded physical RSS reservation')
  if time.monotonic()>=deadline:raise TimeoutError('CPU diagnostic exceeded120-second bound')
  try:child.wait(timeout=min(.5,max(.01,deadline-time.monotonic())))
  except subprocess.TimeoutExpired:pass
 result=analysis_output/'RESULT.json';gate=analysis_output/'GATE.json'
 report.update(exit_code=child.returncode,peak_RSS_sampled_bytes=peak)
 if result.exists():
  value=json.loads(result.read_text())
  report.update(result=pin(result),result_status=value.get('status'),
                result_keys=list(value),analysis_output=str(analysis_output))
 if gate.exists():report['failure']=json.loads(gate.read_text())
 report['status']='passed' if child.returncode==0 and report.get('result_status')=='complete' else 'failed'
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
code=code.replace('@ROOT@',repr(REMOTE_ROOT)).replace('@EXP@',repr(REMOTE_EXPANSION))
manifest_path=E/'research/residual_positive_bce_pooling_v1/MANIFEST.json'
import hashlib
manifest_hash='cc246de7017fdd0f868e584fcb75e453e7519d678804dcd58c96c4536aad95dd'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==manifest_hash
code=code.replace('@MANIFEST@',repr(manifest_hash))
result=json.loads(call(code,timeout=155))
with (E/'POOLING_DIAGNOSTIC_28.json').open('x') as handle:
    json.dump(result,handle,indent=2);handle.write('\n')
print(json.dumps(result))
