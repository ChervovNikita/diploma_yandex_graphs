"""Run exactly the two root-reviewed stochastic CPU checks; no fit or retry."""
import json
from pathlib import Path
from remote_control import call, REMOTE_ROOT, REMOTE_EXPANSION

E = Path(__file__).resolve().parent
code = r'''import json,subprocess,os,time,signal,hashlib,sys
from pathlib import Path
R=Path(@ROOT@);E=Path(@EXP@)
sys.path.insert(0,str(E/'qualification/portablev4_link_queued_parent_authority_repair_v1'))
from supervise import process
p=E/'research/stochastic_private_adapter_source_v1'
result={'scientific_fits':0,'new_GPU_workers':0,'automatic_retries':0,'TEST_accessed':False,'invocations':[]}
record=E/'STOCHASTIC_CPU_EXECUTION_30.json'
assert not record.exists()
def save():
 temporary=record.with_suffix('.partial');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(record)
def terminate(child):
 if child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
save()
for arm in ('fixed','learned'):
 admission=E/('STOCHASTIC_CPU_'+arm.upper()+'_ADMISSION_30.json')
 output=E/('qualification/stochastic_private_'+arm+'_v1')
 assert not output.exists()
 command=[str(R/'.venv/bin/python'),'-B','-u',str(p/'qualify_cpu.py'),'--project-root',str(R),'--arm',arm,'--output',str(output),'--admission',str(admission),'--admission-sha256',hashlib.sha256(admission.read_bytes()).hexdigest()]
 log=E/('qualification/stochastic_private_'+arm+'_v1.log')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 row={'arm':arm,'command':command,'status':'starting','wall_limit_seconds':900,'host_RSS_limit_bytes':8*1024**3,'log':str(log)}
 result['invocations'].append(row);save();started=time.monotonic();peak=0
 child=None
 try:
  with log.open('xb') as stream:child=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  row.update(status='running',worker=process(child.pid));save()
  while child.poll() is None:
   current=process(child.pid)
   if current:peak=max(peak,current['rss_bytes'])
   if time.monotonic()-started>900 or peak>8*1024**3:
    row['limit_hit']='time' if time.monotonic()-started>900 else 'host_RSS';terminate(child);break
   try:child.wait(timeout=2)
   except subprocess.TimeoutExpired:pass
 finally:
  if child is not None:terminate(child)
  row.update(returncode=None if child is None else child.returncode,wall_seconds=time.monotonic()-started,sampled_peak_RSS_bytes=peak)
  report=output/'RESULT.json'
  if report.exists():
   raw=report.read_bytes();value=json.loads(raw)
   row['RESULT']={'path':str(report),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
   row['qualification']=value
  row['status']='passed' if row['returncode']==0 and row.get('qualification',{}).get('status')=='passed' and not row.get('limit_hit') else 'failed'
  save()
 if row['status']!='passed':break
result['status']='passed' if len(result['invocations'])==2 and all(r['status']=='passed'for r in result['invocations']) else 'stopped_after_failure'
save();print(json.dumps(result))
'''
code = code.replace('@ROOT@', repr(REMOTE_ROOT)).replace('@EXP@', repr(REMOTE_EXPANSION))
result = json.loads(call(code, timeout=1850))
with (E/'STOCHASTIC_CPU_EXECUTION_30.json').open('x') as stream:
    json.dump(result, stream, indent=2);stream.write('\n')
print(json.dumps({'status':result['status'],'invocations':[{k:r.get(k) for k in ('arm','status','returncode','wall_seconds','sampled_peak_RSS_bytes')} for r in result['invocations']]}))
