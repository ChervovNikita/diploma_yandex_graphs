"""Root-owned bounded CPU qualifier; no scientific fit or GPU context."""
import argparse, hashlib, json
from pathlib import Path
from remote_control import call, REMOTE_EXPANSION, REMOTE_ROOT
p=argparse.ArgumentParser();p.add_argument('variant',choices=['uniform','structured','permuted']);a=p.parse_args()
E=Path(__file__).resolve().parent
owner=r'''import sys,os,json,subprocess,signal,time,traceback,hashlib
from pathlib import Path
R=Path(@ROOT@);E=Path(@EXP@);variant=@VARIANT@
q=E/'qualification'/('pubmed_cheb_cpu_'+variant+'_23')
sys.path.insert(0,str(E/'qualification/portablev4_link_queued_parent_authority_repair_v1'))
from supervise import preflight,process
started=time.monotonic();child=None
report={'variant':variant,'status':'preflight','scientific_fits':0,'device':'cpu','timeout_seconds':900,'forced_kill_after_seconds':10,'automatic_retries':0,'TEST_accessed':False}
def write():
 report['wall_seconds']=time.monotonic()-started
 temp=q/'SUPERVISOR.partial';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(q/'SUPERVISOR.json')
try:
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==@HEAD@
 path=E/'research/pubmed_cheb_residual_composition_v1/MANIFEST.json'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==@MANIFEST@
 q.mkdir(exist_ok=False)
 # Current scientific worker identity comes from the already-admitted queue.
 outcomes=json.loads((E/'results/structural_private_risk_v1/OUTCOMES.json').read_text())
 rows=outcomes if isinstance(outcomes,list) else outcomes['rows']
 live=[r for r in rows if r.get('status')=='started' and process(r['worker_pid']) is not None]
 assert len(live)==1,'Refresh admission if scientific queue is between fits or finished'
 row=live[0]
 peers=[{'pid':640428,'start_ticks':6055039165,'peak_host_RSS_bytes':16*1024**3},
        {'pid':row['worker_pid'],'start_ticks':row['worker_start_ticks'],'peak_host_RSS_bytes':12*1024**3}]
 admission={'device':'cpu','peers':peers}
 case={'minimum_host_headroom_after_peer_growth_bytes':24*1024**3,'host_RSS_reservation_bytes':24*1024**3,'disk_reservation_bytes':4*1024**3}
 report['preflight']=preflight(admission,case)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 cmd=[str(R/'.venv/bin/python'),'-B','-u',str(path.parent/'qualify.py'),'--project-root',str(R),'--manifest-sha256',@MANIFEST@,'--variant',variant,'--device','cpu','--output',str(q/'native')]
 with (q/'child.log').open('xb') as log:child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 report.update(status='running',command=cmd,child=process(child.pid));write()
 try:code=child.wait(timeout=900)
 except subprocess.TimeoutExpired:
  report['timed_out']=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=10)
  except subprocess.TimeoutExpired:
   report['forced_KILL']=True;os.killpg(child.pid,signal.SIGKILL);code=child.wait()
 result=q/'native/RESULT.json';report['exit_code']=code
 report['native_result']=json.loads(result.read_text()) if result.exists() else None
 report['status']='passed' if code==0 and report['native_result'] and report['native_result']['status']=='passed' and not report.get('timed_out') else 'failed'
except BaseException as error:
 report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=10)})
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();report['forced_KILL']=True
finally:
 if q.exists():write()
 print(json.dumps({'status':report['status'],'variant':variant,'wall_seconds':time.monotonic()-started,'failure':report.get('failure')}),flush=True)
'''
owner=owner.replace('@ROOT@',repr(REMOTE_ROOT)).replace('@EXP@',repr(REMOTE_EXPANSION)).replace('@VARIANT@',repr(a.variant)).replace('@HEAD@',repr('afd61de7cd4ff4fe889d2ee1d7f1ee91232dcf4a')).replace('@MANIFEST@',repr('1c5d3669cec6681bbfbda045a9497805805c287ab3e03b4aca5aa65c0e3cf0dc'))
# Store the exact supervisor source in the project before executing it.
code='''import subprocess,json,os,time,hashlib
from pathlib import Path
E=Path(@EXP@);R=Path(@ROOT@);variant=@VARIANT@
path=E/('pubmed_cpu_owner_'+variant+'_23.py')
assert not path.exists()
path.write_text(@OWNER@)
with (E/('pubmed_cpu_owner_'+variant+'_23.log')).open('xb') as log:
 child=subprocess.Popen(['python3','-B','-u',str(path)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
f=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()
print(json.dumps({'owner_pid':child.pid,'owner_start_ticks':f[19],'variant':variant,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'output':str(E/'qualification'/('pubmed_cheb_cpu_'+variant+'_23')),'launched_unix':time.time()}))
'''
code=code.replace('@EXP@',repr(REMOTE_EXPANSION)).replace('@ROOT@',repr(REMOTE_ROOT)).replace('@VARIANT@',repr(a.variant)).replace('@OWNER@',repr(owner))
receipt=json.loads(call(code));out=E/('PUBMED_CPU_LAUNCH_'+a.variant+'_23.json')
with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
