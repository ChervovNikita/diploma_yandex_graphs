"""Finite normal CPU owner; synthetic state checks, no benchmark fit or scoring."""
import base64,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;P=HERE.parent
sys.path.insert(0,str(P/'training_pipeline_v2_20261012/preparation'))
from prepare_molecular_packets_allocation import GUARD,invoke
REMOTE=GUARD+r'''
payload=json.loads(base64.b64decode(sys.argv[1]));stage=phase/'all_method_gap_ports_root_20261013_v3';out=stage/sys.argv[2];assert not out.exists();out.mkdir()
q=out/'engineering_qa.py';raw=base64.b64decode(payload['data']);assert hashlib.sha256(raw).hexdigest()==payload['sha256'];q.write_bytes(raw)
owner=out/'owner.py';owner.write_text("import os,resource,sys\nresource.setrlimit(resource.RLIMIT_AS,(24*1024**3,24*1024**3))\nos.execv(sys.argv[1],sys.argv[1:])\n")
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env['TMPDIR']=str(out)
cmd=[str(repo/'.venv/bin/python'),'-B',str(owner),str(repo/'.venv/bin/python'),'-B','-u',str(q),str(phase),str(out)]
tick=time.monotonic();child=subprocess.Popen(cmd,cwd=phase,env=env,stdout=(out/'stdout.txt').open('w'),stderr=(out/'stderr.txt').open('w'))
fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split();start={'pid':child.pid,'start_ticks':int(fields[19]),'command':cmd,'source_sha256':payload['sha256'],'timeout_seconds':600,'scientific_fits':0,'GO':False};(out/'START.json').write_text(json.dumps(start,indent=2)+'\n')
print(json.dumps({'owner':start,'out':str(out.relative_to(phase))}),flush=True)
try:child.wait(timeout=600)
except subprocess.TimeoutExpired:child.kill();child.wait()
result={'exit_code':child.returncode,'seconds':time.monotonic()-tick,'stderr':(out/'stderr.txt').read_text(),'result':json.loads((out/'RESULT.json').read_text()) if (out/'RESULT.json').exists() else None,'scientific_fits':0,'GO':False};(out/'TRANSPORT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));sys.exit(0 if child.returncode==0 else 1)
'''
def main():
 name=sys.argv[1];out=HERE/name;out.mkdir();raw=(HERE/'engineering_qa.py').read_bytes();payload={'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()}
 r=invoke(REMOTE,[base64.b64encode(json.dumps(payload).encode()).decode(),name],out,'QA',650);assert r['exit_code']==0,'Preserve engineering failure'
 rows=[json.loads(s) for s in r['stdout'].splitlines() if s.startswith('{')];(out/'RESULT.json').write_text(json.dumps(rows[-1],indent=2)+'\n');print(json.dumps(rows[-1]))
if __name__=='__main__':main()
