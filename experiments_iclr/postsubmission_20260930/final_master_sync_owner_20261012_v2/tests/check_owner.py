"""Synthetic child supervision only; no master/input/source arrays loaded."""
import json,os,sys,tempfile
from pathlib import Path
OUT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(OUT/'runtime'))
from finite_owner import run_child
def main():
 with tempfile.TemporaryDirectory(dir=OUT/'tests') as tmp:
  p=Path(tmp);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
  a=run_child([sys.executable,'-B','-c','print("metadata child complete")'],p,'success',env,5)
  b=run_child([sys.executable,'-B','-c','import sys;print("retained failure",file=sys.stderr);sys.exit(7)'],p,'failure',env,5)
  c=run_child([sys.executable,'-B','-c','import time;time.sleep(10)'],p,'timeout',env,.1)
  assert a['exit_code']==0 and not a['timed_out'];assert b['exit_code']==7 and 'retained failure' in (p/'failure_STDERR.log').read_text()
  assert c['timed_out'] and c['exit_code']!=0 and all((p/(n+'_RESULT.json')).exists() for n in ('success','failure','timeout'))
  assert len({r['owner']['pid'] for r in (a,b,c)})==3 and all(r['partial_outputs_preserved'] for r in (a,b,c))
 print(json.dumps({'synthetic_child_checks':3,'success_failure_timeout_and_exact_owner_preservation':True,'models':0,'scientific_fits':0,'GO':False}))
if __name__=='__main__':main()
