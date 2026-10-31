"""Synthetic finite owner governance; no resource helper/model calls."""
import os,sys,tempfile,json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(OUT/'runtime'))
from finite_hgt_owner import run_child,GIB
def main():
 with tempfile.TemporaryDirectory(dir=OUT/'tests') as tmp:
  p=Path(tmp);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');limits={'wall_seconds':5,'rss_bytes':64*GIB,'cuda_bytes':32*GIB,'minimum_free_cuda_bytes':34*GIB}
  success=run_child([sys.executable,'-B','-c','print("engineering metadata only")'],p,'success',env,limits,measure_gpu=False)
  failed=run_child([sys.executable,'-B','-c','import sys;print("retained",file=sys.stderr);sys.exit(7)'],p,'failure',env,limits,measure_gpu=False)
  timeout=run_child([sys.executable,'-B','-c','import time;time.sleep(10)'],p,'timeout',env,{**limits,'wall_seconds':.1},measure_gpu=False)
  assert success['exit_code']==0 and failed['exit_code']==7 and 'retained' in (p/'failure_STDERR.log').read_text()
  assert timeout['exit_code']!=0 and timeout['limit_violation'] in ('child_wall_limit','child_wall_watchdog')
  assert all((p/(n+'_TERMINAL.json')).is_file() for n in ('success','failure','timeout')) and all(r['partials_preserved'] and not r['automatic_retry'] for r in (success,failed,timeout))
 print(json.dumps({'synthetic_owner_cases':3,'success_failure_timeout_terminal_receipts_preserved':True,'actual_source_callbacks':0,'models':0,'fits':0,'scores':0,'GO':False}))
if __name__=='__main__':main()
