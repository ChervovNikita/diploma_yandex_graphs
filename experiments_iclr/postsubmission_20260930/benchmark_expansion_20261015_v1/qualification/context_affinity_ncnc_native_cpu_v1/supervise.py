"""Finite CPU-only native C3 context checks; stop admission after the first failure."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.root).resolve();out=Path(a.output).resolve();package=Path(__file__).resolve().parent
 assert out.is_relative_to(root) and package.is_relative_to(root) and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 out.mkdir(parents=True,exist_ok=False)
 methods=['context_affinity_ncnc_common_v1','context_affinity_ncnc_split_v1']
 report={'status':'running','started_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':[],
         'max_optimizer_updates':4,'max_extra_AUX_gradient_collections':2,'automatic_retries':0,'GPU_updates':0,'TEST_accessed':False,'full_campaign_GO':False}
 def save():
  t=out/'SUPERVISOR.json.part';t.write_text(json.dumps(report,indent=2)+'\n');t.replace(out/'SUPERVISOR.json')
 save()
 for i,method in enumerate(methods):
  started=time.perf_counter();target=out/str(i)
  with (out/(str(i)+'.log')).open('x') as log:
   try:
    result=subprocess.run([sys.executable,'-B',str(package/'qualify.py'),'--root',str(root),'--method',method,'--output',str(target)],
                          stdout=log,stderr=subprocess.STDOUT,timeout=1200,check=False)
    row={'method':method,'exit_code':result.returncode,'wall_seconds':time.perf_counter()-started}
   except subprocess.TimeoutExpired:
    row={'method':method,'exit_code':None,'timeout':True,'wall_seconds':time.perf_counter()-started}
  report['cases'].append(row);save()
  if row['exit_code']!=0:
   report.update(status='failed',not_started=methods[i+1:]);save();return 1
 report.update(status='complete',finished_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());save();return 0
if __name__=='__main__':raise SystemExit(main())
