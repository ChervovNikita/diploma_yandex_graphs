"""Sequential fresh-process real-source smoke dispatcher. Failures retained, no retry."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from smoke_case import atomic_json,file_receipt,require,require_allocation

def run(manifest,project_root,output,python,timeout,selected=None):
 manifest=Path(manifest).resolve();root=Path(project_root).resolve();output=Path(output).resolve()
 require(manifest.is_relative_to(root) and output.is_relative_to(root),'Manifest/results outside project')
 require(not output.exists(),'Retain earlier smoke attempts; choose fresh output');output.mkdir(parents=True)
 d=json.loads(manifest.read_text());cases=d['cases'];ids=[c['id'] for c in cases];require(len(set(ids))==len(ids),'Repeated source-smoke case id')
 if d.get('expected_case_count') is not None:require(len(cases)==d['expected_case_count'],'Manifest callback count differs')
 if selected:
  require(set(selected)<=set(ids),'Unknown requested case');cases=[c for c in cases if c['id'] in selected]
 summary={'schema':'real-source-sequential-smoke-dispatch-v1','manifest':file_receipt(manifest),'case_count':len(cases),'declared_case_count':len(ids),'cases':[],'actual_source_science_scores':0,'training_is_short_engineering_only':True,'automatic_retries':0,'started_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 atomic_json(output/'SUMMARY.json',summary);script=Path(__file__).with_name('smoke_case.py');env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 for index,case in enumerate(cases):
  folder=output/('case_'+str(index).zfill(3));command=[str(python),'-B','-u',str(script),'--manifest',str(manifest),'--case',case['id'],'--project-root',str(root),'--output',str(folder)];before=time.perf_counter();row={'case_id':case['id'],'procedure':case['procedure'],'output':str(folder),'callback_module':case['callback_module'],'status':'running','started_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
  with (output/('case_'+str(index).zfill(3)+'.log')).open('x') as log:
   child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env);row['pid']=child.pid;summary['cases'].append(row);atomic_json(output/'SUMMARY.json',summary)
   try:code=child.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=None;row['failure']='own smoke timeout; partial log/state retained; no automatic retry'
  row['exit_code']=code;row['wall_seconds']=time.perf_counter()-before
  result=folder/'RESULT.json'
  if result.is_file():
   r=json.loads(result.read_text());row.update(status=r['status'],result=file_receipt(result),cuda_peak_allocated_bytes=max(x.get('cuda_peak_allocated_bytes',0) for x in r.get('resource_observations',[]) or [{}]),cuda_peak_reserved_bytes=max(x.get('cuda_peak_reserved_bytes',0) for x in r.get('resource_observations',[]) or [{}]),actual_optimizer_step_calls=r.get('actual_optimizer_step_calls_total'),failure=r.get('failure',row.get('failure')))
  else:row['status']='failed';row.setdefault('failure','worker ended before terminal result; logs retained')
  atomic_json(output/'SUMMARY.json',summary);print(json.dumps({'case_id':case['id'],'status':row['status'],'completed':index+1,'total':len(cases)}),flush=True)
 summary['status']='passed' if all(c['status']=='passed' for c in summary['cases']) else 'failed';summary['passed_cases']=sum(c['status']=='passed' for c in summary['cases']);summary['failed_cases']=len(cases)-summary['passed_cases'];summary['finished_UTC']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());atomic_json(output/'SUMMARY.json',summary);return summary

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--project-root',required=True);p.add_argument('--output',required=True);p.add_argument('--python',default=sys.executable);p.add_argument('--case-timeout',type=int,default=1800);p.add_argument('--only',nargs='*');a=p.parse_args();require_allocation();r=run(a.manifest,a.project_root,a.output,a.python,a.case_timeout,a.only);sys.exit(0 if r['status']=='passed' else 1)
if __name__=='__main__':main()
