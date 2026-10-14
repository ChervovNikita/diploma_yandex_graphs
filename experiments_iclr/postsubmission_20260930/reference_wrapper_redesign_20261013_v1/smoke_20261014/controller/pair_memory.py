"""Two real-source smoke processes on the one authorized GPU; own PIDs only."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from smoke_case import atomic_json,file_receipt,require,require_allocation

def memory_snapshot(pids):
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10);own=[]
 for line in raw.splitlines():
  fields=[x.strip() for x in line.split(',')]
  if len(fields)==3 and fields[0].isdigit() and int(fields[0]) in pids and fields[2].isdigit():own.append({'pid':int(fields[0]),'gpu_uuid':fields[1],'used_memory_mib':int(fields[2])})
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.free,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).strip();uuid,total,free,used=[s.strip() for s in gpu.split(',')]
 return {'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'own_processes':own,'own_GPU_usage_mib':sum(r['used_memory_mib'] for r in own),'device':{'uuid':uuid,'total_mib':int(total),'free_mib':int(free),'used_mib':int(used)},'other_process_details_recorded':False}

def run(manifest,root,output,python,case_ids,timeout):
 root=Path(root).resolve();manifest=Path(manifest).resolve();output=Path(output).resolve();require(manifest.is_relative_to(root) and output.is_relative_to(root),'Pair source/results outside project');require(not output.exists(),'Preserve previous pair evidence');output.mkdir(parents=True);barrier=output/'barrier';barrier.mkdir();rows=json.loads(manifest.read_text())['cases'];require(all(sum(c['id']==i for c in rows)==1 for i in case_ids),'Two exact source cases required');require(all(next(c for c in rows if c['id']==i).get('device','cuda:0')=='cuda:0' for i in case_ids),'Pair targets the same authorized allocation GPU')
 result={'schema':'two-slot-real-source-pair-memory-v1','manifest':file_receipt(manifest),'case_ids':case_ids,'status':'running','two_jobs_same_gpu':True,'sampling_is_observation_not_complete_upper_bound':True,'samples':[],'workers':[],'automatic_retries':0};logs=[];children=[];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');tick=time.monotonic();script=Path(__file__).with_name('smoke_case.py')
 try:
  result['before']=memory_snapshot([])
  for slot,cid in enumerate(case_ids):
   log=(output/('slot'+str(slot)+'.log')).open('x');logs.append(log);command=[str(python),'-B','-u',str(script),'--manifest',str(manifest),'--case',cid,'--project-root',str(root),'--output',str(output/('slot'+str(slot))),'--barrier',str(barrier),'--slot',str(slot),'--barrier-timeout',str(timeout)];child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env);children.append(child);result['workers'].append({'slot':slot,'case_id':cid,'pid':child.pid,'output':str(output/('slot'+str(slot)))})
  while not all((barrier/('ready'+str(i)+'.json')).exists() for i in range(2)):
   require(all(c.poll() is None for c in children),'Pair child failed before both constructed');require(time.monotonic()-tick<timeout,'Pair construction timeout');time.sleep(.2)
  result['both_constructed']=memory_snapshot([c.pid for c in children]);atomic_json(barrier/'RELEASE.json',{'ready':True,'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
  with (output/'MEMORY_SAMPLES.jsonl').open('x') as samples:
   while any(c.poll() is None for c in children):
    require(time.monotonic()-tick<timeout,'Pair smoke timeout');sample=memory_snapshot([c.pid for c in children]);result['samples'].append(sample);samples.write(json.dumps(sample)+'\n');samples.flush();time.sleep(.2)
  for row,child in zip(result['workers'],children):
   row['exit_code']=child.returncode;q=Path(row['output'])/'RESULT.json'
   if q.is_file():v=json.loads(q.read_text());row['status']=v['status'];row['result']=file_receipt(q)
   else:row['status']='failed';row['failure']='No terminal case result'
  result['status']='passed' if all(r['status']=='passed' and r['exit_code']==0 for r in result['workers']) else 'failed';result['maximum_sampled_own_GPU_usage_mib']=max([result['both_constructed']['own_GPU_usage_mib']]+[s['own_GPU_usage_mib'] for s in result['samples']]);result['minimum_sampled_device_free_mib']=min([result['both_constructed']['device']['free_mib']]+[s['device']['free_mib'] for s in result['samples']])
 except Exception as e:result.update(status='failed',failure={'type':type(e).__name__,'message':str(e)})
 finally:
  for child in children:
   if child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  for log in logs:log.close()
  result['wall_seconds']=time.monotonic()-tick
  try:result['after']=memory_snapshot([])
  except Exception as error:result['final_memory_query_error']=str(error)
  atomic_json(output/'RESULT.json',result)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--project-root',required=True);p.add_argument('--output',required=True);p.add_argument('--python',default=sys.executable);p.add_argument('--cases',nargs=2,required=True);p.add_argument('--timeout',type=int,default=1800);a=p.parse_args();require_allocation();r=run(a.manifest,a.project_root,a.output,a.python,a.cases,a.timeout);print(json.dumps({'status':r['status'],'wall_seconds':r['wall_seconds'],'maximum_sampled_own_GPU_usage_mib':r.get('maximum_sampled_own_GPU_usage_mib')}));sys.exit(0 if r['status']=='passed' else 1)
if __name__=='__main__':main()
