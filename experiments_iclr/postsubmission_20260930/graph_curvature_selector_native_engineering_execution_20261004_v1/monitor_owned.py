"""Read only exact owned engineering receipts and process metadata."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json
HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
PACKET=PHASE/'graph_curvature_selector_native_execution_wrapper_source_20261004_v2'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--sequence',type=int,required=True);args=parser.parse_args();assert args.sequence>0
 out=HERE/('owned_monitor%03d'%args.sequence);out.mkdir(exist_ok=False)
 spec=importlib.util.spec_from_file_location('native_stage_v2',PACKET/'stage_client.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 code='''from pathlib import Path
import hashlib,json,os,subprocess
phase=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
assert Path.cwd()==phase
result=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
rows=result.stdout.strip().splitlines();assert len(rows)==1 and rows[0].split(',')[0].strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
root=phase/'graph_curvature_selector_native_engineering_execution_20261004_v1'
files=[];handles=[]
for name in ('START_GUARD.json','SUPERVISOR_PID.json','EXECUTION_START.json','EXECUTION_FINISH.json','Squirrel_PREFLIGHT.json','Squirrel_PID.json','Squirrel_CUDA_POLICY.json','Squirrel_PHYSICAL_CLOSURE.json','Squirrel_FINISH.json','Photo_PREFLIGHT.json','Photo_PID.json','Photo_CUDA_POLICY.json','Photo_PHYSICAL_CLOSURE.json','Photo_FINISH.json','Squirrel/QUALIFICATION_REPORT.json','Photo/QUALIFICATION_REPORT.json'):
 p=root/name
 if not p.is_file():continue
 assert not p.is_symlink() and p.resolve().is_relative_to(root) and p.stat().st_size<2000000
 raw=p.read_bytes();v=json.loads(raw);files.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),value=v))
 if name in ('SUPERVISOR_PID.json','Squirrel_PID.json','Photo_PID.json'):
  pid=v['pid'];proc=Path('/proc')/str(pid)
  try:
   raw_stat=(proc/'stat').read_text();fields=raw_stat[raw_stat.rfind(')')+2:].split()
   actual=dict(pid=pid,start_ticks=int(fields[19]),pgid=int(fields[2]),sid=int(fields[3]),state=fields[0],argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x])
   assert actual['start_ticks']==v['start_ticks'] and actual['pgid']==v['pgid']
   if actual['state']!='Z':assert actual['argv']==v['argv']
  except FileNotFoundError:actual=None
  handles.append(dict(receipt=name,expected=v,actual=actual))
progress=[]
for graph in ('Squirrel','Photo'):
 p=root/graph/'WARM_TRACE.jsonl'
 if p.is_file():
  with p.open('rb') as h:
   lines=h.read().splitlines()
  progress.append(dict(graph=graph,completed_trace_records=len(lines),quality_values_exposed=False))
print(json.dumps(dict(sole_GPU_uuid=rows[0].split(',')[0].strip(),free_MiB=int(rows[0].split(',')[1]),files=files,handles=handles,warm_progress=progress,predictive_continuation=False,arrays_or_checkpoints_read=False,signals=False)))
'''
 r=m.ssh([m.PYTHON,'-B','-'],code.encode())
 with (out/'TRANSPORT.json').open('x') as h:json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stderr=r.stderr.decode(),stdout_sha256=hashlib.sha256(r.stdout).hexdigest()),h,indent=2)
 assert r.returncode==0,r.stderr.decode()
 value=json.loads(r.stdout)
 with (out/'OBSERVATION.json').open('x') as h:json.dump(value,h,indent=2);h.write('\n')
 finish=next((x['value'] for x in value['files'] if x['path']=='EXECUTION_FINISH.json'),None)
 reports=[dict(path=x['path'],status=x['value']['status'],error=x['value'].get('error'),costs=x['value'].get('costs'),total_wall_seconds=x['value'].get('total_wall_seconds')) for x in value['files'] if x['path'].endswith('/QUALIFICATION_REPORT.json')]
 print(json.dumps(dict(status=finish['status'] if finish else 'ENGINEERING_IN_PROGRESS',finish=finish,reports=reports,warm_progress=value['warm_progress'],free_MiB=value['free_MiB']),indent=2))
if __name__=='__main__':main()
