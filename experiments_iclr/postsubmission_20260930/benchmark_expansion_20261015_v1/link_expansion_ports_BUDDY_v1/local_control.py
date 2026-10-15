"""Guarded own-package stage/finite CPU launch/compact result collection."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
PHASE=HERE.parents[1]
spec=importlib.util.spec_from_file_location('_guarded_allocation',PHASE/'reference_wrapper_redesign_20261013_v1/smoke_20261014/remote_ops.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
REMOTE=remote.REMOTE_ROOT+'/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/link_expansion_ports_BUDDY_v1'
CONTEXT=remote.REMOTE_ROOT+'/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/qualification/link_expansion_ports_BUDDY_v1_context_attempt01'
OUTPUT=remote.REMOTE_ROOT+'/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/qualification/link_expansion_ports_BUDDY_v1_BUDDY_cpu_attempt01'


def save(name,value):
    path=HERE/name
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')


def stage():
    names=[row['path'] for row in json.loads((HERE/'MANIFEST.json').read_text())['files']]+['MANIFEST.json']
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name in names:archive.add(HERE/name,arcname=name,recursive=False)
    packet=buffer.getvalue()
    code=remote.GUARD+'''import io,sys,tarfile,hashlib,json
from pathlib import Path
root=Path(ROOT_LITERAL);root.mkdir(exist_ok=True)
print("READY",flush=True)
packet=sys.stdin.buffer.read(SIZE_LITERAL)
assert len(packet)==SIZE_LITERAL and hashlib.sha256(packet).hexdigest()==HASH_LITERAL
rows=[]
with tarfile.open(fileobj=io.BytesIO(packet),mode="r:gz") as archive:
 for item in archive.getmembers():
  assert item.isfile() and item.name in NAMES_LITERAL
  target=root/item.name;assert target.resolve().is_relative_to(root) and not target.is_symlink()
  raw=archive.extractfile(item).read()
  if target.exists():assert target.read_bytes()==raw,"Preserve existing new-port source"
  else:
   with target.open("xb") as stream:stream.write(raw)
  rows.append({"path":str(target),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
print(json.dumps({"source_files":rows,"fits_started":0}),flush=True)
'''.replace('ROOT_LITERAL',repr(REMOTE)).replace('SIZE_LITERAL',str(len(packet))).replace('HASH_LITERAL',repr(hashlib.sha256(packet).hexdigest())).replace('NAMES_LITERAL',repr(names))
    proc=subprocess.Popen(remote.SSH+['stty raw -echo; python3 -B -c '+shlex.quote(code)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    first=proc.stdout.readline()
    if first.strip()!=b'READY':
        proc.terminate();out,err=proc.communicate(timeout=15);raise RuntimeError((first+out+err).decode(errors='replace'))
    out,err=proc.communicate(input=packet,timeout=60)
    if proc.returncode:raise RuntimeError((out+err).decode(errors='replace'))
    value=json.loads(out);save('TRANSFER_RECEIPT.json',value);return {'source_files':len(value['source_files'])}


def launch():
    expected=hashlib.sha256((HERE/'MANIFEST.json').read_bytes()).hexdigest()
    context_launch=json.loads((HERE/'CONTEXT_LAUNCH.json').read_text())
    code='''import hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
root=Path(ROOT_LITERAL);source=Path(SOURCE_LITERAL);output=Path(OUTPUT_LITERAL);log=Path(str(output)+'.log')
context=Path(CONTEXT_LITERAL);context_manifest=context/'CONTEXT_MANIFEST.json'
context_value=json.loads(context_manifest.read_text())
context_proc=Path('/proc')/str(CONTEXT_PID_LITERAL)/'stat'
assert not context_proc.exists() or context_proc.read_text().split()[2]=='Z','Keep CPU materialization and qualification sequential'
assert context_value['status']=='passed' and context_value['optimizer_transitions']==0 and context_value['TEST_accessed'] is False
assert not output.exists() and not log.exists()
assert hashlib.sha256((source/'MANIFEST.json').read_bytes()).hexdigest()==EXPECTED_LITERAL
for row in json.loads((source/'MANIFEST.json').read_text())['files']:
 p=source/row['path'];assert p.resolve().is_relative_to(source)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'] and p.stat().st_size==row['bytes']
spec=importlib.util.spec_from_file_location('_new_BUDDY_preflight',source/'session.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
pins=module.bindings(root,EXPECTED_LITERAL)
stat={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
limit=int(Path('/sys/fs/cgroup/memory.max').read_text());current=int(Path('/sys/fs/cgroup/memory.current').read_text())
nonfile=stat['anon']+stat.get('kernel',stat.get('slab_unreclaimable',0))+stat.get('shmem',0)
assert limit-nonfile>80*1024**3,'Insufficient cgroup non-file headroom for bounded64GiB child'
command=[str(root/'.venv/bin/python'),'-B','-u',str(source/'supervise_cpu.py'),'--context-manifest',str(context_manifest),'--root',str(root),'--output',str(output)]
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with log.open('xb') as stream:child=subprocess.Popen(command,cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print(json.dumps({'pid':child.pid,'command':command,'output':str(output),'log':str(log),'source_manifest_sha256':EXPECTED_LITERAL,
 'source_receipt_count':len(pins),'maximum_optimizer_transitions':16,'maximum_FULL_VALID_calls':24,'scientific_fits_started':0,'GPU_updates':0,
 'TEST_accessed':False,'memory_preflight':{'memory_max':limit,'memory_current':current,'memory_stat':stat,'nonfile_estimate':nonfile,
 'file_cache_reclamation_not_forced':True,'memory_events_before':Path('/sys/fs/cgroup/memory.events').read_text()},'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}))
'''.replace('ROOT_LITERAL',repr(remote.REMOTE_ROOT)).replace('SOURCE_LITERAL',repr(REMOTE)).replace('OUTPUT_LITERAL',repr(OUTPUT)).replace('EXPECTED_LITERAL',repr(expected)).replace('CONTEXT_LITERAL',repr(CONTEXT)).replace('CONTEXT_PID_LITERAL',str(context_launch['pid']))
    value=json.loads(remote.call(code,timeout=60));save('LAUNCH.json',value);return value


def observe():
    launch_value=json.loads((HERE/'LAUNCH.json').read_text())
    code='''import json
from pathlib import Path
pid=PID_LITERAL;output=Path(OUTPUT_LITERAL);proc=Path('/proc')/str(pid)/'stat'
state=proc.read_text().split()[2] if proc.exists() else None
summary=json.loads((output/'SUMMARY.json').read_text()) if (output/'SUMMARY.json').exists() else None
current=None
if summary and summary['cases']:
 p=Path(summary['cases'][-1]['output'])/'RESULT.json'
 if p.exists():
  value=json.loads(p.read_text());current={k:value.get(k) for k in ('method','status','stage','actual_optimizer_transitions','VALID_calls','elapsed_seconds','process_peak_RSS_KiB','failure')}
print(json.dumps({'pid':pid,'alive':state is not None and state!='Z','summary':summary,'current':current}))
'''.replace('PID_LITERAL',str(launch_value['pid'])).replace('OUTPUT_LITERAL',repr(OUTPUT))
    value=json.loads(remote.call(code));save('OBSERVATION_'+str(len(list(HERE.glob('OBSERVATION_*.json')))+1).zfill(2)+'.json',value);return value


def fetch():
    code='''import json,hashlib
from pathlib import Path
output=Path(OUTPUT_LITERAL);summary=json.loads((output/'SUMMARY.json').read_text())
assert summary['status'] in ('passed','failed')
rows=[]
for case in summary['cases']:
 p=Path(case['output'])/'RESULT.json'
 rows.append({'method':case['method'],'result':json.loads(p.read_text()) if p.exists() else None,'log_tail':(output/(case['method']+'.log')).read_text()[-4000:]})
print(json.dumps({'summary':summary,'cases':rows}))
'''.replace('OUTPUT_LITERAL',repr(OUTPUT))
    value=json.loads(remote.call(code,timeout=60));save('CPU_RESULTS.json',value)
    return value['summary']


def launch_context():
    expected=hashlib.sha256((HERE/'MANIFEST.json').read_bytes()).hexdigest()
    ncnc_launch=json.loads((HERE.parent/'link_expansion_ports_NCNC_v1/LAUNCH.json').read_text())
    code='''import json,os,subprocess,time,hashlib
from pathlib import Path
root=Path(ROOT_LITERAL);source=Path(SOURCE_LITERAL);output=Path(CONTEXT_LITERAL);log=Path(str(output)+'.log')
proc=Path('/proc')/str(NCNC_PID_LITERAL)
if (proc/'stat').exists() and (proc/'stat').read_text().split()[2]!='Z':
 assert b'link_expansion_ports_NCNC_v1' not in (proc/'cmdline').read_bytes(),'Keep CPU source checks sequential; NCNC still active'
assert not output.exists() and not log.exists()
assert hashlib.sha256((source/'MANIFEST.json').read_bytes()).hexdigest()==EXPECTED_LITERAL
for row in json.loads((source/'MANIFEST.json').read_text())['files']:
 p=source/row['path'];assert p.resolve().is_relative_to(source)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'] and p.stat().st_size==row['bytes']
stat={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
limit=int(Path('/sys/fs/cgroup/memory.max').read_text());nonfile=stat['anon']+stat.get('kernel',stat.get('slab_unreclaimable',0))+stat.get('shmem',0)
assert limit-nonfile>80*1024**3,'Insufficient bounded context headroom'
command=[str(root/'.venv/bin/python'),'-B','-u',str(source/'prepare_context_cpu.py'),'--root',str(root),'--output',str(output)]
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with log.open('xb') as stream:child=subprocess.Popen(command,cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print(json.dumps({'pid':child.pid,'command':command,'output':str(output),'log':str(log),'source_manifest_sha256':EXPECTED_LITERAL,'threads':2,'address_space_cap_bytes':64*1024**3,'optimizer_transitions_authorized':0,'model_forwards_authorized':0,'TEST_accessed':False,'memory_stat':stat,'memory_events_before':Path('/sys/fs/cgroup/memory.events').read_text(),'source_checks_remain_sequential':True}))
'''.replace('ROOT_LITERAL',repr(remote.REMOTE_ROOT)).replace('SOURCE_LITERAL',repr(REMOTE)).replace('CONTEXT_LITERAL',repr(CONTEXT)).replace('NCNC_PID_LITERAL',str(ncnc_launch['pid'])).replace('EXPECTED_LITERAL',repr(expected))
    value=json.loads(remote.call(code,timeout=60));save('CONTEXT_LAUNCH.json',value);return value


def observe_context():
    launch_value=json.loads((HERE/'CONTEXT_LAUNCH.json').read_text())
    code='''import json
from pathlib import Path
pid=PID_LITERAL;output=Path(CONTEXT_LITERAL);proc=Path('/proc')/str(pid)/'stat'
state=proc.read_text().split()[2] if proc.exists() else None
path=output/'CONTEXT_MANIFEST.json';value=json.loads(path.read_text()) if path.exists() else None
print(json.dumps({'alive':state is not None and state!='Z','context':None if value is None else {k:value.get(k) for k in ('status','stage','elapsed_seconds','process_peak_RSS_KiB','optimizer_transitions','model_forwards','failure')}}))
'''.replace('PID_LITERAL',str(launch_value['pid'])).replace('CONTEXT_LITERAL',repr(CONTEXT))
    value=json.loads(remote.call(code));save('CONTEXT_OBSERVATION_'+str(len(list(HERE.glob('CONTEXT_OBSERVATION_*.json')))+1).zfill(2)+'.json',value);return value


def fetch_context():
    code='''import json
from pathlib import Path
path=Path(CONTEXT_LITERAL)/'CONTEXT_MANIFEST.json';value=json.loads(path.read_text());assert value['status'] in ('passed','failed')
print(json.dumps(value))
'''.replace('CONTEXT_LITERAL',repr(CONTEXT))
    value=json.loads(remote.call(code));save('NATIVE_CONTEXT_MANIFEST.json',value);return {k:value.get(k) for k in ('status','elapsed_seconds','process_peak_RSS_KiB','context_file')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('stage','launch_context','observe_context','fetch_context','launch','observe','fetch'))
    args=parser.parse_args();print(json.dumps(globals()[args.mode](),sort_keys=True))
