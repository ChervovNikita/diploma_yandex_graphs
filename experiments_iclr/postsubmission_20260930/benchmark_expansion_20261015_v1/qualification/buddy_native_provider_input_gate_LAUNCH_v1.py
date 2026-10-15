"""One admitted seed1 provider input traversal; resource deferral never retries work."""
from pathlib import Path
import argparse,json,sys
E=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(E));import remote_control as r
p=argparse.ArgumentParser();p.add_argument('--admitted-plan-sha256',required=True);a=p.parse_args()
assert a.admitted_plan_sha256=='136da676ed70e20c7e97ff8c487b70e5608a4bcc0f6bdb58aab2021c02380017'
code='''from pathlib import Path
import datetime,hashlib,json,os,subprocess
root=Path(ROOT);E=Path(EXPANSION);q=E/'qualification/buddy_native_provider_input_gate_v1';out=E/'results/buddy_native_provider_input_gate_v1_attempt01'
receipt=q/'LAUNCH_RECEIPT.json';log=out.with_suffix('.driver.log')
assert not receipt.exists() and not out.exists() and not log.exists(), 'Existing attempt must be inspected, never relaunched'
assert json.loads((E/'results/buddy_private_bn_native_cpu_v1_attempt01/SUMMARY.json').read_text())['status']=='passed'
assert json.loads((E/'results/interaction_BUDDY_native_cpu_v1_attempt01/SUPERVISOR.json').read_text())['status']=='complete'
for directory,expected in [(E/'buddy_native_provider_v1','e860e7992b9e93d44bdb66dfd027dc543c1f2a7acafa56c1b26092e65e2060cc'),(q,'231409e8af14993a0d368c4e833d817b0e36a3cad1b7acca30a6c1b288f8cc19')]:
 assert hashlib.sha256((directory/'MANIFEST.json').read_bytes()).hexdigest()==expected
 for row in json.loads((directory/'MANIFEST.json').read_text())['files']:
  raw=(directory/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
for row in json.loads((E/'buddy_native_provider_v1/SOURCES.json').read_text())['source_files']:
 raw=(E.parent/row['path_relative_to_phase']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
assert hashlib.sha256((q/'SMOKE_PLAN.json').read_bytes()).hexdigest()==ADMITTED
assert json.loads((q/'SOURCE_REVIEW.json').read_text())['status']=='passed_source_only'
def tree(pid):
 path=Path('/proc')/str(pid)
 if not path.exists():return []
 try:
  state=(path/'stat').read_text().split(') ')[1].split()[0];status=(path/'status').read_text();rss=next((int(x.split()[1])*1024 for x in status.splitlines() if x.startswith('VmRSS:')),0)
  rows=[{'pid':pid,'state':state,'RSS_bytes':rss}]
  for k in (path/'task'/str(pid)/'children').read_text().split():rows+=tree(int(k))
  return rows
 except FileNotFoundError:return []
peer=tree(837804);rss=sum(x['RSS_bytes'] for x in peer if x['state']!='Z');active=any(x['state']!='Z' for x in peer)
mem={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
available=int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text())+mem['inactive_file']+mem['slab_reclaimable']
reserve=max(0,12*1024**3-rss) if active else 0
record={'schema':'BUDDY-provider-input-gate-launch-v1','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'deferred','output':str(out),'log':str(log),'admitted_plan_sha256':ADMITTED,'gate_manifest_sha256':'231409e8af14993a0d368c4e833d817b0e36a3cad1b7acca30a6c1b288f8cc19','source_manifest_sha256':'e860e7992b9e93d44bdb66dfd027dc543c1f2a7acafa56c1b26092e65e2060cc','preflight':{'cache_aware_available_bytes':available,'peer':peer,'peer_envelope_bytes':12*1024**3,'peer_remaining_growth_bytes':reserve,'available_after_peer_growth_bytes':available-reserve,'required_bytes':12*1024**3,'own_RSS_reservation_bytes':8*1024**3},'bounds':json.loads((q/'SMOKE_PLAN.json').read_text())['bounds'],'query_traversal_not_trained_epoch':True,'TEST_accessed':False,'scientific_fits':0,'memory_events':Path('/sys/fs/cgroup/memory.events').read_text()}
if available-reserve>=12*1024**3:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 command=[str(root/'.venv/bin/python'),'-B','-u',str(q/'supervise.py'),'--root',str(root),'--output',str(out)]
 with log.open('xb') as stream:proc=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 record.update(status='launched',pid=proc.pid,command=command)
 with receipt.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\\n')
print(json.dumps(record))
'''.replace('ROOT',repr(r.REMOTE_ROOT)).replace('EXPANSION',repr(r.REMOTE_EXPANSION)).replace('ADMITTED',repr(a.admitted_plan_sha256))
record=json.loads(r.call(code));path=E/'qualification/buddy_native_provider_input_gate_v1'/('LAUNCH_RECEIPT.json' if record['status']=='launched' else 'DEFER01.json')
assert not path.exists();path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
