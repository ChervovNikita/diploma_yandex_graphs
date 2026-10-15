from pathlib import Path
import sys,json
E=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(E));import remote_control as r
code='''from pathlib import Path
import datetime,hashlib,json,os,subprocess
root=Path(ROOT);E=Path(EXPANSION);q=E/'qualification/interaction_BUDDY_native_cpu_v1';out=E/'results/interaction_BUDDY_native_cpu_v1_attempt01'
receipt=q/'LAUNCH_OR_DEFER.json';log=out.with_suffix('.driver.log')
bn=E/'results/buddy_private_bn_native_cpu_v1_attempt01/SUMMARY.json'
assert bn.exists() and json.loads(bn.read_text())['status']=='passed', 'Complete approved privateBN gate first'
assert not receipt.exists() and not out.exists() and not log.exists(), 'Own attempt already exists; inspect instead of relaunch'
for directory,expected in [(E/'interaction_BUDDY_v1','25e292c5644804da1389983ed383f31bb5a138625fa89bb44cfe112487341c39'),(q,'0e4650a35698afa985c4c803b456f63b3609a232169af540e492f0c6a0a00db1')]:
 assert hashlib.sha256((directory/'MANIFEST.json').read_bytes()).hexdigest()==expected
 for row in json.loads((directory/'MANIFEST.json').read_text())['files']:
  raw=(directory/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
for row in json.loads((E/'interaction_BUDDY_v1/BINDINGS.json').read_text())['native_sources']:
 raw=(E.parent/row['path_relative_to_phase']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
def tree(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return []
 try:
  state=(p/'stat').read_text().split(') ')[1].split()[0];status=(p/'status').read_text();rss=next((int(x.split()[1])*1024 for x in status.splitlines() if x.startswith('VmRSS:')),0)
  rows=[{'pid':pid,'state':state,'RSS_bytes':rss}]
  for k in (p/'task'/str(pid)/'children').read_text().split():rows+=tree(int(k))
  return rows
 except FileNotFoundError:return []
peer=tree(836106);rss=sum(x['RSS_bytes'] for x in peer if x['state']!='Z');active=any(x['state']!='Z' for x in peer)
mem={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
available=int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text())+mem['inactive_file']+mem['slab_reclaimable']
reserve=max(0,20*1024**3-rss) if active else 0
record={'schema':'BUDDY-interaction-CPU-launch-v1','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'deferred','output':str(out),'log':str(log),
 'source_manifest_sha256':'25e292c5644804da1389983ed383f31bb5a138625fa89bb44cfe112487341c39','gate_manifest_sha256':'0e4650a35698afa985c4c803b456f63b3609a232169af540e492f0c6a0a00db1',
 'preflight':{'cache_aware_available_bytes':available,'peer':peer,'peer_envelope_bytes':20*1024**3,'peer_remaining_growth_bytes':reserve,'available_after_peer_growth_bytes':available-reserve,'required_bytes':12*1024**3,'own_RSS_reservation_bytes':8*1024**3},
 'bounds':{'optimizer_updates':8,'native_constructions':8,'full_VALID_calls':12,'native_forwards':128,'extra_gradient_collections':0,'declared_reverse_collections':12,'CPU_threads':2,'per_mode_wall_seconds':600,'per_child_AS_GiB':64},'TEST_accessed':False,'scientific_fits':0,'automatic_retries':0,'memory_events':Path('/sys/fs/cgroup/memory.events').read_text()}
if available-reserve>=12*1024**3:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 command=[str(root/'.venv/bin/python'),'-B','-u',str(q/'supervise.py'),'--root',str(root),'--output',str(out)]
 with log.open('xb') as f:p=subprocess.Popen(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 record.update(status='launched',pid=p.pid,command=command)
 with receipt.open('x') as f:json.dump(record,f,indent=2);f.write('\\n')
print(json.dumps(record))
'''.replace('ROOT',repr(r.REMOTE_ROOT)).replace('EXPANSION',repr(r.REMOTE_EXPANSION))
record=json.loads(r.call(code));p=E/'qualification/interaction_BUDDY_native_cpu_v1'/('LAUNCH_OR_DEFER.json' if record['status']=='launched' else 'DEFER01.json');p.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
