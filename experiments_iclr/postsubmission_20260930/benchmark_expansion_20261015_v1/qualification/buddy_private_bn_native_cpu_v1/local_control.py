"""One explicitly admitted BUDDY BN launch through the literal allocation guard."""
from pathlib import Path
import argparse
import json
import sys

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parents[1]
sys.path.insert(0, str(EXPANSION))
import remote_control

PLAN_SHA = 'd97a4c01faffd9114aa4babafda2662aed0790de6d6bad1804f9f5603dffbcf5'
MANIFEST_SHA = 'b2bfe9e1c2c4050d5f1352864811a0a7fdcc9cd62e7ec6df1e7e217e25bc5cc4'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admitted-plan-sha256', required=True)
    parser.add_argument('--peer-supervisor', type=int, required=True)
    args = parser.parse_args()
    if args.admitted_plan_sha256 != PLAN_SHA:
        raise ValueError('Exact root-admitted frozen plan is required')
    code = r'''from pathlib import Path
import datetime,hashlib,json,os,subprocess
root=Path(ROOT_LITERAL);E=Path(EXPANSION_LITERAL);source=E/'buddy_private_bn_state_v1'
q=E/'qualification/buddy_private_bn_native_cpu_v1';q.mkdir(parents=True,exist_ok=True)
out=E/'results/buddy_private_bn_native_cpu_v1_attempt01';log=out.with_suffix('.driver.log');receipt=q/'LAUNCH.json'
assert not out.exists() and not log.exists() and not receipt.exists(), 'Own attempt already exists; inspect instead of relaunch'
c3=E/'results/context_affinity_BUDDY_native_cpu_v2_attempt01/SUPERVISOR.json'
assert c3.exists() and json.loads(c3.read_text())['status']=='complete', 'C3 gate must complete before BN admission'
assert hashlib.sha256((source/'MANIFEST.json').read_bytes()).hexdigest()==MANIFEST_LITERAL
assert hashlib.sha256((source/'SMOKE_PLAN.json').read_bytes()).hexdigest()==PLAN_LITERAL
for row in json.loads((source/'MANIFEST.json').read_text())['files']:
 raw=(source/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
for row in json.loads((source/'SOURCES.json').read_text())['source_files']:
 raw=(E.parent/row['path_relative_to_phase']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
def tree(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return []
 try:
  state=(p/'stat').read_text().split(') ')[1].split()[0]
  rss=next((int(x.split()[1])*1024 for x in (p/'status').read_text().splitlines() if x.startswith('VmRSS:')),0)
  rows=[{'pid':pid,'state':state,'RSS_bytes':rss}]
  for child in (p/'task'/str(pid)/'children').read_text().split():rows+=tree(int(child))
  return rows
 except FileNotFoundError:return []
peer=tree(PEER_LITERAL);rss=sum(x['RSS_bytes'] for x in peer if x['state']!='Z');active=any(x['state']!='Z' for x in peer)
stat={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
limit=int(Path('/sys/fs/cgroup/memory.max').read_text());current=int(Path('/sys/fs/cgroup/memory.current').read_text())
available=limit-current+stat['inactive_file']+stat['slab_reclaimable'];nonfile=limit-(current-stat['file']);reserve=max(0,20*1024**3-rss) if active else 0
record={'schema':'BUDDY-private-BN-CPU-launch-v1','status':'deferred','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_manifest_sha256':MANIFEST_LITERAL,'root_admitted_plan_sha256':PLAN_LITERAL,'output':str(out),'log':str(log),
 'preflight':{'cache_aware_available_bytes':available,'nonfile_headroom_bytes':nonfile,'peer':peer,'peer_envelope_bytes':20*1024**3,
  'peer_remaining_growth_bytes':reserve,'available_after_peer_growth_bytes':available-reserve,'required_bytes':14*1024**3,
  'required_nonfile_headroom_bytes':80*1024**3,'own_RSS_growth_reservation_bytes':8*1024**3},
 'bounds':{'native_constructions':3,'optimizer_updates':2,'full_VALID_calls':3,'native_forwards':36,'reverse_collections':3,
  'wall_seconds':1800,'CPU_threads':2,'AS_GiB':64},'automatic_retries':0,'GPU_updates':0,'TEST_accessed':False,'scientific_fits':0,
 'memory_events':Path('/sys/fs/cgroup/memory.events').read_text()}
if available-reserve>=14*1024**3 and nonfile-reserve>=80*1024**3:
 context=E/'qualification/link_expansion_ports_BUDDY_v2_context_attempt01/CONTEXT_MANIFEST.json'
 command=[str(root/'.venv/bin/python'),'-B','-u',str(source/'supervise_cpu.py'),'--root',str(root),'--context-manifest',str(context),'--output',str(out),'--admitted-plan-sha256',PLAN_LITERAL]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 with log.open('xb') as stream:child=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 record.update(status='launched',pid=child.pid,command=command)
 with receipt.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps(record))
'''
    for key, value in {'ROOT_LITERAL': remote_control.REMOTE_ROOT, 'EXPANSION_LITERAL': remote_control.REMOTE_EXPANSION,
                       'MANIFEST_LITERAL': MANIFEST_SHA, 'PLAN_LITERAL': PLAN_SHA, 'PEER_LITERAL': args.peer_supervisor}.items():
        code = code.replace(key, repr(value))
    record = json.loads(remote_control.call(code))
    target = HERE/'LAUNCH.json' if record['status']=='launched' else HERE/('DEFER_'+record['UTC'].replace(':','-')+'.json')
    with target.open('x') as stream:
        json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
