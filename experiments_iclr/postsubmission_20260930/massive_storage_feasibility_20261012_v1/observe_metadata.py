"""Guarded metadata-only storage observation. No fits or targets decoded."""
import json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'training_pipeline_v2_20261012/preparation'))
from prepare_molecular_packets_allocation import GUARD,invoke
REMOTE=GUARD+r'''
import collections,gzip
planpath=phase/'massive_practical_master_20261012_v21/PLAN.json'
plan=json.loads(planpath.read_text());canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
assert hashlib.sha256(canonical(plan)).hexdigest()=='1088d651a64f660bff122b6c8c03b92932e5c750da87de42e405ba279748a755'
placement={key:g['gpu_assignment'] for g in plan['groups'] for key in g['run_keys']}
rows=[];adapters={};factories=set()
for pin in plan['studies']:
 p=phase/pin['path'];raw=p.read_bytes();s=json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)
 assert hashlib.sha256(canonical(s)).hexdigest()==pin['spec_hash']
 fits=[r for r in s['runs'] if r['kind']=='model_fit']
 linked=any(any(k in r['adapter'].lower() for k in ('link','ncn','collab','buddy')) for r in fits) or 'collab' in s['dataset'].lower()
 if not linked:continue
 groups=collections.defaultdict(list)
 for r in fits:
  key=(r['adapter'],r['method'],r['budget']['unit'],r['budget']['maximum'],placement[s['study_id']+'/'+r['run_id']])
  groups[key].append(r)
  factories.add(r['adapter'].split(':',1)[0])
 for (adapter,method,unit,maximum,worker),runs in sorted(groups.items()):
  rows.append({'study_id':s['study_id'],'dataset':s['dataset'],'method':method,'factory':adapter,'fit_groups':len(runs),'worker':worker,'budget_unit':unit,'maximum':maximum,'example_config':runs[0]['config'],'example_parents':runs[0]['parents'],'study_file':pin})
for rel in sorted(factories):
 p=phase/rel;lines=p.read_text().splitlines();interesting=[]
 for i,line in enumerate(lines):
  if any(k in line for k in ('source_integer_replay','replay_directory','write_epoch','REPLAY_MASTER','def capture_state','def make_adapter','replay_provider','sourcecount','endpoint')):
   lo=max(0,i-4);hi=min(len(lines),i+14);interesting.append({'start':lo+1,'end':hi,'text':'\n'.join(lines[lo:hi])})
 adapters[rel]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'excerpts':interesting}
fs=os.statvfs(repo)
print(json.dumps({'plan_file_sha256':hashlib.sha256(planpath.read_bytes()).hexdigest(),'plan_hash':hashlib.sha256(canonical(plan)).hexdigest(),'link_rows':rows,'source_excerpts':adapters,'available_bytes':fs.f_bavail*fs.f_frsize,'science_fits':0,'targets_decoded':0}))
'''
r=invoke(REMOTE,[],BASE,'OBSERVE_LINK_METADATA',180)
assert r['exit_code']==0,r
j=json.loads(r['stdout']);(BASE/'LINK_METADATA.json').write_text(json.dumps(j,indent=2,sort_keys=True)+'\n')
print(json.dumps({'link_rows':len(j['link_rows']),'factories':len(j['source_excerpts']),'available_bytes':j['available_bytes'],'science_fits':0}))
