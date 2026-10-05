"""Read only owned identities, resource/progress and launch custody metadata."""
from pathlib import Path
import base64,hashlib,importlib.util,json,zlib
P=Path('/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930')
HERE=P/'shared_private_transfer_gpu77_block_launch_execution_root_20261005_v1'
OUT=P/'shared_private_transfer_gpu77_block_observation_20261005_v4'
PREP=P/'shared_private_transfer_gpu77_block_launch_preparation_20261005_v1'
spec=importlib.util.spec_from_file_location('blockobserve77_transport',P/'shared_private_transfer_gpu77_environment_execution_20261005_v1/remote_transport.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);t.HERE=OUT
launch={b:json.loads((HERE/(b+'.LAUNCH_RECEIPT.json')).read_text()) for b in ('b1','b2')}
expected=json.loads((HERE/'FROZEN_AUTHENTICATION.json').read_text())['blocks']
code=f'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,importlib.util,json,socket,zlib
repo=Path({str(t.REPO)!r});phase=Path({str(t.REMOTE_PHASE)!r});prep=phase/{PREP.name!r};receiptroot=phase/{HERE.name!r};launch={launch!r};expected={expected!r}
assert Path.cwd()==repo and socket.gethostname()=='peptide'
spec=importlib.util.spec_from_file_location('block77_common',prep/'pilot_common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def read(path):
 assert path.is_file() and not path.is_symlink() and path.stat().st_size<262144
 return c.read(path)
def physical(s,owner):
 if owner is None:return None
 current=s.identity(owner['PID'])
 if current is None:return None
 matches=all(current[k]==owner[k] for k in ('PID','start_ticks','argv','pgid','sid'))
 cwd=(Path('/proc')/str(owner['PID'])/'cwd').resolve(strict=True) if current['state']!='Z' else None
 return dict(identity=current,fresh_identity_matches=matches,cwd=str(cwd) if cwd is not None else None)
result=dict(UTC=datetime.now(timezone.utc).isoformat(),scores_read=False,TEST_access=False,new_launches=0,signals_sent=[],blocks={{}},artifacts={{}})
for block in ('b1','b2'):
 c.bind_block([block]);c.physical_host();s=c.supervisor();execution=phase/expected[block]['execution_directory_relative']
 row=dict(queue_physical=physical(s,launch[block]['queue_identity']),physical_GPU_UUID=c.GPU_UUID,whole_block_freeze_exists=(execution/'BLOCK_FREEZE.json').is_file())
 paths=[receiptroot/(block+'.'+name) for name in ('LAUNCH_PREFLIGHT.json','POPEN_STARTED.json','LAUNCH_RECEIPT.json','FREEZE_RECEIPT.json')]
 for name in ('QUEUE_START.json','QUEUE_PROGRESS.json','CURRENT_RESOURCES.json'):
  path=execution/name
  if path.is_file():row[name]=read(path);paths.append(path)
 if (execution/'QUEUE_FAILURE.json').is_file():
  failure=read(execution/'QUEUE_FAILURE.json');row['queue_failure']={{k:failure[k] for k in ('UTC','error','inclusive_seconds','partial_outputs_preserved','retry','scores_read')}}
  paths.append(execution/'QUEUE_FAILURE.json')
 current_path=execution/'CURRENT_PROCESS.json'
 if current_path.is_file():
  current=read(current_path);row['current_cell']=current['cell_id'];row['child_physical']=physical(s,current['identity']);row['owned_tree']=s.owned_tree(current['identity']) if current['identity'] is not None else []
  paths += [current_path,execution/'logs'/(current['cell_id']+'.CHILD_STARTED.json'),execution/'logs'/(current['cell_id']+'.PREFLIGHT.json')]
  progress=execution/'runs'/current['cell_id']/'PROGRESS.json'
  if progress.is_file():
   value=read(progress);assert set(value)=={{'cycle','episode','episodes_in_cycle','counters','inclusive_seconds'}}
   row['fit_progress']=value;paths.append(progress)
  failure=execution/'runs'/current['cell_id']/'FAILURE.json'
  if failure.is_file():
   value=read(failure);row['fit_failure']={{k:value[k] for k in ('error','elapsed_seconds','partial_outputs_preserved','TEST_access') if k in value}}
   paths.append(failure)
 compute=s.query(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],10)
 ownpids={{r['PID'] for r in row.get('owned_tree',[])}}
 row['owned_CUDA_rows']=[r for r in compute if len(r.split(','))==3 and r.split(',')[1].strip().isdigit() and int(r.split(',')[1].strip()) in ownpids]
 row['owned_CUDA_only_assigned_GPU']=all(r.split(',')[0].strip()==c.GPU_UUID for r in row['owned_CUDA_rows'])
 result['blocks'][block]=row
 for path in paths:
  if path.is_file():
   raw=path.read_bytes();result['artifacts'][str(path.relative_to(phase))]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),zlib_base64=base64.b64encode(zlib.compress(raw,9)).decode())
print(json.dumps(result))
'''
r=t.run('private_transfer77_blocks_fourth_owned_observation_20261005_v4',code)
with (OUT/'OBSERVATION_RAW.json').open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
for rel,row in r['artifacts'].items():
 raw=zlib.decompress(base64.b64decode(row['zlib_base64'],validate=True));assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 path=OUT/'authenticated_metadata'/rel;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(raw)
compact={k:v for k,v in r.items() if k!='artifacts'}
with (OUT/'OBSERVATION.json').open('x') as f:json.dump(compact,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'UTC':compact['UTC'],'scores_read':compact['scores_read'],'blocks':{k:{'current_cell':v['current_cell'],'queue_live':v['queue_physical']['fresh_identity_matches'],'child_live':v['child_physical']['fresh_identity_matches'],'fit_progress':v['fit_progress'],'completed':v.get('QUEUE_PROGRESS.json',{}).get('completed',0),'failure':v.get('queue_failure') or v.get('fit_failure')} for k,v in compact['blocks'].items()}}))
