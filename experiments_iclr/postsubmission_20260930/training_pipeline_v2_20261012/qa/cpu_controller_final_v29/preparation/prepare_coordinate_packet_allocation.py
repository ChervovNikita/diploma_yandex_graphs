"""Materialize coordinate source comparisons with explicitly unmeasured costs."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

from prepare_molecular_packets_allocation import GUARD, invoke

PACKAGE=Path(__file__).resolve().parents[1]
HERE=PACKAGE/'preparation'
DESTINATION='massive_coordinate_packet_20261012_v1'
LABEL='coordinate_allocation_20261012_v1'
TRANSFER=GUARD+r'''
transport=phase/(sys.argv[1]+'_transport');target=transport/'payload.b64';offset=int(sys.argv[2])
if offset==0:assert not transport.exists();transport.mkdir()
else:assert target.stat().st_size==offset
with target.open('ab') as stream:stream.write(sys.argv[3].encode())
print(target.stat().st_size)
'''
REMOTE=GUARD+r'''
transport=phase/(sys.argv[1]+'_transport');encoded=(transport/'payload.b64').read_bytes()
assert hashlib.sha256(encoded).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
def atomic(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);target.chmod(0o444)
atomic(stage/'SOURCE_INVENTORY.json',[{k:v for k,v in r.items() if k!='data'} for r in payload['files']])
profiles={dataset+'/coordinate-SAGE-source':{'reservation_mib':65536,'wall_seconds':315360000,
 'estimated_seconds_per_update':10.,'evidence':['PROVISIONAL_UNMEASURED_NOT_RUNTIME_QUALIFIED',
 'Planning placeholder only,not borrowed qualification from native allmap; coordinate permutation buffers,source width matching,GT-sep reference,four optimizers and endpoints require measurements.']}
 for dataset in payload['datasets']}
resources={'profiles':profiles,'worker_capacities_cuda_bytes':{key:78*1024**3 for key in ['allocation/gpu0','server77/gpu0','server77/gpu1']},
 'remote_capacity_live_verified':False,'scientific_launch_enabled':False,'human_GO_required':True}
atomic(stage/'RESOURCE_ESTIMATES.json',resources)
runtime=stage/'runtime';custodies=[str(phase/'training_pipeline_v2_20261012/prepared_data'/dataset/'CUSTODY.json') for dataset in payload['datasets']]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
command=[str(repo/'.venv/bin/python'),'-B','-u',str(runtime/'compile_coordinate_matrix.py'),
 '--project-root',str(phase),'--runtime-package',str(runtime),'--coordinate-source',str(stage/'coordinate/models.py'),
 '--native-model-source',str(stage/'native/models.py'),'--source-specification',str(stage/'coordinate/SPECIFICATION.json'),
 '--resources',str(stage/'RESOURCE_ESTIMATES.json'),'--output',str(stage/'matrix'),
 '--custody',*custodies,'--fixture-receipts',str(stage/'fixtures/cpu_coordinate_source_v14.json'),
 str(stage/'fixtures/cpu_memory_admission_v16.json'),'--frozen-runtime']
r=subprocess.run(command,cwd=runtime,env=env,capture_output=True,text=True,timeout=600)
atomic(stage/'COMPILER_RESULT.json',{'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'command':command,'fits':0,'scores':0})
assert r.returncode==0,r.stderr
summary=json.loads((stage/'matrix/SUMMARY.json').read_text())
from collections import Counter
counts=Counter()
for study in summary['studies']:
 for key,value in study['counts'].items():
  if type(value)==int:counts[key]+=value
 for key in ('atomic_fit_run_groups_by_stage','paid_optimizer_starts_by_stage','physical_bodies_by_stage'):
  for stage_name,value in study['counts'][key].items():counts[key+'_'+stage_name]+=value
value={'schema':'coordinate-disabled-preparation-handoff-v1',
 'status':'DISABLED_REQUIRES_RESOURCE_ENDPOINT_CONTROLLER_QUALIFICATION_AND_HUMAN_GO',
 'plan':receipt(stage/'matrix/PLAN.json'),'summary':receipt(stage/'matrix/SUMMARY.json'),
 'datasets':len(summary['studies']),'counts':dict(counts),'models_created':0,'optimizer_updates':0,
 'predictive_scores':0,'GO_created':False,'all_249_covered':False,'resources':'provisional_unmeasured'}
atomic(stage/'PREPARATION_SUMMARY.json',value);print(json.dumps(value))
'''

def main():
 out=HERE/LABEL;assert not out.exists();out.mkdir()
 files=[]
 def add(source,relative):
  raw=source.read_bytes();files.append({'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
 for relative in ('engine.py','metrics.py','spec.py','registry.py','partition.py','run_study.py',
  'run_endpoint.py','launcher.py','phase_closure.py','adapters/__init__.py','adapters/classifier.py',
  'adapters/node.py','adapters/node_losses.py','adapters/node_endpoints.py','adapters/search.py',
  'adapters/coordinate.py','compile_coordinate_matrix.py','compile_native_matrix.py',
  'prepare_node_data.py','start_massive.sh'):
  add(PACKAGE/relative,'runtime/'+relative)
 add(PACKAGE.parent/'coordinate_ensemble_source_v1/models.py','coordinate/models.py')
 add(PACKAGE.parent/'coordinate_stage1_assessment_source_v2/SPECIFICATION_v2.json','coordinate/SPECIFICATION.json')
 add(PACKAGE.parent/'common_wrapper_native_source_20261010_v1/models.py','native/models.py')
 for label in ('cpu_coordinate_source_v14','cpu_memory_admission_v16'):
  add(PACKAGE/'qa'/label/'TRANSPORT.json','fixtures/'+label+'.json')
 payload={'destination':DESTINATION,'files':files,'datasets':['cora','citeseer','pubmed','wikics','amazon-computers','amazon-photo','coauthor-cs','coauthor-physics']}
 encoded=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
 (out/'SOURCE_INVENTORY.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='data'} for r in files],indent=2)+'\n')
 for offset in range(0,len(encoded),24000):
  result=invoke(TRANSFER,[DESTINATION,str(offset),encoded[offset:offset+24000]],out,'TRANSFER_'+str(offset),45)
  assert result['exit_code']==0,result
 result=invoke(REMOTE,[DESTINATION,hashlib.sha256(encoded.encode()).hexdigest()],out,'TRANSPORT',660)
 assert result['exit_code']==0,result
 print(result['stdout'])

if __name__=='__main__':main()
