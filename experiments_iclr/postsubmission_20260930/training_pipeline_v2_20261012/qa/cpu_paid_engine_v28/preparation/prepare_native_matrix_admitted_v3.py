"""Freeze the resource-admitted native subset; preserve all six blocked cells.

This materializes an oversized prospective design for accounting, not approval.
It does not instantiate any scientific model, optimizer or endpoint.
"""
import base64
import hashlib
import json
from pathlib import Path
import zlib

from prepare_molecular_packets_allocation import GUARD, invoke

PACKAGE = Path(__file__).resolve().parents[1]
HERE = PACKAGE / 'preparation'
DESTINATION = 'massive_native_packet_20261012_v3'
LABEL = 'native_matrix_admitted_v3'
NAMES = ('engine.py','metrics.py','spec.py','registry.py','partition.py',
         'run_study.py','run_endpoint.py','launcher.py','phase_closure.py',
         'adapters/__init__.py','adapters/classifier.py','adapters/node.py',
         'adapters/node_losses.py','adapters/node_endpoints.py','adapters/search.py',
         'start_massive.sh','prepare_node_data.py','compile_native_matrix.py',
         'tests/test_native_resource_deferrals.py')

TRANSFER = GUARD + r'''
transport=phase/(sys.argv[1]+'_transport');target=transport/'payload.b64';offset=int(sys.argv[2])
if offset==0:assert not transport.exists();transport.mkdir()
else:assert target.stat().st_size==offset
with target.open('ab') as stream:stream.write(sys.argv[3].encode())
print(target.stat().st_size)
'''

REMOTE = GUARD + r'''
transport=phase/(sys.argv[1]+'_transport');encoded=(transport/'payload.b64').read_bytes()
assert hashlib.sha256(encoded).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
def atomic(path,value):
 path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def receipt(path):
 raw=path.read_bytes()
 return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
for row in payload['files']:
 relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);target.chmod(0o444)
atomic(stage/'SOURCE_INVENTORY.json',[{k:v for k,v in r.items() if k!='data'} for r in payload['files']])
prior=phase/'massive_native_packet_20261012_v2'
resources=json.loads((prior/'resources_v1/RESOURCES.json').read_text())
for name in ('MANIFEST.json','RESOURCES.json','COST_PROJECTION.json'):
 source=prior/'resources_v1'/name;target=stage/'resource_evidence'/name
 target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target);target.chmod(0o444)
runtime=stage/'runtime'
old=json.loads((prior/'SOURCE_INVENTORY.json').read_text())
old_hashes={r['path']:r['sha256'] for r in old['files']}
for relative in ('adapters/node.py','adapters/node_losses.py'):
 assert receipt(runtime/relative)['sha256']==old_hashes['runtime/'+relative], 'Resource probes do not bind changed model/loss code'
assert receipt(stage/'native/models.py')['sha256']==old_hashes['native/models.py']
ancestry={'schema':'native-resource-model-reuse-v1','model_and_loss_unchanged':True,
 'original_manifest':receipt(prior/'resources_v1/MANIFEST.json'),
 'copied_manifest':receipt(stage/'resource_evidence/MANIFEST.json'),
 'controller_updated_separately':True,'full_runtime_qualification':False,'benchmark_fits':0,'scores':0}
atomic(stage/'RESOURCE_REUSE.json',ancestry)
blocked={}
requested={dataset+'/'+backbone for dataset in payload['datasets'] for backbone in payload['backbones']}
for key in sorted(requested):
 profile=resources['profiles'].get(key)
 if profile is None:blocked[key]='Maximum-shape backward qualification failed; preserve pending cell and OOM evidence. Smaller domains require separate qualification.'
 elif profile['reservation_mib']*1024**2>max(resources['worker_capacities_cuda_bytes'].values()):
  blocked[key]='Measured maximum-shape reservation plus declared headroom exceeds all three planning capacities. No fit is admitted.'
assert len(blocked)==6
atomic(stage/'RESOURCE_DEFERRED_CELLS.json',blocked)
atomic(stage/'RESOURCES.json',resources)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
command=[str(repo/'.venv/bin/python'),'-B','-m','unittest','discover','-s','tests','-p','test_native_resource_deferrals.py','-v']
r=subprocess.run(command,cwd=runtime,env=env,capture_output=True,text=True,timeout=60)
qa={'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'benchmark_fits':0,'predictive_scores':0}
atomic(stage/'CPU_DEFERRAL_QA.json',qa);assert r.returncode==0,r.stderr
custodies=[str(phase/'training_pipeline_v2_20261012/prepared_data'/dataset/'CUSTODY.json') for dataset in payload['datasets']]
command=[str(repo/'.venv/bin/python'),'-B','-u',str(runtime/'compile_native_matrix.py'),
 '--project-root',str(phase),'--custody',*custodies,'--resources',str(stage/'RESOURCES.json'),
 '--output',str(stage/'matrix'),'--runtime-package',str(runtime),
 '--metadata-dir',str(stage/'metadata'),'--native-model-source',str(stage/'native/models.py'),
 '--fixture-receipts',str(stage/'CPU_DEFERRAL_QA.json'),str(stage/'RESOURCE_REUSE.json'),
 str(stage/'fixtures/cpu_multitask_source_v7.json'),str(stage/'fixtures/cpu_packet_import_isolation_v13.json'),
 str(stage/'fixtures/cpu_engine_validation_costs_v15.json'),
 '--frozen-runtime','--scalar-coefficients','neighbors',
 '--resource-deferred-cells',str(stage/'RESOURCE_DEFERRED_CELLS.json')]
r=subprocess.run(command,cwd=runtime,env=env,capture_output=True,text=True,timeout=600)
atomic(stage/'COMPILER_RESULT.json',{'command':command,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'fits':0,'scores':0})
assert r.returncode==0,r.stderr
summary=json.loads((stage/'matrix/SUMMARY.json').read_text())
atomic(stage/'PREPARATION_SUMMARY.json',{'status':'DISABLED_RESOURCE_ADMITTED_SUBSET_OVERSIZED_NOT_READY_FOR_GO',
 'matrix_plan':receipt(stage/'matrix/PLAN.json'),'matrix_summary':receipt(stage/'matrix/SUMMARY.json'),
 'requested_cells':64,'materialized_cells':summary['materialized_native_cells'],'resource_deferred_cells':blocked,
 'fits':0,'scores':0,'GO_created':False,'all_249_covered':False,
 'resources_scope':'zero-update model/loss proxies only; optimizer,serializer,endpoints and concurrency unqualified'})
print(json.dumps({'prepared':True,'plan':receipt(stage/'matrix/PLAN.json'),'summary':receipt(stage/'matrix/SUMMARY.json'),
 'studies':summary['studies'],'counts':summary['fresh_fit_counts'],'deferred':blocked,'training_started':False}))
'''

def main():
    out=HERE/LABEL
    assert not out.exists()
    out.mkdir()
    files=[]
    def add(source,relative):
        raw=source.read_bytes()
        files.append({'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    for name in NAMES:add(PACKAGE/name,'runtime/'+name)
    for name in ('MATRIX_AXES_DRAFT.json','METHOD_INVENTORY.csv','METHOD_SOURCE_ALIASES.csv'):
        add(PACKAGE/'docs'/name,'metadata/'+name)
    add(PACKAGE.parent/'common_wrapper_native_source_20261010_v1/models.py','native/models.py')
    for name in ('cpu_multitask_source_v7','cpu_packet_import_isolation_v13','cpu_engine_validation_costs_v15'):
        add(PACKAGE/'qa'/name/'TRANSPORT.json','fixtures/'+name+'.json')
    payload={'destination':DESTINATION,'files':files,
             'datasets':['cora','citeseer','pubmed','wikics','amazon-computers','amazon-photo','coauthor-cs','coauthor-physics'],
             'backbones':['ResNet','GCN','SAGE','GAT','GAT-sep','GT','GT-sep','TAG']}
    encoded=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    (out/'SOURCE_INVENTORY.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='data'} for r in files],indent=2)+'\n')
    for offset in range(0,len(encoded),24000):
        result=invoke(TRANSFER,[DESTINATION,str(offset),encoded[offset:offset+24000]],out,'TRANSFER_'+str(offset),45)
        assert result['exit_code']==0,result
    result=invoke(REMOTE,[DESTINATION,hashlib.sha256(encoded.encode()).hexdigest()],out,'TRANSPORT',660)
    assert result['exit_code']==0,result
    print(result['stdout'])

if __name__=='__main__':
    main()
