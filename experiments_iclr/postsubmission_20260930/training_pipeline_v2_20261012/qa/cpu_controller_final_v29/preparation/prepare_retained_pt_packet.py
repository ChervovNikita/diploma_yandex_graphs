"""Guarded freeze, synthetic CPU QA, real custody hashes, disabled PT handoff.

Each revision uses a fresh immutable local/allocation packet. No scientific
fit, pilot, CUDA execution, heldout target decoder, GO, or77 access occurs.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib
import prepare_context_exclusion_packet_v1 as transport
from compile_native_matrix import RUNTIME_FILES
from adapters.retained_pt_routes import PINS, SOURCE_ARMS, PT_ARMS, NATIVE_CONTROLS

SOURCES = {
    'pt':'native_meanSAGE_private_continuation_PT_core_source_20261011_v2/core.py',
    'native':'native_meanSAGE_attribute_peer_feedback_core_source_20261011_v1/core.py',
    'original':'common_wrapper_native_source_20261010_v1/models.py',
    'original_driver':'common_wrapper_native_source_20261010_v1/run_common.py',
}
RATIONALE = (
    'private_continuation_PT_architecture_design_20261011_v2/REPORT.md',
    'private_continuation_PT_architecture_design_20261011_v2/ARCHITECTURE_BINDING.json',
    'private_continuation_PT_bias_coordinate_control_assessment_20261011_v1/REPORT.md',
    'private_continuation_PT_bias_coordinate_root_adoption_20261011_v1/ROOT_DECISION.json',
    'private_continuation_PT_mechanism_synthesis_20261012_v2/REPORT.md',
    'private_continuation_PT_mechanism_synthesis_20261012_v2/MECHANISM_AND_DIAGNOSTIC.json',
    'PT_history_decisive_comparison_assessment_20261012_v2/REPORT.md',
    'PT_history_decisive_comparison_assessment_20261012_v2/ADDENDUM.md',
    'PT_history_decisive_comparison_assessment_20261012_v2/ORIGINAL_GNNM_THREE_ROLE_CONTRACT.md',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--revision',type=int,default=1); args = parser.parse_args()
    if args.revision < 1: raise ValueError('Positive fresh packet revision required')
    old = transport.NAME; name = 'retained_pt_packet_20261012_v'+str(args.revision)
    transport.NAME = name; transport.LOCAL = transport.PHASE/name; transport.GUARD = transport.GUARD.replace("phase/'"+old+"'","phase/'"+name+"'")
    assert not transport.LOCAL.exists(); transport.LOCAL.mkdir(); files = []
    def add(relative,raw):
        path = transport.LOCAL/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
        files.append({'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    runtime_files = list(RUNTIME_FILES)+['adapters/retained_pt_routes.py','adapters/retained_pt_endpoints.py','compile_retained_pt_matrix.py','compile_native_matrix.py','prepare_node_data.py',
                    'tests/test_retained_pt_routes.py','docs/RETAINED_PT_STANDARDIZED_PORT.md','compile_master_plan.py']
    for relative in runtime_files: add('runtime/'+relative,(transport.PACKAGE/relative).read_bytes())
    bindings = {}
    for key,relative in SOURCES.items():
        raw = (transport.PHASE/relative).read_bytes(); assert hashlib.sha256(raw).hexdigest()==PINS[key]
        add('sources/'+relative,raw); row = files[-1]; bindings[key] = {'path':name+'/'+row['path'],'bytes':row['bytes'],'sha256':row['sha256']}
    for relative in RATIONALE: add('rationale/'+relative,(transport.PHASE/relative).read_bytes())
    for directory in (Path(SOURCES['pt']).parent,Path(SOURCES['native']).parent):
        for metadata in ('README.md','INTERFACE_CONTRACT.json','STATUS.disabled.json','MANIFEST.json','COUNTS.json'):
            source = transport.PHASE/directory/metadata
            if source.is_file(): add('sources/'+str(directory/metadata),source.read_bytes())
    resources = {'schema':'retained-pt-provisional-resources-v1','resource_scope':'provisional_unmeasured','CUDA_qualified':False,
        'worker_capacities_cuda_bytes':{'allocation/gpu0':78*1024**3},'scientific_launch_enabled':False,'human_GO_required':True,
        'capacity_basis':'78GiB scheduling allowance; sole UUID guard is not measured CUDA admission','profiles':{}}
    for dataset in ('wikics','amazon-photo'):
        for arm in SOURCE_ARMS+PT_ARMS+NATIVE_CONTROLS:
            resources['profiles'][dataset+'/retained-pt-'+arm] = {'reservation_mib':32768,'wall_seconds':250000000,
                'estimated_seconds_per_update':8. if arm=='separable4' else 5. if arm in ('private4','middle_state_mean','original_gnnm','factor_s4') else 2.,
                'estimated_export_seconds':120,'estimated_scalar_cal_seconds':60,'estimated_evaluation_seconds':60,
                'resource_scope':'provisional_unmeasured','CUDA_qualified':False,
                'evidence':'Explicit unmeasured placeholder. CPU source QA does not qualify full-input CUDA sparse kernels, parameter/cache bytes, checkpoint/endpoint serialization or concurrent admission; resource gates pending.'}
    add('SOURCE_BINDINGS.json',(json.dumps(bindings,indent=2)+'\n').encode()); add('RESOURCES_PROVISIONAL.json',(json.dumps(resources,indent=2)+'\n').encode())
    if args.revision > 1:
        for revision in range(1,args.revision):
            prior = transport.PHASE/('retained_pt_packet_20261012_v'+str(revision))
            for label in ('CPU_SOURCE_QA','COMPILE_DISABLED_PACKET','MASTER_INGEST_HANDOFF'):
                previous = prior/'transport'/(label+'.json')
                if previous.exists(): add(f'receipts/prior_v{revision}_{label}_TRANSPORT.json',previous.read_bytes())
    inventory = {'schema':'retained-pt-frozen-footprint-v1','files':[{k:v for k,v in r.items() if k!='data'} for r in files],
                 'source_prototype':'exact_v2','scientific_fits':0,'predictive_scores':0,'GO_created':False,'new_controls_explicit':True}
    (transport.LOCAL/'SOURCE_INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    payload = base64.b64encode(zlib.compress(json.dumps({'files':files,'inventory':inventory}).encode())).decode()
    transfer = r'''
stage=phase/'training_pipeline_v2_20261012/preparation'/sys.argv[3]
offset=int(sys.argv[1])
if offset==0:assert not root.exists() and not stage.exists();stage.mkdir(parents=True)
else:assert (stage/'payload.b64').stat().st_size==offset
with (stage/'payload.b64').open('ab') as f:f.write(sys.argv[2].encode())
print((stage/'payload.b64').stat().st_size)
'''
    stage = name+'_transport'
    for offset in range(0,len(payload),24000): transport.request('TRANSFER_'+str(offset),transfer,[str(offset),payload[offset:offset+24000],stage])
    transport.request('FREEZE',r'''
payload=(phase/'training_pipeline_v2_20261012/preparation'/sys.argv[2]/'payload.b64').read_bytes()
assert hashlib.sha256(payload).hexdigest()==sys.argv[1]
value=json.loads(zlib.decompress(base64.b64decode(payload)));assert not root.exists();root.mkdir()
for row in value['files']:
    relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
    raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
write_json(root/'SOURCE_INVENTORY.json',value['inventory'])
print(json.dumps({'files':len(value['files']),'science':False,'GO':False}))
''',[hashlib.sha256(payload.encode()).hexdigest(),stage])
    transport.request('CPU_SOURCE_QA',r'''
qenv=dict(env,GNNM_REFERENCE_PROJECT=str(root/'sources'))
qa=checked('CPU_SOURCE_QA',[python,'-B','-m','unittest','discover','-s','tests','-p','test_retained_pt_routes.py','-v'],180,cwd=runtime,environment=qenv)
assert 'Ran 7 tests' in qa['stderr'] and '\nOK\n' in qa['stderr'] and 'skipped=' not in qa['stderr']
write_json(root/'receipts/CPU_SOURCE_QA_RESULT.json',{'schema':'retained-pt-source-cpu-qa-v1','passed':True,'tests':7,'skips':0,
    'synthetic_only':True,'scientific_fits':0,'predictive_scores':0,'GO':False,'frozen_footprint':bound(root/'SOURCE_INVENTORY.json'),
    'operation':bound(root/'receipts/CPU_SOURCE_QA.json'),'wall_seconds':qa['wall_seconds']})
print(json.dumps({'passed':True,'tests':7,'skips':0,'wall_seconds':qa['wall_seconds']}))
''',timeout=240)
    transport.request('COMPILE_DISABLED_PACKET',r'''
fixtures=[str(root/'receipts/CPU_SOURCE_QA_RESULT.json'),str(root/'SOURCE_INVENTORY.json'),str(runtime/'tests/test_retained_pt_routes.py'),str(runtime/'docs/RETAINED_PT_STANDARDIZED_PORT.md')]
fixtures += [str(root/row['path']) for row in json.loads((root/'SOURCE_INVENTORY.json').read_text())['files'] if row['path'].startswith('rationale/') or (row['path'].startswith('sources/') and not row['path'].endswith('.py'))]
checked('COMPILE_DISABLED_PACKET',[python,'-B',str(runtime/'compile_retained_pt_matrix.py'),'--project-root',str(phase),
    '--custody',str(phase/'training_pipeline_v2_20261012/prepared_data/wikics/CUSTODY.json'),str(phase/'training_pipeline_v2_20261012/prepared_data/amazon-photo/CUSTODY.json'),
    '--source-bindings',str(root/'SOURCE_BINDINGS.json'),'--runtime-package',str(runtime),'--resources',str(root/'RESOURCES_PROVISIONAL.json'),
    '--fixture-receipts',*fixtures,'--output',str(root/'matrix')],180)
counts=json.loads((root/'matrix/SUMMARY.json').read_text())['counts'];assert counts['atomic_fit_groups']==3140 and counts['all_run_records']==8024
print(json.dumps(counts))
''',timeout=240)
    result = transport.request('MASTER_INGEST_HANDOFF',r'''
sys.path.insert(0,str(runtime))
from compile_master_plan import load_packet
entry={'id':'retained_supervised_PT_routes','plan':bound(root/'matrix/PLAN.json'),'summary':bound(root/'matrix/SUMMARY.json')}
packet=load_packet(phase,entry)
assert len(packet['studies'])==2 and all(row['matched'] for rows in packet['references'].values() for row in rows)
write_json(root/'MASTER_INPUT_ENTRY.json',entry)
write_json(root/'MASTER_INGEST_VERIFICATION.json',{'schema':'retained-pt-master-ingest-v1','verified':True,'plan_hash':packet['plan_hash'],
    'reference_matches':packet['references'],'counts':packet['counts'],'scientific_fits':0,'CAL_TEST_target_tensors_decoded':False,'GO_created':False})
write_json(root/'SUMMARY.json',{'schema':'retained-pt-disabled-handoff-v1','status':'PREPARED_DISABLED_RESOURCE_EXCLUDED_MASTER_AND_GO_PENDING',
    'plan_hash':packet['plan_hash'],'entry':entry,'counts':packet['summary']['counts'],'CPU_source_QA':bound(root/'receipts/CPU_SOURCE_QA_RESULT.json'),
    'master_ingest_verification':bound(root/'MASTER_INGEST_VERIFICATION.json'),'scientific_launch_enabled':False,'human_GO_required':True,
    'science':False,'GO':False,'resource_status':'provisional_unmeasured_EXCLUDED_UNTIL_ENGINEERING_QUALIFICATION'})
mirrors=['MASTER_INPUT_ENTRY.json','MASTER_INGEST_VERIFICATION.json','SUMMARY.json','matrix/PLAN.json','matrix/SUMMARY.json',
         'receipts/CPU_SOURCE_QA.json','receipts/CPU_SOURCE_QA_RESULT.json','receipts/COMPILE_DISABLED_PACKET.json']
mirrors += [str((phase/r['path']).relative_to(root)) for r in packet['plan']['studies']]
values={rel:base64.b64encode((root/rel).read_bytes()).decode() for rel in mirrors}
print(base64.b64encode(zlib.compress(json.dumps(values).encode())).decode())
''',timeout=180)
    values = json.loads(zlib.decompress(base64.b64decode(result.strip())))
    for relative,value in values.items():
        path = transport.LOCAL/relative; path.parent.mkdir(parents=True,exist_ok=True); assert not path.exists(); path.write_bytes(base64.b64decode(value))
    print(json.dumps({'packet':name,'status':'PREPARED_DISABLED','science':False,'GO':False}),flush=True)


if __name__=='__main__': main()
