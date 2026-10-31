"""Freeze new boundary supplement, run synthetic CPU checks, compile disabled."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
import prepare_context_exclusion_packet_v1 as transport
from compile_link_boundary_matrix import RUNTIME

OLD=transport.NAME
NAME='link_boundary_packet_20261012_v3'
transport.NAME=NAME;transport.LOCAL=transport.PHASE/NAME
transport.GUARD=transport.GUARD.replace("phase/'"+OLD+"'","phase/'"+NAME+"'")


def main():
    assert not transport.LOCAL.exists();transport.LOCAL.mkdir();files=[]
    def add(relative,raw):
        p=transport.LOCAL/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        files.append({'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    for name in list(RUNTIME)+['tests/test_link_boundary.py','tests/test_link.py','prepare_link_data.py','compile_master_plan.py']:
        add('runtime/'+name,(transport.PACKAGE/name).read_bytes())
    for relative in ['common_wrapper_native_source_20261010_v1/models.py','graph_link_competing_family_source_20261003_v1/source/ncn__model.py','graph_link_competing_family_source_20261003_v1/source/ncn__utils.py']:
        add('sources/'+relative,(transport.PHASE/relative).read_bytes())
    add('receipts/historical_v1_CPU_QA_FAILURE_TRANSPORT.json',(transport.PHASE/'link_boundary_packet_20261012_v1/transport/CPU_SOURCE_QA.json').read_bytes())
    add('receipts/historical_v2_CPU_QA_FAILURE_TRANSPORT.json',(transport.PHASE/'link_boundary_packet_20261012_v2/transport/CPU_SOURCE_QA.json').read_bytes())
    profile={'reservation_mib':32768,'wall_seconds':250000000,'estimated_seconds_per_epoch':1800,
        'estimated_selection_seconds':60,'estimated_export_seconds':7200,'scope':'provisional_unmeasured',
        'resource_status':'EXCLUDED_UNTIL_ENGINEERING_QUALIFICATION','scientific_launch_enabled':False,
        'evidence':'Explicit unmeasured placeholders authorized by parent. CPU semantics do not qualify CUDA sparse kernels, optimizer state, serializer, endpoints or concurrent admission.'}
    add('RESOURCES_PROVISIONAL.json',(json.dumps(profile,indent=2)+'\n').encode())
    inventory={'schema':'link-boundary-frozen-footprint-v1','files':[{k:v for k,v in r.items() if k!='data'} for r in files],
        'original_task_replication':False,'scientific_fits':0,'GO_created':False}
    (transport.LOCAL/'SOURCE_INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    payload=base64.b64encode(zlib.compress(json.dumps({'files':files,'inventory':inventory}).encode())).decode()
    transfer=r'''
stage=phase/'training_pipeline_v2_20261012/preparation/link_boundary_packet_v3_transport'
offset=int(sys.argv[1])
if offset==0:assert not root.exists() and not stage.exists();stage.mkdir(parents=True)
else:assert (stage/'payload.b64').stat().st_size==offset
with (stage/'payload.b64').open('ab') as s:s.write(sys.argv[2].encode())
print((stage/'payload.b64').stat().st_size)
'''
    for offset in range(0,len(payload),24000):transport.request('TRANSFER_'+str(offset),transfer,[str(offset),payload[offset:offset+24000]])
    transport.request('FREEZE',r'''
payload=(phase/'training_pipeline_v2_20261012/preparation/link_boundary_packet_v3_transport/payload.b64').read_bytes()
assert hashlib.sha256(payload).hexdigest()==sys.argv[1]
value=json.loads(zlib.decompress(base64.b64decode(payload)));assert not root.exists();root.mkdir()
for row in value['files']:
    rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
    raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
write_json(root/'SOURCE_INVENTORY.json',value['inventory'])
print(json.dumps({'files':len(value['files']),'science':False}))
''',[hashlib.sha256(payload.encode()).hexdigest()])
    transport.request('CPU_SOURCE_QA',r'''
qenv=dict(env,GNNM_REFERENCE_PROJECT=str(root/'sources'))
qa=checked('CPU_SOURCE_QA',[python,'-B','-m','unittest','discover','-s','tests','-p','test_link_boundary.py','-v'],180,cwd=runtime,environment=qenv)
assert 'Ran 5 tests' in qa['stderr'] and '\nOK\n' in qa['stderr'] and 'skipped=' not in qa['stderr']
write_json(root/'receipts/CPU_SOURCE_QA_RESULT.json',{'schema':'link-boundary-source-cpu-qa-v1','passed':True,'tests':5,'skips':0,
    'synthetic_only':True,'scientific_fits':0,'GO':False,'frozen_footprint':bound(root/'SOURCE_INVENTORY.json'),
    'operation':bound(root/'receipts/CPU_SOURCE_QA.json'),'wall_seconds':qa['wall_seconds']})
print(json.dumps({'passed':True,'tests':5,'skips':0,'wall_seconds':qa['wall_seconds']}))
''',timeout=240)
    transport.request('COMPILE_DISABLED_PACKET',r'''
checked('COMPILE_DISABLED_PACKET',[python,'-B',str(runtime/'compile_link_boundary_matrix.py'),
    '--project-root',str(phase),'--template-study','training_pipeline_v2_20261012/qa/link_materialized_source_controls_v3/STUDY_DISABLED.json',
    '--boundary-source',str(root/'sources/common_wrapper_native_source_20261010_v1/models.py'),
    '--runtime-package',str(runtime),'--qa-receipt',str(root/'receipts/CPU_SOURCE_QA_RESULT.json'),
    '--resources',str(root/'RESOURCES_PROVISIONAL.json'),'--output',str(root/'matrix')],180)
print(json.dumps(json.loads((root/'matrix/SUMMARY.json').read_text())['counts']))
''',timeout=240)
    result=transport.request('MASTER_INGEST_HANDOFF',r'''
sys.path.insert(0,str(runtime))
from compile_master_plan import load_packet
entry={'id':'original_boundary_link_collab','plan':bound(root/'matrix/PLAN.json'),'summary':bound(root/'matrix/SUMMARY.json')}
packet=load_packet(phase,entry)
assert len(packet['studies'])==1 and packet['summary']['counts']['model_fits']==126
assert all(r['matched'] for rows in packet['references'].values() for r in rows)
write_json(root/'MASTER_INPUT_ENTRY.json',entry)
write_json(root/'MASTER_INGEST_VERIFICATION.json',{'schema':'link-boundary-master-ingest-v1','verified':True,
    'plan_hash':packet['plan_hash'],'reference_matches':packet['references'],'counts':packet['counts'],
    'scientific_fits':0,'TEST_target_tensors_decoded':False,'GO_created':False})
write_json(root/'SUMMARY.json',{'schema':'link-boundary-disabled-handoff-v1','status':'PREPARED_DISABLED_RESOURCE_EXCLUDED_MASTER_AND_GO_PENDING',
    'plan_hash':packet['plan_hash'],'entry':entry,'counts':packet['summary']['counts'],
    'CPU_source_QA':bound(root/'receipts/CPU_SOURCE_QA_RESULT.json'),'master_ingest_verification':bound(root/'MASTER_INGEST_VERIFICATION.json'),
    'scientific_launch_enabled':False,'human_GO_required':True,'science':False,'GO':False,
    'old_link_v1_unchanged':True,'resource_status':'provisional_unmeasured_EXCLUDED_UNTIL_ENGINEERING_QUALIFICATION'})
mirrors=['MASTER_INPUT_ENTRY.json','MASTER_INGEST_VERIFICATION.json','SUMMARY.json','matrix/PLAN.json','matrix/SUMMARY.json',
    'receipts/CPU_SOURCE_QA.json','receipts/CPU_SOURCE_QA_RESULT.json','receipts/COMPILE_DISABLED_PACKET.json']
mirrors += [str((phase/r['path']).relative_to(root)) for r in packet['plan']['studies']]
value={rel:base64.b64encode((root/rel).read_bytes()).decode() for rel in mirrors}
print(base64.b64encode(zlib.compress(json.dumps(value).encode())).decode())
''',timeout=180)
    data=json.loads(zlib.decompress(base64.b64decode(result.strip())))
    for relative,value in data.items():
        p=transport.LOCAL/relative;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(base64.b64decode(value))
    print(json.dumps({'packet':NAME,'status':'PREPARED_DISABLED','science':False,'GO':False}),flush=True)


if __name__=='__main__':main()
