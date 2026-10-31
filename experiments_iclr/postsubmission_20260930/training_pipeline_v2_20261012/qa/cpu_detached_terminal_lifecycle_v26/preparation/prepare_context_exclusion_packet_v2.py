"""Preserve v1 ingest failure; fix comparison IDs and reuse exact target bytes."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import zlib

import prepare_context_exclusion_packet_v1 as base

OLD = base.NAME
NAME = "massive_context_exclusion_packet_20261012_v2"
PREVIOUS = base.LOCAL
base.NAME = NAME
base.LOCAL = base.PHASE / NAME
base.GUARD = base.GUARD.replace("phase/'" + OLD + "'", "phase/'" + NAME + "'")


def main():
    assert not base.LOCAL.exists()
    base.LOCAL.mkdir()
    files = []
    inventory = json.loads((PREVIOUS / "SOURCE_INVENTORY.json").read_text())
    for original in inventory["files"]:
        relative = original["path"]
        raw = (PREVIOUS / relative).read_bytes()
        assert len(raw) == original["bytes"] and hashlib.sha256(raw).hexdigest() == original["sha256"]
        if relative == "runtime/compile_context_exclusion_matrix.py":
            raw = (base.PACKAGE / "compile_context_exclusion_matrix.py").read_bytes()
        elif relative == "SOURCE_BINDINGS.json":
            bindings = json.loads(raw)
            for row in bindings.values():
                row["path"] = row["path"].replace(OLD + "/", NAME + "/", 1)
            raw = (json.dumps(bindings, indent=2) + "\n").encode()
        destination = base.LOCAL / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        files.append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "data": base64.b64encode(raw).decode()})
    raw = (PREVIOUS / "transport/VERIFY_AND_HANDOFF.json").read_bytes()
    files.append({"path": "receipts/historical_v1_INGEST_FAILURE_TRANSPORT.json", "bytes": len(raw),
                  "sha256": hashlib.sha256(raw).hexdigest(), "data": base64.b64encode(raw).decode()})
    inventory = {"schema": "context-exclusion-final-freeze-v2", "project_relative_root": NAME,
                 "preserved_failed_predecessor": OLD, "change": "comparison arm labels replaced with exact exported method IDs",
                 "files": [{k: v for k, v in row.items() if k != "data"} for row in files],
                 "scientific_fits": 0, "GO_created": False}
    (base.LOCAL / "SOURCE_INVENTORY.json").write_text(json.dumps(inventory, indent=2) + "\n")
    payload = base64.b64encode(zlib.compress(json.dumps({"files": files, "inventory": inventory}).encode())).decode()
    transfer = r'''
staging=phase/'training_pipeline_v2_20261012/preparation/context_exclusion_packet_v2_transport'
offset=int(sys.argv[1])
if offset==0:
    assert not root.exists() and not staging.exists();staging.mkdir(parents=True)
else:assert (staging/'payload.b64').stat().st_size==offset
with (staging/'payload.b64').open('ab') as stream:stream.write(sys.argv[2].encode())
print((staging/'payload.b64').stat().st_size)
'''
    for offset in range(0, len(payload), 24000):
        base.request("TRANSFER_" + str(offset), transfer, [str(offset), payload[offset:offset + 24000]])
    base.request("FREEZE_AND_REBIND_TARGET_CUSTODY", r'''
encoded=(phase/'training_pipeline_v2_20261012/preparation/context_exclusion_packet_v2_transport/payload.b64').read_bytes()
assert hashlib.sha256(encoded).hexdigest()==sys.argv[1]
value=json.loads(zlib.decompress(base64.b64decode(encoded)));assert not root.exists();root.mkdir()
for row in value['files']:
    relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
    raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
write_json(root/'SOURCE_INVENTORY.json',value['inventory'])
previous=phase/'massive_context_exclusion_packet_20261012_v1'
old_bindings=json.loads((previous/'SOURCE_BINDINGS.json').read_text())
new_bindings=json.loads((root/'SOURCE_BINDINGS.json').read_text())
assert set(old_bindings)==set(new_bindings) and len(new_bindings)==7
for key in old_bindings:
    assert bound(phase/old_bindings[key]['path'])==old_bindings[key]
    assert bound(phase/new_bindings[key]['path'])==new_bindings[key]
    assert old_bindings[key]['bytes']==new_bindings[key]['bytes'] and old_bindings[key]['sha256']==new_bindings[key]['sha256']
targets=json.loads((previous/'targets/CUSTODY.json').read_text())
assert targets['source_bindings']==old_bindings and len(targets['splits'])==20
for row in targets['splits']:assert bound(phase/row['target']['path'])==row['target']
targets['source_bindings']=new_bindings
targets['immutable_target_custody_predecessor']=bound(previous/'targets/CUSTODY.json')
targets['source_rebinding']='all seven source byte/hash identities unchanged; target pack bytes reused without tensor decode'
targets['physical_target_preparations_in_this_revision']=0
write_json(root/'targets/CUSTODY.json',targets)
print(json.dumps({'frozen_files':len(value['files']),'reused_target_splits':20,'target_tensors_decoded':False,'scientific_fits':0,'GO':False}))
''', [hashlib.sha256(payload.encode()).hexdigest()], timeout=120)
    base.request("FINAL_CPU_QA", r'''
qaenv=dict(env,GNNM_REFERENCE_PROJECT=str(root/'sources'))
result=checked('FINAL_CPU_QA',[python,'-B','-m','unittest','discover','-s','tests','-p','test_context_exclusion.py','-v'],120,cwd=runtime,environment=qaenv)
assert 'Ran 7 tests' in result['stderr'] and '\nOK\n' in result['stderr'] and 'skipped=' not in result['stderr']
write_json(root/'receipts/FINAL_CPU_QA_RESULT.json',{'schema':'context-exclusion-final-cpu-fixture-v2',
    'passed':True,'tests':7,'skips':0,'engineering_only':True,'synthetic_graphs_only':True,'scientific_fits':0,
    'predictive_scores':0,'GO_created':False,'runtime_inventory':bound(root/'SOURCE_INVENTORY.json'),
    'operation':bound(root/'receipts/FINAL_CPU_QA.json'),'source_bindings':bound(root/'SOURCE_BINDINGS.json'),
    'wall_seconds':result['wall_seconds']})
print(json.dumps({'tests':7,'skips':0,'passed':True,'scientific_fits':0}))
''', timeout=180)
    base.request("COMPILE_DISABLED_PACKET", r'''
previous=phase/'massive_context_exclusion_packet_20261012_v1'
checked('COMPILE_DISABLED_PACKET',[python,'-B',str(runtime/'compile_context_exclusion_matrix.py'),
    '--project-root',str(phase),'--custody','training_pipeline_v2_20261012/prepared_data/wikics/CUSTODY.json',
    '--source-bindings',str(root/'SOURCE_BINDINGS.json'),'--targets-custody',str(root/'targets/CUSTODY.json'),
    '--runtime-package',str(runtime),'--resources',str(root/'RESOURCES_PROVISIONAL.json'),
    '--fixture-receipts',str(root/'receipts/FINAL_CPU_QA_RESULT.json'),str(root/'receipts/FINAL_CPU_QA.json'),
    str(root/'receipts/historical_qa19_TRANSPORT.json'),str(root/'receipts/historical_qa20_TRANSPORT.json'),
    str(root/'receipts/historical_v1_INGEST_FAILURE_TRANSPORT.json'),str(previous/'targets/CUSTODY.json'),
    str(previous/'receipts/PREPARE_GRAPH_TRAIN_TARGETS.json'),'--output',str(root/'matrix')],180)
print(json.dumps(json.loads((root/'matrix/SUMMARY.json').read_text())['counts']))
''', timeout=240)
    stdout = base.request("VERIFY_AND_HANDOFF", r'''
import importlib.util
sys.path.insert(0,str(runtime))
spec=importlib.util.spec_from_file_location('context_master_ingest',root/'verification/compile_master_plan.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
entry={'id':'context_positive_exclusion_wikics','plan':bound(root/'matrix/PLAN.json'),'summary':bound(root/'matrix/SUMMARY.json')}
packet=module.load_packet(phase,entry)
plan=packet['plan'];counts=packet['summary']['counts'];targets=json.loads((root/'targets/CUSTODY.json').read_text())
assert counts['atomic_fit_groups']==1242 and counts['all_run_records']==3411
assert counts['optimizer_starts_by_stage']=={'search':162,'comparative':1080}
assert counts['full_budget_optimizer_updates']==6210000 and counts['comparative_banks']==720
assert counts['genuine_I4_banks']==180 and counts['reused_I4_body0']==180
assert plan['scientific_launch_enabled'] is False and plan['human_GO_required'] is True
assert plan['worker_capacities_cuda_bytes']=={'allocation/gpu0':78*1024**3}
assert all(g['gpu_assignment']=='allocation/gpu0' and g['allowed_workers']==['allocation/gpu0'] for g in plan['groups'])
assert len(targets['splits'])==20 and targets['SELECT_CAL_TEST_tensors_opened'] is False
assert all(not f.name.startswith('GO') for f in root.rglob('*'))
assert all(all(n>0 for n in row['preparation']['changed_incidence_entries_per_context']) for row in targets['splits'])
write_json(root/'MASTER_INPUT_ENTRY.json',entry)
verification={'schema':'context-exclusion-master-ingest-verification-v2','verified':True,
    'plan_hash':packet['plan_hash'],'counts':packet['counts'],'reference_matches':packet['references'],
    'source_method_scopes':7,'source_files':7,'target_splits':20,'source_target_preparations_paid_once':20,
    'physical_target_preparations_in_v2':0,'immutable_target_pack_bytes_reused':True,
    'actual_scientific_fits':0,'actual_predictive_scores':0,'GO_created':False,
    'SELECT_CAL_TEST_target_tensors_decoded':False,'worker':'allocation/gpu0','resources':'provisional_unmeasured',
    'remaining_gates':['CUDA_resource_and_serializer_endpoint_concurrent_qualification','combined_root_master_and_global_closures','explicit_human_GO']}
write_json(root/'MASTER_INGEST_VERIFICATION.json',verification)
handoff={'schema':'context-exclusion-packet-handoff-v2','status':'PREPARED_DISABLED_RESOURCE_QA_MASTER_AND_GO_PENDING',
    'entry':entry,'plan_hash':packet['plan_hash'],'counts':counts,'targets':bound(root/'targets/CUSTODY.json'),
    'final_cpu_qa':bound(root/'receipts/FINAL_CPU_QA_RESULT.json'),'ingest_verification':bound(root/'MASTER_INGEST_VERIFICATION.json'),
    'source_canonical_methods':7,'scientific_launch_enabled':False,'human_GO_required':True,'scientific_fits':0,
    'GO_created':False,'resource_scope':'provisional_unmeasured','full_249_coverage':False,
    'preserved_failed_predecessor':'massive_context_exclusion_packet_20261012_v1'}
write_json(root/'SUMMARY.json',handoff)
mirrors=['SOURCE_INVENTORY.json','MASTER_INPUT_ENTRY.json','MASTER_INGEST_VERIFICATION.json','SUMMARY.json',
    'matrix/PLAN.json','matrix/SUMMARY.json','targets/CUSTODY.json','receipts/FINAL_CPU_QA.json',
    'receipts/FINAL_CPU_QA_RESULT.json','receipts/COMPILE_DISABLED_PACKET.json']
mirrors += [str((phase/row['path']).relative_to(root)) for row in plan['studies']]
value={relative:base64.b64encode((root/relative).read_bytes()).decode() for relative in mirrors}
previous=phase/'massive_context_exclusion_packet_20261012_v1'
value['receipts/INHERITED_PREPARE_GRAPH_TRAIN_TARGETS.json']=base64.b64encode((previous/'receipts/PREPARE_GRAPH_TRAIN_TARGETS.json').read_bytes()).decode()
print(base64.b64encode(zlib.compress(json.dumps(value).encode())).decode())
''', timeout=180)
    mirrored = json.loads(zlib.decompress(base64.b64decode(stdout.strip())))
    for relative, encoded in mirrored.items():
        target = base.LOCAL / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = base64.b64decode(encoded)
        if target.exists():
            assert json.loads(target.read_bytes()) == json.loads(raw)
        else:
            target.write_bytes(raw)
    print(json.dumps({"packet": NAME, "status": "PREPARED_DISABLED", "scientific_fits": 0,
                      "GO": False, "local_metadata_files": len(mirrored)}), flush=True)


if __name__ == "__main__":
    main()
