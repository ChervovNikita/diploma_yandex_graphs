"""Rebind compiler3 and concrete callable candidates; metadata only, zero children."""
import socket
import subprocess
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
from collections import Counter
import copy
import json
from pathlib import Path
from contracts import HERE,EXPANSION,ROOT,guard,pin,verify,read,digest,compile_release,validate_python_environment,source_hook_pins
from dispatcher import inspect_release

guard('allocation/gpu0')
OUTPUT=HERE/'generated_v4'
assert not OUTPUT.exists(),'Preserve existing successor output'
old_pin=pin(EXPANSION/'expanded_launch_preparation_v1/generated_v1/PLAN.json')
assert old_pin['sha256']=='2f5c9adc876202fe2075fe1bfe0c31b02646bd975f418119c5ebdb7dc3c1d116'
old=read(old_pin['path']);skeleton=copy.deepcopy(old)
ready_pin=pin(EXPANSION/'expanded_job_binding_v1/generated_v4/READINESS.json')
assert ready_pin['sha256']=='80c1084e1d5f07c9df92bbee4933a269910d1b2bb8637fe43fe325e7c34c3fd7'
ready=read(ready_pin['path']);verify(ready['artifacts']['index']);idx=read(ready['artifacts']['index']['path'])['fits']
rows={x['id']:x for x in idx}
assert len(rows)==6460 and all(r['status']=='descriptor_prepared_not_admitted'for r in idx)
assert ready['blocked_fit_intentions']==0
OUTPUT.mkdir()
def write(name,value):
    path=OUTPUT/name
    with path.open('x')as handle:json.dump(value,handle,indent=2,sort_keys=True);handle.write('\n')
    return pin(path)
source_files={}
def add(value):
    assert Path(value['path']).suffix in ('.py','.json','.md','.txt','.xml','.yaml','.yml','.csv','.tsv','.html'),'Source closure must not open tensors/data'
    verify(value);old=source_files.get(value['path']);assert old is None or old==value,'Conflicting source pin'
    source_files[value['path']]=value
for name in ('contracts.py','engine.py','dispatcher.py','qualified_python_entry.py','PYTHON_ENVIRONMENT.json','MANIFEST.json'):
    add(pin(HERE/name))
for value in old['legacy_supervisor_sources'].values():add(value)
# Reuse compiler source receipt closure, including its immutable parent generations.
seen=set()
def compiler_sources(value):
    if value['path']in seen:return
    seen.add(value['path']);add(value);obj=read(value['path'])
    for item in obj['source_and_context_metadata']:add(item)
    if obj.get('parent_sources'):compiler_sources(obj['parent_sources'])
compiler_sources(ready['artifacts']['sources'])
for folder in ('runtime_v4','runtime_link_v9'):
    manifest=pin(EXPANSION/folder/'MANIFEST.json');add(manifest)
    for row in read(manifest['path'])['files']:
        if isinstance(row,dict):
            value=pin(EXPANSION/folder/row['path'])
            assert value['sha256']==row['sha256']and value['bytes']==row['bytes']
            add(value)
for name in ('aggregate_selected.py','common.py'):
    add(pin(EXPANSION/'runtime_v2'/name))
ledger_pin=pin(HERE/'EVIDENCE_APPLICABILITY_v4.json');add(ledger_pin)
for row in read(ledger_pin['path'])['evidence']:add(row['receipt'])
for name in ('MINIMAL_NATIVE_PROPOSALS.json','BLOCKERS_v4.json','SOURCE_CHECKS.json','INTERPRETER_PROBE.json','SOURCE_REPAIR_02.json','SOURCE_REPAIR_03.json','SOURCE_REPAIR_04.json','prepare_release_v2_attempt04.py'):
    add(pin(HERE/name))
interpreter=pin(HERE/'PYTHON_ENVIRONMENT.json');python_spec=read(interpreter['path'])
runtimes={'node':pin(EXPANSION/'runtime_v4/run_fit.py'),'ogb':pin(EXPANSION/'runtime_v4/run_fit.py'),
          'link':pin(EXPANSION/'runtime_link_v9/run_fit.py'),'I4':pin(EXPANSION/'runtime_v2/aggregate_selected.py')}
representatives={};candidates=[];changed=0;rebound=0;cleared=0
original={t['id']:t for t in old['tasks']}
for task in skeleton['tasks']:
    before=original[task['id']]
    if task['kind']=='fit':
        row=rows[task['id']]
        assert (task['cell_id'],task['family'],task['protocol_track'],task['scientific_method_id'],task['cohort_seed'],task['initialization_seed'],task['split'],task['body'])==(
            row['cell_id'],row['family'],row['track'],row['method_id'],row['cohort_seed'],row['initialization_seed'],row['split'],row['body'])
        changed+=task['science_contract_hash']!=row['expanded_contract_hash']
        rebound+=task['source_descriptor']!=row['descriptor']
        cleared+=bool(task.get('source_blocked_reason'))
        old_reason=task.get('source_blocked_reason')
        task.update(science_contract_hash=row['expanded_contract_hash'],source_descriptor=row['descriptor'],runtime_job_hash=row['runtime_job_hash'],
                    source_status=row['status'],source_blocked_record=None,source_blocked_reason=None)
        task['qualification_blocks']=[b for b in task['qualification_blocks']if b not in ('missing_source_context','missing_reporting_statistics_access',old_reason)]
        task['reporting_statistics_access_approved']=pin(EXPANSION/'benchmark_reporting_protocol_v1/ROOT_REVIEW.json')
        case=(row['cell_id'],row['method_id'],row['track'])
        if case not in representatives:
            job=read(row['descriptor']['path']);verify(row['descriptor'])
            assert digest(job)==row['runtime_job_hash']and digest(job['expanded_contract'])==row['expanded_contract_hash']
            assert job['id']==task['id']and job['device']=='cuda:0'and job['TEST_access']is False
            for hook in source_hook_pins(job):
                actual=pin(hook['path']);assert actual['sha256']==hook['sha256'];add(actual)
            representatives[case]={'task':task['id'],'callback':job['callback_module'],'procedure':job['procedure'],'native_config':job['recipe']['native_fixed_config']}
        runtime=runtimes[row['family']];descriptor=row['descriptor']
        components=['coverage_38','source_rng','reporting_ROOT_REVIEW']if row['track']=='native_replica'else['coverage_38','reporting_ROOT_REVIEW']
        if row['family']=='node':components+=['node_v4']
        elif row['family']=='ogb':components+=['molecular_v4']
        elif row['cell_id']=='collab-ncnc':components+=['NCNC_v6']
        elif row['cell_id']=='collab-buddy':components+=['BUDDY_production_seed1']
    else:
        runtime=runtimes['I4'];descriptor=None;components=['coverage_38','reporting_ROOT_REVIEW']
        assert task['science_contract_hash']==before['science_contract_hash']and task['M1_exact_reuse']==task['dependencies'][0]
    execution={'runtime_entrypoint':runtime,'python_executable':python_spec['lexical_executable'],'python_sha256':python_spec['binary_sha256'],
        'python_environment':interpreter,'job_descriptor':descriptor,'scope':'expanded_finite_campaign'}
    record={'id':task['id'],'science_contract_hash':task['science_contract_hash'],'runtime_job_hash':task['runtime_job_hash'],
        'execution':execution,'source_callable_candidate':True,'admitted':False,'resources':None,'measured_cost':None,'gates':{},
        'shared_evidence_ids':components,'evidence_applicability':ledger_pin,
        'remaining':'approved exploratory reporting contract attached;remaining exact source applicability/scope,mature target resources and human GO'}
    candidates.append(record)
    task['functional_evidence_attachment']={'ledger':ledger_pin,'shared_evidence_ids':components,'claim':'component/source scope only; no individual seed native PASS'}
    task['execution_candidate_id']=task['id']
    assert task['worker']==before['worker']and task['output_relative']==before['output_relative']and task['group_id']==before['group_id']and task['dependencies']==before['dependencies']
    assert task.get('resources')is None and task.get('execution_binding')is None
assert len(candidates)==6730 and len(representatives)==565
candidate_pin=write('EXECUTION_CANDIDATES.json',{'schema':'expanded-concrete-runtime-candidates-v2','candidates':candidates,'resource_admission':False,'campaign_GO':False})
for task in skeleton['tasks']:task['execution_candidates']=candidate_pin
skeleton.update(status='compiler3_and_runtime_successor_source_prepared_all_execution_blocked',descriptor_readiness=ready_pin,
    descriptor_index=ready['artifacts']['index'],source_rebinding_parent=old_pin,execution_candidates=candidate_pin,
    evidence_applicability=ledger_pin,source_execution_bridge_enabled=False,human_GO=None)
# Native I4 source science contracts are unchanged after the source overlay.
new_index={t['id']:t for t in skeleton['tasks']}
for t in skeleton['tasks']:
    if t['kind']!='fit':
        expected=digest({'bank_id':t['bank_id'],'fit_contracts':[new_index[x]['science_contract_hash']for x in t['dependencies']],
            'aggregation':'mean_raw_logits_after_four_own_selectors','VALID_only':True})
        assert expected==t['science_contract_hash']
plan_pin=write('PLAN.json',skeleton)
# Exact typed execution objects are usable candidates; resource/runtime admissions remain absent.
validate_python_environment(candidates[0]['execution'],{p['path']:p['sha256']for p in source_files.values()})
assert all(c['execution']['python_environment']==interpreter and c['execution']['python_executable']==python_spec['lexical_executable']and c['execution']['python_sha256']==python_spec['binary_sha256']for c in candidates)
bindings={'schema':'expanded-production-bindings-v1','skeleton':plan_pin,'source_files':list(source_files.values()),'tasks':{},
    'concrete_candidates':candidate_pin,'evidence_applicability':ledger_pin,'scope':'No resource/admission/GO fabricated. Source callable candidates are explicit and separately pinned.'}
binding_pin=write('BINDINGS.json',bindings)
result=compile_release(plan_pin,binding_pin['path'],OUTPUT/'RELEASE.json',EXPANSION/'campaign_outputs/expanded_dispatcher_v2_REQUIRES_REVIEW')
preflight=inspect_release(OUTPUT/'RELEASE.json','allocation/gpu0');assert preflight['qualified_tasks']==0 and preflight['blocked_tasks']==6730 and preflight['children_started']==0
preflight_pin=write('PREFLIGHT.json',preflight)
summary={'schema':'expanded-dispatcher-compiler3-runtime-successor-readiness-v2','status':'source_callable_candidates_complete_production_admission_blocked',
 'release':result['release'],'skeleton':plan_pin,'bindings':binding_pin,'execution_candidates':candidate_pin,'preflight':preflight_pin,
 'compiler_readiness':ready_pin,'dispatcher_manifest':pin(HERE/'MANIFEST.json'),'runtime_link_v9_manifest':pin(EXPANSION/'runtime_link_v9/MANIFEST.json'),
 'collector_repair':pin(HERE/'SOURCE_REPAIR_04.json'),'shared_evidence_applicability':ledger_pin,'blockers':pin(HERE/'BLOCKERS_v4.json'),'minimal_native_proposals':pin(HERE/'MINIMAL_NATIVE_PROPOSALS.json'),
 'counts':{'fit_intentions':6460,'I4_export_only':270,'tasks':6730,'source_descriptors_prepared':6460,'source_context_blocks':0,
 'old_source_context_blocks_closed':cleared,'descriptors_rebound':rebound,'source_binding_contracts_changed':changed,'callable_execution_candidates':6730,
 'unique_native_method_track_cases':len(representatives),'fully_admitted_tasks':0,'blocked_production_tasks':6730,'mature_resource_profiles_admitted':0},
 'readiness_dimensions':read(ledger_pin['path'])['dimensions'],'engine_bytes_unchanged':True,'predecessor38checks_reused_not_rerun':True,
 'new_checks':'authentic declaration source protocols, narrow successor source/AST checks, one real no-model lexical interpreter probe',
 'source_reference_matched_I4_dependencies_placement_and_native_M1_reuse_preserved':True,'runtime_checkpoint_migration':False,
 'models_constructed':0,'native_forwards':0,'optimizer_updates':0,'scientific_fits_started':0,'TEST_accessed':False,'server77_accessed':False,
 'source_or_data_tensors_opened':0,'human_GO':False,'children_started':0,'maximum_owned_children':2,'initial_serial1_is_valid':True,'campaign_ETA':None}
summary_pin=write('SUMMARY.json',summary)
print(json.dumps({'summary':summary_pin,'release':result['release'],'candidates':6730,'source_context_blocks':0,'production_admitted':0,'children_started':0},sort_keys=True))
