"""Stdlib projection/binding checks; no subprocess, host probe, worker or model."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
if str(HERE)not in sys.path:sys.path.insert(0,str(HERE))
import binding_rules as rules
import prepare_projection as projection
import reporting_binding as reporting


def refuses(function):
    try:function()
    except ValueError:return
    raise AssertionError('Expected explicit refusal')


def node(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.Assign))
                and (getattr(n,'name',None)==name or isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets)))


def reporting_fixture():
    candidates=[];tasks=[];profiles={};presets=[]
    source={'path':'/repo/source.json','bytes':1,'sha256':'a'*64}
    scorer={'path':'/repo/scorer.json','bytes':1,'sha256':'b'*64}
    registry={'preset_files':presets,'selected_serving_source':scorer,
        'normalization_rebind':{'schema':'normalization-reporting-registry-rebind-v1','source_manifest':source,
            'migration_preparation':source,'selected_serving_source_bound':True,'changed_specs':2890,
            'unchanged_specs':3570,'native_qualification_granted':False,'human_GO':False,'reporting_delta':source}}
    for i in range(27):
        preset='preset'+str(i);pp={'path':'/repo/'+preset+'.json','bytes':1,'sha256':'d'*64};presets.append(pp)
        profile={'normalization_rebind':{'schema':'normalization-reporting-profile-rebind-v1','source_manifest':source,
                    'reporting_delta':source,'native_qualification_granted':False,'human_GO':False},
            'selected_serving_source':scorer,'registry':{'presets':[{'id':preset}],
                'expected_predictors':[{}for _ in range(230 if i<7 else 229)]},'fit_specs':{},'I4_specs':{}}
        profiles[pp['path']]=profile
        for j in range(240 if i<7 else 239):
            fid=preset+'/fit'+str(j);descriptor={'path':'/repo/'+fid+'.json','bytes':1,'sha256':'e'*64}
            candidate={'id':fid,'worker':'allocation/gpu0','admitted':False,'job_descriptor':descriptor,
                'runtime_job_hash':'f'*64,'runtime_entrypoint':source};candidates.append(candidate)
            task={'id':fid,'kind':'fit','worker':'allocation/gpu0','cell_id':preset,'runtime_job_hash':'f'*64,
                'output_relative':'fits/'+fid};tasks.append(task)
            profile['fit_specs'][fid]={'identity':{'fit_id':fid},'descriptor':descriptor,'job_hash':'f'*64,
                'runtime_entrypoint':source,'output_relative':task['output_relative']}
        for j in range(10):
            fid=preset+'/I4/'+str(j);deps=[preset+'/fit'+str(4*j+k)for k in range(4)]
            candidates.append({'id':fid,'worker':'allocation/gpu0','admitted':False,'job_descriptor':None,'runtime_entrypoint':source})
            tasks.append({'id':fid,'kind':'I4_export','worker':'allocation/gpu0','cell_id':preset,'dependencies':deps,'output_relative':'I4/'+fid})
            profile['I4_specs'][fid]={'identity':{'fit_id':fid},'aggregator_source':source,'component_fit_ids':deps,'output_relative':'I4/'+fid}
    reporting.validate_registry_identity(registry,source,source,scorer)
    checked=reporting.crosscheck_profiles(registry,candidates,tasks,lambda p:profiles[p['path']])
    bad=copy.deepcopy(registry);bad['normalization_rebind']['source_manifest']={'wrong':True}
    refuses(lambda:reporting.validate_registry_identity(bad,source,source,scorer))
    candidates[0]['job_descriptor']={'wrong':True}
    refuses(lambda:reporting.crosscheck_profiles(registry,candidates,tasks,lambda p:profiles[p['path']]))
    candidates[0]['job_descriptor']=profiles[presets[0]['path']]['fit_specs'][candidates[0]['id']]['descriptor']
    export=next(t for t in tasks if t['kind']!='fit');export['dependencies']=list(reversed(export['dependencies']))
    refuses(lambda:reporting.crosscheck_profiles(registry,candidates,tasks,lambda p:profiles[p['path']]))
    return checked


def main():
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    parent=HERE.parent/'portable_dispatcher_v3'
    for name in ('guard','load_legacy','validate_resources','validate_python_environment','source_hook_pins'):
        assert ast.dump(node(HERE/'contracts.py',name),include_attributes=False)==ast.dump(node(parent/'contracts.py',name),include_attributes=False)
    assert ast.dump(node(HERE/'contracts.py','GATES'),include_attributes=False)==ast.dump(node(parent/'contracts.py','GATES'),include_attributes=False)
    job={'id':'migrated','expanded_contract_hash':'b'*64,'normalization_composition':{
        'source_manifest_sha256':rules.NORMALIZATION_SHA,'banks_sha256':rules.BANKS_SHA,
        'parent_descriptor':{'path':'/repo/parent.json','bytes':1,'sha256':'a'*64},'parent_job_sha256':'c'*64}}
    bound=rules.normalized_execution_binding(job,rules.ROOT_LITERAL,rules.PHASE_LITERAL)
    assert bound['canonical_runtime_job_hash']==bound['bound_runtime_job_hash']==rules.digest(job)
    assert bound['launch_admitted']is False and not bound['field_changes']
    refuses(lambda:rules.normalized_execution_binding(job,'/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git','postsubmission_research_20260930'))
    host={'host':'allocation','project_root':rules.ROOT_LITERAL,'phase_relative':rules.PHASE_LITERAL}
    task={'id':'migrated','runtime_job_hash':rules.digest(job),'kind':'fit'}
    result=projection.project_task(task,'allocation/gpu0',host,job)
    assert result['runtime_binding']==bound and result['worker']=='allocation/gpu0'and task=={'id':'migrated','runtime_job_hash':rules.digest(job),'kind':'fit'}
    refuses(lambda:projection.project_task(task,'server77/gpu0',host,job))

    # Full-size scalar metadata only: no JOB files, process, claims or tensor.
    tasks=[];changed=[]
    for i in range(6460):
        old={'id':'fit/'+str(i),'kind':'fit','output_relative':'fits/old'+str(i),'dependencies':[],
             'binding':None,'execution_status':'blocked','group_id':'cohort/'+str(i//24),'worker':'allocation/gpu0'}
        tasks.append(old)
        if i<2890:
            new=copy.deepcopy(old);new['id']+='/normalization-route-private-v1';new['output_relative']='fits/new'+str(i)
            changed.append({'parent_task':copy.deepcopy(old),'successor_task':new})
    for i in range(270):tasks.append({'id':'I4/'+str(i),'kind':'I4_export','dependencies':['fit/'+str(3000+4*i+j)for j in range(4)],
        'output_relative':'I4/'+str(i),'binding':None,'execution_status':'blocked','group_id':'cohort/I4'+str(i),'worker':'allocation/gpu0'})
    original=copy.deepcopy(tasks);candidate_pin={'path':'/repo/CANDIDATES.json','bytes':1,'sha256':'d'*64}
    canonical=projection.canonical_tasks({'tasks':tasks},{'changed_tasks':changed},candidate_pin)
    assert tasks==original and len(canonical)==len({r['id']for r in canonical})==6730
    for before,after in zip(tasks[2890:],canonical[2890:]):
        expected=copy.deepcopy(before);expected.pop('binding');expected.pop('execution_status');expected['execution_candidates']=candidate_pin
        assert after==expected
    assert [r['dependencies']for r in canonical[6460:]]==[r['dependencies']for r in tasks[6460:]]
    bad=copy.deepcopy(changed);bad[-1]=copy.deepcopy(bad[0])
    refuses(lambda:projection.canonical_tasks({'tasks':tasks},{'changed_tasks':bad},candidate_pin))
    bad=copy.deepcopy(changed);bad[0]['parent_task']['worker']='server77/gpu0'
    refuses(lambda:projection.canonical_tasks({'tasks':tasks},{'changed_tasks':bad},candidate_pin))
    # Exercise actual whitelist source while supplying only a scalar metadata reader.
    function=copy.deepcopy(node(HERE/'binding_rules.py','expected_entrypoint'))
    env={'E':rules.E,'require':rules.require,'json_pin':lambda value,root:value}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(HERE/'binding_rules.py'),'exec'),env)
    choose=env['expected_entrypoint']
    for family in ('node','ogb','link'):
        source={'id':'old'};t={'id':'old','kind':'fit','family':family,'source_descriptor':source}
        assert choose(t,Path('/repo'))==rules.E/'portable_dispatcher_v3'/('runtime_link/run_fit.py'if family=='link'else'runtime_node_molecular/run_fit.py')
    t={'id':'new','kind':'fit','family':'ogb','source_descriptor':{'id':'new','normalization_composition':{'declared':True}}}
    assert choose(t,Path('/repo'))==rules.E/'composable_normalization_runtime_v1/runtime/run_fit.py'
    assert choose({'kind':'I4_export'},Path('/repo'))==rules.E/'portable_dispatcher_v3/runtime_I4/aggregate_selected.py'
    reporting_checked=reporting_fixture()
    assert 'torch'not in sys.modules
    print(json.dumps({'status':'PASS','metadata_fixture_tasks':6730,'changed_fit_ids':2890,'unchanged_fit_ids':3570,'unchanged_I4_exports':270,
        'cohort_outputs_and_dependencies_preserved':True,'duplicate_or_changed_parent_refused':True,
        'same_allocation_composition_binding_passed':True,'server77_composition_refused':True,
        'actual_runtime_whitelist_AST_exercised':True,'resource_interpreter_and_native_gate_functions_AST_unchanged':True,
        'reporting_complete_candidate_profile_crosscheck':reporting_checked,
        'reporting_wrong_lineage_descriptor_or_I4_order_refused':True,
        'Torch_imported':False,'host_or_GPU_probes':0,'claims_created':0,'native_jobs':0,'actual_metadata_generation':False},indent=2,sort_keys=True))


if __name__=='__main__':main()
