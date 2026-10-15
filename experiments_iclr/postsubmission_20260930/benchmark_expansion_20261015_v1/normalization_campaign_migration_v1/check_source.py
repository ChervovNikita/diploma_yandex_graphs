"""Compact stdlib checks: counts from the real roster and finite schema fixtures."""
import ast
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_normalization_campaign_generator_check',HERE/'generate.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


def refuses(function):
    try:function()
    except ValueError:return
    raise AssertionError('Expected refusal')


def main():
    for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    roster=json.loads((g.E/'expanded_study_design_v1/PROPOSED_ROSTER.json').read_text())
    proposal=json.loads((g.E/'normalization_policy_decision_v1/COUNTS.json').read_text())
    by_method=Counter();by_family=Counter();fits=0
    for cell in roster['cells']:
        fits+=cell['total_fits']
        for method in cell['shared_method_ids']:
            row={'cell_id':cell['id'],'method_id':method,'track':'long_convergence'}
            if g.affected(row,roster):by_method[method]+=cell['cohorts'];by_family[cell['task_family']]+=cell['cohorts']
    assert fits==6460 and sum(by_method.values())==2890 and fits-sum(by_method.values())==3570
    assert by_family=={'ogb':2720,'link':170} and len(by_method)==22
    assert all(by_method[r['method_id']]==r['private_BN_candidate_fit_contracts_affected']for r in proposal['per_recipe'])
    assert sum(v for k,v in by_method.items()if k.startswith('interaction_'))==680
    for cell in roster['cells']:
        for method in ('native_single','gnnm_boundary_native','gnnm_private_bn_state_v3'):
            assert not g.affected({'cell_id':cell['id'],'method_id':method,'track':'long_convergence'},roster)
        assert not g.affected({'cell_id':cell['id'],'method_id':cell['shared_method_ids'][0],'track':'native_replica'},roster)

    contracts=g.api();fake=lambda name:{'path':'/repo/'+name+'.json','bytes':1,'sha256':'a'*64}
    # Same field shapes as the observed compiler/candidate/reporting records.
    checks=[]
    for preset,family in [('ogb-ogbg-molbace-gin-drop0.0','ogb'),('ogb-ogbg-molhiv-gcn-virtual-drop0.5','ogb'),('collab-buddy','link')]:
        fid=preset+'/long_convergence/split-0/seed-1/wrapper-boundary_global_half_pool'
        job={'schema':'native-full-epoch-fit-v1','id':fid,'recipe_id':preset,'family':family,'TEST_access':False,
             'scientific_method_id':'aggregation_mean_globalJ_v1','procedure':'boundary_global_half_pool',
             'seed':1,'split':0,'cohort_seed':1,'body':None,'cohort_id':preset+'/cohort-1',
             'policy':{'track':'matched_extended','maximum_epochs':1000},'callback_module':'/repo/callback.py',
             'callback_sha256':'c'*64,'recipe':{'dataset':'ogbl-collab'if family=='link'else'ogbg-molhiv'if'molhiv'in preset else'ogbg-molbace','native_fixed_config':{'width':300}},
             'method_settings':{'native_loss_unchanged':True},'expanded_contract':{'id':fid,'seed':1,'native_setting':'unchanged'},
             'expanded_contract_hash':'b'*64}
        original=copy.deepcopy(job)
        child=g.composed_job(job,fake('parent'),roster,contracts,fake('normalization'),fake('banks'))
        assert job==original and child['id']==fid+contracts.SUFFIX
        assert child['scientific_method_id']=='aggregation_mean_globalJ_v1' and child['procedure']=='boundary_global_half_pool'
        assert child['normalization_composition']['parent_job_sha256']==contracts.digest(original)
        for key in job:
            if key not in ('id','expanded_contract','expanded_contract_hash'):assert child[key]==job[key]
        assert child['expanded_contract_hash']==contracts.digest(child['expanded_contract'])
        refuses(lambda: g.composed_job(child,fake('already-migrated'),roster,contracts,fake('normalization'),fake('banks')))
        mapping={fid:child['id'],'ingredient':'ingredient'+contracts.SUFFIX}
        control={'fit_id':fid,'scientific_method_id':job['scientific_method_id'],'controls':[
            {'resolved_control':'gnnm_boundary_native','symbolic_control':'G0','fit_ids':['G0'],'kind':'paired_shared_fit'},
            {'resolved_control':'ingredient','fit_ids':['ingredient'],'kind':'paired_shared_fit'},
            {'kind':'diagnostic_same_selected_fit','fit_ids':[fid]}]}
        unchanged=copy.deepcopy(control)
        moved=contracts.migrated_controls(control,mapping,'private-BN-control')
        assert control==unchanged and moved['fit_id']==child['id']
        assert moved['controls'][0]['fit_ids']==['G0'] and moved['controls'][0]['comparison_role']=='total_extension_vs_original_GNNM'
        assert moved['controls'][1]['fit_ids']==[mapping['ingredient']] and moved['controls'][2]['fit_ids']==[child['id']]
        assert moved['normalization_attribution_reference']['fit_ids']==['private-BN-control']
        for bank in [{'id':'native-M1','fit_ids':['M1']},{'id':'native-I4','fit_ids':['M1','I2','I3','I4']},
                     {'id':'private-BN','fit_ids':['private-BN-control']},{'id':'G0','fit_ids':['G0']}]:
            assert g.migrated_bank(bank,mapping,contracts)==bank
        moved_bank=g.migrated_bank({'id':'candidate','fit_ids':[fid]},mapping,contracts)
        assert moved_bank['id']=='candidate'+contracts.SUFFIX and moved_bank['fit_ids']==[child['id']]
        task={'id':fid,'kind':'fit','group_id':'same-cohort','worker':'allocation/gpu0','dependencies':[],
              'output_relative':'fits/old','resources':None,'qualification_blocks':['prior_block'],
              'functional_evidence_attachment':{'claim':'original component evidence only'}}
        candidate={'id':fid,'admitted':False,'worker':'allocation/gpu0','job_descriptor':fake('parent'),
                   'runtime_entrypoint':fake('oldrun'),'runtime_job_hash':g.digest(job),'canonical_runtime_job_hash':g.digest(job)}
        old=copy.deepcopy((task,candidate));delta=g.dispatch_delta(task,candidate,child,fake('child'),fake('run_fit'))
        assert (task,candidate)==old
        assert delta['successor_task']['group_id']=='same-cohort' and delta['successor_task']['worker']=='allocation/gpu0'
        assert delta['successor_task']['binding'] is None and delta['successor_candidate']['admitted'] is False
        assert 'functional_evidence_attachment'not in delta['successor_task'] and delta['parent_component_evidence_not_transferred_as_qualification']
        bindings={k:k+'-hash'for k in('native_source','data','controller','evaluator','readout')}
        identity={'fit_id':fid,'bindings':copy.deepcopy(bindings),'scientific_method_id':job['scientific_method_id']}
        report={'identity':identity,'bindings':copy.deepcopy(bindings),'wrapper_authority':{'descriptor':fake('parent')},
                'observed_input_files':[],'same_bank_mean_identity':{**copy.deepcopy(identity),'readout':'same_selected_bank_mean'}}
        before=copy.deepcopy(report)
        update=g.reporting_delta(report,child,fake('child'),fake('run_fit'),fake('normalization'),fake('producer'),fake('banks'),'private-BN-control')
        assert report==before and update['identity']['fit_id']==child['id']
        for key in('native_source','data','controller','evaluator'):assert update['identity']['bindings'][key]==bindings[key]
        assert update['identity']['bindings']['readout']!=bindings['readout']
        assert update['same_bank_mean_identity']['fit_id']==child['id'] and update['same_bank_mean_identity']['bindings']['readout']!=update['identity']['bindings']['readout']
        checks.append(preset)
    assert 'torch'not in sys.modules
    print(json.dumps({'status':'PASS','real_roster_counts':{'fits':fits,'changed':sum(by_method.values()),'unchanged':fits-sum(by_method.values()),
        'affected_recipe_ids':len(by_method),'changed_by_family':by_family},'finite_schema_cases':checks,
        'native_source_G0_private_BN_and_BN_free_classification_preserved':True,'aliased_scientific_method_not_inferred_from_fit_suffix':True,
        'G0_total_and_matched_BN_primary_controls_preserved':True,'component_evidence_not_promoted_to_composed_PASS':True,
        'reporting_variant_and_same_bank_mean_identities_changed':True,'native_common_bindings_unchanged':True,
        'actual_remote_generation_executed':False,'Torch_imported':False,'native_work':0},indent=2,sort_keys=True))


if __name__=='__main__':main()
