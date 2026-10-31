"""Stdlib-only disabled retained supervised P/T v2 and matched-control roster."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess

from adapters.retained_pt_routes import CONTRACT, PINS, SOURCE_ARMS, PT_ARMS, NATIVE_CONTROLS, bound
from compile_native_matrix import bind_custody, file_binding, RUNTIME_FILES, resource_for
from prepare_node_data import verify_allocation
from spec import canonical, confined, fingerprint, validate

SEARCH_SEEDS = (73101,73103)
COMPARATIVE_SEEDS = (73201,73203,73207)
DOMAIN = [{'learning_rate':lr,'dropout':dropout} for lr in (.0005,.001,.002) for dropout in (.2,.4,.6)]
I4 = {'ordinary_i4':'ordinary_p1','factor_i4':'factor_f1','mixed_i4':'mixed_single',
      'bias_coordinate_i4':'bias_coordinate_single','continued_p1_i4':'continued_p1','continued_f1_i4':'continued_f1'}
METHODS = {a:'retained_pt_'+a for a in SOURCE_ARMS+PT_ARMS+NATIVE_CONTROLS+tuple(I4)}
METHODS['original_gnnm'] = 'original_boundary_GNNM'


def validate_custody(c):
    expected = {'wikics':20,'amazon-photo':10}
    if c['dataset_id'] not in expected or sorted(r['split_id'] for r in c['splits']) != list(range(expected[c['dataset_id']])):
        raise ValueError('Complete WikiCS20/Photo10 new-role custody required')
    if c['status'] != 'prepared' or c['benchmark_fits'] != 0 or c['predictive_scores'] != 0 or any(r['evaluate_purpose'] != 'development' for r in c['splits']):
        raise ValueError('Actual no-fit development preparation required')


def make_study(root,custody,data_dir,sources,source_bindings,profiles,runtime,validator=validate):
    validate_custody(custody); root = Path(root).resolve(); data_dir = Path(data_dir).resolve()
    dataset = custody['dataset_id']; splits = {r['split_id']:r for r in custody['splits']}; known = {r['path']:r for r in custody['files']}
    def inputs(split,field):
        return {k:{'path':str((data_dir/v).relative_to(root)),'bytes':known[v]['bytes'],'sha256':known[v]['sha256']} for k,v in splits[split][field].items()}
    def endpoints(split): return {k:v['path'] for k,v in inputs(split,'endpoint_inputs').items()}
    runs = []; selections = {}; banks = []
    def add(identifier,kind,arm,split,seed,body,config,parents=()):
        entry = runtime+'/adapters/retained_pt_routes.py:make_adapter' if kind=='model_fit' else runtime+'/adapters/search.py:select_configuration' if kind=='selection' else runtime+'/adapters/retained_pt_endpoints.py:'+{'ensemble_assembly':'export_bank','calibration':'scalar_calibration','evaluation':'evaluate_bank'}[kind]
        row = {'run_id':identifier,'kind':kind,'method':METHODS[arm],'split':split,'seed':seed,'body':body,'adapter':entry,'config':config,'parents':list(parents),
               'budget':{'unit':'optimizer_updates','policy':'full_budget','maximum':5000} if kind=='model_fit' else {},
               'selection':{'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False} if kind=='model_fit' else {}}
        runs.append(row); return identifier
    def common(split,arm,phase,seed,body=0):
        p = resource_for(profiles,dataset,'retained-pt-'+arm)
        members = 4 if arm in ('factor_s4','original_gnnm','head_only','private4','middle_state_mean','separable4') else 1
        return {'architecture':'original_gnnm' if arm=='original_gnnm' else 'retained_pt_routes','pt_stage':'source' if arm in SOURCE_ARMS else 'continuation','pt_arm':arm,
                'members':members,'physical_slow_bodies':4 if arm=='separable4' else 1,'source_optimizer_count':1,
                'backbone':'SAGE','depth':2,'width':128 if arm=='original_gnnm' else 32,'classes':custody['model_graph']['classes'],
                'weight_decay':0.,'pt_sources':source_bindings,'node_inputs':inputs(split,'node_inputs'),
                'fusion':'mean_logits' if arm=='original_gnnm' else 'mean_probabilities','stage':phase,'is_original_gnnm':arm=='original_gnnm',
                'paired_seed':seed,'initialization_seed':seed+1000003*body,
                'reservation_mib':p['reservation_mib'],'estimated_seconds':p['estimated_seconds_per_update']*5000,
                'recipe_scope':'retained_PT_exact_v2_standardized_roles_optimizer_controller_and_explicit_new_controls'}
    def lifecycle(parent):
        return {'schema':'paid-parent-continuation-v1','parent_fit_ids':[parent],'new_optimizer':True,'parameter_initialization':'selected_parent_no_optimizer_reuse'}
    def search(arm,parents=None):
        candidates = []
        for point in DOMAIN:
            cid = fingerprint(point)[:20]; identifiers = []
            for seed in SEARCH_SEEDS:
                parent = parents[seed] if parents else None
                config = {**common(0,arm,'search',seed),**point}
                if parent: config['lifecycle'] = lifecycle(parent)
                identifiers.append(add('search-'+arm+'-'+cid+'-s'+str(seed),'model_fit',arm,0,seed,0,config,[parent] if parent else []))
            candidates.append({'config_id':cid,'parameters':point,'run_ids':identifiers})
        selections[arm] = add('select-'+arm,'selection',arm,0,0,0,
            {'stage':'search_selection','configurations':candidates,'ranking_metric':'accuracy','ranking_direction':'max','reservation_mib':1,'estimated_seconds':1},[i for c in candidates for i in c['run_ids']])
    for arm in SOURCE_ARMS: search(arm)
    anchors = {arm:{} for arm in ('ordinary_p1','factor_f1')}
    for arm in anchors:
        for seed in SEARCH_SEEDS:
            anchors[arm][seed] = add('search-anchor-'+arm+'-s'+str(seed),'model_fit',arm,0,seed,0,
                {**common(0,arm,'search',seed),**DOMAIN[0],'selected_config_from':selections[arm]},[selections[arm]])
    for arm in PT_ARMS+NATIVE_CONTROLS: search(arm,anchors['factor_f1' if arm=='continued_f1' else 'ordinary_p1'])
    for split in sorted(splits):
        for seed in COMPARATIVE_SEEDS:
            fits = {}
            for arm in SOURCE_ARMS:
                fits[arm] = []
                for body in range(4 if arm in ('ordinary_p1','factor_f1') else 1):
                    fits[arm].append(add(f'compare-sp{split}-{arm}-s{seed}-b{body}','model_fit',arm,split,seed,body,
                        {**common(split,arm,'comparative',seed,body),**DOMAIN[0],'selected_config_from':selections[arm]},[selections[arm]]))
            # Retain complete source quartets, but M1 bank serves the fixed body0.
            source_quartets = copy.deepcopy(fits)
            for arm in PT_ARMS+NATIVE_CONTROLS:
                fits[arm] = []
                for body in range(4 if arm in ('mixed_single','bias_coordinate_single')+NATIVE_CONTROLS else 1):
                    parent = source_quartets['factor_f1' if arm=='continued_f1' else 'ordinary_p1'][body]
                    fits[arm].append(add(f'compare-sp{split}-{arm}-s{seed}-b{body}','model_fit',arm,split,seed,body,
                        {**common(split,arm,'comparative',seed,body),**DOMAIN[0],'selected_config_from':selections[arm],'lifecycle':lifecycle(parent)},[selections[arm],parent]))
            for arm,member in I4.items(): fits[arm] = fits[member][:]
            for arm in ('ordinary_p1','factor_f1','mixed_single','bias_coordinate_single')+NATIVE_CONTROLS: fits[arm] = fits[arm][:1]
            for arm,parents in fits.items():
                member = I4.get(arm,arm); fusion = common(split,member,'comparative',seed)['fusion']; p = resource_for(profiles,dataset,'retained-pt-'+member)
                prefix = f'bank-sp{split}-{arm}-s{seed}'; inp = endpoints(split)
                export = add(prefix,'ensemble_assembly',arm,split,seed,0,{'stage':'bank_export','fit_parents':parents,'endpoint_inputs':inp,'fusion':fusion,'reservation_mib':p['reservation_mib'],'estimated_seconds':p.get('estimated_export_seconds',1)},parents)
                cal = add(prefix+'-cal','calibration',arm,split,seed,0,{'stage':'calibration','export_parent':export,'endpoint_inputs':inp,'fusion':fusion,'placement':'after_serving_pool','steps':500,'learning_rate':.01,'reservation_mib':512,'estimated_seconds':p.get('estimated_scalar_cal_seconds',5)},[export])
                evaluation = add(prefix+'-evaluate','evaluation',arm,split,seed,0,{'stage':'evaluation','export_parent':export,'calibration_parent':cal,'endpoint_inputs':inp,'fusion':fusion,'reservation_mib':512,'estimated_seconds':p.get('estimated_evaluation_seconds',5)},[export,cal])
                banks.append({'arm':arm,'split':split,'seed':seed,'fit_parents':parents,'export':export,'calibration':cal,'evaluation':evaluation,'genuine_I4':arm in I4,'body0_reused':arm in I4})
    roles = {role:{str(split):inputs(split,'node_inputs' if role in ('train','select') else 'endpoint_inputs')[role if role in ('train','select') else role+'_targets'] for split in splits} for role in ('train','select','cal','evaluate')}
    study = {'schema_version':1,'study_id':'retained-pt-v1-'+dataset,'stage':'development','dataset':dataset,
        'hypothesis':'Retained supervised PT v2 compared with fresh matched ordinary/factor/shared/independent, history, bias-coordinate, private-capacity and continued-native controls on new provider roles.',
        'data':{'task':'node_classification','preprocessing':custody['request']['graph_policy'],'roles':roles,'prior_exposure':custody['request']['evaluation_exposure'],
                'evidence':{'custody_request_hash':custody['request_hash'],'model_graph':custody['model_graph'],'source_contract':CONTRACT,'source_exact_training_recipe':False}},
        'source':sources,'runs':runs,'comparisons':[{'references':[METHODS[a] for a in ('original_gnnm','ordinary_p1','ordinary_i4','factor_f1','factor_i4','factor_s4','continued_p1','continued_p1_i4','continued_f1','continued_f1_i4')],
            'candidates':[METHODS[a] for a in PT_ARMS+('mixed_i4','bias_coordinate_i4')],'scope':'complete provider split roster and three fixed comparative seeds; paid ancestors charged once; stages separately reported'}],
        'release':{'policy':'whole_roster','primary_endpoint':'closed raw and scalar-calibrated banks','global_plan_gates_required':True},
        'resources':{'wall_seconds':max(p['wall_seconds'] for p in profiles.values()),'status':'PROVISIONAL_DISABLED_REQUIRES_FULL_INPUT_RESOURCE_QA_AND_GO'}}
    validator(study); fits = [r for r in runs if r['kind']=='model_fit']; sources_count = sum(r['config']['pt_stage']=='source' for r in fits)
    counts = {'atomic_fit_groups':len(fits),'paid_optimizer_objects':len(fits),'physical_body_instances':sum(r['config']['physical_slow_bodies'] for r in fits),
        'model_fits':len(fits),'full_budget_optimizer_updates':5000*len(fits),'optimizer_starts_by_stage':dict(Counter(r['config']['stage'] for r in fits)),
        'source_acquisition_fits':sources_count,'source_acquisition_optimizer_updates':sources_count*5000,'continuations':len(fits)-sources_count,
        'continuation_optimizer_updates':(len(fits)-sources_count)*5000,'extensions':0,'search_selection_jobs':len(selections),'comparative_banks':len(banks),
        'bank_exports':len(banks),'scalar_CAL_fits':len(banks),'evaluation_jobs':len(banks),'genuine_I4_banks':sum(b['genuine_I4'] for b in banks),
        'reused_I4_body0':sum(b['body0_reused'] for b in banks),'all_run_records':len(runs),'actual_optimizer_updates':0,'actual_predictive_scores':0}
    return study,counts,banks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','source-bindings','runtime-package','resources','output'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--custody',nargs=2,required=True); parser.add_argument('--fixture-receipts',nargs='+',required=True); parser.add_argument('--source-commit')
    args = parser.parse_args(); verify_allocation()
    root = Path(args.project_root).resolve()
    def path(value):
        v = Path(value); return confined(root,str(v.resolve().relative_to(root)) if v.is_absolute() else str(v))
    runtime,output,bindings_path,resource_path = map(path,(args.runtime_package,args.output,args.source_bindings,args.resources))
    if output.exists() or runtime.is_relative_to(root/'training_pipeline_v2_20261012'): raise ValueError('Fresh packet and frozen external runtime required')
    bindings = json.loads(bindings_path.read_text())
    if set(bindings)!=set(PINS) or any(bindings[k]['sha256']!=v for k,v in PINS.items()): raise ValueError('Exact preserved source pins required')
    for row in bindings.values(): bound(root,row)
    resources = json.loads(resource_path.read_text()); capacities = resources['worker_capacities_cuda_bytes']
    if set(capacities)!= {'allocation/gpu0'}: raise ValueError('Allocation-only disabled packet')
    custodies = [bind_custody(root,path(v)) for v in args.custody]
    if {v[0]['dataset_id'] for v in custodies} != {'wikics','amazon-photo'}: raise ValueError('Both complete new providers required')
    files = [file_binding(root,runtime/v) for v in list(RUNTIME_FILES)+['adapters/retained_pt_routes.py','adapters/retained_pt_endpoints.py','compile_retained_pt_matrix.py','compile_native_matrix.py','prepare_node_data.py']]
    files += list(bindings.values())+[file_binding(root,bindings_path),file_binding(root,resource_path)]
    files += [file_binding(root,path(v)) for v in args.fixture_receipts]
    for c,d,b in custodies: files += b
    sources = {'git_commit':args.source_commit or subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'files':list({r['path']:r for r in files}.values()),'working_tree_fingerprints_authoritative':True,'frozen_runtime_package':str(runtime.relative_to(root)),'official_author_protocol_replication':False}
    schema_spec = importlib.util.spec_from_file_location('retained_pt_frozen_schema',runtime/'spec.py'); schema = importlib.util.module_from_spec(schema_spec); schema_spec.loader.exec_module(schema)
    output.mkdir(parents=True); (output/'studies').mkdir(); prefix = str(output.relative_to(root)); studies = []; groups = []; total = Counter(); banks = []
    for c,d,b in custodies:
        study,counts,cohort = make_study(root,c,d,sources,bindings,resources['profiles'],str(runtime.relative_to(root)),schema.validate)
        for k,v in counts.items():
            if type(v)==int: total[k]+=v
        for row in study['runs']:
            if row['config']['reservation_mib']*1024**2>capacities['allocation/gpu0']: raise ValueError('Provisional reservation exceeds capacity')
        relative = prefix+'/studies/'+study['study_id']+'.json.gz'; (root/relative).write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0))
        studies.append({'path':relative,'spec_hash':fingerprint(study)})
        groups.append({'id':study['study_id'],'gpu_assignment':'allocation/gpu0','allowed_workers':['allocation/gpu0'],'maximum_cuda_bytes':max(r['config']['reservation_mib'] for r in study['runs'])*1024**2,'estimated_seconds':sum(r['config']['estimated_seconds'] for r in study['runs']),'run_keys':[study['study_id']+'/'+r['run_id'] for r in study['runs']]})
        banks += [{'dataset':c['dataset_id'],**v} for v in cohort]
    plan = {'schema_version':1,'slots_per_gpu':2,'torch_threads_per_fit':2,'output_root':prefix+'/execution','runtime_launcher':str((runtime/'launcher.py').relative_to(root)),'runtime_package':str(runtime.relative_to(root)),
        'worker_capacities_cuda_bytes':capacities,'studies':studies,'groups':groups,'global_gates':{},'barriers':{},'scientific_launch_enabled':False,'human_GO_required':True,
        'host_bindings':{'allocation':{'status':'verified_at_compilation','hostname':'anogena-2-0','gpu_uuids':['GPU-44039938-fd82-41d2-fefd-de71514e2fac']}}}
    for name,stages,required in [('whole_search',['search','search_selection'],['comparative']),('whole_comparative',['comparative'],['bank_export','calibration','evaluation']),('whole_calibration',['calibration'],['evaluation'])]:
        plan['global_gates'][name] = {'path':prefix+'/gate_receipts/'+name+'.json','source_stages':stages,'required_for_stages':required,'required_hosts':['allocation']}
    summary = {'schema':'retained-pt-disabled-summary-v1','status':'DISABLED_PROVISIONAL_RESOURCE_MASTER_AND_GO_PENDING','plan_hash':fingerprint(plan),'counts':dict(total),'banks':banks,'source_contract':CONTRACT,'domain':DOMAIN,
        'search_seeds':SEARCH_SEEDS,'comparative_seeds':COMPARATIVE_SEEDS,'scientific_launch_enabled':False,'human_GO_required':True,'scientific_fits':0,'GO_created':False,'full_249_coverage':False,
        'per_PT_body_total_paid_updates':10000,'source_ancestry_receipts_and_H0_deferred_until_paid_source_fit':True}
    (output/'PLAN.json').write_text(canonical(plan)+'\n'); (output/'SUMMARY.json').write_text(canonical(summary)+'\n')
    print(canonical({'counts':dict(total),'plan_hash':summary['plan_hash'],'science':False,'GO':False}))


if __name__=='__main__': main()
