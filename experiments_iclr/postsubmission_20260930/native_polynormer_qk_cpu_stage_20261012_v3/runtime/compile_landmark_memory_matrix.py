"""Stdlib-only disabled landmark memory and matched-control roster."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess

from adapters.landmark_memory import CONTRACT, PINS, SOURCE_ARMS, MEMORY_ARMS, NATIVE_CONTROLS, bound, prepare_graph_capsule, tensor_hash
from compile_native_matrix import bind_custody, file_binding, RUNTIME_FILES, resource_for
from prepare_node_data import verify_allocation
from spec import canonical, confined, fingerprint, validate

SEARCH_SEEDS = (75101,75103)
COMPARATIVE_SEEDS = (75201,75203,75207)
DOMAIN = [{'learning_rate':lr,'dropout':dropout} for lr in (.0025,.005,.01) for dropout in (.3,.5,.7)]
I4 = {'ordinary_i4':'ordinary_m1','factor_i4':'factor_m1','continued_m1_i4':'continued_m1','continued_f1_i4':'continued_f1'}
MEMORY_I4 = {'I4[S64]':'S64','I4[FS64]':'FS64'}
METHODS = {a:'landmark_standardized_'+a for a in SOURCE_ARMS+MEMORY_ARMS+NATIVE_CONTROLS+tuple(I4)+tuple(MEMORY_I4)}
METHODS['original_boundary'] = 'original_boundary_GNNM'



def validate_custody(c):
    expected = {'wikics':20,'amazon-photo':10}
    if c['dataset_id'] not in expected or sorted(r['split_id'] for r in c['splits']) != list(range(expected[c['dataset_id']])):
        raise ValueError('Complete WikiCS20/Photo10 new-role custody required')
    if c['status'] != 'prepared' or c['benchmark_fits'] != 0 or c['predictive_scores'] != 0 or any(r['evaluate_purpose'] != 'development' for r in c['splits']):
        raise ValueError('Actual no-fit development preparation required')


def make_study(root,custody,data_dir,graph_custody,sources,source_bindings,profiles,runtime,validator=validate):
    validate_custody(custody); root = Path(root).resolve(); data_dir = Path(data_dir).resolve()
    dataset = custody['dataset_id']; graph_rows = {r['seed']:r for r in graph_custody['capsules']}; splits = {r['split_id']:r for r in custody['splits']}; known = {r['path']:r for r in custody['files']}
    def inputs(split,field):
        return {k:{'path':str((data_dir/v).relative_to(root)),'bytes':known[v]['bytes'],'sha256':known[v]['sha256']} for k,v in splits[split][field].items()}
    def endpoints(split): return {k:v['path'] for k,v in inputs(split,'endpoint_inputs').items()}
    if set(graph_rows)!=set(SEARCH_SEEDS+COMPARATIVE_SEEDS) or graph_custody['dataset_id']!=dataset: raise ValueError('Complete source-bound five-seed landmark graph custody required')
    runs = []; selections = {}; banks = []
    def add(identifier,kind,arm,split,seed,body,config,parents=()):
        entry = runtime+'/adapters/landmark_memory.py:make_adapter' if kind=='model_fit' else runtime+'/adapters/search.py:select_configuration' if kind=='selection' else runtime+'/adapters/landmark_memory_endpoints.py:'+{'ensemble_assembly':'export_bank','calibration':'scalar_calibration','evaluation':'evaluate_bank'}[kind]
        row = {'run_id':identifier,'kind':kind,'method':METHODS[arm],'split':split,'seed':seed,'body':body,'adapter':entry,'config':config,'parents':list(parents),
               'budget':{'unit':'optimizer_updates','policy':'full_budget','maximum':5000} if kind=='model_fit' else {},
               'selection':{'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False} if kind=='model_fit' else {}}
        runs.append(row); return identifier
    def common(split,arm,phase,seed,body=0):
        p = resource_for(profiles,dataset,'landmark-'+arm)
        members = 4 if arm in ('B4','T16','LM4','shared_factor4','original_boundary') else 1
        return {'architecture':'original_gnnm' if arm=='original_boundary' else 'landmark_memory','landmark_stage':'source' if arm in SOURCE_ARMS else 'continuation','landmark_arm':arm,
                'members':members,'physical_slow_bodies':1,'source_optimizer_count':1,
                'backbone':'PolyFormer_mono_K2','depth':2,'width':256,'classes':custody['model_graph']['classes'],
                'weight_decay':0.,'landmark_sources':source_bindings,'node_inputs':inputs(split,'node_inputs'),
                'landmark_graph':{k:graph_rows[seed][k] for k in ('capsule','payload_hash')} ,
                'fusion':'mean_logits' if arm=='original_boundary' else 'mean_probabilities','stage':phase,'is_original_gnnm':arm=='original_boundary',
                'paired_seed':seed,'initialization_seed':seed+1000003*body,
                'reservation_mib':p['reservation_mib'],'estimated_seconds':p['estimated_seconds_per_update']*5000,
                'recipe_scope':'pinned_factorized_PolyFormer_inducing_write_grouped_read_standardized_roles_optimizer_controller_and_CAL'}
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
    anchors = {arm:{} for arm in ('ordinary_m1','factor_m1')}
    for arm in anchors:
        for seed in SEARCH_SEEDS:
            anchors[arm][seed] = add('search-anchor-'+arm+'-s'+str(seed),'model_fit',arm,0,seed,0,
                {**common(0,arm,'search',seed),**DOMAIN[0],'selected_config_from':selections[arm]},[selections[arm]])
    for arm in MEMORY_ARMS+NATIVE_CONTROLS: search(arm,anchors['ordinary_m1' if arm=='continued_m1' else 'factor_m1'])
    for split in sorted(splits):
        for seed in COMPARATIVE_SEEDS:
            fits = {}
            for arm in SOURCE_ARMS:
                fits[arm] = []
                for body in range(4 if arm in ('ordinary_m1','factor_m1') else 1):
                    fits[arm].append(add(f'compare-sp{split}-{arm}-s{seed}-b{body}','model_fit',arm,split,seed,body,
                        {**common(split,arm,'comparative',seed,body),**DOMAIN[0],'selected_config_from':selections[arm]},[selections[arm]]))
            # Retain complete source quartets, but M1 bank serves the fixed body0.
            source_quartets = copy.deepcopy(fits)
            for arm in MEMORY_ARMS+NATIVE_CONTROLS:
                fits[arm] = []
                for body in range(4 if arm in ('S64','FS64')+NATIVE_CONTROLS else 1):
                    parent = source_quartets['ordinary_m1' if arm=='continued_m1' else 'factor_m1'][body]
                    fits[arm].append(add(f'compare-sp{split}-{arm}-s{seed}-b{body}','model_fit',arm,split,seed,body,
                        {**common(split,arm,'comparative',seed,body),**DOMAIN[0],'selected_config_from':selections[arm],'lifecycle':lifecycle(parent)},[selections[arm],parent]))
            for arm,member in {**I4,**MEMORY_I4}.items(): fits[arm] = fits[member][:]
            for arm in ('ordinary_m1','factor_m1','S64','FS64')+NATIVE_CONTROLS: fits[arm] = fits[arm][:1]
            for arm,parents in fits.items():
                member = {**I4,**MEMORY_I4}.get(arm,arm); fusion = common(split,member,'comparative',seed)['fusion']; p = resource_for(profiles,dataset,'landmark-'+member)
                prefix = f'bank-sp{split}-{arm}-s{seed}'; inp = endpoints(split)
                export = add(prefix,'ensemble_assembly',arm,split,seed,0,{'stage':'bank_export','fit_parents':parents,'endpoint_inputs':inp,'fusion':fusion,'reservation_mib':p['reservation_mib'],'estimated_seconds':p.get('estimated_export_seconds',1),'landmark_memory_I4':arm in MEMORY_I4},parents)
                cal = add(prefix+'-cal','calibration',arm,split,seed,0,{'stage':'calibration','export_parent':export,'endpoint_inputs':inp,'fusion':fusion,'placement':'after_serving_pool','steps':500,'learning_rate':.01,'reservation_mib':512,'estimated_seconds':p.get('estimated_scalar_cal_seconds',5)},[export])
                evaluation = add(prefix+'-evaluate','evaluation',arm,split,seed,0,{'stage':'evaluation','export_parent':export,'calibration_parent':cal,'endpoint_inputs':inp,'fusion':fusion,'reservation_mib':512,'estimated_seconds':p.get('estimated_evaluation_seconds',5)},[export,cal])
                banks.append({'arm':arm,'split':split,'seed':seed,'fit_parents':parents,'export':export,'calibration':cal,'evaluation':evaluation,'genuine_I4':arm in I4 or arm in MEMORY_I4,'landmark_memory_I4':arm in MEMORY_I4,'body0_reused':arm in I4 or arm in MEMORY_I4})
    roles = {role:{str(split):inputs(split,'node_inputs' if role in ('train','select') else 'endpoint_inputs')[role if role in ('train','select') else role+'_targets'] for split in splits} for role in ('train','select','cal','evaluate')}
    study = {'schema_version':1,'study_id':'landmark-memory-v1-'+dataset,'stage':'development','dataset':dataset,
        'hypothesis':'Pinned factorized PolyFormer landmark write/read source with B1/B4,T16,S64,FS64,LM4 and four own S64/FS64 controls; ordinary/factor/shared/original and full-native continuation references on complete new roles. Joint normalization/view-count/objective confounds remain.',
        'data':{'task':'node_classification','preprocessing':custody['request']['graph_policy'],'roles':roles,'prior_exposure':custody['request']['evaluation_exposure'],
                'evidence':{'custody_request_hash':custody['request_hash'],'model_graph':custody['model_graph'],'source_contract':CONTRACT,'source_exact_training_recipe':False}},
        'source':sources,'runs':runs,'comparisons':[{'references':[METHODS[a] for a in ('original_boundary','ordinary_m1','ordinary_i4','factor_m1','factor_i4','shared_factor4','continued_m1','continued_m1_i4','continued_f1','continued_f1_i4')],
            'candidates':[METHODS[a] for a in MEMORY_ARMS+tuple(MEMORY_I4)],'scope':'complete provider split roster and three fixed comparative seeds; paid ancestors charged once; stages separately reported'}],
        'release':{'policy':'whole_roster','primary_endpoint':'closed raw and scalar-calibrated banks','global_plan_gates_required':True},
        'resources':{'wall_seconds':max(p['wall_seconds'] for p in profiles.values()),'status':'PROVISIONAL_DISABLED_REQUIRES_FULL_INPUT_RESOURCE_QA_AND_GO'}}
    validator(study); fits = [r for r in runs if r['kind']=='model_fit']; sources_count = sum(r['config']['landmark_stage']=='source' for r in fits)
    counts = {'atomic_fit_groups':len(fits),'paid_optimizer_objects':len(fits),'physical_body_instances':sum(r['config']['physical_slow_bodies'] for r in fits),
        'model_fits':len(fits),'full_budget_optimizer_updates':5000*len(fits),'optimizer_starts_by_stage':dict(Counter(r['config']['stage'] for r in fits)),
        'source_acquisition_fits':sources_count,'source_acquisition_optimizer_updates':sources_count*5000,'continuations':len(fits)-sources_count,
        'continuation_optimizer_updates':(len(fits)-sources_count)*5000,'extensions':0,'search_selection_jobs':len(selections),'comparative_banks':len(banks),
        'bank_exports':len(banks),'scalar_CAL_fits':len(banks),'evaluation_jobs':len(banks),'genuine_I4_banks':sum(b['genuine_I4'] for b in banks),
        'landmark_memory_I4_banks':sum(b['landmark_memory_I4'] for b in banks),'reused_body0':sum(b['body0_reused'] for b in banks),'all_run_records':len(runs),'actual_optimizer_updates':0,'actual_predictive_scores':0}
    return study,counts,banks


def prepare_graphs(root,custodies,bindings,output):
    import torch
    if output.exists(): raise ValueError('Fresh immutable graph preparation required')
    output.mkdir(parents=True); datasets=[]
    for custody,data_dir,data_bindings in custodies:
        validate_custody(custody); graph_relative=custody['splits'][0]['node_inputs']['graph']
        graph=file_binding(root,data_dir/graph_relative); rows=[]; directory=output/custody['dataset_id']; directory.mkdir()
        for seed in SEARCH_SEEDS+COMPARATIVE_SEEDS:
            capsule=prepare_graph_capsule(torch,root,graph,bindings,seed); path=directory/('seed_'+str(seed)+'.pt'); torch.save(capsule,path)
            rows.append({'seed':seed,'capsule':file_binding(root,path),'payload_hash':capsule['payload_hash'],'nodes':capsule['nodes'],'features':capsule['features'],'anchors':64,'support_hash':capsule['geometry']['support_hash'],'anchor_hash':tensor_hash(capsule['geometry']['anchors']),'tokens':3})
        datasets.append({'dataset_id':custody['dataset_id'],'provider_custody':data_bindings[0],'graph':graph,'capsules':rows})
    result={'schema':'landmark-graph-custody-v1','datasets':datasets,'source_bindings':bindings,'source_contract':CONTRACT,
        'GRAPH_only':True,'TRAIN_SELECT_CAL_TEST_tensors_decoded':False,'models_constructed':0,'optimizers_created':0,'scientific_fits':0,'GO':False}
    (output/'CUSTODY.json').write_text(canonical(result)+'\n'); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','source-bindings','output'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--custody',nargs=2,required=True); parser.add_argument('--prepare-graphs',action='store_true')
    for name in ('runtime-package','resources','graphs-custody','source-commit'): parser.add_argument('--'+name)
    parser.add_argument('--fixture-receipts',nargs='+'); args=parser.parse_args(); verify_allocation()
    root=Path(args.project_root).resolve()
    def path(value):
        v=Path(value); return confined(root,str(v.resolve().relative_to(root)) if v.is_absolute() else str(v))
    output,bindings_path=map(path,(args.output,args.source_bindings)); bindings=json.loads(bindings_path.read_text())
    if set(bindings)!=set(PINS) or any(bindings[k]['sha256']!=v for k,v in PINS.items()): raise ValueError('Exact landmark/native/factor/original source pins required')
    for row in bindings.values(): bound(root,row)
    custodies=[bind_custody(root,path(v)) for v in args.custody]
    if {v[0]['dataset_id'] for v in custodies}!={'wikics','amazon-photo'}: raise ValueError('Both complete new providers required')
    if args.prepare_graphs:
        result=prepare_graphs(root,custodies,bindings,output); print(canonical({'graph_capsules':sum(len(v['capsules']) for v in result['datasets']),'fits':0,'GO':False})); return
    if not all((args.runtime_package,args.resources,args.graphs_custody,args.fixture_receipts)): raise ValueError('Frozen runtime/landmark graph custody/resources/CPU QA receipts required')
    runtime,resource_path,graphs_path=map(path,(args.runtime_package,args.resources,args.graphs_custody))
    if output.exists() or runtime.is_relative_to(root/'training_pipeline_v2_20261012'): raise ValueError('Fresh disabled packet and frozen external runtime required')
    resources=json.loads(resource_path.read_text()); capacities=resources['worker_capacities_cuda_bytes']
    if set(capacities)!={'allocation/gpu0'}: raise ValueError('Allocation-only disabled packet')
    graphs=json.loads(graphs_path.read_text()); graphs_by_dataset={v['dataset_id']:v for v in graphs['datasets']}
    if graphs['schema']!='landmark-graph-custody-v1' or graphs['source_bindings']!=bindings or graphs['source_contract']!=CONTRACT or graphs['GRAPH_only'] is not True or graphs['TRAIN_SELECT_CAL_TEST_tensors_decoded'] is not False: raise ValueError('Actual GRAPH-only immutable landmark graph custody required')
    files=[file_binding(root,runtime/v) for v in list(RUNTIME_FILES)+['adapters/retained_pt_routes.py','adapters/polyformer.py','adapters/polyformer_boundary.py','adapters/landmark_memory.py','adapters/landmark_memory_endpoints.py','compile_landmark_memory_matrix.py','compile_native_matrix.py','prepare_node_data.py']]
    files+=list(bindings.values())+[file_binding(root,bindings_path),file_binding(root,resource_path),file_binding(root,graphs_path)]
    files+=[file_binding(root,path(v)) for v in args.fixture_receipts]
    for c,d,b in custodies:
        files+=b; tc=graphs_by_dataset[c['dataset_id']]
        if tc['provider_custody']!=b[0] or tc['graph']!=file_binding(root,d/c['splits'][0]['node_inputs']['graph']): raise ValueError('Tour/provider graph receipt differs')
        for row in tc['capsules']:
            if file_binding(root,path(row['capsule']['path']))!=row['capsule']: raise ValueError('Tour capsule bytes changed')
            files.append(row['capsule'])
    sources={'git_commit':args.source_commit or subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'files':list({r['path']:r for r in files}.values()),'working_tree_fingerprints_authoritative':True,'frozen_runtime_package':str(runtime.relative_to(root)),'official_author_protocol_replication':False}
    schema_spec=importlib.util.spec_from_file_location('landmark_frozen_schema',runtime/'spec.py'); schema=importlib.util.module_from_spec(schema_spec); schema_spec.loader.exec_module(schema)
    output.mkdir(parents=True); (output/'studies').mkdir(); prefix=str(output.relative_to(root)); studies=[]; groups=[]; total=Counter(); banks=[]
    for c,d,b in custodies:
        study,counts,cohort=make_study(root,c,d,graphs_by_dataset[c['dataset_id']],sources,bindings,resources['profiles'],str(runtime.relative_to(root)),schema.validate)
        for k,v in counts.items():
            if type(v)==int: total[k]+=v
        for row in study['runs']:
            if row['config']['reservation_mib']*1024**2>capacities['allocation/gpu0']: raise ValueError('Provisional reservation exceeds capacity')
        relative=prefix+'/studies/'+study['study_id']+'.json.gz'; (root/relative).write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0)); studies.append({'path':relative,'spec_hash':fingerprint(study)})
        groups.append({'id':study['study_id'],'gpu_assignment':'allocation/gpu0','allowed_workers':['allocation/gpu0'],'maximum_cuda_bytes':max(r['config']['reservation_mib'] for r in study['runs'])*1024**2,
            'estimated_seconds':sum(r['config']['estimated_seconds'] for r in study['runs']),'run_keys':[study['study_id']+'/'+r['run_id'] for r in study['runs']]})
        banks += [{'dataset':c['dataset_id'],**v} for v in cohort]
    plan={'schema_version':1,'slots_per_gpu':2,'torch_threads_per_fit':2,'output_root':prefix+'/execution','runtime_launcher':str((runtime/'launcher.py').relative_to(root)),'runtime_package':str(runtime.relative_to(root)),
        'worker_capacities_cuda_bytes':capacities,'studies':studies,'groups':groups,'global_gates':{},'barriers':{},'scientific_launch_enabled':False,'human_GO_required':True,
        'host_bindings':{'allocation':{'status':'verified_at_compilation','hostname':'anogena-2-0','gpu_uuids':['GPU-44039938-fd82-41d2-fefd-de71514e2fac']}}}
    for name,stages,required in [('whole_search',['search','search_selection'],['comparative']),('whole_comparative',['comparative'],['bank_export','calibration','evaluation']),('whole_calibration',['calibration'],['evaluation'])]:
        plan['global_gates'][name]={'path':prefix+'/gate_receipts/'+name+'.json','source_stages':stages,'required_for_stages':required,'required_hosts':['allocation']}
    summary={'schema':'landmark-disabled-summary-v1','status':'DISABLED_PROVISIONAL_RESOURCE_MASTER_AND_GO_PENDING','plan_hash':fingerprint(plan),'counts':dict(total),'banks':banks,'source_contract':CONTRACT,'domain':DOMAIN,
        'search_seeds':SEARCH_SEEDS,'comparative_seeds':COMPARATIVE_SEEDS,'graph_custody':file_binding(root,graphs_path),'scientific_launch_enabled':False,'human_GO_required':True,'scientific_fits':0,'GO_created':False,'full_249_coverage':False,
        'per_landmark_body_total_paid_updates':10000,'source_ancestry_receipts_and_H_U_deferred_until_paid_source_fit':True}
    (output/'PLAN.json').write_text(canonical(plan)+'\n'); (output/'SUMMARY.json').write_text(canonical(summary)+'\n'); print(canonical({'counts':dict(total),'plan_hash':summary['plan_hash'],'science':False,'GO':False}))


if __name__=='__main__': main()
