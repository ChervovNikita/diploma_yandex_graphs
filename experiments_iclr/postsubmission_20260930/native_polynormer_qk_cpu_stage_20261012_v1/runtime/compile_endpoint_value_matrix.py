"""Compile two complete new-data endpoint VALUE source studies, disabled."""
import argparse
from collections import Counter
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess

from adapters.endpoint_value import ARMS,I4,SOURCES,CONTRACT,bound
from compile_native_matrix import bind_custody,file_binding,RUNTIME_FILES
from prepare_node_data import verify_allocation
from spec import canonical,confined,fingerprint,validate

DOMAIN=[{'learning_rate':lr,'dropout':p} for lr in (.0005,.001,.002) for p in (.1,.2,.4)]
SEARCH_SEEDS=(76101,76103)
COMPARATIVE_SEEDS=(76201,76203,76207)
METHODS={'BASE':'endpoint_conditioned_sage_base','E4':'endpoint_conditioned_sage_e4','N4':'endpoint_conditioned_sage_n4',
 'equipped_M1':'endpoint_conditioned_sage_equipped_m1','equipped_I4':'endpoint_conditioned_sage_equipped_i4',
 'ordinary_M1':'ordinary_native_single','ordinary_I4':'ordinary_genuine_independent4','original_gnnm':'original_boundary_GNNM'}

def make_study(root,custody,data_dir,source,sources,runtime):
    dataset=custody['dataset_id'];expected={'wikics':20,'amazon-photo':10}
    if dataset not in expected or sorted(r['split_id'] for r in custody['splits'])!=list(range(expected[dataset])):raise ValueError('Complete existing WikiCS/Photo split roster required')
    split_rows={r['split_id']:r for r in custody['splits']};files={r['path']:r for r in custody['files']}
    def binding(relative):
        row=files[relative];return {'path':str((data_dir/relative).relative_to(root)),'bytes':row['bytes'],'sha256':row['sha256']}
    def inputs(split,field):return {k:binding(v) for k,v in split_rows[split][field].items()}
    runs=[];selections={};banks=[]
    def common(arm,split,stage):
        members=4 if arm in ('BASE','E4','N4','original_gnnm') else 1
        architecture='original_gnnm' if arm=='original_gnnm' else 'single' if arm=='ordinary_M1' else 'factorized_single' if arm=='equipped_M1' else 'internal_be'
        return {'endpoint_arm':arm,'architecture':architecture,'members':members,'physical_slow_bodies':1,'source_optimizer_count':1,'backbone':'SAGE','depth':2,'width':128,
          'classes':custody['model_graph']['classes'],'weight_decay':0.,'endpoint_sources':sources,'node_inputs':inputs(split,'node_inputs'),
          'is_original_gnnm':arm=='original_gnnm','fusion':'mean_logits' if arm=='original_gnnm' else 'mean_probabilities',
          'stage':stage,'source_contract':CONTRACT,'reservation_mib':32768,'estimated_seconds':5000000,
          'resource_evidence':'unmeasured explicit scheduling placeholder; no CUDA admission or ETA','recipe_scope':'exact source operator/controller with standardized new roles/search/common CAL'}
    def add(identifier,kind,arm,split,seed,body,config,parents=()):
        adapter=runtime+'/adapters/'+('endpoint_value.py:make_adapter' if kind=='model_fit' else 'search.py:select_configuration' if kind=='selection' else
          'node_endpoints.py:'+{'ensemble_assembly':'export_bank','calibration':'scalar_calibration','evaluation':'evaluate_bank'}[kind])
        row={'run_id':identifier,'kind':kind,'method':METHODS[arm],'split':split,'seed':seed,'body':body,'adapter':adapter,'config':config,'parents':list(parents),
          'budget':{'unit':'optimizer_updates','policy':'full_budget','maximum':5000} if kind=='model_fit' else {},
          'selection':{'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,'patience_steps':5000} if kind=='model_fit' else {}}
        runs.append(row);return row
    for arm in ARMS:
        candidates=[]
        for point in DOMAIN:
            key=fingerprint(point)[:20];ids=[]
            for seed in SEARCH_SEEDS:
                ids.append(add('search-'+arm+'-'+key+'-s'+str(seed),'model_fit',arm,0,seed,0,{**common(arm,0,'search'),**point})['run_id'])
            candidates.append({'config_id':key,'parameters':point,'run_ids':ids})
        selections[arm]=add('select-'+arm,'selection',arm,0,0,0,{'stage':'search_selection','configurations':candidates,'ranking_metric':'accuracy',
          'ranking_direction':'max','reservation_mib':1,'estimated_seconds':1},[r for c in candidates for r in c['run_ids']])['run_id']
    for split in sorted(split_rows):
        for seed in COMPARATIVE_SEEDS:
            fits={}
            for arm in ARMS:
                row=add('compare-sp'+str(split)+'-'+arm+'-s'+str(seed),'model_fit',arm,split,seed,0,
                  {**common(arm,split,'comparative'),**DOMAIN[0],'selected_config_from':selections[arm]},[selections[arm]])
                fits[arm]=[row['run_id']]
            for arm,member in I4.items():
                first=next(r for r in runs if r['run_id']==fits[member][0]);fits[arm]=[first['run_id']]
                for body in (1,2,3):fits[arm].append(add(first['run_id']+'-i4-b'+str(body),'model_fit',member,split,seed,body,copy.deepcopy(first['config']),first['parents'])['run_id'])
            for arm,parents in fits.items():
                single=I4.get(arm,arm);fusion=common(single,split,'comparative')['fusion'];prefix='bank-sp'+str(split)+'-'+arm+'-s'+str(seed)
                endpoints={k:v['path'] for k,v in inputs(split,'endpoint_inputs').items()}
                export=add(prefix,'ensemble_assembly',arm,split,seed,0,{'stage':'bank_export','fit_parents':parents,'endpoint_inputs':endpoints,
                  'fusion':fusion,'reservation_mib':32768,'estimated_seconds':10000},parents)['run_id']
                cal=add(prefix+'-cal','calibration',arm,split,seed,0,{'stage':'calibration','export_parent':export,'endpoint_inputs':endpoints,'fusion':fusion,
                  'placement':'after_serving_pool','steps':500,'learning_rate':.01,'reservation_mib':512,'estimated_seconds':1000},[export])['run_id']
                evaluation=add(prefix+'-evaluate','evaluation',arm,split,seed,0,{'stage':'evaluation','export_parent':export,'calibration_parent':cal,
                  'endpoint_inputs':endpoints,'fusion':fusion,'reservation_mib':512,'estimated_seconds':1000},[export,cal])['run_id']
                banks.append({'arm':arm,'split':split,'seed':seed,'fit_parents':parents,'export':export,'calibration':cal,'evaluation':evaluation,
                  'genuine_I4':arm in I4,'body0_reused':arm in I4,'served_views':4 if arm in I4 else common(single,split,'comparative')['members']})
    roles={role:{str(split):inputs(split,'node_inputs' if role in ('train','select') else 'endpoint_inputs')[role if role in ('train','select') else role+'_targets'] for split in split_rows}
      for role in ('train','select','cal','evaluate')}
    study={'schema_version':1,'study_id':'endpoint-value-standardized-v3-'+dataset,'stage':'development','dataset':dataset,
      'hypothesis':'Exact factual endpoint VALUE with same-capacity node conditioning, source-equipped own M1/I4 and original GNNM/native references on new-role providers.',
      'data':{'task':'node_classification','preprocessing':custody['request']['graph_policy'],'roles':roles,'prior_exposure':custody['request']['evaluation_exposure'],
        'evidence':{'custody_request_hash':custody['request_hash'],'model_graph':custody['model_graph'],'source_exact_training_recipe':False}},
      'source':source,'runs':runs,'comparisons':[{'references':[METHODS[a] for a in ('BASE','N4','equipped_M1','equipped_I4','ordinary_M1','ordinary_I4','original_gnnm')],
        'candidates':[METHODS['E4']],'scope':'all prepared splits and fixed comparative seeds; no independent-graph inference or acceptance promise'}],
      'release':{'policy':'whole_roster','primary_endpoint':'all raw and common scalar-calibrated banks','global_plan_gates_required':True},
      'resources':{'wall_seconds':1000000000,'status':'PROVISIONAL_UNMEASURED_DISABLED'}}
    validate(study)
    fits=[r for r in runs if r['kind']=='model_fit'];starts=Counter(r['config']['stage'] for r in fits)
    counts={'optimizer_starts_by_stage':dict(starts),'atomic_fit_groups':len(fits),'paid_optimizer_objects':len(fits),'physical_body_instances':len(fits),
      'maximum_optimizer_updates':len(fits)*5000,'full_budget_optimizer_updates':len(fits)*5000,'search_selection_jobs':len(selections),
      'comparative_banks':len(banks),'bank_exports':len(banks),'scalar_CAL_fits':len(banks),'evaluation_jobs':len(banks),
      'genuine_I4_banks':sum(b['genuine_I4'] for b in banks),'reused_I4_body0':sum(b['body0_reused'] for b in banks),
      'all_run_records':len(runs),'actual_optimizer_updates':0,'actual_predictive_scores':0}
    return study,counts,banks

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('project-root','custody','source-bindings','runtime-package','output'):p.add_argument('--'+field,required=True,nargs='+' if field=='custody' else None)
    p.add_argument('--fixture-receipts',nargs='+',required=True);args=p.parse_args();verify_allocation()
    root=Path(args.project_root).resolve()
    def path(value):
        p=Path(value);return confined(root,str(p.resolve().relative_to(root)) if p.is_absolute() else str(p))
    runtime=path(args.runtime_package);output=path(args.output);bindings=json.loads(path(args.source_bindings).read_text())
    if runtime.is_relative_to(root/'training_pipeline_v2_20261012') or output.exists():raise ValueError('Frozen runtime and fresh disabled output required')
    for key,(relative,sha) in SOURCES.items():
        if bindings[key]['sha256']!=sha:raise ValueError('Source pin differs')
        bound(root,bindings[key])
    owned=list(RUNTIME_FILES)+['adapters/endpoint_value.py','compile_endpoint_value_matrix.py','compile_native_matrix.py','prepare_node_data.py']
    source_files=[file_binding(root,runtime/name) for name in owned]+list(bindings.values())+[file_binding(root,path(args.source_bindings))]
    source_files += [file_binding(root,path(relative)) for relative in args.fixture_receipts]
    source_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    studies=[];all_counts=Counter();stage_counts=Counter();bank_rows=[];prefix=str(output.relative_to(root));output.mkdir();(output/'studies').mkdir()
    for custody_path in map(path,args.custody):
        custody,data_dir,data_bindings=bind_custody(root,custody_path)
        source={'git_commit':source_commit,'files':source_files+data_bindings,'working_tree_fingerprints_authoritative':True,
          'frozen_runtime_package':str(runtime.relative_to(root)),'official_author_protocol_replication':False}
        study,counts,banks=make_study(root,custody,data_dir,source,bindings,str(runtime.relative_to(root)))
        spec_path=prefix+'/studies/'+study['study_id']+'.json.gz';(root/spec_path).write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0))
        studies.append({'path':spec_path,'spec_hash':fingerprint(study)});bank_rows.extend({'dataset':study['dataset'],**r} for r in banks)
        stage_counts.update(counts['optimizer_starts_by_stage']);all_counts.update({k:v for k,v in counts.items() if type(v)==int})
    if len(studies)!=2:raise ValueError('Exact two-provider packet required')
    all_counts['optimizer_starts_by_stage']=dict(stage_counts)
    groups=[]
    for row in studies:
        study=json.loads(gzip.decompress((root/row['path']).read_bytes()));groups.append({'id':study['study_id'],'gpu_assignment':'allocation/gpu0',
          'allowed_workers':['allocation/gpu0'],'maximum_cuda_bytes':32768*1024**2,'estimated_seconds':sum(r['config']['estimated_seconds'] for r in study['runs']),
          'run_keys':[study['study_id']+'/'+r['run_id'] for r in study['runs']]})
    plan={'schema_version':1,'slots_per_gpu':2,'torch_threads_per_fit':2,'output_root':prefix+'/execution','runtime_launcher':str((runtime/'launcher.py').relative_to(root)),
      'runtime_package':str(runtime.relative_to(root)),'worker_capacities_cuda_bytes':{'allocation/gpu0':78*1024**3},'studies':studies,'groups':groups,
      'global_gates':{},'barriers':{},'scientific_launch_enabled':False,'human_GO_required':True,
      'host_bindings':{'allocation':{'status':'verified_at_compilation','hostname':'anogena-2-0','gpu_uuids':['GPU-44039938-fd82-41d2-fefd-de71514e2fac']}}}
    for name,source_stages,target in [('whole_search',['search','search_selection'],['comparative']),('whole_comparative',['comparative'],['bank_export','calibration','evaluation']),('whole_calibration',['calibration'],['evaluation'])]:
        plan['global_gates'][name]={'path':prefix+'/gate_receipts/'+name+'.json','source_stages':source_stages,'required_for_stages':target,'required_hosts':['allocation']}
    summary={'schema':'endpoint-value-disabled-source-summary-v1','plan_hash':fingerprint(plan),'counts':dict(all_counts),'banks':bank_rows,'domain':DOMAIN,
      'source_contract':CONTRACT,'newly_scoped_canonical_methods':4,'BASE_duplicate_mechanism_alias':'allmap_factor_shared4; recipe equality not asserted',
      'genuine_I4_body0_reused':True,'scientific_training_started':False,'GO_created':False,'source_exact_recipe':False,
      'resource_scope':'provisional_unmeasured','status':'DISABLED_NEW_SOURCE_SCOPES_RESOURCE_PROTOCOL_MASTER_AND_GO_PENDING'}
    (output/'PLAN.json').write_text(canonical(plan)+'\n');(output/'SUMMARY.json').write_text(canonical(summary)+'\n')
    print(canonical({'counts':summary['counts'],'scientific_fits':0,'GO':False}))

if __name__=='__main__':main()
