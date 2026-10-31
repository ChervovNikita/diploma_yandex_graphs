"""Two disabled new-data paid-native/capture/label studies; metadata only."""
import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path
import subprocess

from adapters.label_stage import ARMS,SOURCE_PINS
from adapters.native_polynormer_qk import SOURCES,CONTRACT,receipt,bound
from compile_native_matrix import bind_custody,file_binding
from prepare_node_data import verify_allocation
from spec import canonical,fingerprint,validate,confined
from compile_master_plan import typed_counts

SEARCH=(81101,81103);COMPARE=(81201,81203,81207)
NATIVE_DOMAIN=[{'learning_rate':lr,'in_dropout':a,'dropout':b,'global_dropout':g}
               for lr in (.0005,.001,.002) for a,b,g in ((.5,.5,.5),(.5,.3,.5),(.5,.5,.3))]
LABEL_DOMAIN=[{'label_learning_rate':lr} for lr in (.0005,.001,.002)]
NATIVE_ARMS={'native_m1':'bare_native_single','original_gnnm':'original_gnnm'}
CONTROL_METHODS=('native_m1','native_independent4','original_gnnm')
PUBLIC_FACTOR_MANIFEST=('portable_internal_be_public_interface_20261007_v2/MANIFEST.json','190940ca9f8141ac45f739d064cb1ef76aaf1965ba8c91adb544ad8e38fef724')


def make_study(root,custody,data_dir,source,native_sources,label_sources,runtime):
    dataset=custody['dataset_id'];expected={'wikics':20,'amazon-photo':10}
    if dataset not in expected or sorted(r['split_id'] for r in custody['splits'])!=list(range(expected[dataset])):
        raise ValueError('All WikiCS20/Photo10 masks required')
    splits={r['split_id']:r for r in custody['splits']};files={r['path']:r for r in custody['files']}
    def binding(relative):
        row=files[relative];return {'path':str((data_dir/relative).relative_to(root)),'bytes':row['bytes'],'sha256':row['sha256']}
    def inputs(split,field):return {k:binding(v) for k,v in splits[split][field].items()}
    runs=[];banks=[]
    def native_config(method,split,stage):
        arm=NATIVE_ARMS[method];original=method=='original_gnnm';members=4 if original else 1
        return {'polynormer_qk_arm':arm,'polynormer_qk_operator':'original_boundary' if original else 'bare',
                'source_ownership_kind':'boundary4' if original else 'single','architecture':'original_gnnm' if original else 'single',
                'backbone':'native_Polynormer','width':512,'depth':7,'global_layers':2,'heads':1,'local_transition_updates':100,
                'members':members,'physical_slow_bodies':1,'source_optimizer_count':1,'source_optimizer_calls_per_cycle':1,
                'source_TRAIN_forwards_per_update':2*members,'source_TRAIN_backwards_per_update':2*members,
                'is_original_gnnm':original,'classes':custody['model_graph']['classes'],'weight_decay':0.,
                'polynormer_qk_sources':native_sources,'node_inputs':inputs(split,'node_inputs'),
                'fusion':'mean_logits' if original else 'mean_probabilities','stage':stage,'source_contract':CONTRACT,
                'paid_native_acquisition':True,'historical_native_horizon':1100,'prospective_native_horizon':5000,
                'historical_authorization':'unresolved; no old outcome/checkpoint/cost inherited',
                'reservation_mib':32768,'estimated_seconds':5000000,'resource_evidence':'unmeasured disabled placeholder; not admission or ETA'}
    def label_config(arm,split,stage,native_fit,capture):
        spec=ARMS[arm]
        return {'paid_label_arm':arm,'architecture':'paid_native_label_stage','backbone':'native_Polynormer_frozen_own_selected_capture',
                'members':spec[3],'physical_slow_bodies':0,'source_optimizer_count':spec[6],'label_cycles':5000,
                'classes':custody['model_graph']['classes'],'node_inputs':inputs(split,'node_inputs'),'label_sources':label_sources,
                'fusion':'mean_probabilities','stage':stage,'capture_parent':capture,
                'lifecycle':{'schema':'paid-native-capture-label-stage-v1','parent_fit_ids':[native_fit],'new_label_optimizer':True,
                             'native_initialization':'own_selected_native_frozen_capture','label_initialization':'fresh_source_maps_no_native_optimizer_reuse'},
                'historical_native_horizon':1100,'historical_label_horizon':1100,'historical_gate_new_native_updates':0,
                'prospective_native_horizon':5000,'prospective_label_cycles':5000,'historical_authorization':'unresolved; no historical results or ancestors reused',
                'native_capture_reuse':'one exact paid native fit/capture reused across all9 arm fits in a declared seed/mask cohort',
                'native_capture_is_not_cross_fitted':True,'TRAIN_query_fraction':.5,'TRAIN_value_scale':'(A-1)/(A-Q) exactly once',
                'label_width':64,'attention_width':64,'native_probability_weight':.2,'label_probability_weight':.8,
                'shared_B_untied_correctors_backward_CE_scale':.25,'reservation_mib':8192,'estimated_seconds':5000000,
                'resource_evidence':'unmeasured disabled placeholder; not resource admission or ETA'}
    def add(identifier,kind,method,split,seed,body,config,parents=()):
        if kind=='model_fit':
            config['initialization_seed']=seed+1009*body
            adapter='label_stage.py:make_adapter' if method in ARMS else 'native_polynormer_qk.py:make_adapter'
        elif kind=='selection':adapter='label_stage_endpoints.py:capture_native' if config['stage']=='native_capture' else 'search.py:select_configuration'
        else:
            prefix='label_stage_endpoints.py:' if method in ARMS else 'node_endpoints.py:'
            adapter=prefix+{'ensemble_assembly':'export_bank','calibration':'scalar_calibration','evaluation':'evaluate_bank'}[kind]
        row={'run_id':identifier,'kind':kind,'method':method,'split':split,'seed':seed,'body':body,'adapter':runtime+'/adapters/'+adapter,
             'config':config,'parents':list(parents),'budget':{'unit':'epochs' if method in ARMS else 'optimizer_updates','policy':'full_budget','maximum':5000} if kind=='model_fit' else {},
             'selection':{'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,'patience_steps':5000} if kind=='model_fit' else {}}
        runs.append(row);return row
    def selector(method,points,fit_builder):
        configs=[]
        for point in points:
            key=fingerprint(point)[:20];ids=[fit_builder(point,key,seed)['run_id'] for seed in SEARCH]
            configs.append({'config_id':key,'parameters':point,'run_ids':ids})
        return add('select-'+method,'selection',method,0,0,0,
                   {'stage':'search_selection','configurations':configs,'ranking_metric':'accuracy','ranking_direction':'max','reservation_mib':1,'estimated_seconds':1},
                   [i for c in configs for i in c['run_ids']])['run_id']
    native_select={}
    for method in NATIVE_ARMS:
        native_select[method]=selector(method,NATIVE_DOMAIN,lambda point,key,seed,m=method:add('search-'+m+'-'+key+'-s'+str(seed),'model_fit',m,0,seed,0,{**native_config(m,0,'search'),**point}))
    def capture(native_fit,split,seed):
        identifier='capture-'+native_fit
        add(identifier,'selection','paid_native_capture',split,seed,0,
            {'stage':'native_capture','native_capture':True,'native_fit_parent':native_fit,'capture_source':label_sources['capture'],
             'nodes':custody['model_graph']['nodes'],'classes':custody['model_graph']['classes'],'reservation_mib':32768,'estimated_seconds':10000},[native_fit])
        return identifier
    search_sources={}
    for seed in SEARCH:
        row=add('label-source-search-s'+str(seed),'model_fit','native_m1',0,seed,0,
                {**native_config('native_m1',0,'search'),**NATIVE_DOMAIN[0],'selected_config_from':native_select['native_m1']},[native_select['native_m1']])
        search_sources[seed]=(row['run_id'],capture(row['run_id'],0,seed))
    label_select={}
    for arm in ARMS:
        def builder(point,key,seed,a=arm):
            native_fit,cap=search_sources[seed]
            return add('search-'+a+'-'+key+'-s'+str(seed),'model_fit',a,0,seed,0,{**label_config(a,0,'search',native_fit,cap),**point},[native_fit,cap])
        label_select[arm]=selector(arm,LABEL_DOMAIN,builder)
    for split in splits:
        for seed in COMPARE:
            base={}
            for method in NATIVE_ARMS:
                base[method]=[add('compare-sp'+str(split)+'-'+method+'-s'+str(seed),'model_fit',method,split,seed,0,
                     {**native_config(method,split,'comparative'),**NATIVE_DOMAIN[0],'selected_config_from':native_select[method]},[native_select[method]])['run_id']]
            base['native_independent4']=[base['native_m1'][0]]
            first=next(r for r in runs if r['run_id']==base['native_m1'][0])
            for body in (1,2,3):
                base['native_independent4'].append(add(first['run_id']+'-I4-b'+str(body),'model_fit','native_m1',split,seed,body,copy.deepcopy(first['config']),first['parents'])['run_id'])
            cap=capture(base['native_m1'][0],split,seed)
            for arm in ARMS:
                row=add('compare-sp'+str(split)+'-'+arm+'-s'+str(seed),'model_fit',arm,split,seed,0,
                        {**label_config(arm,split,'comparative',base['native_m1'][0],cap),**LABEL_DOMAIN[0],'selected_config_from':label_select[arm]},
                        [base['native_m1'][0],cap,label_select[arm]])
                base[arm]=[row['run_id']]
            for method,parents in base.items():
                prefix='bank-sp'+str(split)+'-'+method+'-s'+str(seed);endpoints={k:v['path'] for k,v in inputs(split,'endpoint_inputs').items()}
                fusion='mean_logits' if method=='original_gnnm' else 'mean_probabilities'
                export=add(prefix,'ensemble_assembly',method,split,seed,0,
                    {'stage':'bank_export','fit_parents':parents,'endpoint_inputs':endpoints,'fusion':fusion,'reservation_mib':32768,'estimated_seconds':10000},parents)['run_id']
                cal=add(prefix+'-cal','calibration',method,split,seed,0,
                    {'stage':'calibration','export_parent':export,'endpoint_inputs':endpoints,'fusion':fusion,'placement':'after_serving_pool','steps':500,'learning_rate':.01,'reservation_mib':512,'estimated_seconds':1000},[export])['run_id']
                ev=add(prefix+'-evaluate','evaluation',method,split,seed,0,
                    {'stage':'evaluation','export_parent':export,'calibration_parent':cal,'endpoint_inputs':endpoints,'fusion':fusion,'reservation_mib':512,'estimated_seconds':1000},[export,cal])['run_id']
                banks.append({'method':method,'split':split,'seed':seed,'fit_parents':parents,'export':export,'calibration':cal,'evaluation':ev})
    roles={role:{str(split):inputs(split,'node_inputs' if role in ('train','select') else 'endpoint_inputs')[role if role in ('train','select') else role+'_targets'] for split in splits} for role in ('train','select','cal','evaluate')}
    study={'schema_version':1,'study_id':'paid-native-label-stage-newdata-v1-'+dataset,'stage':'development','dataset':dataset,
           'hypothesis':'Exact staged posterior and query/value-gated label maps on one new paid native Polynormer capture, versus capable own-selected native M1/I4/original RSB and source map ownership controls.',
           'data':{'task':'node_classification','preprocessing':custody['request']['graph_policy'],'roles':roles,'prior_exposure':custody['request']['evaluation_exposure'],
                   'evidence':{'custody_request_hash':custody['request_hash'],'model_graph':custody['model_graph'],'historical_results_inherited':False,'native_capture_cross_fitted':False}},
           'source':source,'runs':runs,'comparisons':[
               {'references':list(CONTROL_METHODS),'candidates':list(ARMS),'scope':'all20WikiCS/all10Photo masks;2search/3comparative fresh paired seeds;explicit paid source reuse;whole-bank source controls and own native I4 distinct'},
               {'references':['staged_posterior_s_joint4head','staged_posterior_u4_sharedb'],'candidates':['staged_posterior_c4'],'scope':'source whole-bank ownership/control contrasts on the exact same paid capture'},
               *[{'references':[plain],'candidates':[gate],'scope':'matched source architecture with receiver gate; separate own fixed-domain label LR selection retained'}
                 for plain,gate in (('staged_posterior_c4','query_value_gate_c4_gate'),('staged_posterior_s_joint4head','query_value_gate_s_joint4head_gate'),
                                    ('staged_posterior_u4_sharedb','query_value_gate_u4_sameb_full_untied_gate'),('staged_posterior_s_one_path','query_value_gate_s_one_path_gate'))],
               {'references':['query_value_gate_c4_gate'],'candidates':['query_value_gate_c4_gate_identity_erased'],'scope':'permanent value-class erasure control; original TRAIN targets are retained'}],
           'release':{'policy':'whole_roster','primary_endpoint':'raw/common scalar CAL all banks','global_plan_gates_required':True},
           'resources':{'wall_seconds':1000000000,'status':'DISABLED_SOURCE_STATE_QA_AND_RESOURCE_PENDING','whole_roster_no_free_teachers':True}}
    validate(study);return study,banks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','runtime-package','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--custody',nargs='+',required=True);args=p.parse_args();verify_allocation();root=Path(args.project_root).resolve()
    runtime=confined(root,args.runtime_package);output=confined(root,args.output)
    if output.exists():raise ValueError('Fresh disabled packet required')
    native_sources={k:receipt(root,root/path) for k,(path,sha) in SOURCES.items()};label_sources={k:receipt(root,root/path) for k,(path,sha) in SOURCE_PINS.items()}
    if any(native_sources[k]['sha256']!=v[1] for k,v in SOURCES.items()) or any(label_sources[k]['sha256']!=v[1] for k,v in SOURCE_PINS.items()):raise ValueError('Original native/label source changed')
    factor_manifest=receipt(root,root/PUBLIC_FACTOR_MANIFEST[0])
    if factor_manifest['sha256']!=PUBLIC_FACTOR_MANIFEST[1]:raise ValueError('Original public factor manifest changed')
    runtime_files=[file_binding(root,f) for f in sorted(runtime.rglob('*')) if f.is_file()]
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();output.mkdir();(output/'studies').mkdir()
    studies=[];all_studies=[];groups=[]
    for value in args.custody:
        custody,data_dir,files=bind_custody(root,confined(root,value))
        source={'git_commit':commit,'files':runtime_files+list(native_sources.values())+list(label_sources.values())+[factor_manifest]+files,
                'working_tree_fingerprints_authoritative':True,'frozen_runtime_package':args.runtime_package,
                'historical_authorization':'unresolved and excluded; this is independent new-data source acquisition'}
        study,banks=make_study(root,custody,data_dir,source,native_sources,label_sources,args.runtime_package);all_studies.append(study)
        path=output/'studies'/(study['study_id']+'.json.gz');path.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0));studies.append({'path':str(path.relative_to(root)),'spec_hash':fingerprint(study)})
        groups.append({'id':study['study_id'],'run_keys':[study['study_id']+'/'+r['run_id'] for r in study['runs']],
                       'allowed_workers':['allocation/gpu0'],'gpu_assignment':'allocation/gpu0','estimated_seconds':sum(r['config']['estimated_seconds'] for r in study['runs']),'maximum_cuda_bytes':32768*1024**2})
    if len(all_studies)!=2:raise ValueError('Both bridged new providers required')
    counts=typed_counts(all_studies)
    plan={'schema_version':1,'slots_per_gpu':2,'studies':studies,'groups':groups,'barriers':{},'scientific_launch_enabled':False,'human_GO_required':True,'resources_are_provisional':True}
    summary={'schema':'paid-native-label-stage-disabled-packet-v1','plan_hash':fingerprint(plan),'counts':counts,'status':'DISABLED_SOURCE_STATE_QA_AND_RESOURCE_PENDING',
             'GO_receipt_created':False,'benchmark_training_started':False,'historical_native_and_label_horizons':1100,'prospective_native_and_label_full_budgets':5000,
             'historical_gate_extra_native_updates':0,'paid_native_acquisition_fit_records':sum(r['kind']=='model_fit' and r['method'] in NATIVE_ARMS for s in all_studies for r in s['runs']),
             'paid_label_fit_records':sum(r['kind']=='model_fit' and r['method'] in ARMS for s in all_studies for r in s['runs']),
             'shared_native_capture_reuse_is_declared_and_paid':True,'new_data_Photo_class_binding_is_explicit':True,'actual_fits':0,'actual_CUDA_executions':0,'GO':False}
    for name,value in (('PLAN.json',plan),('SUMMARY.json',summary)):(output/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'fits_planned':counts['totals']['atomic_fit_run_groups'],'optimizer_starts':counts['totals']['paid_optimizer_starts'],'records':counts['totals']['run_records'],'fits_executed':0,'GO':False}))


if __name__=='__main__':main()
