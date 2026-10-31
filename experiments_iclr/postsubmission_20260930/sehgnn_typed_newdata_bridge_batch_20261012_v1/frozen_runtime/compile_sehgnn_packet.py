"""Compile one disabled finite native5 new-data packet; never train or score."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import subprocess

from adapters.sehgnn import BASIC,SUPPLY,JOINT
from source_utils import receipt,bound,bind_source
from prepare_node_data import verify_allocation
from spec import canonical,fingerprint,validate,confined
from compile_master_plan import typed_counts,graph_check,bank_roster,SEHGNN_PREFIXES,SEHGNN_JOINT

SEARCH=(83101,83103);COMPARE=(83201,83203,83207)
DOMAIN=[{'learning_rate':lr,'dropout':d} for lr in (.0005,.001,.002) for d in (.3,.5,.7)]
ARMS=(*BASIC,*SUPPLY,*JOINT)
I4={'native_independent4':'native_single','factorized_native_independent4':'factorized_native_single','rank1_P_independent4':'rank1_single_P'}


def make_study(root,custody,custody_dir,source,sources,runtime):
    if custody['dataset_id']!='hgb-imdb-native5-development' or sorted(r['split_id'] for r in custody['splits'])!=list(range(10)) or custody['class_dim']!=5:raise ValueError('Exact complete native5 ten-mask data required')
    rows={r['path']:r for r in custody['files']};splits={r['split_id']:r for r in custody['splits']};runs=[]
    def binding(relative):
        r=rows[relative];return {'path':str((custody_dir/relative).relative_to(root)),'bytes':r['bytes'],'sha256':r['sha256']}
    def typed_inputs(split):
        row=splits[split]
        return {**{k:binding(v) for k,v in row['node_inputs'].items()},'features':{k:binding(v) for k,v in custody['graph_feature_caches'].items()},'labels':{k:binding(v) for k,v in row['label_caches'].items()}}
    def config(arm,split,stage):
        if arm in SEHGNN_JOINT:members,owners,*unused=SEHGNN_JOINT[arm]
        else:members,owners=(1,1) if arm in ('native_single','factorized_native_single') else (4,1)
        return {'sehgnn_arm':arm,'sehgnn_epochs':1000,'architecture':'native_sehgnn_source','backbone':'native_SeHGNN','width':512,'classes':5,'feature_channels':25,'label_channels':12,
                'members':members,'physical_slow_bodies':owners,'source_optimizer_count':owners,'is_original_gnnm':arm=='original_gnnm','historical_state_inherited':False,
                'sehgnn_sources':sources,'typed_inputs':typed_inputs(split),'fusion':'mean_logits' if arm=='original_gnnm' else 'mean_probabilities','weight_decay':0.,'stage':stage,
                'selection_roles':'TRAIN/SELECT only; heldout-style development endpoints remain globally closed','historical_source_horizon':200,'prospective_complete_native_epochs':1000,
                'diagnostic_prefixes':[200,500],'prefixes_are_not_additional_heldout_endpoints':True,'native_source_own_batch_size':10000,'native_source_TRAIN_batches_per_epoch':1,
                'source_correction_fraction':.1,'source_correction_start_epoch':15,'source_correction_cadence':5,'source_trial_scales':[1.,.5,.25,.125],
                'reservation_mib':32768,'estimated_seconds':5000000,'resource_evidence':'unmeasured disabled reservation placeholder; no admission or ETA'}
    def add(identifier,kind,method,split,seed,body,c,parents=()):
        if kind=='model_fit':
            c['initialization_seed']=seed+1009*body;adapter='sehgnn.py:make_adapter';budget={'unit':'epochs','policy':'full_budget','maximum':1000,'prefixes':copy.deepcopy(SEHGNN_PREFIXES)}
            selection={'metric':'pooled_BCE','direction':'min','ties':'first','evaluation_interval':1,'checkpoint0':False,'patience_steps':1001}
        else:
            adapter='search.py:select_configuration' if kind=='selection' else 'sehgnn_endpoints.py:'+{'ensemble_assembly':'export_bank','calibration':'scalar_calibration','evaluation':'evaluate_bank'}[kind];budget={};selection={}
        row={'run_id':identifier,'kind':kind,'method':method,'split':split,'seed':seed,'body':body,'adapter':runtime+'/adapters/'+adapter,'config':c,'parents':list(parents),'budget':budget,'selection':selection};runs.append(row);return identifier
    selectors={}
    for arm in ARMS:
        configs=[]
        for point in DOMAIN:
            key=fingerprint(point)[:16];ids=[add('search-'+arm+'-'+key+'-s'+str(seed),'model_fit',arm,0,seed,0,{**config(arm,0,'search'),**point}) for seed in SEARCH]
            configs.append({'config_id':key,'parameters':point,'run_ids':ids})
        selectors[arm]=add('select-'+arm,'selection',arm,0,0,0,{'stage':'search_selection','configurations':configs,'ranking_metric':'objective','ranking_direction':'min','reservation_mib':1,'estimated_seconds':1},[i for row in configs for i in row['run_ids']])
    for split in range(10):
        for seed in COMPARE:
            fits={}
            for arm in ARMS:
                fits[arm]=[add('compare-sp'+str(split)+'-'+arm+'-s'+str(seed),'model_fit',arm,split,seed,0,{**config(arm,split,'comparative'),**DOMAIN[0],'selected_config_from':selectors[arm]},[selectors[arm]])]
            for method,single in I4.items():
                fits[method]=[fits[single][0]]
                for body in (1,2,3):fits[method].append(add(fits[single][0]+'-I4-b'+str(body),'model_fit',single,split,seed,body,{**config(single,split,'comparative'),**DOMAIN[0],'selected_config_from':selectors[single]},[selectors[single]]))
            endpoints={k:binding(v)['path'] for k,v in splits[split]['endpoint_inputs'].items()}
            for method,parents in fits.items():
                prefix='bank-sp'+str(split)+'-'+method+'-s'+str(seed);fusion='mean_logits' if method=='original_gnnm' else 'mean_probabilities'
                export=add(prefix,'ensemble_assembly',method,split,seed,0,{'stage':'bank_export','fit_parents':parents,'endpoint_inputs':endpoints,'fusion':fusion,'reservation_mib':32768,'estimated_seconds':10000},parents)
                cal=add(prefix+'-cal','calibration',method,split,seed,0,{'stage':'calibration','export_parent':export,'endpoint_inputs':endpoints,'fusion':fusion,'placement':'after_serving_pool','steps':500,'learning_rate':.01,'reservation_mib':512,'estimated_seconds':1000},[export])
                add(prefix+'-evaluate','evaluation',method,split,seed,0,{'stage':'evaluation','export_parent':export,'calibration_parent':cal,'endpoint_inputs':endpoints,'fusion':fusion,'reservation_mib':512,'estimated_seconds':1000},[export,cal])
    roles={role:{str(split):binding(splits[split]['node_inputs' if role in ('train','select') else 'endpoint_inputs'][role if role in ('train','select') else role+'_targets']) for split in range(10)} for role in ('train','select','cal','evaluate')}
    study={'schema_version':1,'study_id':'native-SeHGNN-source-seven-conditions-newdata-v1','stage':'development','dataset':custody['dataset_id'],
           'hypothesis':'Private contextual source credit and additive native corrections improve shared-native five-label predictions versus own capable native/original GNNM and genuine separately selected independent controls; all source work is paid.',
           'data':{'task':'multilabel_node_classification','preprocessing':custody['preprocessing'],'roles':roles,'prior_exposure':custody['prior_evaluation_exposure'],
                   'evidence':{'custody_schema':custody['schema'],'native_nodes':custody['node_counts'],'raw_byte_authority':custody['raw_input_files'],'class_dim':5,'model_parameters_native_single':83659532,'historical_results_inherited':False,'official_TEST_access':False}},
           'source':source,'runs':runs,'comparisons':[{'references':['native_single','native_independent4','factorized_native_single','factorized_native_independent4','original_gnnm'],
               'candidates':list(SUPPLY)+list(JOINT),'scope':'All10custom paired development masks;3fresh comparative seeds;9points×2fresh HPO seeds per source arm;1000complete native epochs;200/500convergence diagnostics only'},
               {'references':['shared_own_only'],'candidates':list(SUPPLY[1:]),'scope':'Exact private source supply policies including COMMON and assigned family dose/cost accounting'},
               {'references':['BE_P','ADD_P','ADD_P_copied_dictionary','ADD_PS_neutral_all_source_average','independent_ADD_P','independent_ADD_PS','native_single_P','rank4_single_P','rank1_single_P','rank1_P_independent4'],
                'candidates':['BE_PS','ADD_PS'],'scope':'Source operator/context/dictionary/ownership/rank controls; grouped independent source learners remain distinct from genuine separately selected I4'}],
           'release':{'policy':'whole_roster','primary_endpoint':'1000epoch own SELECT bank: raw BCE/microF1/macroF1/macroAUC and common scalar CAL; seed/mask paired inference','global_plan_gates_required':True},
           'resources':{'wall_seconds':1000000000,'status':'DISABLED_ACTUAL_INPUT_RESOURCE_ADMISSION_AND_HUMAN_GO_PENDING','all_replays_leaf_reverses_private_VJPs_and_trials_paid':True}}
    validate(study);graph_check([study],{});return study


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('project-root','runtime-package','custody','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();verify_allocation();root=Path(a.project_root).resolve();runtime=confined(root,a.runtime_package);out=confined(root,a.output);cp=confined(root,a.custody);custody=json.loads(cp.read_text());sources=bind_source(root)
    if out.exists():raise ValueError('Fresh disabled packet required')
    files=[]
    for row in custody['files']:
        observed=receipt(root,cp.parent/row['path'])
        if any(observed[k]!=row[k] for k in ('bytes','sha256')):raise ValueError('New independent typed custody bytes changed')
        files.append(observed)
    provider=json.loads(bound(root,sources['provider_manifest']).read_text());provider_files=[receipt(root,row['path']) for row in provider['files']]
    if provider_files!=provider['files']:raise ValueError('Isolated provider files changed')
    runtime_files=[receipt(root,f) for f in sorted(runtime.rglob('*.py')) if '__pycache__' not in f.parts]
    source_rows=runtime_files+list(sources['modules'].values())+list(sources['metadata'].values())+[sources['provider_manifest']]+provider_files+files+[receipt(root,cp)]
    source={'git_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'files':source_rows,'working_tree_fingerprints_authoritative':True,'frozen_runtime_package':a.runtime_package,'historical_admission_or_fits_inherited':False}
    study=make_study(root,custody,cp.parent,source,sources,a.runtime_package);counts=typed_counts([study]);banks=bank_roster(study);out.mkdir();(out/'studies').mkdir();path=out/'studies'/(study['study_id']+'.json.gz');path.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0))
    plan={'schema_version':1,'slots_per_gpu':2,'studies':[{'path':str(path.relative_to(root)),'spec_hash':fingerprint(study)}],
          'groups':[{'id':study['study_id'],'run_keys':[study['study_id']+'/'+r['run_id'] for r in study['runs']],'allowed_workers':['allocation/gpu0'],'gpu_assignment':'allocation/gpu0','estimated_seconds':sum(r['config']['estimated_seconds'] for r in study['runs']),'maximum_cuda_bytes':32768*1024**2}],
          'barriers':{},'scientific_launch_enabled':False,'human_GO_required':True,'resources_are_provisional':True}
    summary={'schema':'native-SeHGNN-seven-source-conditions-disabled-packet-v1','plan_hash':fingerprint(plan),'status':'DISABLED_ACTUAL_INPUT_RESOURCE_ADMISSION_AND_GO_PENDING','counts':counts,'arms':list(ARMS),'bank_methods':list(ARMS)+list(I4),'HPO_domain':DOMAIN,'HPO_seeds':list(SEARCH),'comparative_seeds':list(COMPARE),'complete_masks':10,'primary_complete_native_epochs':1000,'diagnostic_prefix_caps':[200,500],'prefix_heldout_banks':0,
             'genuine_I4_banks':sum(b['genuine_I4'] for b in banks),'grouped_independent_source_banks_are_not_genuine_I4':True,'original_gnnm_fit_records':sum(r['kind']=='model_fit' and r['method']=='original_gnnm' for r in study['runs']),
             'GO_receipt_created':False,'benchmark_training_started':False,'actual_fits':0,'actual_CUDA_executions':0,'GO':False,'source_provider_manifest':sources['provider_manifest']}
    for name,value in (('PLAN.json',plan),('SUMMARY.json',summary)):(out/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    entry={'id':'native-SeHGNN-source-seven-conditions-newdata-v1','plan':receipt(root,out/'PLAN.json'),'summary':receipt(root,out/'SUMMARY.json'),'plan_hash':fingerprint(plan)};(out/'MASTER_INPUT_ENTRY.json').write_text(json.dumps(entry,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'disabled_prepared','fits_planned':counts['totals']['atomic_fit_run_groups'],'optimizer_owners':counts['totals']['paid_optimizer_starts'],'banks':len(banks),'records':len(study['runs']),'fits_executed':0,'GO':False}))


if __name__=='__main__':main()
