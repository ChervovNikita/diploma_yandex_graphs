"""Complete fixed native HGT6/mixed4 DBLP10x3 disabled finite roster."""
import argparse,copy,gzip,json,subprocess
from pathlib import Path
from adapters.hgt_native import PINS,CONTRACT,BUDGET,SELECTION,expected_parameters,receipt
from compile_master_plan import graph_check,typed_counts,reference_matches
from spec import canonical,fingerprint,validate

SEEDS=(99601,99603,99607)
CANONICAL={'global_BE':'typed_hgt_global_be','shared_relation':'typed_hgt_shared_relation','CP':'typed_hgt_cp','unrestricted':'typed_hgt_unrestricted',
 'untied_HGT':'typed_hgt_untied_hgt','wider_BE':'typed_hgt_wider_be','own/own':'typed_mixed_objectives_own_own','pool/pool':'typed_mixed_objectives_pool_pool',
 'pool/own':'typed_mixed_objectives_pool_own','own/pool':'typed_mixed_objectives_own_pool'}
METHODS={'native_M1':'control_HGT_ordinary_M1','factor_M1':'control_HGT_factorized_M1','original_gnnm':'control_HGT_original_GNNM',**CANONICAL}

def make_study(root,packet,custody,sources,source_files,commit):
 prefix=str(packet.relative_to(root));runtime=prefix+'/runtime';data=prefix+'/data';known={r['path']:r for r in custody['files']}
 def inputs(split,field):return {k:{**known[v],'path':data+'/'+v} for k,v in custody['splits'][split][field].items()}
 def endpoints(split):return {k:v['path'] for k,v in inputs(split,'endpoint_inputs').items()}
 runs=[];banks=[];params=expected_parameters(custody['model_graph']['input_dims'],4);params['wider_BE']=expected_parameters(custody['model_graph']['input_dims'],4,width=72)['global_BE']
 def add(identifier,kind,method,split,seed,body,config,parents=()):
  fn='hgt_native.py:make_adapter' if kind=='model_fit' else 'hgt_endpoints.py:'+{'ensemble_assembly':'export_bank','calibration':'scalar_calibration','evaluation':'evaluate_bank'}[kind]
  runs.append({'run_id':identifier,'kind':kind,'method':method,'split':split,'seed':seed,'body':body,'adapter':runtime+'/adapters/'+fn,'config':config,
   'budget':copy.deepcopy(BUDGET) if kind=='model_fit' else {},'selection':copy.deepcopy(SELECTION) if kind=='model_fit' else {},'parents':list(parents)})
  return identifier
 for split in range(10):
  for seed in SEEDS:
   fits={}
   for arm in METHODS:
    fits[arm]=[];members=1 if arm in ('native_M1','factor_M1') else 4
    for body in range(4 if arm in ('native_M1','factor_M1') else 1):
     c={'hgt_arm':arm,'hgt_sources':sources,'architecture':'native_HGT_typed','backbone':'native_HGB_HGT','width':72 if arm=='wider_BE' else 64,
      'depth':3,'heads':8,'classes':4,'dropout':.2,'weight_decay':.0001,'learning_rate':.001,'fusion':'mean_logits','members':members,
      'source_optimizer_count':1,'physical_slow_bodies':4 if arm=='untied_HGT' else 1,'initialization_seed':seed+1000003*body,
      'source_complete_body_constructors':1 if arm=='original_gnnm' else 5 if arm=='untied_HGT' else 3 if arm=='wider_BE' else 2,
      'source_scheduler_rule':'OneCycle300 maxlr.001 pct_start.05 cosine step(update)','source_TRAIN_forwards_per_update':members,
      'source_TRAIN_backwards_per_update':2 if arm in ('pool/own','own/pool') else 1,'historical_state_inherited':False,'is_original_gnnm':arm=='original_gnnm',
      'node_inputs':inputs(split,'node_inputs'),'stage':'comparative','parameter_count':params[arm],
      'recipe_scope':'exact native HGT optimizer/scheduler/selector/equations; explicit fresh masks/paid seeds and typed original boundary adaptation',
      'reservation_mib':32768,'estimated_seconds':300.,'resource_profile_status':'UNQUALIFIED_PROVISIONAL; root actual-input source qualification required'}
     fits[arm].append(add(f'compare-sp{split}-{arm.replace("/","_")}-s{seed}-b{body}','model_fit',METHODS[arm],split,seed,body,c))
   cohorts=[(arm,METHODS[arm],[ids[0]] if arm in ('native_M1','factor_M1') else ids) for arm,ids in fits.items()]
   cohorts += [(arm,'control_HGT_'+name+'_I4',fits[arm]) for arm,name in (('native_M1','ordinary'),('factor_M1','factorized'))]
   for arm,method,parents in cohorts:
    name=f'bank-sp{split}-{method}-s{seed}';inp=endpoints(split)
    export=add(name,'ensemble_assembly',method,split,seed,0,{'stage':'bank_export','fit_parents':parents,'endpoint_inputs':inp,'fusion':'mean_logits','reservation_mib':32768,'estimated_seconds':30.},parents)
    cal=add(name+'-cal','calibration',method,split,seed,0,{'stage':'calibration','export_parent':export,'endpoint_inputs':inp,'fusion':'mean_logits',
     'placement':'after_native_FP32_mean_logits','steps':500,'learning_rate':.01,'reservation_mib':512,'estimated_seconds':10.},[export])
    evaluation=add(name+'-evaluate','evaluation',method,split,seed,0,{'stage':'evaluation','export_parent':export,'calibration_parent':cal,'endpoint_inputs':inp,'fusion':'mean_logits','reservation_mib':512,'estimated_seconds':10.},[export,cal])
    banks.append({'method':method,'arm':arm,'split':split,'seed':seed,'fit_parents':parents,'export':export,'calibration':cal,'evaluation':evaluation,
     'genuine_I4':len(parents)==4,'source_joint_untied_not_I4':arm=='untied_HGT','logical_views':4 if len(parents)==4 or arm not in ('native_M1','factor_M1') else 1})
 roles={r:{str(split):inputs(split,'node_inputs' if r in ('train','select') else 'endpoint_inputs')[r if r in ('train','select') else r+'_targets'] for split in range(10)} for r in ('train','select','cal','evaluate')}
 study={'schema_version':1,'study_id':'native-HGT-mixed-DBLP-newroles-v2','stage':'development','dataset':'HGB-DBLP',
  'hypothesis':'Full exact HGT6 and mixed-block4 source conditions on fresh encountered-development DBLP roles, with fresh native/factor M1/I4 and compatible actual original boundary GNNM; source untied remains one joint learner.',
  'data':{'task':'node_classification','preprocessing':custody['request']['native_graph'],'roles':roles,'prior_exposure':custody['request']['evaluation_exposure'],
   'evidence':{'actual_custody':receipt(root,packet/'data/CUSTODY.json'),'complete_model_graph':custody['model_graph'],'development_only_pool':1217,'source_contract':CONTRACT,
    'official_test_labels_opened':False,'native_fixed_recipe_no_search':True}},
  'source':{'git_commit':commit,'files':source_files,'working_tree_fingerprints_authoritative':True,'frozen_runtime_package':runtime,'official_author_protocol_replication':False},
  'runs':runs,'comparisons':[{'references':['control_HGT_ordinary_M1','control_HGT_ordinary_I4','control_HGT_factorized_M1','control_HGT_factorized_I4','control_HGT_original_GNNM'],
   'candidates':list(CANONICAL.values()),'scope':'complete fresh10 masks x3paired model seeds; fixed native OneCycle300/p30; latest SELECT mean-logit CE; bounded update costs and all outcomes retained'}],
  'release':{'policy':'whole_roster','primary_endpoint':'all raw FP32 mean-logit and scalar CAL500 banks','global_plan_gates_required':True},
  'resources':{'wall_seconds':172800,'status':'DISABLED_SOURCE_RESOURCE_MASTER_AND_GO_PENDING'}}
 validate(study);graph_check([study],{});return study,banks

def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);p.add_argument('--packet',required=True);args=p.parse_args();root=Path(args.project_root).resolve();packet=root/args.packet
 custody=json.loads((packet/'data/CUSTODY.json').read_text());sources=json.loads((packet/'EXACT_SOURCE_BINDINGS.json').read_text());sources={k:{x:v for x,v in row.items() if x in ('path','bytes','sha256')} for k,row in sources.items()}
 if custody['status']!='prepared' or custody['test_label_member_opened'] or custody['benchmark_fits'] or custody['predictive_scores'] or any(sources[k]['sha256']!=v for k,v in PINS.items()):raise ValueError('Exact source and actual raw full graph/fresh role custody')
 files=[]
 for directory in ('runtime','sources'):
  files += [receipt(root,path) for path in sorted((packet/directory).rglob('*')) if path.is_file() and path.suffix=='.py']
 files += [receipt(root,packet/name) for name in ('EXACT_SOURCE_BINDINGS.json','DBLP_INPUT_METADATA.json','SYNTHETIC_CPU_PROOF.json','ROOT_V16_COMPOSITION.json','data/CUSTODY.json')]
 files += [{**r,'path':args.packet+'/data/'+r['path']} for r in custody['files']]+[custody['source_archive'],custody['source_development_labels']]
 files=list({r['path']:r for r in files}.values());commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();study,banks=make_study(root,packet,custody,sources,files,commit)
 counts=typed_counts([study]);refs=reference_matches(study)
 if any(not r['matched'] for r in refs):raise ValueError('Missing complete matched source references')
 matrix=packet/'matrix';matrix.mkdir();(matrix/'studies').mkdir();path=matrix/'studies'/(study['study_id']+'.json.gz');path.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0));prefix=args.packet+'/matrix'
 plan={'schema_version':1,'slots_per_gpu':2,'torch_threads_per_fit':2,'output_root':prefix+'/execution','runtime_launcher':args.packet+'/runtime/launcher.py','runtime_package':args.packet+'/runtime',
  'worker_capacities_cuda_bytes':{'allocation/gpu0':85899345920},'studies':[{'path':str(path.relative_to(root)),'spec_hash':fingerprint(study)}],
  'groups':[{'id':study['study_id'],'gpu_assignment':'allocation/gpu0','allowed_workers':['allocation/gpu0'],'maximum_cuda_bytes':34359738368,
   'estimated_seconds':sum(r['config']['estimated_seconds'] for r in study['runs']),'run_keys':[study['study_id']+'/'+r['run_id'] for r in study['runs']]}],
  'global_gates':{'whole_comparative':{'path':prefix+'/gate_receipts/whole_comparative.json','source_stages':['comparative'],'required_for_stages':['bank_export','calibration','evaluation'],'required_hosts':['allocation']},
   'whole_calibration':{'path':prefix+'/gate_receipts/whole_calibration.json','source_stages':['calibration'],'required_for_stages':['evaluation'],'required_hosts':['allocation']}},
  'barriers':{},'scientific_launch_enabled':False,'human_GO_required':True,'host_bindings':{'allocation':{'status':'verified_at_compilation','hostname':'anogena-2-0','gpu_uuids':['GPU-44039938-fd82-41d2-fefd-de71514e2fac']}}}
 summary={'schema':'native-HGT-mixed-disabled-summary-v2','status':'SOURCE_PREPARED_DISABLED_RESOURCE_MASTER_AND_GO_PENDING','plan_hash':fingerprint(plan),'canonical_ids':list(CANONICAL.values()),
  'counts':counts,'banks':banks,'source_contract':CONTRACT,'fixed_comparative_seeds':list(SEEDS),'fixed_native_recipe_no_tuning_fits':True,'training_started':False,'actual_model_fits':0,'actual_predictive_scores':0,'GO_receipt_created':False,
  'data_closure':'full immutable rawfeature2 DBLP26,128nodes/239,566directedrelations;1217releaseddevelopment target labels,10freshrole masks; no officialtestlabels',
  'resource_closure':'UNQUALIFIED_ROOT_ACTUAL_INPUT_CALLBACKS_PENDING; no actual-input models or optimizer updates executed'}
 (matrix/'PLAN.json').write_text(canonical(plan)+'\n');(matrix/'SUMMARY.json').write_text(canonical(summary)+'\n')
 entry={'id':'native_HGT_mixed_DBLP_newroles_v2','plan':receipt(root,matrix/'PLAN.json'),'plan_hash':summary['plan_hash'],'summary':receipt(root,matrix/'SUMMARY.json'),
  'coverage':[receipt(root,packet/'CANONICAL_SOURCE_COMPLETION.json'),receipt(root,packet/'EXACT_SOURCE_BINDINGS.json')]}
 (packet/'MASTER_INPUT_ENTRIES.json').write_text(json.dumps([entry],indent=2,sort_keys=True)+'\n');(packet/'COMPILATION_METADATA.json').write_text(json.dumps({'schema':'native-HGT-mixed-typed-compilation-v2','entry':entry,'counts':counts,'references':refs,'source_files':len(files),'canonical_ids':list(CANONICAL.values()),'fits':0,'scores':0,'GO':False},indent=2,sort_keys=True)+'\n')
 print(json.dumps({'entry':entry,'counts':counts,'source_files':len(files),'fits':0,'scores':0,'GO':False}))

if __name__=='__main__':main()
