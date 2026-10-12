"""Price the current roster from existing random-source receipts only."""
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD,invoke

OUT=Path(__file__).resolve().parent/'v17_resource_storage_catalog_root_20261012_v1'
REMOTE=GUARD+r'''
import gzip,collections
master_path=phase/'massive_practical_master_20261012_v17/PLAN.json'
plan=json.loads(master_path.read_text())
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
assert hashlib.sha256(canonical(plan)).hexdigest()=='88a995423d627baa159e24d794311db355e55d7c471eccb9b112c68f08e09aa1'
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def bound(row):
 p=phase/row['path'];assert p.resolve().is_relative_to(phase) and receipt(p)==row;return p
def study_at(p):
 raw=p.read_bytes();return json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)
catalog=[];failures=[]
manifest=json.loads((phase/'practical_resource_qualifier_manifest_20261012_v2/MANIFEST.json').read_text())
observations=json.loads((phase/'practical_resource_qualifier_remaining_20261012_v2/OBSERVATIONS.json').read_text())
for case in manifest['cases']:
 binding=observations[case['case_id']]['result_binding'];r=json.loads(bound(binding).read_text())
 assert r['actual_optimizer_updates']==r['actual_optimizer_kernel_calls']==r['scores_emitted']==r['heldout_original_file_accesses']==0
 catalog.append({'case_id':case['case_id'],'study_id':case['study_id'],'method':case['run']['method'],
  'source_run':case['run'],'result':binding,'shape_coverage':case.get('coverage_relation'),
  'covered_config_ids':case.get('covered_config_ids_shape_proxy_only',[]),
  'checkpoint_bytes':r['serialization']['bytes'],
  'reserved_bytes':r['CUDA_peak_reserved_bytes'],'allocated_bytes':r['CUDA_peak_allocated_bytes'],
  'gradient_callback_seconds':r.get('warm_TRAIN_median_seconds'),
  'serializer_seconds':r['serialization']['save_seconds'],
  'source_case_scope':'declared maximal native shape envelope; learned state and optimizer kernels unmeasured'})
sets=[('native_wrapper_rotation_resource_stage_20261012_v1/source/PROPOSAL.json','wrapper_rotation_resource_results_root_20261012_v1/CLOSURE.json'),
 ('native_combinations_resource_stage_root_20261012_v1/source/PROPOSAL.json','native_combinations_resource_results_root_20261012_v1/CLOSURE.json'),
 ('closed15_resource_stage_root_20261012_v1/source/PROPOSAL.json','closed15_resource_results_root_20261012_v1/CLOSURE.json'),
 ('closed15_resource_stage_root_20261012_v2/source/PROPOSAL.json','closed15_resource_results_root_20261012_v2/CLOSURE.json')]
for proposal_path,closure_path in sets:
 proposal=json.loads((phase/proposal_path).read_text());cases={c['case_id']:c for c in proposal['requested_cases']}
 closure=json.loads((phase/closure_path).read_text());assert closure['all_children_closed']
 for a in closure['attempts']:
  if a['exit_code']!=0:
   failures.append({'case_id':a['case_id'],'closure':closure_path,'failure':a['failure_log_tail'].splitlines()[-1]});continue
  case=cases[a['case_id']];r=json.loads(bound(a['result']).read_text())
  assert r['actual_optimizer_updates']==r['scores_emitted']==r['heldout_target_files_opened']==0
  study_path=case.get('study_path') or case.get('study_receipt',{}).get('path')
  s=study_at(phase/study_path);run=next(v for v in s['runs'] if v['run_id']==case['source_run_id'])
  catalog.append({'case_id':a['case_id'],'study_id':s['study_id'],'method':run['method'],'source_run':run,
   'result':a['result'],'checkpoint_bytes':r['random_engine_restore']['bytes'],
   'reserved_bytes':r['CUDA_peak_reserved_bytes'],'allocated_bytes':r['CUDA_peak_allocated_bytes'],
   'gradient_callback_seconds':r['warm_median_seconds'],
   'serializer_seconds':r['random_engine_restore']['save_seconds'],
   'bank_shape_bytes':r['CPU_bank_export_proxy'].get('retained_route_logits_bytes'),
   'source_case_scope':'exact source/provider random callback; other methods or geometries are not qualified'})
profiles={}
for c in catalog:
 key=(c['study_id'],c['method']);profiles.setdefault(key,[]).append(c)
placement={k:g['gpu_assignment'] for g in plan['groups'] for k in g['run_keys']}
covered=[];missing=collections.Counter();method_rows=[];roster=[]
for entry in plan['studies']:
 s=study_at(phase/entry['path']);methods={}
 for run in s['runs']:
  if run['kind']!='model_fit':continue
  key=(s['study_id'],run['method']);candidates=profiles.get(key,[])
  methods.setdefault(run['method'],[]).append(run)
  if not candidates:
   missing[key]+=1;continue
  # This is explicitly a shape-envelope projection, not whole-run admission.
  state=max(c['checkpoint_bytes'] for c in candidates)
  prefixes=len(run['budget'].get('prefixes',[]));copies=4+2*prefixes
  secs=[c['gradient_callback_seconds'] for c in candidates if c['gradient_callback_seconds'] is not None]
  covered.append({'study_id':s['study_id'],'run_id':run['run_id'],'method':run['method'],
   'worker':placement[s['study_id']+'/'+run['run_id']],
   'budget_unit':run['budget']['unit'],'budget_maximum':run['budget']['maximum'],
   'checkpoint_shape_proxy_bytes':copies*state,'retained_state_multiplier':copies,
   'random_callback_weight_seconds':None if not secs else max(secs)*run['budget']['maximum'],
   'maximum_random_CUDA_bytes':max(max(c['reserved_bytes'],c['allocated_bytes']) for c in candidates),
   'evidence':[c['result'] for c in candidates]})
 for method,runs in sorted(methods.items()):
  fields=('backbone','architecture','width','depth','heads','members','learning_rate','weight_decay','dropout','normalization','ffn_multiplier','objective_variant','objective_coefficients','mechanism_arm','view_arm','pt_arm','ported_family','ported_arm','local_transition_updates','physical_slow_bodies','source_optimizer_count','fusion','recipe_scope')
  domain={k:sorted({json.dumps(r['config'][k],sort_keys=True) for r in runs if k in r['config']}) for k in fields}
  method_rows.append({'study_id':s['study_id'],'dataset':s['dataset'],'method':method,'fit_groups':len(runs),
   'search_fit_groups':sum(r['config']['stage']=='search' for r in runs),
   'comparative_fit_groups':sum(r['config']['stage']=='comparative' for r in runs),
   'budgets':sorted({json.dumps(r['budget'],sort_keys=True) for r in runs}),
   'hyperparameters_and_architectures':domain,
   'source_factories':sorted({r['adapter'] for r in runs}),
   'seeds':sorted({r['seed'] for r in runs}),'splits':sorted({r['split'] for r in runs},key=str),
   'shape_proxy_fit_groups':sum(c['study_id']==s['study_id'] and c['method']==method for c in covered)})
 roster.append({'study_id':s['study_id'],'dataset':s['dataset'],'source_spec_hash':entry['spec_hash'],
  'workers':sorted({placement[s['study_id']+'/'+r['run_id']] for r in s['runs']}),
  'model_fit_groups':sum(r['kind']=='model_fit' for r in s['runs']),
  'author_groups':sum(r['kind']=='author_cli_group' for r in s['runs'])})
assert len(covered)+sum(missing.values())==70692
fs=os.statvfs(repo)
summary={'schema':'current-master-resource-storage-projection-v1','plan_hash':'88a995423d627baa159e24d794311db355e55d7c471eccb9b112c68f08e09aa1',
 'existing_passed_resource_receipts':len(catalog),'preserved_failed_resource_attempts':len(failures),
 'covered_fit_groups':len(covered),'unpriced_fit_groups':sum(missing.values()),
 'covered_checkpoint_shape_proxy_bytes':sum(c['checkpoint_shape_proxy_bytes'] for c in covered),
 'trace_scenario_bytes_per_registered_unit':1024,
 'covered_trace_scenario_bytes':1024*sum(c['budget_maximum'] for c in covered),
 'filesystem_available_bytes_at_observation':fs.f_bavail*fs.f_frsize,
 'filesystem_available_inodes_at_observation':fs.f_favail,
 'whole_run_memory_time_storage_admitted':False,'ETA_available':False,
 'bank_bytes_are_dummy_shape_proxies':True,'epoch_batch_and_source_cycle_units_remain_distinct':True,
 'known_gradient_callback_weights_are_not_duration_estimates':True,
 'by_worker':[{'worker':w,'covered_fit_groups':sum(c['worker']==w for c in covered),
  'covered_checkpoint_shape_proxy_bytes':sum(c['checkpoint_shape_proxy_bytes'] for c in covered if c['worker']==w)} for w in sorted(plan['worker_capacities_cuda_bytes'])],
 'unpriced_study_methods':[{'study_id':s,'method':m,'fit_groups':n} for (s,m),n in sorted(missing.items())],
 'excluded_costs':['unpriced checkpoint families','prediction banks','author artifacts','atomic-write transients','failed/recovery attempts','calibration','disk quota','server77 available disk','optimizer kernels','SELECT and convergence-dependent writes'],
 'no_new_models_constructed':True,'tensor_arrays_or_scores_opened':False,'scientific_fits':0,'GO':False}
out=phase/'v17_resource_storage_catalog_root_20261012_v1';assert not out.exists();out.mkdir()
value={'summary':summary,'catalog':catalog,'preserved_failures':failures,'methods':method_rows,'studies':roster}
(out/'CATALOG.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
print(json.dumps(value))
'''

def main():
    OUT.mkdir();(OUT/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
    r=invoke(REMOTE,[],OUT,'CATALOG',180);assert r['exit_code']==0,'Preserve catalog failure'
    value=json.loads(r['stdout'])
    (OUT/'CATALOG.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in value['summary'].items() if k not in ('unpriced_study_methods','excluded_costs')}))

if __name__=='__main__':main()
