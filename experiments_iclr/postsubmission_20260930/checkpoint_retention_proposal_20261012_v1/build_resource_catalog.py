"""Parameterised metadata catalogue; existing random callbacks are not ETA."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'training_pipeline_v2_20261012/preparation'))
from prepare_molecular_packets_allocation import GUARD,invoke

REMOTE=GUARD+r'''
import collections,gzip
relative,expected,destination=sys.argv[1:4]
master=phase/relative;assert master.resolve().is_relative_to(phase)
plan=json.loads(master.read_text());canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
assert hashlib.sha256(canonical(plan)).hexdigest()==expected
out=phase/destination;assert out.resolve().is_relative_to(phase) and not out.exists();out.mkdir()
def receipt(p):
 raw=p.read_bytes();return {'path':str(p.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def bound(row):
 p=phase/row['path'];assert p.resolve().is_relative_to(phase) and receipt(p)==row;return p
def load_study(row):
 p=phase/row['path'];raw=p.read_bytes();value=json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)
 assert hashlib.sha256(canonical(value)).hexdigest()==row['spec_hash'];return value
previous_path=phase/'v17_resource_storage_catalog_root_20261012_v1/CATALOG.json'
previous=json.loads(previous_path.read_text());catalog=previous['catalog'];corrected=[]
for row in catalog:
 r=json.loads(bound(row['result']).read_text())
 assert r.get('scientific_fits',0)==r.get('actual_optimizer_updates',0)==r.get('scores_emitted',0)==0
 timing=r.get('warm_train_median_seconds',r.get('warm_TRAIN_median_seconds',r.get('warm_median_seconds')))
 revised=dict(row);revised['gradient_callback_seconds']=timing
 revised['callback_timing_field']='warm_train_median_seconds' if 'warm_train_median_seconds' in r else ('warm_median_seconds' if 'warm_median_seconds' in r else 'warm_TRAIN_median_seconds')
 revised['callback_timing_scope']='measured zero-update random-source callbacks only; optimizer kernels, learned buffers and SELECT not admitted'
 revised['source_factory_receipt']=receipt(phase/row['source_run']['adapter'].split(':',1)[0])
 corrected.append(revised)
profiles=collections.defaultdict(list)
for row in corrected:profiles[(row['study_id'],row['method'])].append(row)
placement={k:g['gpu_assignment'] for g in plan['groups'] for k in g['run_keys']}
methods=[];studies=[];queue=[];fit_total=0;covered=0;missing=0;changed_factory=0;storage=0;measured=0
structural=('backbone','architecture','width','depth','heads','members','classes','ffn_multiplier','normalization','objective_variant','objective_coefficients','source_optimizer_count','physical_slow_bodies','fusion','local_transition_updates','nsd_arm','nsd_stage','growth_arm','growth_stage','label_stage','label_arm','mechanism_arm','view_arm','pt_arm','ported_family','ported_arm','curvature_arm','curvature_stage','stalk_dim')
for spec in plan['studies']:
 s=load_study(spec)
 assert s['dataset'].strip().lower().replace('_',' ') not in {'roman empire','amazon ratings','minesweeper','questions','tolokers'}
 by_method=collections.defaultdict(list)
 for run in s['runs']:
  if run['kind']=='model_fit':by_method[run['method']].append(run);fit_total+=1
 for method,runs in sorted(by_method.items()):
  candidates=profiles.get((s['study_id'],method),[])
  factories=sorted({r['adapter'] for r in runs});factory_receipts={f:receipt(phase/f.split(':',1)[0]) for f in factories}
  same=[c for c in candidates if c['source_run']['adapter'] in factories and c['source_factory_receipt']==factory_receipts[c['source_run']['adapter']]]
  if same:
   covered+=len(runs);state=max(c['checkpoint_bytes'] for c in same);storage+=sum((4+2*len(r['budget'].get('prefixes',[])))*state for r in runs)
   measured+=sum(c['gradient_callback_seconds'] is not None for c in same)
  else:
   missing+=len(runs);changed_factory+=int(bool(candidates))
  row={'study_id':s['study_id'],'dataset':s['dataset'],'method':method,'fit_groups':len(runs),
   'factories':[{'entry':f,'file':factory_receipts[f]} for f in factories],
   'existing_same_study_method_factory_shape_receipts':[c['result'] for c in same],
   'shape_relation':'predeclared shape envelope only; mask/seed/config/runtime admission is not implied' if same else 'missing exact current-family receipt',
   'maximum_existing_random_callback_seconds':max([c['gradient_callback_seconds'] for c in same if c['gradient_callback_seconds'] is not None],default=None),
   'maximum_existing_checkpoint_proxy_bytes':max([c['checkpoint_bytes'] for c in same],default=None),
   'source_geometries':{k:sorted({json.dumps(r['config'][k],sort_keys=True) for r in runs if k in r['config']}) for k in structural},
   'workers':sorted({placement[s['study_id']+'/'+r['run_id']] for r in runs}),
   'whole_run_admitted':False,'ETA_available':False}
  methods.append(row)
  if same:continue
  # A finite root-supervised proposal, not executable scientific run or automatic
  # admission. Distinct structural branches survive; within each branch retain
  # all Pareto-maximal width/depth/head/route/FFN geometries.
  groups=collections.defaultdict(list)
  dims=('width','depth','heads','members','ffn_multiplier','stalk_dim')
  for r in runs:
   c=r['config'];law={k:c[k] for k in structural if k in c and k not in dims}
   law['factory']=r['adapter'];groups[json.dumps(law,sort_keys=True)].append(r)
  for law,variants in sorted(groups.items()):
   shapes={}
   for r in variants:
    shape=tuple(r['config'].get(k,1) if type(r['config'].get(k,1)) in (int,float) else 1 for k in dims)
    shapes.setdefault(shape,r)
   maxima=[shape for shape in shapes if not any(all(a>=b for a,b in zip(other,shape)) and other!=shape for other in shapes)]
   for shape in maxima:
    r=shapes[shape];phases=['cold_TRAIN','warm_TRAIN','TRAIN_only_serving','selected_state_serialization_restore']
    if r['config'].get('local_transition_updates',0):phases+=['last_local_stage','first_global_stage','steady_global_stage']
    if any(k in r['config'] for k in ('growth_stage','nsd_stage','label_stage','curvature_stage')):phases+=['exact_paid_parent_constructor_and_state_transition_contract']
    queue.append({'case_id':hashlib.sha256(canonical({'study':s['study_id'],'run':r['run_id'],'phases':phases})).hexdigest()[:24],
     'study_receipt':receipt(phase/spec['path']),'study_id':s['study_id'],'dataset':s['dataset'],'method':method,
     'source_run_id':r['run_id'],'factory':r['adapter'],'factory_receipt':factory_receipts[r['adapter']],
     'shape':dict(zip(dims,shape)),'structural_law':json.loads(law),'source_run':r,
     'requested_phases':phases,'root_supervision_required':True,'optimizer_kernels_denied':True,
     'real_TRAIN_graph_and_roles_required':True,'heldout_targets_forbidden':True,
     'scientifically_paid_parent_reconstruction_not_yet_available':bool(r['parents']),
     'old_random_or_synthetic_ancestor_must_not_become_scientific_evidence':True,
     'no_human_GO_or_fit_is_created':True,'automatic_execution':False})
 studies.append({'study_id':s['study_id'],'dataset':s['dataset'],'spec_hash':spec['spec_hash'],'model_fit_groups':sum(len(v) for v in by_method.values())})
assert covered+missing==fit_total
fs=os.statvfs(repo)
summary={'schema':'parameterized-current-master-resource-catalogue-v2','plan':receipt(master),'plan_hash':expected,
 'planned_model_fit_groups':fit_total,'studies':len(studies),'existing_passed_random_source_receipts':len(corrected),
 'same_study_method_factory_shape_proxy_fit_groups':covered,'missing_current_family_shape_proxy_fit_groups':missing,
 'same_method_but_changed_factory_rows':changed_factory,'corrected_callback_timings_present':sum(c['gradient_callback_seconds'] is not None for c in corrected),
 'engineering_queue_maximal_branch_cases':len(queue),'engineering_queue_is_metadata_only':True,
 'covered_current_duplicate_retention_proxy_bytes':storage,'filesystem_available_bytes_at_observation':fs.f_bavail*fs.f_frsize,
 'whole_run_memory_time_storage_admitted':False,'ETA_available':False,'scientific_fits':0,'GO':False,
 'scope_limits':['exact file/factory matching is necessary, not sufficient for a source dependency/geometry proof',
 'old shape envelopes are not fresh fits, trained buffer bounds or optimizer-kernel timing',
 'source maxima must retain branch-specific allocation laws; Pareto widths are a proposal, not a verified monotonic bound',
 'queued ancestor construction must follow actual future paid source law; prototypes never certify scientific acquisition',
 'server77 disk/GPU runtime was not accessed or admitted','all original-paper five datasets excluded']}
value={'summary':summary,'existing_random_source_receipts':corrected,'preserved_prior_failures':previous['preserved_failures'],'method_rows':methods,'study_rows':studies,'root_supervised_engineering_queue':queue}
(out/'CATALOG.json').write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps(value))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--plan-hash',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    relative=Path(a.output)
    if relative.is_absolute() or len(relative.parts)!=1 or '..' in relative.parts:raise ValueError('Fresh explicit one-component output name required')
    out=HERE/relative
    out.mkdir();(out/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
    r=invoke(REMOTE,[a.plan,a.plan_hash,a.output],out,'CATALOG',180);assert r['exit_code']==0,'Preserve metadata-catalogue failure'
    value=json.loads(r['stdout']);(out/'CATALOG.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps(value['summary']))


if __name__=='__main__':main()
