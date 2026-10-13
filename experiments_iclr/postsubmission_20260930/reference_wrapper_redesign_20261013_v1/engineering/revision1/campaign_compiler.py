"""Metadata-only reference/wrapper campaign compiler. Never imports learning code."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path

WORKERS=('allocation/gpu0','server77/gpu0','server77/gpu1')
RETIRED_V29='0a35daf95e7f1ad8e8df1037923a5a38638ed37e8c0a65c9dbcc3bf73f55488b'
EXCLUDED={'roman-empire','amazon-ratings','minesweeper','questions','tolokers'}

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def require(v,message):
 if not v:raise ValueError(message)
def normalized(x):return x.lower().replace('_','-').replace(' ','-')
def require_fixed(x):
 """Reject old adaptive selectors/grids, without banning native list-valued settings."""
 if isinstance(x,dict):
  require(not ({'selected_config_from','search_space','hpo_domain','best_config','tuning_grid'}&set(x)), 'No HPO/adaptive config dependency')
  for value in x.values():require_fixed(value)
 elif isinstance(x,list):
  for value in x:require_fixed(value)
def validate_training(x,track):
 require(set(x)>={'unit','maximum','optimizer','selection','native_schedule'},'Complete fixed training contract required')
 require(x['unit'] in ('epochs','optimizer_updates') and type(x['maximum']) is int and x['maximum']>0,'Finite source-native clock/cap required')
 require_fixed(x)
 if track=='long_convergence':
  stop=x.get('stopping')
  require(isinstance(stop,dict) and set(stop)>={'minimum_units','patience_validation_checks','required_gates'},'Explicit long-convergence stopping required')
  require(0<=stop['minimum_units']<x['maximum'] and stop['patience_validation_checks']>0,'Nonempty convergence stopping interval')
  require(isinstance(x.get('schedule_adaptations'),list),'Explicit unchanged/extended native schedule disposition required')

def compile_campaign(roster):
 require(roster.get('schema')=='reference-wrapper-roster-v1','Unexpected source roster schema')
 require(roster.get('scientific_launch_enabled') is False,'Preparation must remain disabled')
 require(roster.get('slots_per_gpu',2)==2,'Exactly maximum two slots per GPU')
 require_fixed(roster)
 jobs=[];groups=[];recipe_ids=set();load={w:0. for w in WORKERS};logical_banks=[]
 for recipe in roster['recipes']:
  rid=recipe['id'];require(rid not in recipe_ids,'Repeated recipe id');recipe_ids.add(rid)
  require(normalized(recipe['dataset']) not in EXCLUDED,'Original paper dataset excluded')
  require(set(recipe)>={'source_bindings','native_factory','native_fixed_config','official_split_bindings','official_evaluator','native_seed_offsets','seeds','split_ids','published_native_training','tracks','wrappers'},'Incomplete source-faithful recipe')
  require(all(type(v) is int for v in recipe['native_seed_offsets']) and all(type(v) is int for v in recipe['seeds']),'Explicit integer source seed policy required')
  require(len(recipe['native_seed_offsets'])==4 and recipe['native_seed_offsets'][0]==0 and len(set(recipe['native_seed_offsets']))==4,'Four distinct independent-body seed offsets, paired body0 required')
  require(len(set(recipe['seeds']))==len(recipe['seeds']) and len(set(recipe['split_ids']))==len(recipe['split_ids']),'Unique finite seed/split rosters required')
  require(recipe['source_bindings'] and recipe['official_evaluator'] and recipe['official_split_bindings'],'Source/provider/evaluator metadata required')
  require(set(recipe['tracks'])=={'native_replica','long_convergence'},'Separate strict and long tracks required')
  for track,training in recipe['tracks'].items():
   validate_training(training,track)
   require(track!='native_replica' or training==recipe['published_native_training'],'Strict replica must equal frozen published training contract')
   for split in recipe['split_ids']:
    require(str(split) in recipe['official_split_bindings'],'Official source split metadata missing')
    for seed in recipe['seeds']:
     cohort=f'{rid}/{track}/split-{split}/seed-{seed}';fit_ids=[]
     common={'recipe_id':rid,'dataset':recipe['dataset'],'split':split,'cohort_seed':seed,'track':track,'source_bindings':recipe['source_bindings'],'role_bindings':recipe['official_split_bindings'][str(split)],'official_evaluator':recipe['official_evaluator'],'native_config':recipe['native_fixed_config'],'training':training,'historical_state_inherited':False}
     for body,offset in enumerate(recipe['native_seed_offsets']):
      jid=cohort+f'/native-body-{body}';c={**copy.deepcopy(common),'body':body,'initialization_seed':seed+offset,'factory':recipe['native_factory'],'model_procedure':'native_single','own_optimizer':True,'own_selection':True,'source_replication_claim':track=='native_replica'}
      jobs.append({'id':jid,'kind':'fit','contract':c,'contract_hash':digest(c),'dependencies':[],'resources':copy.deepcopy(recipe.get('resources',{}))});fit_ids.append(jid)
     # M1 is the exact native-body0 authority, not a fifth independently fitted model.
     logical_banks.extend([{'id':cohort+'/native-M1','fit_ids':[fit_ids[0]],'selection':'own selected state','extra_fits':0},{'id':cohort+'/native-I4','fit_ids':list(fit_ids),'selection':'four own selected states','extra_fits':0,'exact_M1_body0_reuse':True}])
     for wrapper in recipe['wrappers']:
      if track not in wrapper['tracks']:continue
      require(wrapper.get('native_operator_preservation') and wrapper.get('training_objective'),'Explicit wrapper delta and source objective required')
      jid=cohort+'/wrapper-'+wrapper['id'];c={**copy.deepcopy(common),'body':0,'initialization_seed':seed,'factory':wrapper['factory'],'model_procedure':wrapper['id'],'wrapper':wrapper,'own_optimizer':True,'own_selection':True,'source_replication_claim':False,'claim':'new wrapper comparison under source recipe' if track=='native_replica' else 'new wrapper comparison under declared longer convergence recipe'}
      jobs.append({'id':jid,'kind':'fit','contract':c,'contract_hash':digest(c),'dependencies':[],'resources':copy.deepcopy(recipe.get('resources',{}))});fit_ids.append(jid)
      logical_banks.append({'id':cohort+'/wrapper-bank-'+wrapper['id'],'fit_ids':[jid],'selection':'own selected complete shared bank','extra_fits':0})
     endpoint=cohort+'/evaluate-roster';jobs.append({'id':endpoint,'kind':'evaluate_roster','contract':{'cohort':cohort,'official_evaluator':recipe['official_evaluator'],'native_replication_claim':track=='native_replica','all_declared_fit_outcomes_required':True},'dependencies':list(fit_ids),'resources':copy.deepcopy(recipe.get('endpoint_resources',{}))})
     eligible=recipe.get('allowed_workers',list(WORKERS));require(set(eligible)<=set(WORKERS) and eligible,'Allowed worker roster invalid')
     weight=float(recipe.get('planning_seconds_per_fit',1.))*len(fit_ids);require(weight>0,'Positive planning weight required')
     worker=min(eligible,key=lambda w:(load[w],WORKERS.index(w)));load[worker]+=weight
     groups.append({'id':cohort,'worker':worker,'job_ids':fit_ids+[endpoint],'planning_weight':weight,'planning_weight_is_measured':recipe.get('planning_weight_is_measured',False)})
 plan={'schema':'reference-wrapper-campaign-v1','scientific_launch_enabled':False,'human_GO_required':True,'retired_plan_hashes':[RETIRED_V29],'source_roster_hash':digest(roster),'slots_per_gpu':2,'workers':list(WORKERS),'preparation_host':'allocation','jobs':jobs,'groups':groups,'logical_banks':logical_banks,'whole_roster_reporting':True,'automatic_retries':0,'runtime_adapter_implementation_required_before_GO':True}
 fit_count=sum(j['kind']=='fit' for j in jobs);summary={'schema':'reference-wrapper-campaign-count-v1','plan_hash':digest(plan),'fits':fit_count,'evaluation_rosters':sum(j['kind']=='evaluate_roster' for j in jobs),'logical_banks':len(logical_banks),'recipes':len(recipe_ids),'fits_per_worker':{w:sum(sum(next(j for j in jobs if j['id']==jid)['kind']=='fit' for jid in g['job_ids']) for g in groups if g['worker']==w) for w in WORKERS},'new_fits_executed':0,'GO':False,'scheduler_weights_are_not_ETA':True}
 return plan,summary

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roster',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 roster=json.loads(Path(a.roster).read_text());plan,summary=compile_campaign(roster);out=Path(a.output);require(not out.exists(),'Preserve previous proposal; use new output directory');out.mkdir(parents=True)
 for name,value in [('PLAN.json',plan),('SUMMARY.json',summary)]: (out/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(canonical(summary))
if __name__=='__main__':main()
