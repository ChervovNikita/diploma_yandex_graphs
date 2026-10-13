"""Disabled metadata compiler: M1 source horizon + full longer-track comparisons."""
from __future__ import annotations
import argparse, copy, hashlib, json, math
from collections import Counter
from pathlib import Path

WORKERS=('allocation/gpu0','server77/gpu0','server77/gpu1')
TRACKS=('native_replica','long_convergence')
RETIRED_V29='0a35daf95e7f1ad8e8df1037923a5a38638ed37e8c0a65c9dbcc3bf73f55488b'
EXCLUDED={'roman-empire','amazon-ratings','minesweeper','questions','tolokers'}

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def require(v,message):
 if not v:raise ValueError(message)
def normalized(x):return x.lower().replace('_','-').replace(' ','-')
def require_fixed(x):
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
  require(isinstance(x.get('schedule_adaptations'),list),'Explicit native schedule disposition required')
def fidelity(recipe):
 f=recipe['source_seed_fidelity']
 require(set(f)>={'status','source_global_rng_known','source_seed_policy_known','winner_dropout_known','unresolved'},'Explicit native seed/dropout fidelity required')
 require(all(type(f[k]) is bool for k in ('source_global_rng_known','source_seed_policy_known','winner_dropout_known')) and isinstance(f['unresolved'],list),'Fidelity flags must be explicit')
 full=(f['status']=='whole_paper_protocol_verified' and all(f[k] for k in ('source_global_rng_known','source_seed_policy_known','winner_dropout_known')) and not f['unresolved'])
 require(f['status']!='whole_paper_protocol_verified' or full,'Unknown source RNG/seed/winner dropout cannot claim exact whole-paper protocol')
 return f,full

def relative_cost(recipe,track,procedure):
 raw=recipe.get('fit_relative_cost_weights',{}).get(track+'/'+procedure,{'value':1.,'basis':'unmeasured equal-weight placeholder','measured':False})
 require(isinstance(raw,dict) and set(raw)>={'value','basis','measured'},'Per-fit weight needs value/basis/measured annotation')
 value=float(raw['value']);require(math.isfinite(value) and value>0 and isinstance(raw['basis'],str) and type(raw['measured']) is bool,'Invalid relative fit cost')
 return {'value':value,'basis':raw['basis'],'measured':raw['measured'],'not_ETA_seconds':True}

def compile_campaign(roster):
 require(roster.get('schema')=='reference-wrapper-roster-v2','Use paired-cohort roster v2')
 require(roster.get('scientific_launch_enabled') is False,'Preparation must remain disabled')
 require(roster.get('slots_per_gpu',2)==2,'Maximum two slots per GPU')
 require_fixed(roster);jobs=[];groups=[];recipe_ids=set();load={w:0. for w in WORKERS};banks=[]
 for recipe in roster['recipes']:
  rid=recipe['id'];require(rid not in recipe_ids,'Repeated source cell');recipe_ids.add(rid)
  require(normalized(recipe['dataset']) not in EXCLUDED,'Original paper dataset excluded')
  require(set(recipe)>={'source_bindings','native_factory','native_fixed_config','official_split_bindings','official_evaluator','native_seed_offsets','cohorts_by_track','published_native_training','tracks','wrappers','source_seed_fidelity'},'Incomplete source-faithful recipe')
  offsets=recipe['native_seed_offsets'];require(len(offsets)==4 and all(type(v) is int for v in offsets) and offsets[0]==0 and len(set(offsets))==4,'Four disjoint own-body seed offsets and exact body0 pairing required')
  require(set(recipe['tracks'])==set(TRACKS) and set(recipe['cohorts_by_track'])==set(TRACKS),'Both explicit track/cohort lists required')
  require(recipe['source_bindings'] and recipe['official_evaluator'] and recipe['official_split_bindings'],'Source/provider/evaluator metadata required')
  sf,whole_claim=fidelity(recipe)
  wrapper_ids=[w['id'] for w in recipe['wrappers']];require(len(set(wrapper_ids))==len(wrapper_ids),'Repeated wrapper condition')
  for wrapper in recipe['wrappers']:
   require(wrapper.get('tracks',['long_convergence'])==['long_convergence'],'No wrapper or I4 run in source-horizon M1 track')
   require(wrapper.get('native_operator_preservation') and wrapper.get('training_objective'),'Explicit source preservation and joint objective required')
  for track in TRACKS:
   training=recipe['tracks'][track];validate_training(training,track)
   require(track!='native_replica' or training==recipe['published_native_training'],'Source horizon reference must equal declared published training recipe')
   pairs=recipe['cohorts_by_track'][track];require(isinstance(pairs,list) and pairs,'Explicit nonempty paired split/seed list required')
   require(all(set(c)=={'split','seed'} and type(c['seed']) is int for c in pairs),'Use explicit split/seed pairs, not a Cartesian grid')
   require(len({(str(c['split']),c['seed']) for c in pairs})==len(pairs),'Repeated paired cohort')
   for pair in pairs:
    split,seed=pair['split'],pair['seed'];require(str(split) in recipe['official_split_bindings'],'Official source split metadata missing')
    cohort=f'{rid}/{track}/split-{split}/seed-{seed}';fit_ids=[]
    common={'recipe_id':rid,'dataset':recipe['dataset'],'split':split,'cohort_seed':seed,'track':track,'source_bindings':recipe['source_bindings'],'role_bindings':recipe['official_split_bindings'][str(split)],'official_evaluator':recipe['official_evaluator'],'native_config':recipe['native_fixed_config'],'training':training,'source_seed_fidelity':sf,'historical_state_inherited':False,'selected_prediction_export_in_fit':True}
    body_offsets=offsets[:1] if track=='native_replica' else offsets
    for body,offset in enumerate(body_offsets):
     jid=cohort+f'/native-body-{body}';c={**copy.deepcopy(common),'body':body,'initialization_seed':seed+offset,'factory':recipe['native_factory'],'model_procedure':'native_single','own_optimizer':True,'own_selection':True,'whole_paper_replication_claim':track=='native_replica' and whole_claim,'claim':'native source-horizon M1 reference; fidelity explicitly declared' if track=='native_replica' else 'native longer-convergence reference'}
     jobs.append({'id':jid,'kind':'fit','contract':c,'contract_hash':digest(c),'dependencies':[],'resources':copy.deepcopy(recipe.get('resources',{})),'relative_cost':relative_cost(recipe,track,'native_single')});fit_ids.append(jid)
    banks.append({'id':cohort+'/native-M1','track':track,'fit_ids':[fit_ids[0]],'selection':'own selected state','extra_fits':0})
    if track=='long_convergence':
     banks.append({'id':cohort+'/native-I4','track':track,'fit_ids':list(fit_ids),'selection':'four own selected states','extra_fits':0,'exact_M1_body0_reuse':True})
     for wrapper in recipe['wrappers']:
      jid=cohort+'/wrapper-'+wrapper['id'];c={**copy.deepcopy(common),'body':0,'initialization_seed':seed,'factory':wrapper['factory'],'model_procedure':wrapper['id'],'wrapper':wrapper,'own_optimizer':True,'own_selection':True,'whole_paper_replication_claim':False,'claim':'shared wrapper under declared longer convergence recipe'}
      jobs.append({'id':jid,'kind':'fit','contract':c,'contract_hash':digest(c),'dependencies':[],'resources':copy.deepcopy(wrapper.get('resources',recipe.get('resources',{}))),'relative_cost':relative_cost(recipe,track,wrapper['id'])});fit_ids.append(jid)
      banks.append({'id':cohort+'/wrapper-bank-'+wrapper['id'],'track':track,'fit_ids':[jid],'selection':'own selected complete shared bank','extra_fits':0})
    endpoint=cohort+'/evaluate-roster';jobs.append({'id':endpoint,'kind':'evaluate_roster','contract':{'cohort':cohort,'track':track,'official_evaluator':recipe['official_evaluator'],'whole_paper_replication_claim':track=='native_replica' and whole_claim,'all_declared_fit_outcomes_required':True,'statistical_only':True,'imports_or_executes_native_model':False,'inputs':'authenticated selected prediction banks from own fit finalization'},'dependencies':list(fit_ids),'resources':copy.deepcopy(recipe.get('endpoint_resources',{}))})
    eligible=recipe.get('allowed_workers',list(WORKERS));require(set(eligible)<=set(WORKERS) and eligible,'Allowed worker roster invalid')
    own_jobs=[j for j in jobs if j['id'] in fit_ids];weight=sum(j['relative_cost']['value'] for j in own_jobs);worker=min(eligible,key=lambda w:(load[w],WORKERS.index(w)));load[worker]+=weight
    groups.append({'id':cohort,'track':track,'worker':worker,'job_ids':fit_ids+[endpoint],'relative_planning_weight':weight,'all_fit_cost_weights_measured':all(j['relative_cost']['measured'] for j in own_jobs)})
 plan={'schema':'reference-wrapper-campaign-v2','scientific_launch_enabled':False,'human_GO_required':True,'retired_plan_hashes':[RETIRED_V29],'source_roster_hash':digest(roster),'slots_per_gpu':2,'workers':list(WORKERS),'preparation_host':'allocation','jobs':jobs,'groups':groups,'logical_banks':banks,'whole_roster_reporting':True,'automatic_retries':0,'runtime_adapter_implementation_required_before_GO':True}
 index={j['id']:j for j in jobs};fit_count=sum(j['kind']=='fit' for j in jobs)
 summary={'schema':'reference-wrapper-campaign-count-v2','plan_hash':digest(plan),'fits':fit_count,'fits_by_track':dict(Counter(j['contract']['track'] for j in jobs if j['kind']=='fit')),'statistical_evaluation_rosters':sum(j['kind']=='evaluate_roster' for j in jobs),'logical_banks':len(banks),'logical_banks_by_track':dict(Counter(b['track'] for b in banks)),'source_cells':len(recipe_ids),'cohorts_by_track':dict(Counter(g['track'] for g in groups)),'fits_per_worker':{w:sum(sum(index[jid]['kind']=='fit' for jid in g['job_ids']) for g in groups if g['worker']==w) for w in WORKERS},'relative_planning_load_per_worker':load,'method_count_per_long_cohort':{r['id']:2+len(r['wrappers']) for r in roster['recipes']},'source_track_M1_only':True,'new_fits_executed':0,'GO':False,'relative_weights_are_not_ETA':True}
 return plan,summary

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roster',required=True);p.add_argument('--output',required=True);a=p.parse_args();roster=json.loads(Path(a.roster).read_text());plan,summary=compile_campaign(roster);out=Path(a.output);require(not out.exists(),'Preserve previous proposal; use new output directory');out.mkdir(parents=True)
 for name,value in [('PLAN.json',plan),('SUMMARY.json',summary)]: (out/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(canonical(summary))
if __name__=='__main__':main()
