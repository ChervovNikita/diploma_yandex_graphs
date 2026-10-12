"""Prepare the literal six-condition hop panel and matched fresh controls."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;P=HERE.parent
sys.path.insert(0,str(P/'training_pipeline_v2_20261012/preparation'))
from prepare_molecular_packets_allocation import GUARD,invoke
REMOTE=GUARD+r'''
import copy,gzip,importlib.util
out=phase/'all_method_gap_ports_root_20261013_v1';runtime=out/'runtime'
def load(q,name):
 s=importlib.util.spec_from_file_location(name,q);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def receipt(q):
 b=q.read_bytes();return {'path':str(q.relative_to(phase)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
sem=load(runtime/'hop_complete_semantics.py','root_complete_hop_semantics');module=load(runtime/'adapters/private_hop_complete.py','root_complete_hop_adapter')
prior=phase/'remaining_fitted_label20_source_batch_20261012_v3/packets/private_hop/matrix';plan=json.loads((prior/'PLAN.json').read_text());study=json.loads(gzip.decompress((phase/plan['studies'][0]['path']).read_bytes()))
prototype=next(r for r in study['runs'] if r['kind']=='model_fit' and r['config']['family_arm']=='private_missinghop')
endpoint=next(r for r in study['runs'] if r['kind']=='ensemble_assembly');old=copy.deepcopy(study);study.update(study_id='literal-complete-private-hop-pubmed-source2000-v1',runs=[],comparisons=[],hypothesis='All six literal source HopCredit conditions, fullsource2000/p250, matched native/original and genuinely independent controls; unchanged existing broad study is separate.')
for name in study['data']['roles']:study['data']['roles'][name]={'0':study['data']['roles'][name]['0']}
study['data']['evidence'].update(split_roster=[0],source_operator_objective_stopping_fixed=True,source_exact_training_recipe=False,source_role_adaptation='Current PubMed complete graph and disjoint Planetoid-derived TRAIN/SELECT/CAL/evaluate roles replace prior source split; source policy and geometry unchanged')
rt=str(runtime.relative_to(phase));banks=[];fitcount=0
METHODS={a:'literal_hop_'+a for a in sem.CONDITIONS};METHODS.update(native_I4='literal_hop_genuine_native_I4',same_info_allview_I4='literal_hop_genuine_factor1_allview_I4')
for seed in (9101,9203,9307):
 fits={}
 def addfit(arm,body=0):
  global fitcount
  c=copy.deepcopy(prototype['config']);c.update(stage='comparative',complete_hop_arm=arm,complete_hop_source_contract=module.CONTRACT,
   hop_base_adapter='remaining_fitted_label20_source_batch_20261012_v3/runtime/adapters/private_hop_port.py',
   architecture='original_gnnm' if arm=='original_GNNM' else 'single' if arm in ('native_M1','factor1_allview') else 'internal_be',
   members=1 if arm in ('native_M1','factor1_allview') else 4,physical_slow_bodies=1,source_optimizer_count=1,learning_rate=.005,dropout=.5,
   initialization_seed=seed+1009*body,is_original_gnnm=arm=='original_GNNM',historical_state_inherited=False,
   fusion='mean_logits' if arm=='original_GNNM' else 'mean_probabilities',family_arm=arm)
  # GRAPH role is normally not split-keyed; use exact inherited model input.
  c['node_inputs']=copy.deepcopy(prototype['config']['node_inputs'])
  for k in ('train','select'):c['node_inputs'][k]=old['data']['roles'][k]['0']
  c.pop('source_arm',None)
  rid=f'compare-{arm}-s{seed}-b{body}'
  r={'run_id':rid,'kind':'model_fit','method':METHODS[arm],'split':0,'seed':seed,'body':body,'adapter':rt+'/adapters/private_hop_complete.py:make_adapter',
   'config':c,'parents':[],'budget':{'unit':'optimizer_updates','policy':sem.POLICY,'maximum':2000,'minimum':251,'patience':250},
   'selection':{'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,'patience_steps':250}}
  sem.validate_hop_run(r);study['runs'].append(r);fitcount+=1;return rid
 for arm in sem.CONDITIONS:fits[arm]=[addfit(arm)]
 fits['native_I4']=fits['native_M1']+[addfit('native_M1',b) for b in (1,2,3)]
 fits['same_info_allview_I4']=fits['factor1_allview']+[addfit('factor1_allview',b) for b in (1,2,3)]
 for arm,parents in fits.items():
  method=METHODS[arm];prefix=f'bank-{arm}-s{seed}';fusion='mean_logits' if arm=='original_GNNM' else 'mean_probabilities'
  e=copy.deepcopy(endpoint);e.update(run_id=prefix,method=method,split=0,seed=seed,parents=parents);e['config'].update(stage='bank_export',fit_parents=parents,fusion=fusion)
  if arm in ('native_I4','same_info_allview_I4'):e['config']['genuine_independent_ensemble']=True
  study['runs'].append(e)
  for kind,suffix in [('calibration','-cal'),('evaluation','-evaluate')]:
   r=copy.deepcopy(next(v for v in old['runs'] if v['kind']==kind));r.update(run_id=prefix+suffix,method=method,split=0,seed=seed,parents=[prefix] if kind=='calibration' else [prefix,prefix+'-cal']);r['config'].update(stage=kind,export_parent=prefix,fusion=fusion)
   if kind=='evaluation':r['config']['calibration_parent']=prefix+'-cal'
   study['runs'].append(r)
  banks.append({'arm':arm,'seed':seed,'fit_parents':parents,'bank':prefix})
study['comparisons']=[{'references':[METHODS[a] for a in ('native_M1','native_I4','original_GNNM','factor1_allview','same_info_allview_I4')],
 'candidates':[METHODS[a] for a in ('shared4_own','private_missinghop','full_aux','common_nonfull','allblock_missinghop')],
 'scope':'source fixedsixconditions plus native/original/same-infoI4 references;one registered mask x3source seeds;source2000/p250;noHPO',
 'interpretation':'different information/gradient recipients/route counts explicitly separated; existing broad5000/HPO private-hop study preserved'}]
files={r['path']:r for r in study['source']['files']}
for q in sorted(runtime.rglob('*.py')):files[str(q.relative_to(phase))]=receipt(q)
study['source']['files']=[files[k] for k in sorted(files)]
sys.path.insert(0,str(phase/'master_controller_source_20261013_v22/runtime'));from spec import canonical,fingerprint
matrix=out/'complete_hop_packet';assert not matrix.exists();matrix.mkdir();(matrix/'studies').mkdir();q=matrix/'studies'/(study['study_id']+'.json.gz');q.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0))
plan=copy.deepcopy(plan);g=copy.deepcopy(plan['groups'][0]);g.update(id=study['study_id'],run_keys=[study['study_id']+'/'+r['run_id'] for r in study['runs']],estimated_seconds=sum(r['config']['estimated_seconds'] for r in study['runs']));plan.update(studies=[{'path':str(q.relative_to(phase)),'spec_hash':fingerprint(study)}],groups=[g],output_root=str(matrix.relative_to(phase))+'/execution')
for gate in plan['global_gates'].values():gate['path']=str(matrix.relative_to(phase))+'/gate_receipts/'+Path(gate['path']).name
assert fitcount==42 and len(banks)==30
summary={'schema':'literal-complete-hop-source2000-p250-packet-v1','plan_hash':fingerprint(plan),'canonical_method_ids':['private_hop_native_private_hop','private_hop_private_hop_controls'],
 'counts':{'atomic_fit_groups':fitcount,'banks':len(banks),'run_records':len(study['runs']),'optimizer_map_bounds':[fitcount*251,fitcount*2000],'scalar_CAL_fit_groups':len(banks),'CAL_maps':len(banks)*500},
 'source2000_patience250_preserved':True,'same_information_allview_I4_added':True,'existing_private_hop_broad_study_unchanged':True,
 'root_controller_semantics_amendment_required':True,'scientific_fits':0,'GO':False}
for name,value in [('PLAN.json',plan),('SUMMARY.json',summary),('BANK_ROSTER.json',{'banks':banks,'genuine_I4_member0_reuse':6,'GO':False})]:(matrix/name).write_text(canonical(value)+'\n')
entry={'id':'literal_complete_private_hop_source2000','plan':receipt(matrix/'PLAN.json'),'summary':receipt(matrix/'SUMMARY.json'),'plan_hash':fingerprint(plan),'coverage':[receipt(matrix/'BANK_ROSTER.json')]}
(matrix/'MASTER_INPUT_ENTRY.json').write_text(canonical(entry)+'\n');print(json.dumps({'entry':entry,'counts':summary['counts'],'scientific_fits':0}))
'''
def main():
 out=HERE/'complete_hop_prepare_01';out.mkdir();(out/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes());r=invoke(REMOTE,[],out,'PREPARE',120);assert r['exit_code']==0,'Preserve failed source proposal'
 v=json.loads(r['stdout']);(out/'PREPARED_ENTRY.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v['counts']))
if __name__=='__main__':main()
