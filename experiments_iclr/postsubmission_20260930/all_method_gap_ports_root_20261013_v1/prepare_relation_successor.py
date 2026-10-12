"""Add source relation policies to the existing fresh literal Session cohort.

An exclusive successor reuses exactly its newly planned native controls;
historical fitted states and scores are never donors.
"""
import copy,gzip,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=HERE.parent
sys.path.insert(0,str(P/'training_pipeline_v2_20261012/preparation'))
from prepare_molecular_packets_allocation import GUARD,invoke

REMOTE=GUARD+r'''
import ast,gzip,importlib.util
out=phase/'all_method_gap_ports_root_20261013_v1';assert out.exists()
prior=phase/'unresolved_source_ports_20261012_v1/label_corrector_ready_v2'
entry=json.loads((prior/'MASTER_INPUT_ENTRY.json').read_text());plan=json.loads((phase/entry['plan']['path']).read_text());ref=plan['studies'][0]
original=json.loads(gzip.decompress((phase/ref['path']).read_bytes()));study=json.loads(json.dumps(original));oldrows={r['run_id']:r for r in study['runs']}
base=phase/'retained_prototype_batch_20261012_v1/polynormer_ready_v1/matrix/studies/retained-polynormer-mechanisms-v1-wikics.json.gz'
prototype=json.loads(gzip.decompress(base.read_bytes()));B=next(r['config']['mechanism_sources'] for r in prototype['runs'] if r['kind']=='model_fit')
def receipt(p):
 raw=p.read_bytes();return {'path':str(p.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
sem=load(out/'runtime/relation_semantics.py','root_relation_semantics')
contract=load(out/'runtime/adapters/configured_relation.py','root_relation_source_adapter').CONTRACT
runtime=str((out/'runtime').relative_to(phase));adapter=runtime+'/adapters/configured_relation.py:make_adapter';added=[];banks=[]
for seed in (6101,6203,6307):
 native=next(r for r in original['runs'] if r['kind']=='model_fit' and r['seed']==seed and r['config'].get('label_corrector_arm')=='native_single' and r['body']==0)
 endpoint=next(r for r in original['runs'] if r['kind']=='ensemble_assembly' and r['seed']==seed and r['method']=='label_only_corrector_native_single')
 for policy in sem.POLICIES:
  cfg={k:v for k,v in native['config'].items() if k in ('node_inputs','provider_graph','reservation_mib','estimated_seconds')}
  cfg.update(stage='comparative',configured_relation_policy=policy,configured_relation_source_contract=contract,mechanism_sources=B,
   relation_base_adapter='retained_prototype_batch_20261012_v1/runtime/adapters/polynormer_mechanisms.py',
   backbone='literal_publicV2_WikiCS_Session_Polynormer',architecture='internal_be',members=4,physical_slow_bodies=1,source_optimizer_count=1,
   width=512,depth=7,global_layers=2,classes=10,learning_rate=.001,weight_decay=0.,dropout=.5,local_transition_updates=100,
   initialization_seed=seed,is_original_gnnm=False,historical_state_inherited=False,fusion='mean_probabilities')
  method='source_copied_relation_credit_'+policy;rid='compare-copied_relation-'+policy+'-s'+str(seed)
  r={'run_id':rid,'kind':'model_fit','method':method,'split':0,'seed':seed,'body':0,'adapter':adapter,'config':cfg,'parents':[],
   'budget':{'unit':'optimizer_updates','policy':'full_budget','maximum':1100},'selection':{'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False}}
  sem.validate_relation_run(r);study['runs'].append(r);added.append(r)
  prefix='bank-copied_relation-'+policy+'-s'+str(seed)
  e=json.loads(json.dumps(endpoint));e.update(run_id=prefix,method=method,parents=[rid]);e['config'].update(fit_parents=[rid],fusion='mean_probabilities');study['runs'].append(e)
  previous=endpoint['run_id']
  for kind,suffix in [('calibration','-cal'),('evaluation','-evaluate')]:
   old=next(x for x in original['runs'] if x['kind']==kind and x['seed']==seed and x['method']==endpoint['method'])
   v=json.loads(json.dumps(old));v.update(run_id=prefix+suffix,method=method)
   v['parents']=[prefix if x==previous else prefix+'-cal' if x==previous+'-cal' else x for x in v['parents']]
   v['config']['export_parent']=prefix
   if kind=='evaluation':v['config']['calibration_parent']=prefix+'-cal'
   study['runs'].append(v)
  banks.append({'policy':policy,'seed':seed,'fit':rid,'bank':prefix})
study['hypothesis']+=' Four copied-native relation-credit policies retain the same current-role native controls; no old outcomes or extra HPO.'
study['comparisons'].append({'references':['label_only_corrector_native_single','label_only_corrector_genuine_native_I4','label_only_corrector_original_GNNM'],
 'candidates':['source_copied_relation_credit_'+p for p in sem.POLICIES],
 'scope':'source12 fourpolicies x3seeds; relation18=those12banks plus3nativeM1+3genuineI4; originalGNNM additional reference',
 'interpretation':'copiednative private local scorers;source1100/local100;F/J mixed block fields;no embeddingauxiliary,no optimizationnoveltyclaim'})
newfiles=[receipt(q) for q in sorted((out/'runtime').rglob('*.py'))]
bindings={r['path']:r for r in study['source']['files']}
for row in prototype['source']['files']+newfiles:
 if row['path'] in bindings:assert bindings[row['path']]==row
 bindings[row['path']]=row
study['source']['files']=[bindings[k] for k in sorted(bindings)]
sys.path.insert(0,str(phase/'master_controller_source_20261013_v22/runtime'))
from spec import canonical,fingerprint,validate
validate(study)
matrix=out/'relation_successor';assert not matrix.exists();matrix.mkdir();(matrix/'studies').mkdir();q=matrix/'studies'/(study['study_id']+'.json.gz');q.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0))
plan=json.loads(json.dumps(plan));plan.update(studies=[{'path':str(q.relative_to(phase)),'spec_hash':fingerprint(study)}],output_root=str(matrix.relative_to(phase))+'/execution')
plan['groups'][0]['run_keys']=[study['study_id']+'/'+r['run_id'] for r in study['runs']]
plan['groups'][0]['estimated_seconds']=sum(r['config']['estimated_seconds'] for r in study['runs'])
for gate in plan['global_gates'].values():gate['path']=str(matrix.relative_to(phase))+'/gate_receipts/'+Path(gate['path']).name
summary=json.loads((phase/entry['summary']['path']).read_text());summary.update(plan_hash=fingerprint(plan),relation_source_policy_count=4,relation_fit_groups_added=12,
 canonical_relation_panels=['graph_relation_private_credit_configured_relation12','graph_relation_private_credit_configured_relation18'],
 source_controls_reused_as_fresh_registered_fits=True,old_numerical_donors=False,root_controller_semantics_amendment_required=True)
summary['counts']['atomic_fit_groups']+=12;summary['counts']['banks']+=12;summary['counts']['run_records']+=48
summary['counts']['model_optimizer_maps']+=13200;summary['counts']['scalar_CAL_fits']+=12;summary['counts']['CAL_maps']+=6000;summary['counts']['evaluations']+=12
for name,value in [('PLAN.json',plan),('SUMMARY.json',summary),('RELATION_BANK_ROSTER.json',{'banks':banks,'extra_optimizer_maps':13200,'source_fullgraph_reverse_collections':211200,'source_output_cotangent_collections':26400,'existing_exact_native_controls_unchanged':True,'GO':False})]:
 (matrix/name).write_text(canonical(value)+'\n')
lineage={'prior_entry':receipt(prior/'MASTER_INPUT_ENTRY.json'),'old_model_fit_contracts_bytecanonical_unchanged':all(oldrows[r['run_id']]==r for r in study['runs'] if r['run_id'] in oldrows),
 'new_fits':12,'total_fits':39,'new_banks':12,'total_banks':33,'source_role_adaptation_disclosed':True,'exclusive_replacement_packet_id':entry['id'],'GO':False}
assert lineage['old_model_fit_contracts_bytecanonical_unchanged'];(matrix/'EXCLUSIVE_LINEAGE.json').write_text(canonical(lineage)+'\n')
newentry={'id':entry['id'],'plan':receipt(matrix/'PLAN.json'),'summary':receipt(matrix/'SUMMARY.json'),'plan_hash':fingerprint(plan),
 'coverage':entry['coverage']+[receipt(matrix/'RELATION_BANK_ROSTER.json'),receipt(matrix/'EXCLUSIVE_LINEAGE.json')]}
(matrix/'MASTER_INPUT_ENTRY.json').write_text(canonical(newentry)+'\n')
print(json.dumps({'entry':newentry,'lineage':lineage,'fit_adapter':adapter,'new_source_files':newfiles,'scientific_fits':0}))
'''

def main():
 out=HERE/'relation_prepare_02';out.mkdir();(out/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
 r=invoke(REMOTE,[],out,'PREPARE',120);assert r['exit_code']==0,'Preserve failing proposal'
 v=json.loads(r['stdout']);(out/'PREPARED_ENTRY.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'lineage':v['lineage'],'plan_hash':v['entry']['plan_hash'],'scientific_fits':0}))
if __name__=='__main__':main()
