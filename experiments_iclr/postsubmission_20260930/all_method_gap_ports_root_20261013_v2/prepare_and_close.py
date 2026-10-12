"""Immutable source/metadata successors and actual denied pre-fit closure."""
import base64,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;P=HERE.parent
sys.path.insert(0,str(P/'training_pipeline_v2_20261012/preparation'))
from prepare_molecular_packets_allocation import GUARD,invoke

REMOTE=GUARD+r'''
import ast,copy,gzip,importlib.util
payload=json.loads(base64.b64decode(sys.argv[1]));stage=phase/'all_method_gap_ports_root_20261013_v2';assert not stage.exists();stage.mkdir()
def receipt(p):
 b=p.read_bytes();return {'path':str(p.relative_to(phase)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
for row in payload:
 p=stage/row['path'];assert not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
 b=base64.b64decode(row['data']);assert hashlib.sha256(b).hexdigest()==row['sha256'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
base=phase/'master_controller_source_20261013_v22/MANIFEST.json';prior=json.loads(base.read_text());controller=phase/'master_controller_source_20261013_v23';assert not controller.exists();controller.mkdir();runtime=controller/'runtime';runtime.mkdir();before={}
for row in prior['files']:
 p=phase/row['path'];assert receipt(p)==row;(runtime/p.name).write_bytes(p.read_bytes());before[p.name]=p.read_bytes()
load(stage/'amend_controller.py','root_gap_amend').install(runtime)
def functions(raw):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
changes={};allowed={'compile_master_plan.py':{'fit_semantics','typed_counts'},'spec.py':{'validate'},'native_stopping.py':{'require_completion'},'run_study.py':{'execute'},'engine.py':{'_validate_spec','run_trajectory'}}
for name,b in before.items():
 if name.endswith('.py'):
  a,c=functions(b),functions((runtime/name).read_bytes());d={k for k in a.keys()|c.keys() if a.get(k)!=c.get(k)}
  if d:assert d==allowed[name],(name,d);changes[name]=sorted(d)
 else:assert (runtime/name).read_bytes()==b
assert set(changes)==set(allowed)
sys.path.insert(0,str(runtime));import spec,compile_master_plan as master
rt=str((stage/'runtime').relative_to(phase));source_new=[receipt(p) for p in sorted((stage/'runtime').rglob('*.py'))]
def successor(entry,out,change):
 plan=json.loads((phase/entry['plan']['path']).read_text());studies=[];out.mkdir();(out/'studies').mkdir()
 for ref in plan['studies']:
  s=json.loads(gzip.decompress((phase/ref['path']).read_bytes()));change(s)
  files={r['path']:r for r in s['source']['files']};files.update({r['path']:r for r in source_new});s['source']['files']=[files[k] for k in sorted(files)]
  spec.validate(s);q=out/'studies'/(s['study_id']+'.json.gz');q.write_bytes(gzip.compress((spec.canonical(s)+'\n').encode(),mtime=0));studies.append((s,q))
 plan['studies']=[{'path':str(q.relative_to(phase)),'spec_hash':spec.fingerprint(s)} for s,q in studies];plan['output_root']=str(out.relative_to(phase))+'/execution'
 for gate in plan['global_gates'].values():gate['path']=str(out.relative_to(phase))+'/gate_receipts/'+Path(gate['path']).name
 summary=json.loads((phase/entry['summary']['path']).read_text());summary.update(plan_hash=spec.fingerprint(plan),exact_actual_controller_counts=master.typed_counts([s for s,q in studies]),scientific_fits=0,GO=False)
 for name,v in [('PLAN.json',plan),('SUMMARY.json',summary)]:(out/name).write_text(spec.canonical(v)+'\n')
 lineage={'prior_entry':entry,'new_entry_same_id':True,'source_numerical_operators_unchanged':True,'source_capture_serialization_amended':True,'scientific_fits':0,'GO':False}
 (out/'LINEAGE.json').write_text(spec.canonical(lineage)+'\n')
 new=dict(entry,plan=receipt(out/'PLAN.json'),summary=receipt(out/'SUMMARY.json'),plan_hash=spec.fingerprint(plan));new['coverage']=entry.get('coverage',[])+[receipt(out/'LINEAGE.json')]
 (out/'MASTER_INPUT_ENTRY.json').write_text(spec.canonical(new)+'\n');return new,studies

relation=json.loads((phase/'all_method_gap_ports_root_20261013_v1/relation_successor/MASTER_INPUT_ENTRY.json').read_text())
def relation_change(s):
 for r in s['runs']:
  if 'configured_relation_source_contract' in r['config']:
   r['adapter']=rt+'/adapters/configured_relation.py:make_adapter';r['config']['plain_numpy_helper']=rt+'/plain_numpy_state.py'
 s['source']['description']=s['source'].get('description','')+' Complete copied-native relation-credit plus exact plain NumPy RNG restoration.'
entry_relation,relation_studies=successor(relation,stage/'relation_successor',relation_change)
q=stage/'relation_successor/SUMMARY.json';v=json.loads(q.read_text());v['counts']['additional_relation_work']={'optimizer_maps':13200,'member_reverse_collections':211200,'output_cotangent_collections':26400};q.write_text(spec.canonical(v)+'\n');entry_relation['summary']=receipt(q);(q.parent/'MASTER_INPUT_ENTRY.json').write_text(spec.canonical(entry_relation)+'\n')

hop=json.loads((phase/'all_method_gap_ports_root_20261013_v1/complete_hop_packet/MASTER_INPUT_ENTRY.json').read_text())
def hop_change(s):
 for r in s['runs']:
  if r['kind']!='model_fit':continue
  r['adapter']=rt+'/adapters/private_hop_complete.py:make_adapter';c=r['config']
  for k in ('full_budget_extension','wrapper_arm','source_only_independent_new_data','source_feature_normalization_gate'):c.pop(k,None)
  c['source_contract']=copy.deepcopy(c['complete_hop_source_contract'])
  c['source_native_recipe']='Literal source PubMed monoK2/H256/L2/heads8/FFN32/q1.6/multi2/dprate.8;dropout.5;Adam base.005/attention.0005,decays.001/1e-8;fixed source2000/p250'
  c['source_optimizer_constructor_count']=2 if c['complete_hop_arm']=='factor1_allview' else 1
  c['source_controller']={'cap':2000,'patience':250,'policy':'hop_complete_native_selector_stop','source_old5000_HPO_claim_removed':True}
entry_hop,hop_studies=successor(hop,stage/'complete_hop_packet',hop_change)

index=json.loads((phase/'practical_master_inputs_20261013_v24/INPUT_INDEX.json').read_text());old=next(e for e in index['packets'] if 'single8' in e['plan']['path'] and 'successor' in e['plan']['path'])
def plain_change(s):
 for r in s['runs']:
  if r['kind']=='model_fit' and r['adapter'].split(':')[0].endswith(('/polynormer_mechanisms.py','/single8_attribution.py')):
   r['config']['original_plain_state_adapter']=r['adapter'];r['config']['plain_numpy_helper']=rt+'/plain_numpy_state.py';r['adapter']=rt+'/adapters/plain_state_wrapper.py:make_adapter'
 s['source']['description']=s['source'].get('description','')+' Plain checkpoint NumPy RNG representation; source operator and stopping unchanged.'
entry_plain,plain_studies=successor(old,stage/'plain_state_exclusive_successor',plain_change)
entries=[entry_relation,entry_hop,entry_plain];all_studies=[s for s,q in relation_studies+hop_studies+plain_studies]
runner=load(runtime/'run_study.py','root_gap_v23_runner');engine=load(runtime/'engine.py','root_gap_v23_engine')
class BoundaryReached(RuntimeError):pass
runner.require_approval=lambda *_:{'engineering_metadata_only':True};runner.require_phase_approval=lambda *_:None
runner.Registry=lambda *_:(_ for _ in ()).throw(BoundaryReached())
proof=[];mutants=0
for s in all_studies:
 for r in s['runs']:
  if r['kind']!='model_fit':continue
  runner.load=lambda *_,study=s: (study,'metadata_only')
  try:runner.execute(phase,'unused',r['run_id'],'unused','unused','unused','cpu')
  except BoundaryReached:pass
  else:raise AssertionError('Registry boundary not reached')
  sem=master.fit_semantics(r)
  if 'complete_hop_source_contract' in r['config'] or 'configured_relation_source_contract' in r['config']:
   e={'fit_id':r['run_id'],'budget_steps':r['budget']['maximum'],'budget_unit':r['budget']['unit'],'validation_interval':1,'validate_initial':False,'prefixes':[],'inherited_fit_ids':[],'adapter_spec':{'run':r},'completion_policy':r['budget']['policy']};engine._validate_spec(e)
   for mutate in ('maximum','dropout','fusion','initialization_seed','body'):
    bad=copy.deepcopy(r)
    if mutate=='maximum':bad['budget']['maximum']+=1
    elif mutate=='body':bad['body']=99
    else:bad['config'][mutate]=99
    try:master.fit_semantics(bad)
    except ValueError:mutants+=1
    else:raise AssertionError('Mutant admitted '+mutate)
  proof.append({'study':s['study_id'],'run_id':r['run_id'],'actual_prefit_denied_registry':True,'semantics':sem})
assert not any(n in sys.modules for n in ('torch','numpy','dgl','ogb'))
dispatch={'cases':len(proof),'rejected_mutants':mutants,'numerical_imports':0,'models':0,'scientific_fits':0,'GO':False,'per_fit':proof};(controller/'ROOT_SOURCE_DISPATCH.json').write_text(spec.canonical(dispatch)+'\n')
manifest=dict(prior,runtime_package=str(runtime.relative_to(phase)),files=[receipt(p) for p in sorted(runtime.iterdir()) if p.is_file()],prior_controller_manifest=receipt(base),root_source_amendment=receipt(stage/'amend_controller.py'),root_dispatch_proof=receipt(controller/'ROOT_SOURCE_DISPATCH.json'),changed_AST_functions_against_v22=changes,scientific_fits=0,GO_created=False)
(controller/'MANIFEST.json').write_text(spec.canonical(manifest)+'\n');(stage/'MASTER_INPUT_ENTRIES.json').write_text(spec.canonical(entries)+'\n')
for p in runtime.iterdir():
 if p.is_file():p.chmod(0o444)
print(json.dumps({'entries':entries,'controller_manifest':manifest,'dispatch_cases':len(proof),'rejected_mutants':mutants,'fits':0,'GO':False}))
'''

def main():
 out=HERE/'prepare_closure_01';out.mkdir();(out/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes());payload=[]
 for q in sorted(HERE.rglob('*.py')):
  if q.name=='prepare_and_close.py' or 'prepare_closure_01' in q.parts:continue
  b=q.read_bytes();payload.append({'path':str(q.relative_to(HERE)),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()})
 r=invoke(REMOTE,[base64.b64encode(json.dumps(payload).encode()).decode()],out,'PREPARE',180);assert r['exit_code']==0,'Retain failure;successor only'
 v=json.loads(r['stdout']);(out/'RESULT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v[k] for k in ('dispatch_cases','rejected_mutants','fits','GO')}))
if __name__=='__main__':main()
