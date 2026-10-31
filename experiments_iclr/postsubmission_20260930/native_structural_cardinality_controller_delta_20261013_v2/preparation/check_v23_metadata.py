import copy,importlib.abc,json,sys
from pathlib import Path
S=Path(__file__).resolve().parent.parent;P=S.parent;A=P/'native_structural_cardinality_source_ports_20261013_v1';sys.path.insert(0,str(S/'runtime'))
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','numpy','pandas','scipy','torch_geometric','torch_sparse','dgl','ogb'):raise AssertionError('No numerical imports in v23 delta:'+fullname)
sys.meta_path.insert(0,Deny());import compile_master_plan as c,run_study as runner
value=c.load_packet(P,json.loads((A/'MASTER_INPUT_ENTRY.json').read_text()));assert value['counts']['model_optimizer_step_call_total']==408240;rows=[(s,r) for s in value['studies'] for r in s['runs'] if r['kind']=='model_fit'];rejected=0
for study,run in rows:
 for group,key,bad in [('budget','maximum',1),('selection','checkpoint0',True),('selection','evaluation_interval',99),('config','members',2),('config','initialization_seed',919)]:
  r=copy.deepcopy(run);r[group][key]=bad
  try:c.fit_semantics(r)
  except (ValueError,KeyError):rejected+=1
  else:raise AssertionError('Changed native/source contract admitted')
class Denied(RuntimeError):pass
class Registry:
 def __init__(self,*args,**kw):raise Denied('Registry denied')
saved={k:getattr(runner,k) for k in ('load','require_approval','require_phase_approval','Registry')};runner.require_approval=lambda *args:{};runner.require_phase_approval=lambda *args:None;runner.Registry=Registry;passed=0
try:
 for study,run in rows:
  runner.load=lambda *args,study=study:(study,'memory_only_digest')
  try:runner.execute(project_root=P,study_path='memory_only',run_id=run['run_id'],approval_path='memory_only',registry_path='denied',output_root='denied',device='denied')
  except Denied:passed+=1
  else:raise AssertionError('Did not reach denied Registry boundary')
finally:
 for k,v in saved.items():setattr(runner,k,v)
assert passed==60 and rejected==300
v={'schema':'v23-additive-sourceports-controller-metadata-proof-v1','status':'passed','base_controller':'master_controller_source_20261013_v23/runtime','actual_execute_prefit_cases':passed,'rejected_mutants':rejected,'typed_counts':value['counts'],'self_contained_helper':True,'numerical_imports':0,'Registry_objects':0,'source_arrays_decoded':0,'scientific_fits':0,'GO':False};(S/'V23_ADDITIVE_PREFIT_METADATA_QA.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'status':'passed','prefit_cases':passed,'rejected_mutants':rejected,'numerical_imports':0,'GO':False}))
