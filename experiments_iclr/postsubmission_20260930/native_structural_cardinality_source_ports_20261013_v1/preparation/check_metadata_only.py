"""Actual frozen metadata and pre-Registry dispatch; numerical imports denied."""
import copy,importlib.abc,json,sys
from pathlib import Path
S=Path(__file__).resolve().parent.parent;P=S.parent;sys.path.insert(0,str(S/'runtime'))
class NumericalDeny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','numpy','pandas','scipy','torch_geometric','torch_sparse','dgl','ogb'):raise AssertionError('Numerical import forbidden in metadata proof:'+fullname)
sys.meta_path.insert(0,NumericalDeny());import compile_master_plan as c,run_study as runner
entry=json.loads((S/'MASTER_INPUT_ENTRY.json').read_text());value=c.load_packet(P,entry);counts=value['counts'];assert counts['totals']['atomic_fit_run_groups']==60 and counts['totals']['served_banks']==51 and counts['model_optimizer_step_call_total']==408240;assert counts['totals']['genuine_separately_selected_I4_banks']==6
rows=[(study,row) for study in value['studies'] for row in study['runs'] if row['kind']=='model_fit'];rejects=0
for _,run in rows:
 for group,key,bad in [('budget','maximum',1),('selection','checkpoint0',True),('selection','evaluation_interval',99),('config','members',2),('config','initialization_seed',919)]:
  clone=copy.deepcopy(run);clone[group][key]=bad
  try:c.fit_semantics(clone)
  except (ValueError,KeyError):rejects+=1
  else:raise AssertionError('Changed exact source procedure admitted:'+run['run_id']+'/'+key)
class RegistryDenied(RuntimeError):pass
class DeniedRegistry:
 def __init__(self,*args,**kwargs):raise RegistryDenied('Reached actual execute prefit boundary')
saved={name:getattr(runner,name) for name in ('load','require_approval','require_phase_approval','Registry')};runner.require_approval=lambda *args:{};runner.require_phase_approval=lambda *args:None;runner.Registry=DeniedRegistry;passed=[]
try:
 for study,run in rows:
  runner.load=lambda *args,study=study:(study,'synthetic_memory_only_metadata_digest')
  try:runner.execute(project_root=P,study_path='memory-only',run_id=run['run_id'],approval_path='memory-only',registry_path='DENIED',output_root='DENIED',device='DENIED')
  except RegistryDenied:passed.append(run['run_id'])
  else:raise AssertionError('Actual execute failed to reach denied Registry boundary')
finally:
 for name,old in saved.items():setattr(runner,name,old)
report={'schema':'literal-sourceports-actual-controller-typed-and-prefit-proof-v1','status':'passed','counts':counts,'actual_execute_prefit_cases':len(passed),'rejected_mutant_source_procedures':rejects,'registry_objects_created':0,'numerical_imports':0,'actual_input_arrays_opened':0,'model_constructors':0,'optimizer_objects':0,'scientific_fits':0,'GO_receipts_created':0,'parent_modified':False};(S/'METADATA_AND_EXECUTE_PREFIT_QA.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','prefit_cases':len(passed),'rejected_mutants':rejects,'numerical_imports':0,'GO':False}))
