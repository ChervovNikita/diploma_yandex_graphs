import ast,copy,gzip,importlib.util,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;sys.path.insert(0,str(P/'runtime'));import compile_master_plan as c
from engine import _validate_spec
from postfit_semantics import validate_postfit_run
v=c.load_packet(R,json.loads((P/'MASTER_INPUT_ENTRY.json').read_text()));t=v['counts']['totals'];assert t['atomic_fit_run_groups']==1251 and t['served_banks']==324
fits=[r for s in v['studies'] for r in s['runs'] if r['kind']=='model_fit'];reject=0
for family,phase in [('direct','native'),('direct','head'),('affine','native'),('affine','head'),('serving','native'),('serving','head')]:
 r=next(r for r in fits if r['config']['postfit_family']==family and r['config']['postfit_phase']==phase)
 es={'fit_id':r['run_id'],'budget_steps':r['budget']['maximum'],'budget_unit':r['budget']['unit'],'validation_interval':r['selection']['evaluation_interval'],'validate_initial':False,'prefixes':[],'inherited_fit_ids':r['parents'],'adapter_spec':{'run':r}}
 if r['budget']['policy']=='native_selector_stop':es['completion_policy']='native_selector_stop'
 _validate_spec(es)
 for group,key,value in [('config','members',9),('config','learning_rate',.002),('config','physical_slow_bodies',99),('budget','maximum',17),('selection','checkpoint0',True)]:
  q=copy.deepcopy(r);q[group][key]=value
  try:validate_postfit_run(q)
  except ValueError:reject+=1
  else:raise AssertionError((family,phase,group,key,'Changed source law admitted'))
 # Changed selected donor identity must be rejected by the actual master ancestry path.
 if phase=='head':
  spec=next(s for s in v['studies'] if r in s['runs']);bad=copy.deepcopy(spec);q=next(x for x in bad['runs'] if x['run_id']==r['run_id']);wrong=next(x['run_id'] for x in bad['runs'] if x['kind']=='model_fit' and x['config']['postfit_phase']=='native' and x['seed']!=q['seed']);q['config']['lifecycle']['donor_fit_ids'][0]=wrong
  try:c.graph_check([bad],{})
  except ValueError:reject+=1
  else:raise AssertionError('Wrong source seed donor admitted')
q=P/'amend_controller.py';sp=importlib.util.spec_from_file_location('postfit_amend',q);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);proof={}
for name,fun,expected in [('native_stopping.py',m.amend_native,{'validate_native_run'}),('compile_master_plan.py',m.amend_compiler,{'fit_semantics','bank_roster','graph_check','typed_counts'}),('run_study.py',m.amend_run_study,{'paid_parent_ids','paid_parent_checkpoints','execute'})]:
 before=(P/'preparation/actual_v22_base'/name).read_text();after=fun(before);assert fun(after)==after
 def functions(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
 a,b=functions(before),functions(after);changed={k for k in a if a[k]!=b[k]};assert changed==expected,(name,changed);proof[name]={'changed_functions':sorted(changed),'unrelated_functions_unchanged':True}
report={'schema':'literal-postfit-actual-controller-ingest-v1','status':'passed','plan_hash':v['plan_hash'],'counts':v['counts'],'mutants_rejected':reject,'AST_delta':proof,'helper_idempotent':True,'model_fit_groups':len(fits),'banks':324,'numerical_imports':0,'optimizer_maps':0,'tensor_reads':0,'scientific_fits':0,'GO':False};(P/'METADATA_INGEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','fits':len(fits),'banks':324,'rejects':reject,'GO':False}))
