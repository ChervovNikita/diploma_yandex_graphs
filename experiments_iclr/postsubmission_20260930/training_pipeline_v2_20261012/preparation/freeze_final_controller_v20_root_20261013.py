"""Compose typed-context source policy onto v19 and test the actual pre-fit path."""
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD, invoke

OUT=Path(__file__).resolve().parent/'controller_v20_final_ports_root_20261013_v1'
REMOTE=GUARD+r'''
import ast,copy,gzip,importlib.util
def receipt(p):
 raw=p.read_bytes();return {'path':str(p.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def checked(row):
 p=phase/row['path'];assert receipt(p)==row;return p
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
base=phase/'master_controller_source_20261012_v19/MANIFEST.json';prior=json.loads(base.read_text())
packet=phase/'unresolved_source_ports_20261012_v1/typed_context_ready_v2'
entry=packet/'MASTER_INPUT_ENTRY.json'
assert receipt(entry)['sha256']=='484c0b1039193c25bb9c193853f08556db31fb2f1d5111020baacd9aa1532ecf'
helper=packet/'amend_controller.py';assert receipt(helper)['sha256']=='0e8125f031c343a16bae030eccc88816f83a6fb0c98539cdca6ea45c60767308'
stage=phase/'master_controller_source_20261013_v20';assert not stage.exists();stage.mkdir();runtime=stage/'runtime';runtime.mkdir();before={}
for row in prior['files']:
 p=checked(row);(runtime/p.name).write_bytes(p.read_bytes());before[p.name]=p.read_bytes()
m=load(helper,'root_typed_context_amendment_v20');m.install(runtime)
policy='typed_context_native_selector_stop'
# The source helper predates TABM/BSNN additions. Extend the current explicit
# policy sets additively, preserving every previously supported policy.
def extend_policy_sets(text):
 tree=ast.parse(text);lines=text.splitlines(True);edits=[]
 for n in ast.walk(tree):
  if not isinstance(n,ast.Set):continue
  values={v.value for v in n.elts if isinstance(v,ast.Constant) and isinstance(v.value,str)}
  if 'hgt_native_selector_stop' in values and policy not in values:
   assert n.lineno==n.end_lineno
   edits.append((n.lineno-1,n.end_col_offset-1))
 for line,col in sorted(edits,reverse=True):lines[line]=lines[line][:col]+', '+repr(policy)+lines[line][col:]
 return ''.join(lines)
for name in ('engine.py','spec.py','run_study.py'):
 p=runtime/name;p.write_text(extend_policy_sets(p.read_text()))
p=runtime/'run_study.py';text=p.read_text();anchor='    from native_stopping import require_completion\n';assert text.count(anchor)==1
text=text.replace(anchor,anchor+"    if run['budget']['policy']=='typed_context_native_selector_stop':\n        from typed_context_semantics import validate_typed_context_run\n        validate_typed_context_run(run)\n",1);p.write_text(text)
# Epoch counts and AMP commits are separate quantities, including native skips.
p=runtime/'engine.py';text=p.read_text();anchor='                result["actual_completed_optimizer_updates"] = result["trajectory_operation_counts"]["optimizer_updates"]\n';assert text.count(anchor)==1
text=text.replace(anchor,anchor+"                if spec.get('completion_policy')=='typed_context_native_selector_stop':\n                    result['declared_source_epoch_cap']=result.pop('declared_optimizer_update_cap')\n                    result['actual_completed_source_epochs']=result['completed_steps']\n                    result['actual_committed_optimizer_maps']=result.pop('actual_completed_optimizer_updates')\n",1);p.write_text(text)
p=runtime/'compile_master_plan.py';text=p.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONTROLLER_FILES' for t in n.targets));names=list(ast.literal_eval(node.value));names.extend(sorted(q.name for q in runtime.iterdir() if q.is_file() and q.name not in names));lines=text.splitlines(True);lines[node.lineno-1:node.end_lineno]=['CONTROLLER_FILES = '+repr(tuple(names))+'\n'];p.write_text(''.join(lines));assert len(names)==len(set(names))
def funcs(raw):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
allowed={'compile_master_plan.py':{'fit_semantics','graph_check','typed_counts','bank_roster'},'engine.py':{'_validate_spec','run_trajectory'},'run_study.py':{'execute'},'spec.py':{'validate'}};changes={}
for name,raw in before.items():
 if not name.endswith('.py'):assert (runtime/name).read_bytes()==raw;continue
 a,b=funcs(raw),funcs((runtime/name).read_bytes());changed={k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
 if changed:assert changed==allowed[name],(name,changed);changes[name]=sorted(changed)
assert set(changes)==set(allowed)
for p in runtime.glob('*.py'):compile(p.read_text(),str(p),'exec')
subprocess.run(['bash','-n',str(runtime/'start_massive.sh')],check=True)
sys.path.insert(0,str(runtime));runner=load(runtime/'run_study.py','root_final_v20_runner');engine=load(runtime/'engine.py','root_final_v20_engine')
class BoundaryReached(RuntimeError):pass
def denied_registry(*a,**kw):raise BoundaryReached('Actual execute reached denied Registry; no fit/model/import')
runner.require_approval=lambda *_:{'engineering_memory_stub':True};runner.require_phase_approval=lambda *_:None;runner.Registry=denied_registry
typed=json.loads((packet/'MASTER_INPUT_ENTRY.json').read_text());plan=json.loads((phase/typed['plan']['path']).read_text());s=json.loads(gzip.decompress((phase/plan['studies'][0]['path']).read_bytes()));proof=[]
for r in s['runs']:
 if r['kind']!='model_fit':continue
 runner.load=lambda *_,run=r:({'runs':[run]},'metadata_only')
 try:runner.execute(phase,'unused',r['run_id'],'unused','unused','unused','cpu')
 except BoundaryReached:pass
 else:raise AssertionError('Actual pre-fit path crossed denied Registry')
 e={'fit_id':r['run_id'],'budget_steps':r['budget']['maximum'],'budget_unit':r['budget']['unit'],'validation_interval':r['selection']['evaluation_interval'],'validate_initial':False,'prefixes':[],'inherited_fit_ids':r['parents'],'adapter_spec':{'run':r},'completion_policy':policy,'native_amp_cycle_policy':{'schema':'typed-context-AMP-complete-cycle-v1','optimizer_owner_attempts':1}}
 engine._validate_spec(e)
 proof.append({'run_id':r['run_id'],'actual_execute_prefit':True,'actual_engine_spec_validation':True,'registry_denied':True})
bad={'kind':'model_fit','run_id':'bad','budget':{'policy':'unknown'},'config':{}};runner.load=lambda *_:({'runs':[bad]},'metadata_only')
try:runner.execute(phase,'unused','bad','unused','unused','unused','cpu')
except ValueError as err:assert str(err)=='Unsupported completion policy'
else:raise AssertionError('Unknown policy accepted')
assert not any(name in sys.modules for name in ('torch','numpy','dgl','ogb'))
dispatch={'cases':proof,'unknown_policy_rejected':True,'numeric_imports':False,'models':0,'fits':0,'GO':False,'scope':'All45 typed fits through exact pre-fit dispatch and engine spec; no actual adapters or numerical work.'}
(stage/'TYPED_DISPATCH_PROOF.json').write_text(json.dumps(dispatch,indent=2)+'\n')
manifest=dict(prior);manifest.update(runtime_package=str(runtime.relative_to(phase)),files=[receipt(p) for p in sorted(runtime.iterdir()) if p.is_file()],prior_controller_manifest=receipt(base),typed_context_source_entry=receipt(entry),typed_context_helper=receipt(helper),typed_dispatch_proof=receipt(stage/'TYPED_DISPATCH_PROOF.json'),changed_AST_functions_against_v19=changes,new_scope='Complete cumulative ports; typed-context200 source policy/AMP/master-scaler chain added onto v19; previous special policies and bank-export gate retained.',scientific_fits=0,GO_created=False)
(stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
for p in runtime.iterdir():
 if p.is_file():p.chmod(0o444)
print(json.dumps({'manifest':manifest,'dispatch_cases':len(proof),'fits':0,'GO':False}))
'''

def main():
 OUT.mkdir();(OUT/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
 result=invoke(REMOTE,[],OUT,'FREEZE',120)
 assert result['exit_code']==0,'Retain failed composition; use a successor after repair'
 value=json.loads(result['stdout']);(OUT/'MANIFEST.json').write_text(json.dumps(value['manifest'],indent=2)+'\n')
 print(json.dumps({'runtime':value['manifest']['runtime_package'],'files':len(value['manifest']['files']),'dispatch_cases':value['dispatch_cases'],'fits':0}))

if __name__=='__main__':main()
