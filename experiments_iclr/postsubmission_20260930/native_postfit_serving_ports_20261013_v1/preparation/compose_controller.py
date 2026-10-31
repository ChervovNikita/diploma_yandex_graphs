import ast,hashlib,json,shutil,sys
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent;R=P.parent;sys.path.insert(0,str(P));import amend_controller as m
base=R/'native_structural_cardinality_controller_delta_20261013_v2/runtime';out=P/'cumulative_controller/runtime';out.mkdir(parents=True,exist_ok=True);proof={}
for q in sorted(base.iterdir()):
 if q.is_file():shutil.copy2(q,out/q.name)
for name,f,allowed in [('compile_master_plan.py',m.amend_compiler,{'fit_semantics','bank_roster','graph_check','typed_counts'}),('native_stopping.py',m.amend_native,{'validate_native_run'}),('run_study.py',m.amend_run_study,{'paid_parent_ids','paid_parent_checkpoints','execute'})]:
 old=(base/name).read_text();new=f(old);assert f(new)==new
 def nodes(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
 a,b=nodes(old),nodes(new);changed={k for k in a if a[k]!=b[k]};assert changed==allowed,(name,changed);(out/name).chmod(0o644);(out/name).write_text(new);proof[name]={'base_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest(),'changed_functions':sorted(changed),'unrelated_functions_unchanged':True};(P/'runtime'/name).write_text(new)
for q in sorted(base.glob('*.py')):
 if q.name not in proof:shutil.copy2(q,P/'runtime'/q.name)
shutil.copy2(P/'runtime/postfit_semantics.py',out/'postfit_semantics.py')
manifest={'schema':'cumulative-v23-structural-postfit-controller-v1','base_manifest_sha256':hashlib.sha256((base.parent/'MANIFEST.json').read_bytes()).hexdigest(),'files':[{'path':q.name,'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in sorted(out.iterdir()) if q.is_file()],'GO':False};(out.parent/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');(P/'CUMULATIVE_CONTROLLER_DELTA_PROOF.json').write_text(json.dumps({'schema':'v23-plus-structural-postfit-AST-proof-v1','proof':proof,'unrelated_v22_v23_structural_branches_unchanged':True,'idempotent':True,'GO':False},indent=2)+'\n');print(json.dumps({'status':'composed','base':str(base),'changes':proof,'GO':False}))
