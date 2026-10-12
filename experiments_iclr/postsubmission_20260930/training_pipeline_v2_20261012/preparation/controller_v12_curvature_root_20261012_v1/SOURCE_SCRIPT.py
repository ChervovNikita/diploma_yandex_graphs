"""Freeze the qualified curvature composition on exact v11 source bytes."""
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD,invoke
OUT=Path(__file__).resolve().parent/'controller_v12_curvature_root_20261012_v1'
REMOTE=GUARD+r'''
import ast
prior_path=phase/'master_controller_source_20261012_v11/MANIFEST.json'
prior=json.loads(prior_path.read_text())
base=phase/'retained_prototype_batch_20261012_v1/curvature_controller_v11_composition_v1'
handoff=json.loads((base/'ROOT_MERGE_HANDOFF.json').read_text())
assert handoff['scientific_fits']==0 and handoff['GO'] is False and handoff['NSD_metadata_preserved_groups']==2886
stage=phase/'master_controller_source_20261012_v12';assert not stage.exists();stage.mkdir();runtime=stage/'runtime';runtime.mkdir()
def receipt(p):
 raw=p.read_bytes();return {'path':str(p.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def functions(raw):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
for row in prior['files']:
 p=phase/row['path'];assert receipt(p)==row;(runtime/p.name).write_bytes(p.read_bytes())
changes={}
for row in handoff['v11_composed_files']:
 source=phase/row['path'];assert receipt(source)==row;target=runtime/source.name
 before,after=functions(target.read_bytes()),functions(source.read_bytes())
 altered=sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k))
 expected=['fit_semantics','graph_check','typed_counts'] if source.name=='compile_master_plan.py' else ['paid_parent_checkpoints','paid_parent_ids','require_curvature_paid_parent']
 assert altered==sorted(expected),(source.name,altered)
 target.write_bytes(source.read_bytes());changes[source.name]=altered
assert receipt(phase/handoff['v11_engine_exact']['path'])==handoff['v11_engine_exact']
assert hashlib.sha256((runtime/'engine.py').read_bytes()).hexdigest()==handoff['v11_engine_exact']['sha256']
for p in runtime.glob('*.py'):compile(p.read_text(),str(p),'exec')
subprocess.run(['bash','-n',str(runtime/'start_massive.sh')],check=True)
manifest=dict(prior)
manifest.pop('native_learning_and_standardized_model_engine_changed',None)
manifest.update(runtime_package=str(runtime.relative_to(phase)),files=[receipt(p) for p in sorted(runtime.iterdir()) if p.is_file()],
 prior_controller_manifest=receipt(prior_path),changed_existing_files=sorted(changes),
 unchanged_controller_files=[r for r in prior['files'] if Path(r['path']).name not in changes],
 curvature_composition_handoff=receipt(base/'ROOT_MERGE_HANDOFF.json'),changed_AST_functions=changes,
 new_scope='Paid curvature warm/trial/continuation ancestry and bounded trial-cost typing; v11 NSD and lossless shared checkpoint storage preserved',
 native_learning_laws_changed=False,standard_engine_changed_in_this_successor=False,
 standard_engine_diff_from_v10='v11 inactive fast-factor ownership, terminal-training receipt and lossless identical-view clone; unchanged in v12',
 ambiguous_inherited_combined_engine_flag_removed=True,scientific_fits=0,benchmark_scores=0,start_commands_are_disabled_until_GO=True)
(stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
for p in runtime.iterdir():
 if p.is_file():p.chmod(0o444)
print(json.dumps({'manifest':manifest,'scientific_fits':0,'GO':False}))
'''
def main():
    OUT.mkdir();(OUT/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
    r=invoke(REMOTE,[],OUT,'FREEZE',90);assert r['exit_code']==0,'Preserve freeze failure'
    v=json.loads(r['stdout']);(OUT/'MANIFEST.json').write_text(json.dumps(v['manifest'],indent=2)+'\n')
    print(json.dumps({'controller':'v12','files':len(v['manifest']['files']),'GO':False}))
if __name__=='__main__':main()
