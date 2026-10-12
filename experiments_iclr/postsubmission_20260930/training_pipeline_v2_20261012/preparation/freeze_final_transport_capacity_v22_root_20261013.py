"""Expand the bounded digest-only host receipt tree for the new export cohort."""
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD,invoke
OUT=Path(__file__).resolve().parent/'controller_v22_transport_capacity_root_20261013_v1'
REMOTE=GUARD+r'''
import ast
def receipt(p):
 raw=p.read_bytes();return {'path':str(p.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
base=phase/'master_controller_source_20261013_v21/MANIFEST.json';prior=json.loads(base.read_text());stage=phase/'master_controller_source_20261013_v22';assert not stage.exists();stage.mkdir();runtime=stage/'runtime';runtime.mkdir()
for row in prior['files']:
 p=phase/row['path'];assert receipt(p)==row;(runtime/p.name).write_bytes(p.read_bytes())
p=runtime/'phase_rendezvous.py';text=p.read_text();old='MAX_COMPRESSED_BYTES = 32 * 1024**2';assert text.count(old)==1;text=text.replace(old,'MAX_COMPRESSED_BYTES = 64 * 1024**2',1);p.write_text(text)
oldtree=ast.parse((phase/'master_controller_source_20261013_v21/runtime/phase_rendezvous.py').read_text());newtree=ast.parse(text)
oldfuncs={n.name:ast.dump(n,include_attributes=False) for n in oldtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};newfuncs={n.name:ast.dump(n,include_attributes=False) for n in newtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};assert oldfuncs==newfuncs
limits=dict(prior['phase_rendezvous']);assert limits['max_compressed_host_bytes']==33554432 and limits['max_uncompressed_proof_bytes']==67108864;limits['max_compressed_host_bytes']=67108864
for q in runtime.glob('*.py'):compile(q.read_text(),str(q),'exec')
proof={'scope':'Bounded hash-only receipt tree gains bank-export closure; raw single-proof limit unchanged64MiB; compressed host-tree cap64MiB. Final exact capacity report still required. No model, role, prediction or credential payload admitted.', 'prior_host_tree_cap_bytes':33554432,'host_tree_cap_bytes':67108864,'single_raw_proof_cap_bytes':67108864,'functions_unchanged':True,'source_learning_and_fit_counts_unchanged':True,'fits':0,'GO':False}
(stage/'TRANSPORT_CAPACITY_DECISION.json').write_text(json.dumps(proof,indent=2)+'\n')
manifest=dict(prior);manifest.update(runtime_package=str(runtime.relative_to(phase)),files=[receipt(q) for q in sorted(runtime.iterdir()) if q.is_file()],prior_controller_manifest=receipt(base),phase_rendezvous=limits,transport_capacity_decision=receipt(stage/'TRANSPORT_CAPACITY_DECISION.json'),changed_AST_functions_against_v21={},scientific_fits=0,GO_created=False)
(stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
for q in runtime.iterdir():
 if q.is_file():q.chmod(0o444)
print(json.dumps({'manifest':manifest,'proof':proof}))
'''
def main():
 OUT.mkdir();(OUT/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes());r=invoke(REMOTE,[],OUT,'FREEZE',90);assert r['exit_code']==0
 v=json.loads(r['stdout']);(OUT/'MANIFEST.json').write_text(json.dumps(v['manifest'],indent=2)+'\n');print(json.dumps({'runtime':v['manifest']['runtime_package'],'host_tree_cap_bytes':v['proof']['host_tree_cap_bytes']}))
if __name__=='__main__':main()
