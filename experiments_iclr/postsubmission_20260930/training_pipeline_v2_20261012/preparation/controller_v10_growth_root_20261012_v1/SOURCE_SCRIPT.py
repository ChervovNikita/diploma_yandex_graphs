"""Freeze the exact qualified growth delta without changing learning or startup."""
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD, invoke

OUT = Path(__file__).resolve().parent / 'controller_v10_growth_root_20261012_v1'
REMOTE = GUARD + r'''
import ast
prior_path=phase/'master_controller_source_20261012_v9/MANIFEST.json'
prior=json.loads(prior_path.read_text())
amendment=phase/'preaggregation_growth_controller_v9_amendment_20261012_v1'
handoff=json.loads((amendment/'ALLOCATION_HANDOFF.json').read_text())
qa=json.loads((amendment/'REMOTE_METADATA_QA_RESULT.json').read_text())
assert handoff['metadata_QA_checks_passed']==38 and handoff['GO'] is False
stage=phase/'master_controller_source_20261012_v10';assert not stage.exists();stage.mkdir()
runtime=stage/'runtime';runtime.mkdir()
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
for row in prior['files']:
 p=phase/row['path'];assert receipt(p)==row
 (runtime/p.name).write_bytes(p.read_bytes())
changes={}
for row in handoff['exact_base_files']:
 target=runtime/row['file'];old=target.read_bytes()
 assert hashlib.sha256(old).hexdigest()==row['frozen_v9_sha256']
 successor=amendment/'runtime'/row['file'];raw=successor.read_bytes()
 assert hashlib.sha256(raw).hexdigest()==row['successor_sha256'] and len(raw)==row['bytes']
 def functions(value):
  return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(value).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
 before,after=functions(old),functions(raw)
 altered=sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k))
 assert altered==sorted(row['changed_AST_functions']),(row['file'],altered)
 changes[row['file']]=altered;target.write_bytes(raw)
for path in runtime.glob('*.py'):compile(path.read_text(),str(path),'exec')
subprocess.run(['bash','-n',str(runtime/'start_massive.sh')],check=True)
manifest=dict(prior)
manifest.update(runtime_package=str(runtime.relative_to(phase)),files=[receipt(p) for p in sorted(runtime.iterdir()) if p.is_file()],
 prior_controller_manifest=receipt(prior_path),changed_existing_files=sorted(changes),
 unchanged_controller_files=[r for r in prior['files'] if Path(r['path']).name not in changes],
 growth_exact_amendment=receipt(amendment/'ALLOCATION_HANDOFF.json'),growth_metadata_QA=receipt(amendment/'REMOTE_METADATA_QA_RESULT.json'),
 changed_AST_functions=changes,new_scope='Paid preaggregation-growth ancestry and exact optimizer-cost typing; unchanged native learning, startup, rendezvous and standard engine',
 passed_growth_metadata_checks=38,scientific_fits=0,benchmark_scores=0,start_commands_are_disabled_until_GO=True)
(stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
for path in runtime.iterdir():
 if path.is_file():path.chmod(0o444)
print(json.dumps({'manifest':manifest,'GO':False,'scientific_fits':0}))
'''

def main():
    OUT.mkdir()
    (OUT/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
    result=invoke(REMOTE,[],OUT,'FREEZE',90)
    assert result['exit_code']==0,'Preserve freeze failure'
    value=json.loads(result['stdout'])
    (OUT/'MANIFEST.json').write_text(json.dumps(value['manifest'],indent=2)+'\n')
    print(json.dumps({'controller':'v10','files':len(value['manifest']['files']),'GO':False}))

if __name__=='__main__':main()
