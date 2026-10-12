"""Freeze qualified NSD lifecycle typing and lossless clone storage together."""
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD,invoke
OUT=Path(__file__).resolve().parent/'controller_v11_nsd_alias_root_20261012_v1'
REMOTE=GUARD+r'''
import ast
prior_path=phase/'master_controller_source_20261012_v10/MANIFEST.json'
prior=json.loads(prior_path.read_text())
packet=phase/'native_nsd_five_new_data_packet_20261012_v2'
bindings=json.loads((packet/'CONTROLLER_TYPE_AMENDMENT.json').read_text())
alias=phase/'clone_alias_storage_root_20261012_v1'
qa=json.loads((alias/'RESULT.json').read_text());assert qa['exit_code']==0 and qa['scientific_fits']==0
checks=json.loads(qa['stdout']);assert len(checks['checks_passed'])==6 and checks['CUDA_initialized'] is False
stage=phase/'master_controller_source_20261012_v11';assert not stage.exists();stage.mkdir()
runtime=stage/'runtime';runtime.mkdir()
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def functions(raw):
 return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
for row in prior['files']:
 p=phase/row['path'];assert receipt(p)==row;(runtime/p.name).write_bytes(p.read_bytes())
changes={}
for row in bindings['files']:
 p=runtime/row['file'];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==row['base_sha256']
 successor=packet/'runtime'/row['file'];new=successor.read_bytes()
 assert hashlib.sha256(new).hexdigest()==row['successor_sha256'] and len(new)==row['bytes']
 before,after=functions(raw),functions(new)
 altered=sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k))
 assert altered==sorted(row['changed_AST_functions']),(row['file'],altered)
 p.write_bytes(new);changes[row['file']]=altered
p=runtime/'engine.py';raw=p.read_bytes();new=(alias/'engine.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()==qa['NSD_base_engine_sha256'] and hashlib.sha256(new).hexdigest()==qa['successor_engine_sha256']
before,after=functions(raw),functions(new)
assert [k for k in before if before[k]!=after[k]]==['_clone_tree']
p.write_bytes(new);changes['engine.py'].append('_clone_tree')
for p in runtime.glob('*.py'):compile(p.read_text(),str(p),'exec')
subprocess.run(['bash','-n',str(runtime/'start_massive.sh')],check=True)
manifest=dict(prior)
manifest.update(runtime_package=str(runtime.relative_to(phase)),files=[receipt(p) for p in sorted(runtime.iterdir()) if p.is_file()],
 prior_controller_manifest=receipt(prior_path),changed_existing_files=sorted(changes),
 unchanged_controller_files=[r for r in prior['files'] if Path(r['path']).name not in changes],
 NSD_type_amendment=receipt(packet/'CONTROLLER_TYPE_AMENDMENT.json'),NSD_v10_handoff=receipt(packet/'ALLOCATION_HANDOFF.json'),
 identical_view_clone_QA=receipt(alias/'RESULT.json'),changed_AST_functions=changes,
 new_scope='Paid NSD warm/screen lifecycle, inactive fast-factor ownership and terminal checkpoint authority; lossless identical-view CPU snapshots. Unchanged native learning laws and startup.',
 scientific_fits=0,benchmark_scores=0,start_commands_are_disabled_until_GO=True,
 snapshot_values_and_restore_unchanged=True,checkpoint_pruning=False,native_learning_and_standardized_model_engine_changed=False)
(stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
for p in runtime.iterdir():
 if p.is_file():p.chmod(0o444)
print(json.dumps({'manifest':manifest,'scientific_fits':0,'GO':False}))
'''
def main():
    OUT.mkdir();(OUT/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
    r=invoke(REMOTE,[],OUT,'FREEZE',90);assert r['exit_code']==0,'Preserve freeze failure'
    v=json.loads(r['stdout']);(OUT/'MANIFEST.json').write_text(json.dumps(v['manifest'],indent=2)+'\n')
    print(json.dumps({'controller':'v11','files':len(v['manifest']['files']),'GO':False}))
if __name__=='__main__':main()
