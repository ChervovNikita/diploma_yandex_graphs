"""Correct inherited administrative flags; preserve v7 runtime bytes and QA."""
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD,invoke
HERE=Path(__file__).resolve().parent/'controller_manifest_correction_root_20261012_v8'
REMOTE=GUARD+r'''
prior_dir=phase/'master_controller_source_20261012_v7'
prior=json.loads((prior_dir/'MANIFEST.json').read_text())
out=phase/'master_controller_source_20261012_v8';assert not out.exists();out.mkdir();runtime=out/'runtime';runtime.mkdir()
def receipt(p):return {'path':str(p.relative_to(phase)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
for row in prior['files']:
 source=phase/row['path'];assert receipt(source)==row
 target=runtime/source.name;target.write_bytes(source.read_bytes());target.chmod(0o555 if target.suffix=='.sh' else 0o444)
manifest=dict(prior)
stale={}
for key in ('common_master_semantics_AST_unchanged_except_controller_file_catalogue','changed_existing_files','source_snapshot'):
 if key in manifest:stale[key]=manifest.pop(key)
manifest.update(runtime_package=str(runtime.relative_to(phase)),files=[receipt(p) for p in sorted(runtime.iterdir())],
 prior_controller_manifest=receipt(prior_dir/'MANIFEST.json'),
 administrative_correction={'inherited_historical_fields':stale,
  'reason':'v6 deliberately extended native QK cost typing; the v5 unchanged-AST administrative flag was inherited outside its valid scope',
  'v7_delta_from_v6':'Only16to32MiB aggregate hash metadata capacity constant',
  'v8_delta_from_v7':'Manifest interpretation only; every runtime file byte-identical'},
 unchanged_controller_files=list(prior['files']),new_engineering_tests_run=0,
 engineering_QA_inherited_exact_runtime=True,source_scope='Byte-identical v7 runtime with corrected inherited metadata scope',
 scientific_fits=0,benchmark_scores=0)
for name,value in [('MANIFEST.json',manifest),('RENDEZVOUS_POLICY.json',manifest['phase_rendezvous']),('HOST_BINDINGS.json',manifest['host_bindings'])]:
 (out/name).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'manifest':manifest,'unchanged_runtime_files':len(manifest['files']),'fits':0,'GO':False}))
'''
def main():
    assert not HERE.exists();HERE.mkdir()
    (HERE/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
    v=invoke(REMOTE,[],HERE,'METADATA_SUCCESSOR',90);assert v['exit_code']==0
    x=json.loads(v['stdout']);(HERE/'MANIFEST.json').write_text(json.dumps(x['manifest'],indent=2)+'\n')
    print(json.dumps({'runtime':x['manifest']['runtime_package'],'unchanged_runtime_files':x['unchanged_runtime_files'],'fits':0,'GO':False}))
if __name__=='__main__':main()
