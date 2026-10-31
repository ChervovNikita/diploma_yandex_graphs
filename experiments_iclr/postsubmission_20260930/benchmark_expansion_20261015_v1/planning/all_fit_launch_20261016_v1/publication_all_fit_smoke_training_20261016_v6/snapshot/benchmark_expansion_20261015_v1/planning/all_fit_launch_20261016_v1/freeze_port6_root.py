"""Freeze root-owned cache/coordinator helpers after final repair package manifests exist."""
from pathlib import Path
import ast,hashlib,json
HERE=Path(__file__).resolve().parent;E=HERE.parents[1]
def pin(p):
 b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def verify(p):
 m=json.loads((p/'MANIFEST.json').read_text())
 for row in m['files']:
  f=p/row['path'];assert all(pin(f)[k]==row[k] for k in ('bytes','sha256')),(p,row['path'])
 return pin(p/'MANIFEST.json')
packages=['server51_campaign_port_v6','qualification/all_fit_smoke_20261016_v6','campaign_execution_20261016_v7']
pins={name:verify(E/name) for name in packages};port_sha=pins[packages[0]]['sha256']
cache=E/'server51_campaign_projection_cache_v6'
for f in cache.iterdir():
 if f.is_file():f.write_text(f.read_text().replace('FINAL_PORT6_MANIFEST_SHA',port_sha))
wrapper=pin(cache/'prepare_cached.py')
f=HERE/'prepare_port6.py';f.write_text(f.read_text().replace('FINAL_PORT6_MANIFEST_SHA',port_sha).replace('FINAL_CACHE6_WRAPPER_SHA',wrapper['sha256']))
for f in list(cache.glob('*.py'))+list(HERE.glob('*port6*.py')):
 ast.parse(f.read_text(),filename=str(f))
 if f != Path(__file__).resolve():assert 'FINAL_' not in f.read_text(),f
checks=json.loads((HERE/'PORT6_ROOT_FOCUSED_CHECKS.json').read_text());assert all(r['exit_code']==0 for r in checks)
manifest={'schema':'server51-projection-cache-wrapper-source-v1','source_checks':'PASS','frozen_parent_manifest_sha256':port_sha,'upstream_port_manifest_sha256':port_sha,'predecessor_manifest_sha256':pin(E/'server51_campaign_projection_cache_v5/MANIFEST.json')['sha256'],'frozen_parent_modified':False,'source_rebinding_only':True,'native_work':0,'files':[{'path':f.name,**pin(f)} for f in sorted(cache.iterdir()) if f.is_file()]}
with (cache/'MANIFEST.json').open('x') as stream:json.dump(manifest,stream,indent=2,sort_keys=True)
pins[cache.name]=verify(cache)
receipt={'schema':'port6-downstream-source-freeze-v1','status':'PASS','sources':pins,'cache_wrapper':wrapper,'root_focused_checks':pin(HERE/'PORT6_ROOT_FOCUSED_CHECKS.json'),'helpers':{f.name:pin(f) for f in sorted(HERE.glob('*port6*.py'))},'scientific_settings_changed':False,'TEST_accessed':False,'remote_launches':0,'full_training_claimed':False}
with (HERE/'PORT6_DOWNSTREAM_SOURCE_FREEZE.json').open('x') as stream:json.dump(receipt,stream,indent=2,sort_keys=True)
print(json.dumps(receipt))
