"""Snapshot exact closed source and metadata for Git; exclude datasets/checkpoints."""
import base64,gzip,hashlib,json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD,invoke
P=Path(__file__).resolve().parents[2]
OUT=P/'publication/Master_v18_source_storage_20261012_v1'
REMOTE=GUARD+r'''
import gzip
plan_path=phase/'massive_practical_master_20261012_v18/PLAN.json'
plan=json.loads(plan_path.read_text())
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
assert hashlib.sha256(canonical(plan)).hexdigest()=='7966f239f12c3acf7c116595f67a09518b4e56a3793acaf4483c7e6dfd7b117d'
paths=set();sources={};excluded=[]
for version in (14,15,17,18):
 base=phase/('massive_practical_master_20261012_v'+str(version))
 for name in ('PLAN.json','SUMMARY.json','READINESS.json','ACTUAL_CONTROLLER_LOAD_VERIFICATION.json','WORKER_COSTS.csv','COMPACT_SUMMARY.json'):
  p=base/name
  if p.exists():paths.add(str(p.relative_to(phase)))
def bind(row):
 p=phase/row['path'];assert p.resolve().is_relative_to(phase) and not p.is_symlink()
 raw=p.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];return p
for row in plan['immutable_source_bundle']['files']:bind(row);sources[row['path']]=row
for entry in plan['studies']:
 p=phase/entry['path'];raw=p.read_bytes();s=json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)
 assert hashlib.sha256(canonical(s)).hexdigest()==entry['spec_hash'];paths.add(entry['path'])
 for row in s['source']['files']:sources[row['path']]=row
for row in sources.values():
 suffix=Path(row['path']).suffix
 if suffix in {'.py','.json','.md','.txt','.yaml','.yml','.sh','.csv','.html','.source','.patch','.diff','.gz'} or Path(row['path']).name in {'LICENSE','PENCIL_LICENSE','PYG_LICENSE','.project-root'}:
  bind(row);paths.add(row['path'])
 else:excluded.append({'path':row['path'],'reason':'Nontext/source binary is excluded from Git publication; transfer/data custody stays separate'})
for version in (9,10,11):paths.add('master_controller_source_20261012_v'+str(version)+'/MANIFEST.json')
for base,names in {
 'native_nsd_five_new_data_packet_20261012_v2':['MASTER_INPUT_ENTRIES.json','ALLOCATION_HANDOFF.json','CUSTODY.json','SOURCE_BINDINGS.json','CONTROLLER_TYPE_AMENDMENT.json','ROOT_V10_COMPATIBILITY.json','RESOURCE_AND_SETUP_HANDOFF.json','SETUP_COST_AND_FEATURE_SCOPE_SUPPLEMENT.json'],
 'preaggregation_growth_new_data_packet_20261012_v2':['MASTER_INPUT_ENTRY.json','SOURCE_BINDINGS.json','CUSTODY.json','LEARNING_PROTOCOL_AND_COST_DECISIONS.json','SETUP_COST_AND_FEATURE_SCOPE_SUPPLEMENT.json'],
 'clone_alias_storage_root_20261012_v1':['engine.py','patch_clone.py','test_clone.py','RESULT.json'],
 'v17_resource_storage_catalog_root_20261012_v1':['CATALOG.json']}.items():
 for name in names:paths.add(base+'/'+name)
files=[]
for relative in sorted(paths):
 p=phase/relative;assert p.resolve().is_relative_to(phase) and not p.is_symlink()
 assert p.suffix not in {'.pt','.pth','.npy','.npz','.zip','.tar','.so','.pdf'}
 raw=p.read_bytes();files.append({'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'payload':base64.b64encode(gzip.compress(raw,mtime=0)).decode()})
readme=subprocess.run(['git','show','HEAD:README.md'],cwd=repo,capture_output=True,text=True,check=True).stdout
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert head=='18f522d36061321ffe6948d24024762aa763f02e'
print(json.dumps({'files':files,'excluded_non_source_files':excluded,'README':readme,'head':head,'datasets_or_checkpoints_included':False,'scientific_fits':0,'GO':False}))
'''
def main():
    OUT.mkdir();(OUT/'SNAPSHOT_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    r=invoke(REMOTE,[],OUT,'SNAPSHOT',180);assert r['exit_code']==0,'Preserve snapshot failure'
    value=json.loads(r['stdout']);archives=OUT/'archives';archives.mkdir()
    rows=[]
    for row in value['files']:
        raw=gzip.decompress(base64.b64decode(row['payload']))
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        wrapper={'schema':'exact-project-source-archive-v1','original_project_relative_path':row['path'],'original_bytes':row['bytes'],'original_sha256':row['sha256'],'encoding':'base64-gzip','payload':row['payload']}
        path=archives/(hashlib.sha256(row['path'].encode()).hexdigest()[:24]+'.archive.json')
        path.write_text(json.dumps(wrapper,separators=(',',':'))+'\n')
        assert path.stat().st_size<2_000_000,'Archive split required; preserve attempt'
        rows.append({k:v for k,v in wrapper.items() if k!='payload'}|{'archive_project_relative_path':str(path.relative_to(P)),'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT/'RESTORATION_INDEX.json').write_text(json.dumps({'archives':rows},indent=2)+'\n')
    (OUT/'ARCHIVE_INDEX.json').write_text(json.dumps({'archives':[{k:v for k,v in row.items() if not k.startswith('archive_')} for row in rows],'excluded_non_source_files':value['excluded_non_source_files'],'source_bytes_preserved':True,'datasets_or_checkpoints_included':False},indent=2)+'\n')
    donor=P/'publication/New_data_source_integration_20261012_v1/rehydrate_source_archives.py'
    (OUT/'rehydrate_source_archives.py').write_bytes(donor.read_bytes())
    (OUT/'README_PREDECESSOR.md').write_text(value['README'])
    (OUT/'SNAPSHOT_SUMMARY.json').write_text(json.dumps({'head':value['head'],'archives':len(rows),'source_bytes':sum(r['original_bytes'] for r in rows),'datasets_or_checkpoints_included':False,'scientific_fits':0,'GO':False},indent=2)+'\n')
    print(json.dumps({'archives':len(rows),'source_bytes':sum(r['original_bytes'] for r in rows),'scientific_fits':0,'GO':False}))
if __name__=='__main__':main()
