"""Guarded source/metadata seal; no array copies, numerical imports or fits."""
import base64,hashlib,json,zlib,textwrap
from transport import OUT,request
def main():
 names=['preparation/build_handoff.py','preparation/seal_packet.py','SOURCE_SETUP_COSTS.json','RESOURCE_PROPOSAL.json','CONTROLLER_HANDOFF.json','ALLOCATION_HANDOFF.json',
  'SOURCE_COPY_ATTEMPT1_FAILURE.json','COMPILER_INGESTION_RESULT.json','transport/WIKICS_EXACT_GRAPH_PATH_AND_PHYSICS_INHERITANCE.json']
 rows=[]
 for name in names:
  raw=(OUT/name).read_bytes();rows.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
 value=request(textwrap.dedent(r'''
 stage=phase/sys.argv[1];rows=json.loads(zlib.decompress(base64.b64decode(sys.argv[2])))
 for row in rows:
  rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts;p=stage/rel;raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  if p.exists():assert p.read_bytes()==raw
  else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 def receipt(p):
  raw=p.read_bytes();return {'path':str(p.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 handoff=json.loads((stage/'ALLOCATION_HANDOFF.json').read_text());audit=json.loads((stage/'FINAL_TYPED_INGESTION.json').read_text());assert handoff['entry']==audit['entry'] and audit['verified']
 def nested(v):
  if isinstance(v,dict):
   if set(v)=={'path','bytes','sha256'}:yield v
   else:
    for item in v.values():yield from nested(item)
  elif isinstance(v,list):
   for item in v:yield from nested(item)
 checked={}
 for row in nested(handoff):
  p=phase/row['path'];assert not p.is_symlink() and p.is_file()
  if row['path'] not in checked:checked[row['path']]=receipt(p)
  assert checked[row['path']]==row
 files=[]
 for folder in ('runtime','sources','preparation','tests','controller_base'):
  files += [p for p in sorted((stage/folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.txt','.sh','.patch')]
 files += [p for p in sorted((stage/'matrix').rglob('*')) if p.is_file()]
 files += [p for p in sorted(stage.glob('*.json')) if p.name not in ('FINAL_SOURCE_SEAL.json','SOURCE_INVENTORY.json')]
 inventory=[]
 for p in files:
  row=checked.get(str(p.relative_to(phase)))
  if row is None:row=receipt(p)
  inventory.append(row);p.chmod(0o444)
 inventory=list({r['path']:r for r in inventory}.values());target=stage/'SOURCE_INVENTORY.json';assert not target.exists();target.write_text(json.dumps({'schema':'Physics-OIPG-frozen-source-metadata-inventory-v1','files':inventory,'opaque_bytes':sum(r['bytes'] for r in inventory),'numerical_arrays_not_copied_or_rehashed':True,'scientific_launch_enabled':False,'GO':False},indent=2,sort_keys=True)+'\n');target.chmod(0o444)
 custody=json.loads((stage/'DATA_CUSTODY.json').read_text());data=[]
 for dataset in custody['datasets'].values():
  for split in dataset['splits']:
   for row in list(split['roles'].values())+list(split['endpoint_inputs'].values()):
    p=phase/row['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'];p.chmod(0o444);data.append(row)
 data=list({r['path']:r for r in data}.values())
 seal={'schema':'Physics-OIPG-exact-source-roster-seal-v1','status':'SEALED_SOURCE_AND_DATA_DISABLED','packet':stage.name,'inventory':receipt(target),'entry':handoff['entry'],
  'canonical_methods':15,'source_metadata_files':len(inventory),'source_metadata_bytes':sum(r['bytes'] for r in inventory),'owned_data_role_files':len(data),'owned_data_role_bytes':sum(r['bytes'] for r in data),
  'data_hashes_inherited_from_exact_passed_custody':True,'existing_large_graph_aliases_not_copied_or_rehashed':True,'preserved_revision_and_attempt_history':True,
  'root_resource_profile_qualified':False,'actual_input_fits':0,'actual_optimizer_updates':0,'actual_predictive_scores':0,'GO':False}
 target=stage/'FINAL_SOURCE_SEAL.json';assert not target.exists();target.write_text(json.dumps(seal,indent=2,sort_keys=True)+'\n');target.chmod(0o444)
 artifacts=[]
 for name in ('SOURCE_INVENTORY.json','FINAL_SOURCE_SEAL.json','METADATA_COMPILATION_ATTEMPT1.json'):
  p=stage/name;raw=p.read_bytes();artifacts.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
 print(json.dumps({'seal':seal,'handoff':receipt(stage/'ALLOCATION_HANDOFF.json'),'artifacts':artifacts,'GO':False}))
 '''),[OUT.name,base64.b64encode(zlib.compress(json.dumps(rows).encode(),9)).decode()],'FINAL_SOURCE_AND_METADATA_SEAL_V1',180)
 for row in value.pop('artifacts',[]):
  raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];p=OUT/row['path'];assert not p.exists();p.write_bytes(raw);p.chmod(0o444)
 inventory=json.loads((OUT/'SOURCE_INVENTORY.json').read_text())
 for row in inventory['files']:
  p=OUT.parent/row['path']
  if p.is_file():
   raw=p.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];p.chmod(0o444)
 print(json.dumps(value))
if __name__=='__main__':main()
