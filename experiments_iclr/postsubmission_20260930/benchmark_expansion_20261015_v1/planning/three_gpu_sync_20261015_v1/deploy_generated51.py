"""Install the exact generated metadata archive in server51's project checkout."""
from pathlib import Path
import json
import remote51

HERE = Path(__file__).resolve().parent
CODE = r'''
from pathlib import Path
import hashlib,json,tarfile,time,os
root=Path(ROOT_VALUE);record=RECORD_VALUE
archive=root/RELATIVE_VALUE;folder=archive.parent
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert archive.resolve().is_relative_to(root) and not archive.is_symlink()
assert archive.stat().st_size==record['archive']['bytes'] and digest(archive)==record['archive']['sha256']
receipt=folder/'SERVER51_ARCHIVE_DEPLOYMENT.json';assert not receipt.exists()
start=time.monotonic();changed=existing=0
with tarfile.open(archive,'r:gz') as tar:
 members=tar.getmembers();names=[m.name for m in members]
 assert len(names)==len(set(names)) and all(m.isfile() for m in members)
 name='server51/STAGING_RECEIPTS.json';raw=tar.extractfile(name).read()
 assert len(raw)==record['staging_receipts']['bytes'] and hashlib.sha256(raw).hexdigest()==record['staging_receipts']['sha256']
 stage=json.loads(raw);assert stage['host']['host']=='server51' and stage['host']['project_root']==str(root)
 expected={name:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}
 for row in stage['staged_files']:
  p=Path(row['execution']['path']);assert p.is_relative_to(folder/'server51')
  rel=str(p.relative_to(folder));assert rel not in expected
  expected[rel]=row['execution']
 assert set(names)==set(expected) and len(names)==record['members']
 for member in members:
  row=expected[member.name];p=folder/member.name
  assert p.resolve().is_relative_to(folder/'server51') and not p.is_symlink() and member.size==row['bytes']
  if p.exists():
   assert p.is_file() and p.stat().st_size==row['bytes'] and digest(p)==row['sha256']
   existing+=1;continue
  p.parent.mkdir(parents=True,exist_ok=True);part=p.with_name(p.name+'.unpack-part')
  h=hashlib.sha256();total=0
  with tar.extractfile(member) as src,part.open('xb') as dst:
   for b in iter(lambda:src.read(1048576),b''):dst.write(b);h.update(b);total+=len(b)
  assert total==row['bytes'] and h.hexdigest()==row['sha256']
  part.chmod(0o444);part.replace(p);changed+=1
out={'status':'PASS','archive_sha256':record['archive']['sha256'],'staging_receipts_sha256':record['staging_receipts']['sha256'],'members':len(names),'created':changed,'preserved_exact':existing,'wall_seconds':time.monotonic()-start,'full_campaign_GO':False,'native_work':0}
with receipt.open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
'''

if __name__ == '__main__':
    record = json.loads((HERE / 'CAMPAIGN51_GENERATED_ARCHIVE.json').read_text())
    plan = json.loads((HERE / 'CAMPAIGN51_GENERATED_ARCHIVE_TRANSFER_PLAN.json').read_text())
    transfer = json.loads((HERE / 'CAMPAIGN51_GENERATED_ARCHIVE_TRANSFER_RECEIPT.json').read_text())
    assert transfer['returncode'] == 0 and len(plan) == 1
    target = HERE / 'CAMPAIGN51_GENERATED_DEPLOYMENT.json'
    assert not target.exists()
    code = CODE.replace('ROOT_VALUE', repr(remote51.ROOT)).replace('RECORD_VALUE', repr(record)).replace('RELATIVE_VALUE', repr(plan[0]['target_relative']))
    result = json.loads(remote51.call(code, timeout=600))
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
