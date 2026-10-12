"""Stage the exact eight-case native Q/K proposal; metadata only, no GPU."""
import base64,hashlib,json,zlib
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD,invoke
BASE=Path(__file__).resolve().parent
SOURCE=BASE/'native_polynormer_qk_resource_proposal_20261012_v1'
HERE=BASE/'native_qk_resource_staging_root_20261012_v1'
DEST='native_polynormer_qk_resource_source_20261012_v1'
def main():
 assert not HERE.exists();HERE.mkdir()
 names=['HANDOFF.md','PROPOSAL.json','SOURCE_BINDINGS.json','ENGINEERING_ADMISSION_TEMPLATE.json',
  'qualify_native_qk_resources.py','MANIFEST.json','SOURCE_ONLY_QA.json','test_source_only.py']
 rows=[]
 for name in names:
  raw=(SOURCE/name).read_bytes();rows.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
 assert next(r for r in rows if r['path']=='qualify_native_qk_resources.py')['sha256']=='d752f4994627dad2a03dd4e19a59032f124ba6afdabbdc66d2dcc3ee3b3d2c7d'
 payload=base64.b64encode(zlib.compress(json.dumps(rows).encode(),9)).decode()
 transfer=GUARD+r'''
out=phase/sys.argv[1];offset=int(sys.argv[2]);p=out/'PAYLOAD.b64'
if offset==0:assert not out.exists();out.mkdir()
else:assert p.stat().st_size==offset
with p.open('ab') as f:f.write(sys.argv[3].encode())
print(json.dumps({'bytes':p.stat().st_size,'models':0,'GO':False}))
'''
 for offset in range(0,len(payload),24000):
  v=invoke(transfer,[DEST+'_root_transport',str(offset),payload[offset:offset+24000]],HERE,'TRANSFER_'+str(offset),60);assert v['exit_code']==0
 remote=GUARD+r'''
raw=(phase/(sys.argv[1]+'_root_transport/PAYLOAD.b64')).read_bytes();assert hashlib.sha256(raw).hexdigest()==sys.argv[2]
source=phase/sys.argv[1];assert not source.exists();source.mkdir()
for row in json.loads(zlib.decompress(base64.b64decode(raw))):
 assert len(Path(row['path']).parts)==1
 b=base64.b64decode(row['data']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 p=source/row['path'];p.write_bytes(b);p.chmod(0o444)
def receipt(p):return {'path':str(p.relative_to(phase)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
admission=json.loads((source/'ENGINEERING_ADMISSION_TEMPLATE.json').read_text())
proposal=json.loads((source/'PROPOSAL.json').read_text());assert len(proposal['requested_cases'])==8
assert receipt(source/'PROPOSAL.json')==admission['proposal_receipt']
for row in admission['runtime_dependency_receipts']+[admission['role_optimizer_cap_helper'],admission['tensor_commit_tree_helper'],admission['frozen_controller_manifest']]:
 assert receipt(phase/row['path'])==row
for case in proposal['requested_cases']:
 assert receipt(phase/case['study_receipt']['path'])==case['study_receipt']
out=phase/'native_qk_resource_admission_root_20261012_v1';assert not out.exists();out.mkdir()
admission.update(root_source_review_confirmed=True,approved_case_ids=[c['case_id'] for c in proposal['requested_cases']],
 root_review_scope='Exact native source callbacks with inherited denial/moments/workspace/RoleIO/serializer; explicit random branch and missing learned local geometry; no scientific GO')
(out/'ENGINEERING_ADMISSION.json').write_text(json.dumps(admission,indent=2)+'\n')
result={'approved_cases':8,'helper_sha256':admission['helper_sha256'],'models_created':0,'CUDA':0,'fits':0,'GO':False,
 'admission':receipt(out/'ENGINEERING_ADMISSION.json'),'missing_learned_local_snapshot_geometry_qualified':False}
(out/'ROOT_SOURCE_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''
 v=invoke(remote,[DEST,hashlib.sha256(payload.encode()).hexdigest()],HERE,'METADATA_STAGE',120);assert v['exit_code']==0
 x=json.loads(v['stdout']);(HERE/'ROOT_SOURCE_REVIEW.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
if __name__=='__main__':main()
