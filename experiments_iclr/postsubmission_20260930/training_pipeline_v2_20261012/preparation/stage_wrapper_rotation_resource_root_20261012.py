"""Stage ten reviewed source resource checks; no model, GPU callback or fit."""
import base64,hashlib,json,zlib
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD,invoke

BASE=Path(__file__).resolve().parent
SOURCE=BASE.parents[1]/'native_wrapper_rotation_resource_proposal_20261012_v1'
HERE=BASE/'wrapper_rotation_resource_staging_root_20261012_v1'
DEST='native_wrapper_rotation_resource_proposal_20261012_v1'

def main():
    assert not HERE.exists()
    HERE.mkdir()
    names=['README.md','PROPOSAL.json','qualify_wrapper_rotation.py','stage_metadata.py',
        'SOURCE_INVENTORY.json','LOCAL_METADATA_PROOF.json','PROBE_REUSE_PROOF.json','OBSERVATIONS_SCHEMA.json']
    rows=[]
    for name in names:
        raw=(SOURCE/name).read_bytes();rows.append({'path':name,'bytes':len(raw),
            'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    assert next(r for r in rows if r['path']=='qualify_wrapper_rotation.py')['sha256']=='1aca793f94d10314375374bbe8e1ec165ce875dc0caf7b650391e540ec6f7937'
    (HERE/'ROOT_REVIEW.json').write_text(json.dumps({'reviewed_probe_sha256':rows[2]['sha256'],
        'finite_cases':10,'complete_source_callbacks':40,'actual_optimizer_updates':0,'scientific_fits':0,
        'GO':False,'scope':'Inherited denial/state helpers; exact native exchange and paired-local work hooks; legal GRAPH/TRAIN and full engine restore'},indent=2)+'\n')
    payload=base64.b64encode(zlib.compress(json.dumps(rows,separators=(',',':')).encode(),9)).decode()
    transfer=GUARD+r'''
out=phase/sys.argv[1];offset=int(sys.argv[2]);target=out/'PAYLOAD.b64'
if offset==0:assert not out.exists();out.mkdir()
else:assert target.stat().st_size==offset
with target.open('ab') as f:f.write(sys.argv[3].encode())
print(json.dumps({'bytes':target.stat().st_size,'fits':0,'GO':False}))
'''
    for offset in range(0,len(payload),24000):
        v=invoke(transfer,[DEST+'_root_transport',str(offset),payload[offset:offset+24000]],HERE,'TRANSFER_'+str(offset),60)
        assert v['exit_code']==0
    remote=GUARD+r'''
raw=(phase/(sys.argv[1]+'_root_transport/PAYLOAD.b64')).read_bytes();assert hashlib.sha256(raw).hexdigest()==sys.argv[2]
rows=json.loads(zlib.decompress(base64.b64decode(raw)));source=phase/sys.argv[1];assert not source.exists();source.mkdir()
for row in rows:
 assert len(Path(row['path']).parts)==1
 b=base64.b64decode(row['data']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 p=source/row['path'];p.write_bytes(b);p.chmod(0o444)
cmd=[str(repo/'.venv/bin/python'),'-B',str(source/'stage_metadata.py'),'--project-root',str(phase),
 '--source-directory',sys.argv[1],'--output','native_wrapper_rotation_resource_stage_20261012_v1','--root-reviewed']
v=subprocess.run(cmd,cwd=source,capture_output=True,text=True,timeout=90)
value={'exit_code':v.returncode,'stdout':v.stdout,'stderr':v.stderr,'models':0,'CUDA':0,'fits':0,'GO':False}
(source/'ROOT_STAGE_RECEIPT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value));sys.exit(v.returncode)
'''
    v=invoke(remote,[DEST,hashlib.sha256(payload.encode()).hexdigest()],HERE,'METADATA_STAGE',120)
    assert v['exit_code']==0,'Metadata staging failure retained; no engineering callback launched'
    x=json.loads(v['stdout']);(HERE/'STAGE_RECEIPT.json').write_text(json.dumps(x,indent=2)+'\n')
    print(json.dumps(x))

if __name__=='__main__':main()
