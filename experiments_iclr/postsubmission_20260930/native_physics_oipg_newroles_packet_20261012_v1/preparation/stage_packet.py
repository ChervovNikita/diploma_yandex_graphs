"""Fresh guarded source-only stage and data custody; no actual-input models."""
import base64,hashlib,json,zlib
from pathlib import Path
from transport import OUT,PHASE,helper,GUARD
def main():
    label='PHYSICS_OIPG_SOURCE_STAGE_V1';directory=OUT/'transport'
    if (directory/(label+'_REMOTE.json')).exists():raise FileExistsError('Preserve prior attempt')
    files=[]
    for folder in ('runtime','sources','preparation','tests'):
        for p in sorted((OUT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.sh','.txt'):
                raw=p.read_bytes();files.append({'path':str(p.relative_to(OUT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    for name in ('DESIGN_DECISION.json','DATA_INPUT_BINDINGS.json','ROOT_V17_BASE.json','SOURCE_SNAPSHOT_RECEIPTS.json','CONTROLLER_TYPE_AMENDMENT.json','SOURCE_COPY_ATTEMPT1_FAILURE.json'):
        p=OUT/name;raw=p.read_bytes();files.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    encoded=base64.b64encode(zlib.compress(json.dumps({'destination':OUT.name,'files':files}).encode())).decode()
    for offset in range(0,len(encoded),16000):
        r=helper.invoke(helper.TRANSFER,[label,str(offset),encoded[offset:offset+16000]],directory,label+'_TRANSFER_'+str(offset),60)
        if r['exit_code']!=0:raise RuntimeError('Guarded source transfer failure retained')
    code=GUARD+r'''
encoded=(phase/(sys.argv[1]+'_transport')/'payload.b64').read_text();assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)));stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
rows=[]
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 q=stage/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw);rows.append({k:v for k,v in row.items() if k!='data'})
(stage/'PROVISIONAL_SOURCE_INVENTORY.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
command=[str(repo/'.venv/bin/python'),'-B',str(stage/'preparation/prepare_data.py'),'--project-root',str(phase),'--package',payload['destination']]
started=time.monotonic();child=subprocess.run(command,cwd=stage/'preparation',env=env,capture_output=True,text=True,timeout=600)
result={'command':command,'exit_code':child.returncode,'wall_seconds':time.monotonic()-started,'stdout':child.stdout,'stderr':child.stderr,'actual_input_models':0,'scientific_fits':0,'GO':False}
(stage/'DATA_PREPARATION_ATTEMPT1.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result),flush=True);sys.exit(0 if child.returncode==0 else 1)
'''
    r=helper.invoke(code,[label,hashlib.sha256(encoded.encode()).hexdigest()],directory,label+'_REMOTE',700)
    print(json.dumps(r,indent=2))
    if r['exit_code']!=0:raise RuntimeError('Data preparation failed; source and attempt preserved')
if __name__=='__main__':main()
