"""Preserve owned provisional revisions and run synthetic source engineering."""
import base64,hashlib,json,zlib
from pathlib import Path
from transport import OUT,helper,GUARD
def main():
    label='PHYSICS_OIPG_SYNTHETIC_ATTEMPT1';directory=OUT/'transport'
    if (directory/(label+'_REMOTE.json')).exists():raise FileExistsError('Preserve prior attempt')
    files=[]
    for folder in ('runtime','sources','preparation','tests'):
        for p in sorted((OUT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.sh','.txt'):
                raw=p.read_bytes();files.append({'path':str(p.relative_to(OUT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    for name in ('DESIGN_DECISION.json','DATA_INPUT_BINDINGS.json','ROOT_V17_BASE.json','SOURCE_SNAPSHOT_RECEIPTS.json','CONTROLLER_TYPE_AMENDMENT.json'):
        p=OUT/name;raw=p.read_bytes();files.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    encoded=base64.b64encode(zlib.compress(json.dumps({'destination':OUT.name,'files':files}).encode())).decode()
    for offset in range(0,len(encoded),16000):
        r=helper.invoke(helper.TRANSFER,[label,str(offset),encoded[offset:offset+16000]],directory,label+'_TRANSFER_'+str(offset),60)
        if r['exit_code']!=0:raise RuntimeError('Guarded source transfer failure retained')
    code=GUARD+r'''
encoded=(phase/(sys.argv[1]+'_transport')/'payload.b64').read_text();assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2];payload=json.loads(zlib.decompress(base64.b64decode(encoded)));stage=phase/payload['destination'];assert stage.exists()
changes=[]
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts;q=stage/rel;raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if q.exists() and q.read_bytes()!=raw:
  old=q.read_bytes();saved=stage/'source_revisions'/sys.argv[1]/rel;saved.parent.mkdir(parents=True,exist_ok=True);assert not saved.exists();saved.write_bytes(old);saved.chmod(0o444);changes.append({'path':row['path'],'prior_sha256':hashlib.sha256(old).hexdigest(),'next_sha256':row['sha256']})
 q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw)
(stage/'SYNTHETIC_SOURCE_REVISION.json').write_text(json.dumps(changes,indent=2,sort_keys=True)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');command=[str(repo/'.venv/bin/python'),'-B',str(stage/'tests/check_source_synthetic.py')];start=time.monotonic();child=subprocess.Popen(command,cwd=stage,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split();(stage/'SYNTHETIC_ATTEMPT1_START.json').write_text(json.dumps({'pid':child.pid,'start_ticks':int(fields[19]),'command':command,'actual_input_models':0,'fits':0})+'\n')
try:stdout,stderr=child.communicate(timeout=900)
except subprocess.TimeoutExpired:child.kill();stdout,stderr=child.communicate();stderr+='\nBounded synthetic-only CPU source check timeout preserved.\n'
result={'command':command,'exit_code':child.returncode,'wall_seconds':time.monotonic()-start,'stdout':stdout,'stderr':stderr,'actual_input_models':0,'scientific_fits':0,'GO':False};target=stage/'SYNTHETIC_ATTEMPT1_RESULT.json';assert not target.exists();target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result));sys.exit(0 if child.returncode==0 else 1)
'''
    r=helper.invoke(code,[label,hashlib.sha256(encoded.encode()).hexdigest()],directory,label+'_REMOTE',1000);print(json.dumps(r,indent=2))
    if r['exit_code']!=0:raise RuntimeError('Synthetic-only source check failed; attempt retained')
if __name__=='__main__':main()
