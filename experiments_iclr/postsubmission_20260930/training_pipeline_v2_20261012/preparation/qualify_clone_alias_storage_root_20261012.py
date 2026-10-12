"""Qualify a lossless clone fix on an owned source copy; never change a live engine."""
import base64,json,zlib
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD,invoke
HERE=Path(__file__).resolve().parent
OUT=HERE/'clone_alias_storage_root_20261012_v1'
REMOTE=GUARD+r'''
payload=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])))
stage=phase/'clone_alias_storage_root_20261012_v1';assert not stage.exists();stage.mkdir()
for name,encoded in payload.items():(stage/name).write_bytes(base64.b64decode(encoded))
source=phase/'native_nsd_five_new_data_packet_20261012_v2/runtime/engine.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='b5fc64b7ef0c1ddd26ff6f81840ad982192d9e95de0b493c0079b3deadf34679'
engine=stage/'engine.py';engine.write_bytes(source.read_bytes())
sys.path.insert(0,str(stage));import patch_clone
amendment=patch_clone.apply(engine)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
result=subprocess.run([str(repo/'.venv/bin/python'),'-B',str(stage/'test_clone.py'),str(engine)],env=env,cwd=stage,capture_output=True,text=True,timeout=90)
value={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'amendment':amendment,'NSD_base_engine_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'successor_engine_sha256':hashlib.sha256(engine.read_bytes()).hexdigest(),'live_or_frozen_engine_changed':False,'scientific_fits':0,'GO':False}
(stage/'RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
'''
def main():
    assert not (OUT/'RESULT.json').exists()
    payload={name:base64.b64encode((OUT/name).read_bytes()).decode() for name in ('patch_clone.py','test_clone.py')}
    result=invoke(REMOTE,[base64.b64encode(zlib.compress(json.dumps(payload).encode(),9)).decode()],OUT,'QA',120)
    assert result['exit_code']==0,'Preserve transport failure'
    value=json.loads(result['stdout']);(OUT/'RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    assert value['exit_code']==0,'Preserve qualification failure'
    print(value['stdout'])
if __name__=='__main__':main()
