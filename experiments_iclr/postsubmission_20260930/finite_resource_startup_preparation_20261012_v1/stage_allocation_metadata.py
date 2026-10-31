"""Stage finite disabled resource/preflight source; zero callbacks and jobs."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'training_pipeline_v2_20261012/preparation'))
from prepare_molecular_packets_allocation import GUARD,TRANSFER,invoke
REMOTE=GUARD+r'''
encoded=(phase/(sys.argv[1]+'_transport/payload.b64')).read_bytes();assert hashlib.sha256(encoded).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
out=phase/sys.argv[1];assert not out.exists();out.mkdir()
def receipt(p):return {'path':str(p.relative_to(phase)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
for row in payload['files']:
 p=out/row['name'];raw=base64.b64decode(row['data']);assert p.parent==out and len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];p.write_bytes(raw)
assert json.loads((out/'HOST_BUDGET_MANIFEST_TEMPLATE.json').read_text())['scientific_GO'] is False
assert json.loads((out/'RESOURCE_SCOPE_SUPPLEMENT.json').read_text())['whole_master_storage_and_all_source_geometry_coverage_proved'] is False
manifest={'schema':'finite-source-resource-startup-package-v1','files':[receipt(p) for p in sorted(out.iterdir()) if p.is_file()],
 'scientific_fits':0,'actual_input_callbacks':0,'source_metadata_only':True,'physical_77_not_accessed':True,'GO':False}
(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
for p in out.iterdir():p.chmod(0o444)
print(json.dumps({'manifest':receipt(out/'MANIFEST.json'),'scientific_fits':0,'actual_input_callbacks':0,'GO':False}))
'''


def main():
    files=[]
    for name in ('source_geometry.py','preflight_host_budget.py','catalogue_final_geometries.py','build_supplement.py',
                 'stage_allocation_metadata.py','THREE_GPU_STARTUP_SCOPE.md','RESOURCE_SCOPE_SUPPLEMENT.json',
                 'SOURCE_METADATA_QA.json','HOST_BUDGET_MANIFEST_TEMPLATE.json'):
        p=HERE/name;raw=p.read_bytes()
        if p.suffix=='.py':ast.parse(raw)
        files.append({'name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    encoded=base64.b64encode(zlib.compress(json.dumps({'files':files}).encode(),9)).decode()
    for offset in range(0,len(encoded),48000):
        result=invoke(TRANSFER,[HERE.name,str(offset),encoded[offset:offset+48000]],HERE,'TRANSFER_'+str(offset),90)
        assert result['exit_code']==0
    result=invoke(REMOTE,[HERE.name,hashlib.sha256(encoded.encode()).hexdigest()],HERE,'STAGE_METADATA',180)
    assert result['exit_code']==0
    value=json.loads(result['stdout']);(HERE/'ALLOCATION_HANDOFF.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps(value))

if __name__=='__main__':main()
