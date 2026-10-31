"""Stage root-readable disabled source proposal; no source proof is executed."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'training_pipeline_v2_20261012/preparation'))
from prepare_molecular_packets_allocation import GUARD,invoke

REMOTE=GUARD+r'''
payload=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])))
out=phase/payload['destination'];assert not out.exists();out.mkdir()
for row in payload['files']:
 target=out/row['name'];raw=base64.b64decode(row['data'])
 assert target.parent==out and len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 target.write_bytes(raw)
template=json.loads((out/'ROOT_RELEASE_TEMPLATE.json').read_text());assert template['execution_enabled'] is False
files={'qualification':'graph_ncNC_train_paired_replay_provider_preparation_20261004_v1/replay_qualification.py',
 'factory_adapter':'ncnc_completion_successor_root_20261012_v3/runtime/adapters/link_prototypes.py',
 'factory_bridge':'ncnc_completion_successor_root_20261012_v3/runtime/adapters/link.py',
 'factory_common':'ncnc_completion_successor_root_20261012_v3/runtime/adapters/common_native_wrapper.py',
 'shared_store':'shared_link_replay_proposal_20261012_v1/shared_replay_store.py'}
def receipt(path):return {'path':str(path.relative_to(phase)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
dependencies={k:receipt(phase/rel) for k,rel in files.items()}
assert dependencies['qualification']==template['additional_source_files']['qualification']
template['additional_source_files']=dependencies
(out/'ROOT_RELEASE_TEMPLATE_WITH_SOURCE_PINS.json').write_text(json.dumps(template,indent=2,sort_keys=True)+'\n')
(out/'SOURCE_DEPENDENCY_METADATA.json').write_text(json.dumps(dependencies,indent=2,sort_keys=True)+'\n')
manifest={'schema':'finite-shared-replay-epoch1-admission-source-manifest-v1',
          'files':[receipt(p) for p in sorted(out.iterdir()) if p.is_file()],
          'actual_input_callbacks_executed':0,'source_epochs_generated':0,'scientific_fits':0,'optimizer_updates':0,
          'execution_enabled':False,'root_read_and_admission_pending':True,'GO':False}
(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
for p in out.iterdir():
 if p.is_file():p.chmod(0o444)
print(json.dumps({'manifest':receipt(out/'MANIFEST.json'),'source_dependencies':dependencies,
                  'template_with_source_pins':template,'source_epochs_generated':0,'scientific_fits':0}))
'''

if __name__=='__main__':
    files=[]
    for path in sorted(HERE.iterdir()):
        if path.suffix not in ('.py','.json','.md') or not path.is_file() or path.name in {'STAGE_SOURCE.json','ALLOCATION_HANDOFF.json'}:
            continue
        raw=path.read_bytes();files.append({'name':path.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    encoded=base64.b64encode(zlib.compress(json.dumps({'destination':HERE.name,'files':files}).encode(),9)).decode()
    assert len(encoded)<48000,'Use chunked transport if proposal grows'
    r=invoke(REMOTE,[encoded],HERE,'STAGE_SOURCE',180);assert r['exit_code']==0,'Preserve preparation failure'
    value=json.loads(r['stdout']);(HERE/'ALLOCATION_HANDOFF.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'manifest':value['manifest'],'source_epochs_generated':0,'scientific_fits':0}))
