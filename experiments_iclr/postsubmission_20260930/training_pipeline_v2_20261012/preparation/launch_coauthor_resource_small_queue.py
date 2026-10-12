"""Launch the ten reviewed finite engineering cases, not scientific fits."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

from prepare_molecular_packets_allocation import GUARD, TRANSFER, invoke

PACKAGE = Path(__file__).resolve().parents[1]
LABEL = 'coauthor_resource_small_queue_root_20261012_v2'
DEST = 'coauthor_resource_small_queue_20261012_v2'

REMOTE = GUARD + r'''
encoded=(phase/(sys.argv[1]+'_transport')/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
source=stage/'queue.py';raw=base64.b64decode(payload['queue']['data'])
assert len(raw)==payload['queue']['bytes'] and hashlib.sha256(raw).hexdigest()==payload['queue']['sha256']
source.write_bytes(raw);source.chmod(0o444)
config=payload['config'];config['python']=str(repo/'.venv/bin/python');config['project_root']=str(phase)
for key in ('helper','manifest'):
 path=phase/config[key]['path'];raw=path.read_bytes()
 assert len(raw)==config[key]['bytes'] and hashlib.sha256(raw).hexdigest()==config[key]['sha256']
config_path=stage/'CONFIG.json';config_path.write_text(json.dumps(config,indent=2)+'\n');config_path.chmod(0o444)
with (stage/'owner.log').open('xb') as log:
 child=subprocess.Popen([config['python'],'-B',str(source),str(config_path)],cwd=phase,
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
receipt={'owner':{'pid':child.pid,'pgid':int(fields[2]),'start_ticks':int(fields[19])},
 'cases':len(config['case_ids']),'source_sha256':payload['queue']['sha256'],
 'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
 'scientific_fits':0,'GO_created':False,'detached':True}
(stage/'DETACHED_LAUNCH.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def main():
    out=PACKAGE/'preparation'/LABEL
    out.mkdir(exist_ok=False)
    review=json.loads((PACKAGE/'preparation/coauthor_resource_review_root_20261012_v3/REVIEW_ROSTER.json').read_text())
    methods=('polyformer_original_boundary_own','polyformer_native_own', 'polyformer_shared4_own',
             'polyformer_shared4_paired_class_full','polyformer_shared4_gncl_supcon')
    cases=[]
    for dataset in ('coauthor-cs','coauthor-physics'):
        for method in methods:
            candidates=[r for r in review['cases'] if dataset+'-mono' in r['study_id'] and r['method']==method]
            assert len(candidates)==2
            cases.append(max(candidates,key=lambda r:r['dimensions'][2]))
    assert len(cases)==10
    raw=(PACKAGE/'qa/coauthor_resource_small_queue.py').read_bytes()
    config={'helper':review['helper'],'manifest':review['manifest'],'case_ids':[r['case_id'] for r in cases],
            'output':DEST+'/execution_v1',
            'runtime':'massive_polyformer_coauthor_practical_packet_20261012_v1/sources/training_pipeline_v2_20261012'}
    payload={'destination':DEST,'config':config,'queue':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                                                       'data':base64.b64encode(raw).decode()}}
    (out/'REVIEWED_TEN_CASES.json').write_text(json.dumps({'cases':cases,'config':config,'source_sha256':payload['queue']['sha256'],
        'authorization':'user_authorized_source_preparation_and_engineering;root_reviewed_existing_zero_update_helper',
        'actual_model_updates':0,'scientific_GO':False,'scope':'ten_new_data_random_resource_cases_only'},indent=2)+'\n')
    encoded=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    result=invoke(TRANSFER,[DEST,'0',encoded],out,'TRANSFER',60)
    assert result['exit_code']==0
    result=invoke(REMOTE,[DEST,hashlib.sha256(encoded.encode()).hexdigest()],out,'DETACHED_LAUNCH',60)
    assert result['exit_code']==0, result['stdout']
    print(result['stdout'].strip())


if __name__=='__main__':
    main()
