"""Prepare a small new-data engineering roster; no constructors or CUDA calls."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

from prepare_molecular_packets_allocation import GUARD, TRANSFER, invoke

PACKAGE = Path(__file__).resolve().parents[1]
LABEL = 'coauthor_resource_review_root_20261012_v3'
DESTINATION = 'coauthor_resource_review_root_20261012_v3'
MASTER = 'massive_practical_master_20261012_v7/PLAN.json'
MASTER_HASH = 'c58fd5b0b590d581a2cbc5e12f7da3090c57ee781f2e1434ab835874c6874c08'

REMOTE = GUARD + r'''
encoded=(phase/(sys.argv[1]+'_transport')/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
helper=stage/'practical_zero_update_resources.py'
raw=base64.b64decode(payload['source']['data'])
assert len(raw)==payload['source']['bytes'] and hashlib.sha256(raw).hexdigest()==payload['source']['sha256']
helper.write_bytes(raw);helper.chmod(0o444)
import importlib.util
module_spec=importlib.util.spec_from_file_location('coauthor_resource_metadata',helper)
module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
plan=json.loads((phase/payload['master']).read_text())
assert module.digest(plan)==payload['master_hash'] and plan['scientific_launch_enabled'] is False
# Reuse the reviewed envelope builder, but restrict its metadata enumeration to
# the two new Coauthor studies. The returned manifest is rebound to the unchanged
# full master before any numerical command can be admitted.
expected_paths={payload['master'].rsplit('/',1)[0]+'/studies/polyformer-coauthor-practical-v1-'+dataset+'-mono.json.gz'
 for dataset in ('coauthor-cs','coauthor-physics')}
subset=dict(plan,studies=[r for r in plan['studies'] if r['path'] in expected_paths])
assert len(subset['studies'])==2
temporary=stage/'COAUTHOR_METADATA_SUBSET_PLAN.json'
temporary.write_text(json.dumps(subset,sort_keys=True,separators=(',',':'))+'\n')
manifest=module.prepare_manifest(phase,temporary,module.digest(subset))
manifest['master_plan']=module.receipt(phase,phase/payload['master'])
manifest['master_hash']=payload['master_hash']
roster=[]
for case in manifest['cases']:
 c=case['run']['config']
 roster.append({'case_id':case['case_id'],'study_id':case['study_id'],
   'method':case['run']['method'],'dimensions':case['representative_dimensions'],
   'members':c.get('members'),'loss':c.get('loss'),'config_id':case.get('source_search_run_id'),
   'prototype_bodies':1,'actual_TRAIN_rows':case['actual_TRAIN_rows'],
   'scientific_fits':0,'CUDA_executions':0})
path=stage/'MANIFEST.json';path.write_text(json.dumps(manifest,indent=2)+'\n');path.chmod(0o444)
metadata={'schema':'coauthor-resource-metadata-review-v1','master_hash':payload['master_hash'],
 'helper':module.receipt(phase,helper),'manifest':module.receipt(phase,path),
 'cases':roster,'unqualified_studies':manifest['unqualified_studies'],
 'scientific_fits':0,'CUDA_executions':0,'GO_created':False,
 'scope':'random_STATE_TRAIN_callback_and_serializer_only;not_final_all_method_admission',
 'original_master_and_sources_modified':False,'new_datasets_only':True}
path=stage/'REVIEW_ROSTER.json';path.write_text(json.dumps(metadata,indent=2)+'\n');path.chmod(0o444)
print(json.dumps(metadata))
'''


def main():
    out=PACKAGE/'preparation'/LABEL
    out.mkdir(exist_ok=False)
    helper=PACKAGE/'qa/practical_zero_update_resources.py'
    raw=helper.read_bytes()
    payload={'destination':DESTINATION,'master':MASTER,'master_hash':MASTER_HASH,
             'source':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                       'data':base64.b64encode(raw).decode()}}
    encoded=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    result=invoke(TRANSFER,[DESTINATION,'0',encoded],out,'TRANSFER',60)
    assert result['exit_code']==0
    result=invoke(REMOTE,[DESTINATION,hashlib.sha256(encoded.encode()).hexdigest()],out,'PREPARATION_TRANSPORT',120)
    assert result['exit_code']==0, result['stderr']
    value=json.loads(result['stdout'])
    (out/'REVIEW_ROSTER.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'cases':len(value['cases']), 'source_and_master_bound':True,
                      'CUDA_executions':0,'scientific_fits':0}))


if __name__=='__main__':
    main()
