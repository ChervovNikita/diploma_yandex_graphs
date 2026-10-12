"""Preserve low-cap failures and prepare two larger-reservation engineering cases."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

from prepare_molecular_packets_allocation import GUARD, TRANSFER, invoke

PACKAGE = Path(__file__).resolve().parents[1]
LABEL = 'coauthor_capacity_deferral_review_root_20261012_v1'
DEST = 'coauthor_resource_larger_reservation_20261012_v1'

REMOTE = GUARD + r'''
encoded=(phase/(sys.argv[1]+'_transport')/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
previous=phase/'coauthor_resource_small_queue_20261012_v2'
closure=json.loads((previous/'execution_v1/CLOSURE.json').read_text())
assert closure['status']=='CLOSED' and closure['passed']==8 and closure['scientific_fits']==0
failed=[r for r in closure['cases'] if r['status']=='RESOURCE_DEFERRED']
assert len(failed)==2
source=phase/'coauthor_resource_review_root_20261012_v3/practical_zero_update_resources.py'
old=source.read_bytes();assert hashlib.sha256(old).hexdigest()==payload['old_helper_sha256']
body=old.decode()
assert body.count('total*2//5')==1 and body.count("'maximum_fraction_of_total': .4")==1
body=body.replace('total*2//5','total*3//4').replace("'maximum_fraction_of_total': .4","'maximum_fraction_of_total': .75")
helper=stage/'practical_zero_update_resources.py';helper.write_text(body);helper.chmod(0o444)
def receipt(path):
 raw=path.read_bytes()
 return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
manifest=json.loads((phase/'coauthor_resource_review_root_20261012_v3/MANIFEST.json').read_text())
assert manifest['helper_sha256']==payload['old_helper_sha256']
manifest['helper_sha256']=receipt(helper)['sha256']
manifest['cases']=[c for c in manifest['cases'] if c['case_id'] in {r['case_id'] for r in failed}]
assert len(manifest['cases'])==2
manifest_path=stage/'MANIFEST.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');manifest_path.chmod(0o444)
rows=[]
for r in closure['cases']:
 result_path=phase/r['result'];value=json.loads(result_path.read_text())
 row={'case_id':r['case_id'],'status':r['status'],'result':receipt(result_path)}
 for key in ('parameter_bytes','optimizer_moment_bytes','CUDA_peak_allocated_bytes','CUDA_peak_reserved_bytes',
   'warm_train_median_seconds','constructor_seconds','serialization','actual_optimizer_updates','actual_optimizer_kernel_calls',
   'heldout_original_file_accesses','scores_emitted'):
  if key in value:row[key]=value[key]
 rows.append(row)
review={'schema':'coauthor-larger-resource-reservation-review-v1','previous_queue':receipt(previous/'execution_v1/CLOSURE.json'),
 'previous_cases':rows,'next_helper':receipt(helper),'next_manifest':receipt(manifest_path),
 'old_helper':receipt(source),'changed_behavior':'current_child_Torch_allocator_cap40pct_to75pct_only',
 'scientific_numerical_source_changed':False,'host_settings_changed':False,'old_failures_preserved':True,
 'next_scope':'two_distinct_random_TRAIN_resource_cases_with_larger_reservation;not_a_scientific_retry',
 'single_child_at_a_time':True,'keeps_at_least4GiB_current_free_headroom':True,
 'scientific_fits':0,'GO_created':False,'case_ids':[c['case_id'] for c in manifest['cases']]}
(stage/'ROOT_CAPACITY_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps(review))
'''


def main():
    out=PACKAGE/'preparation'/LABEL;out.mkdir(exist_ok=False)
    previous=json.loads((PACKAGE/'preparation/coauthor_resource_review_root_20261012_v3/REVIEW_ROSTER.json').read_text())
    payload={'destination':DEST,'old_helper_sha256':previous['helper']['sha256']}
    encoded=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    result=invoke(TRANSFER,[DEST,'0',encoded],out,'TRANSFER',60)
    assert result['exit_code']==0
    result=invoke(REMOTE,[DEST,hashlib.sha256(encoded.encode()).hexdigest()],out,'REVIEW',90)
    assert result['exit_code']==0, result['stdout']
    review=json.loads(result['stdout']);(out/'ROOT_CAPACITY_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps({'previous_passed':sum(r['status']=='TRAIN_PROXY_COMPLETE' for r in review['previous_cases']),
                      'larger_cases_prepared':len(review['case_ids']),'models_executed_now':0}))


if __name__=='__main__':main()
