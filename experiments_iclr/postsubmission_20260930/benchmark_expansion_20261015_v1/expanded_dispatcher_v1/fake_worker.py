"""Bounded fake subprocess. It imports no task/model/data libraries."""
import hashlib,json,os
from pathlib import Path
import sys,time

case=json.loads(sys.argv[1]);output=Path(case['output']);output.mkdir(parents=True,exist_ok=True)
assert 'FAKE_DISPATCH_FIXTURE' in str(output.resolve())
time.sleep(float(case.get('sleep_seconds',.04)))
mode=case.get('mode','complete')
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
# These .pt files are deliberately plain text fake receipts, never real tensors.
checkpoint=output/'recovery.pt';checkpoint.write_text('FAKE COMPLETE EPOCH; NO TENSOR\n')
if mode=='interrupt':raise SystemExit(9)
status='failed'if mode=='fail'else'no_selected_state'if mode=='no_selected'else'complete'
raw={'schema':'native-full-epoch-fit-result-v1','status':status,'job_id':case['id'],'job_hash':case.get('job_hash'),
 'TEST_accessed':False,'invocation_wall_seconds':float(case.get('sleep_seconds',.04)),
 'optimizer_calls_this_invocation':{'attempts':[0],'completed':[0]},'resources':{'FAKE_METADATA_ONLY':True},
 'fake_metadata_only':True,'resume_argument_recorded':case.get('resume')}
if case['kind']=='fit'and status=='complete':
 selected=output/'selected_VALID.pt';selected.write_text('FAKE VALID AUTHORITY; NO LOGITS OR TENSOR\n');raw['selected_VALID_export']=receipt(selected)
 if mode=='tampered_export':selected.write_text('FAKE TAMPERED EXPORT; NO TENSOR\n')
elif case['kind']!='fit':
 raw={'schema':'independent-own-selected-VALID-bank-result-v1','status':status,'TEST_accessed':False,
      'members':case['members'],'extra_training_fits':0,'fake_metadata_only':True}
if mode=='wrong_identity':raw['job_hash']='wrong'
(output/'RESULT.json').write_text(json.dumps(raw,sort_keys=True)+'\n')
raise SystemExit(3 if mode=='fail'else 0)
