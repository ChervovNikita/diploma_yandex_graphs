"""Adapt a used root client for the exact independently reviewed zero-update stage."""
import ast
from datetime import datetime,timezone
import difflib
import hashlib
import json
from pathlib import Path

P=Path(__file__).resolve().parent
OLD=P/'pubmed_native_only_continuation_control_execution_root_20261004_v1/stage_admit_control.py'
E=P/'pubmed_shared4_zero_update_first_batch_repeat_execution_root_20261004_v1'
S=P/'pubmed_shared4_zero_update_first_batch_repeat_source_20261004_v2'
R=P/'pubmed_shared4_zero_update_first_batch_repeat_independent_source_review_20261004_v2'
PIN='7219d7d7863b431d4d4c2b8775810211ce3ebdd426ec05de8c58e970efb1a36f'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(S/'MANIFEST.json')==PIN
review=json.loads((R/'REVIEW.json').read_text())
assert review['status']=='PASS' and review['candidate_manifest_sha256']==PIN and review['execution_authorized'] is False
assert not review.get('blocking_findings')
E.mkdir()
original=OLD.read_text()
code=original.replace('pubmed_native_only_continuation_control_source_20261004_v2',S.name)
code=code.replace('456a22a28420268e9ae47fd49f6ac322e36856b40a313340da4b31bde954ce7e',PIN)
code=code.replace('pubmed_native_control_v1_','pubmed_zero_update_repeat_v1_')
code=code.replace('native_continuation_control','first_batch_repeat')
code=code.replace('NATIVE_CONTROL_ATTEMPT_SPENT.json','ZERO_UPDATE_REPEAT_ATTEMPT_SPENT.json')
start=code.index(" save('ROOT_ADMISSION.json',")
end=code.index(' rows=[]\n',start)
code=code[:start]+''' save('ROOT_ADMISSION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),
  status='APPROVED_ONE_FROZEN_FOUR_PREFIX_ZERO_UPDATE_SHARED4_DIAGNOSTIC',
  source_manifest_sha256=PIN,independent_source_review_sha256=rb['sha256'],
  optimizer_updates=0,prefixes=4,complete_native_epochs=0,complete_VALID_serves=0,
  scope='Two independently restored units from the exact owned failed epoch5 engineering state; two first-batch prefixes each. TRAIN/raw features only.',
  original_failed_shared4_qualification_preserved=True,original_rule_and_workload_unchanged=True,
  scientific_fit_admitted=False,state_donor_allowed=False,VALID_TEST_access=False,
  storage_alias_source_scope_reviewed=True,actual_storage_API_unverified_before_execution=True,
  invalid_or_opaque_preforward_state_isolated_before_numeric_interpretation=True,
  no_causal_kernel_or_bridge_attribution=True,no_extra_prefix_or_favourable_fallback=True,
  automatic_retry=False))
''' +code[end:]
code=code.replace('engineering_updates_planned=252','engineering_updates_planned=0')
code=code.replace('OWNED_FRESH_NATIVE_REPEAT_CONTROL_LAUNCHED','OWNED_FOUR_PREFIX_ZERO_UPDATE_SHARED4_DIAGNOSTIC_LAUNCHED')
code=code.replace('EXACT_REVIEWED_NATIVE_CONTROL_SOURCE_STAGED','EXACT_REVIEWED_ZERO_UPDATE_SHARED4_SOURCE_STAGED')
code=code.replace('Exact independently reviewed native control staged; no numerical execution.',
                  'Exact independently reviewed zero-update source staged; no numerical execution.')
anchor=" code+='print(json.dumps(dict(status=\"EXACT_REVIEWED_ZERO_UPDATE_SHARED4_SOURCE_STAGED\""
assert code.count(anchor)==1
pos=code.index(anchor)
addition=''' code+='owned_metadata={}\\nfor name,row in '+repr(binding['owned_failure_metadata'])+'.items():\\n p=phase/row["path"];assert p.resolve().is_relative_to(phase/'+repr(binding['failed_qualification_execution_root'])+');b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==row["sha256"];assert row["bytes"] is None or len(b)==row["bytes"];owned_metadata[name]=dict(path=row["path"],bytes=len(b),sha256=row["sha256"])\\n'
'''
code=code[:pos]+addition+code[pos:]
code=code.replace('prerequisites=prerequisites,numerical_execution=False)))',
                  'prerequisites=prerequisites,owned_metadata=owned_metadata,numerical_execution=False)))')
anchor2=" stage='first_batch_repeat';release=json.loads((SOURCE/'ROOT_RELEASE_TEMPLATE.json').read_text())\n"
assert code.count(anchor2)==1
code=code.replace(anchor2,anchor2+''' for name,row in binding['owned_failure_metadata'].items():
  observed=staged['owned_metadata'][name]
  assert observed['path']==row['path'] and observed['sha256']==row['sha256']
  assert release[name+'_sha256']==observed['sha256']
''')
old_resource='gpus=[s.split(",") for s in r.stdout.splitlines() if s.split(",")[0].strip()=='
new_resource='assert {s.split(",")[0].strip() for s in r.stdout.splitlines()}=={"GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998","GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced"}\\ngpus=[s.split(",") for s in r.stdout.splitlines() if s.split(",")[0].strip()=='
assert code.count(old_resource)==1
code=code.replace(old_resource,new_resource)
ast.parse(code)
(E/'stage_admit_control.py').write_text(code)
(E/'CLIENT_VS_NATIVE_CONTROL.diff').write_text(''.join(difflib.unified_diff(
    original.splitlines(True),code.splitlines(True),fromfile=str(OLD.relative_to(P)),
    tofile=E.name+'/stage_admit_control.py')))
(E/'CLIENT_PREPARATION.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),
    source_manifest_sha256=PIN,review_sha256=sha(R/'REVIEW.json'),
    client_sha256=sha(E/'stage_admit_control.py'),previous_client_sha256=sha(OLD),
    stage_release_launch=False,AST_parse=True,
    admitted_workload='none_until_separate_stage_and_root_admission',
    modifications=['Exact v2 source/review identity and stage paths.',
      'Frozen four prefixes and zero updates in root record.',
      'Authenticate existing owned failure scalar metadata.',
      'Authenticate exact two physical GPU UUIDs; retain selected GPU memory guard.']),indent=2)+'\n')
print(json.dumps(dict(prepared=E.name,source_review=review['status'],stage_or_launch=False)))
