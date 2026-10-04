"""Metadata/source sealing only; never imports numerical modules or dispatches."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import ast
import difflib
import json
import shlex
P=Path(__file__).resolve().parent.parent
R=Path(__file__).resolve().parent
S=P/'ncnc_frozen_all25_heldout_source_preparation_20261004_v2'
OLD_S=P/'ncnc_frozen_all25_heldout_source_preparation_20261004_v1'
OLD_R=P/'ncnc_frozen_all25_heldout_release_preparation_20261004_v1'
REMOTE_P=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930')
REMOTE_S=REMOTE_P/S.name; REMOTE_R=REMOTE_P/R.name

def sha(p): return sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(v if isinstance(v,str) else json.dumps(v,indent=2,allow_nan=False)+'\n')
def pin(p,remote=None): return dict(path=str(remote or p),bytes=p.stat().st_size,sha256=sha(p))
def replace_paths(v):
    if isinstance(v,str): return v.replace(OLD_R.name,R.name).replace(OLD_S.name,S.name)
    if isinstance(v,list): return [replace_paths(x) for x in v]
    if isinstance(v,dict): return {k:replace_paths(x) for k,x in v.items()}
    return v

original=[]
for row in json.loads((OLD_S/'MANIFEST.json').read_text())['files']:
    p=OLD_S/row['path']; assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']; original.append(pin(p))
old_handoff=json.loads((OLD_R/'ROOT_PREPARATION_HANDOFF.json').read_text())
for row in old_handoff['local_artifact_pins']:
    p=Path(row['local_path']); assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
checks=[]
for p in sorted([*S.glob('*.py'),*R.glob('*.py')]):
    ast.parse(p.read_text()); compile(p.read_text(),str(p),'exec'); checks.append(str(p.relative_to(P)))
write(S/'SOURCE_ONLY_CHECK.json',dict(schema='ncnc-frozen-all25-heldout-source-only-check-v2',UTC=datetime.now(timezone.utc).isoformat(),AST_and_compile_PASS=checks,original_v1_seal_verified_unchanged=True,original_v1_source_pins=original,source_entry_executed=False,Torch_imported=False,TEST_arrays_scores_read=False,study_checkpoint_tensors_loaded=False,numerical_qualification_run=False,remote_dispatch_or_staging=False,GPU_work=False))
rows=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(S.iterdir()) if p.is_file() and p.name not in ('MANIFEST.json','SEAL.json')]
write(S/'MANIFEST.json',dict(schema='ncnc-frozen-all25-heldout-source-manifest-v2',files=rows))
source_sha=sha(S/'MANIFEST.json')
write(S/'SEAL.json',dict(schema='ncnc-frozen-all25-heldout-source-seal-v2',manifest_sha256=source_sha,payloads=len(rows),source_only=True,execution_authorized=False,successor_of_manifest_sha256=sha(OLD_S/'MANIFEST.json')))
# Bind the separate stdlib fallback even if the child's imports/admission fail.
p=R/'supervise_heldout_once.py'; value=p.read_text()
import re
value=re.sub(r"EXPECTED_SOURCE_SHA = '[^']+'", "EXPECTED_SOURCE_SHA = '"+source_sha+"'",value)
value=re.sub(r"EXPECTED_ACCOUNTING_SHA = '[^']+'", "EXPECTED_ACCOUNTING_SHA = '"+sha(S/'heldout_accounting.py')+"'",value)
write(p,value)
p=R/'dispatch_heldout_once.py'; value=re.sub(r"RUNNER_SHA256 = '[^']+'", "RUNNER_SHA256 = '"+sha(R/'supervise_heldout_once.py')+"'",p.read_text());write(p,value)
for p in (R/'supervise_heldout_once.py',R/'dispatch_heldout_once.py'):
    ast.parse(p.read_text()); compile(p.read_text(),str(p),'exec')
original_release=json.loads((OLD_R/'ROOT_HELDOUT_RELEASE_DISABLED_CANDIDATE.json').read_text())
release=replace_paths(original_release)
policy=release['policy']; identity=release['identity']; invocation=release['authorized_invocations'][0]
qa_invocation=dict(stage='fabricated_heldout_wrapper_qualification',output_directory=str(REMOTE_R/'qualification/run01'),cuda_visible_devices=release['cuda_visible_devices'])
common=dict(identity=identity,heldout_source_manifest_sha256=source_sha,family_lock_sha256=release['family_lock']['sha256'],v4_audit_result_sha256=release['v4_audit_result']['sha256'],invocation=invocation,qualification_invocation=qa_invocation,policy=policy,supervisor_source=pin(R/'supervise_heldout_once.py',REMOTE_R/'supervise_heldout_once.py'),dispatcher_source=pin(R/'dispatch_heldout_once.py',REMOTE_R/'dispatch_heldout_once.py'))
resource=replace_paths(json.loads((OLD_R/'RUNTIME_RESOURCE_ADMISSION_DISABLED_CANDIDATE.json').read_text()));resource.update(common);resource.update(status='PENDING_ROOT_ADMISSION',TEST_execution_authorized=False,basis='Original v4 TRAIN-only VALID reserved peak 16550723584 bytes is context only. Successor TRAIN+VALID official-query serving and new private evidence writes are not qualified by source preparation.',fabricated_qualification_actual_PASS_required=True)
review=replace_paths(json.loads((OLD_R/'INDEPENDENT_HELDOUT_SOURCE_REVIEW_DISABLED_CANDIDATE.json').read_text()));review.update(common);review.update(status='PENDING_INDEPENDENT_DELTA_REVIEW',reviewed_parent=pin(P/'ncnc_all25_heldout_fresh_source_review_20261004_v1/REPORT.md'),required_delta_review=['B1 all25 bounded/import failure closure; entry events persisted before calls; lower-bound event counts and unknown actual in-flight work','B2 cell phase independent of member validation; actual official evaluator attempt/return separate from strict helper return/validation','Raw returns, served arrays, evaluator returns, candidate summary and partial private write bytes kept with exact identities; failed public metrics all null','18 minimal fabricated cases: source only, original40 scorer baseline and52 stub fault entries declared','Actual fabricated PASS required by heldout metadata admission; separate source/runtime/resource/root enabled release'],execution_authorized=False)
write(R/'RUNTIME_RESOURCE_ADMISSION_DISABLED_CANDIDATE.json',resource);write(R/'INDEPENDENT_HELDOUT_SOURCE_REVIEW_DISABLED_CANDIDATE.json',review)
plan=json.loads((S/'FABRICATED_QUALIFICATION_PLAN.json').read_text())
qa_pending=dict(schema='ncnc-heldout-wrapper-fabricated-qualification-v2',status='PENDING_ROOT_EXECUTION',heldout_source_manifest_sha256=source_sha,source_entry=pin(S/'heldout_qualification.py',REMOTE_S/'heldout_qualification.py'),runtime_authority=release['runtime_authority'],original_family_identity=identity,invocation=qa_invocation,policy=policy,case_count=len(plan['cases']),cases=[dict(case=n,status='NOT_ATTEMPTED') for n in plan['cases']],actual_original_scorer_calls=None,actual_stub_scorer_entries=None,fabricated_inputs_only=True,TEST_opened=False,study_data_or_selected_checkpoint_accessed=False,training_updates=0,automatic_retry=False,run_performed=False,execution_receipt=False)
write(R/'FABRICATED_QUALIFICATION_PENDING_CANDIDATE.json',qa_pending)
release.update(heldout_source_manifest_sha256=source_sha,heldout_wrapper_fabricated_qualification=pin(R/'FABRICATED_QUALIFICATION_PENDING_CANDIDATE.json',REMOTE_R/'FABRICATED_QUALIFICATION_PENDING_CANDIDATE.json'),independent_heldout_source_review=pin(R/'INDEPENDENT_HELDOUT_SOURCE_REVIEW_DISABLED_CANDIDATE.json',REMOTE_R/'INDEPENDENT_HELDOUT_SOURCE_REVIEW_DISABLED_CANDIDATE.json'),runtime_resource_admission=pin(R/'RUNTIME_RESOURCE_ADMISSION_DISABLED_CANDIDATE.json',REMOTE_R/'RUNTIME_RESOURCE_ADMISSION_DISABLED_CANDIDATE.json'),TEST_data_authority=pin(R/'TEST_DATA_AUTHORITY_METADATA_CANDIDATE.json',REMOTE_R/'TEST_DATA_AUTHORITY_METADATA_CANDIDATE.json'),execution_enabled=False,root_authorization_reference=None,TEST_access_authorized=False)
write(R/'ROOT_HELDOUT_RELEASE_DISABLED_CANDIDATE.json',release)
v4=json.loads((P/'ncnc_v4_scientific_audit_release_preparation_20261004_v1/ROOT_ORDINARY_AUDIT_RELEASE_DISABLED_CANDIDATE.json').read_text())
qa=dict(schema='ncnc-heldout-wrapper-fabricated-root-release-v2',execution_enabled=False,root_authorization_reference=None,fabricated_inputs_only=True,TEST_access_authorized=False,study_checkpoint_access_authorized=False,heldout_source_manifest_sha256=source_sha,authorized_stages=[qa_invocation['stage']],authorized_invocations=[qa_invocation],output_directory=qa_invocation['output_directory'],cuda_visible_devices=release['cuda_visible_devices'],original_family_identity=identity,policy=policy,runtime_authority=release['runtime_authority'],independent_source_review=release['independent_heldout_source_review'],supervisor_source=common['supervisor_source'],dispatcher_source=common['dispatcher_source'],minimum_GPU_free_MiB=16384,minimum_host_MemAvailable_bytes=16*1024**3,maximum_child_wall_seconds=1800,maximum_sampled_owned_session_RSS_bytes=32*1024**3,root_owned_physical_supervision_required=True)
for k in ('driver_root','design_root','prototype_root','resource_root','driver_manifest_sha256','design_manifest_sha256','prototype_manifest_sha256','resource_manifest_sha256'): qa[k]=v4[k]
write(R/'ROOT_FABRICATED_QUALIFICATION_DISABLED_CANDIDATE.json',qa)
heldout_command='cd '+shlex.quote(str(REMOTE_P.parent.parent))+' && /usr/bin/python3 -I -S -B '+shlex.quote(str(REMOTE_R/'dispatch_heldout_once.py'))+' --root-release '+shlex.quote(str(REMOTE_R/'ROOT_HELDOUT_RELEASE_DISABLED_CANDIDATE.json'))+' --release-sha256 '+sha(R/'ROOT_HELDOUT_RELEASE_DISABLED_CANDIDATE.json')+'\n'
write(R/'GPU77_DISABLED_DISPATCH_CANDIDATE.txt',heldout_command)
qa_command=shlex.quote(str(json.loads((P/'graph_ncNC_predictive_runtime_authority_root_20261003_v1/RUNTIME_AUTHORITY.json').read_text())['interpreter_path']))+' -B '+shlex.quote(str(REMOTE_S/'heldout_qualification.py'))+' --root-release '+shlex.quote(str(REMOTE_R/'ROOT_FABRICATED_QUALIFICATION_DISABLED_CANDIDATE.json'))+' --release-sha256 '+sha(R/'ROOT_FABRICATED_QUALIFICATION_DISABLED_CANDIDATE.json')+'\n'
write(R/'DISABLED_FABRICATED_QUALIFICATION_ENTRY_CANDIDATE.txt',qa_command)
diffs=[]
for old,new in ((OLD_S,S),(OLD_R,R)):
    names=sorted(set(p.name for p in old.glob('*.py')) | set(p.name for p in new.glob('*.py')))
    for name in names:
        if name in ('build_preparation.py','stage_disabled_preparation.py','build_successor.py'): continue
        before=(old/name).read_text() if (old/name).exists() else ''
        after=(new/name).read_text() if (new/name).exists() else ''
        if before!=after: diffs.append(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old.name+'/'+name if before else '/dev/null',tofile=new.name+'/'+name)))
write(R/'SUCCESSOR_SOURCE_DIFF.patch',''.join(diffs))
write(R/'SOURCE_ONLY_PREPARATION_CHECK.json',dict(schema='ncnc-heldout-successor-preparation-static-check-v2',checks='AST parse/compile without module execution; all v1 source and handoff artifact pins verified unchanged',source_manifest_sha256=source_sha,AST_compile_files=checks,heldout_execution=False,qualification_execution=False,remote_staging=False,TEST_arrays_scores_opened=False,checkpoint_tensors_loaded=False,GPU_work=False))
write(R/'REPORT.md',f'''# Frozen all25 heldout source successor v2

Source preparation only. v1 source and every v1 handoff artifact pin were verified unchanged. No heldout or qualification entry was executed, no TEST array/score or checkpoint tensor was opened, and no remote staging, dispatch or GPU work occurred.

B1 now has fsynced all25 identities, independent cell/member phases and events before calls. A pinned stdlib helper gives the supervisor an all25, null-metric failed closure for missing/import/killed child terminals. Interrupted counters are persisted event lower bounds; an entry event can precede actual entry, so actual in-flight work is unknown. Existing child result/status bytes are preserved privately before replacement. The claim remains persistent and the dispatcher rejects its reuse.

B2 now tracks the cell phase independently of the last member's validated flag. Pooling and metric faults mark the affected cell FAILED. Strict helper attempts/returns are distinct from actual authenticated official evaluator attempts/returns and validated values. Returned scorer pools, served arrays, original evaluator values, computed candidate summary and partial private write bytes have exact release/source/cell/selected-state bindings. Public failed metrics and summary stay null.

Scientific design is unchanged: 35 original fits, 25 cells, 40 frozen selected scorer calls, 25 official metrics, True2, the original scorer/evaluator/pooling bodies, TRAIN then VALID topology, complete original TEST query order and private-minus-pooled contrast. No retry or new selection.

The sealed 18-case qualification source uses one 40-call original-scorer baseline with tiny sentinel models and fabricated in-memory snapshots, then 52 declared stub scorer entries across wrapper faults. It checks native/factorized shared topology, retained overlap/no removal/no TEST addition, query/member order, I4 raw-logit mean, three-positive normalization, official ties, exact loader same-open CPU authentication and wrong dtype/year/hash/order/pool rejection, pre-TEST/scorer/pooling/metric/final custody failures, child import failure, identified owned timeout/reap, physical failure after a successful logical child and claim reuse rejection. Original model kernels and the already qualified study selection contracts are outside this narrow QA. No numerical PASS is presumed.

Successor source manifest SHA256: `{source_sha}`. Supervisor SHA256: `{sha(R/'supervise_heldout_once.py')}`. Dispatcher SHA256: `{sha(R/'dispatch_heldout_once.py')}`. `SUCCESSOR_SOURCE_DIFF.patch` contains only changed/new Python source. `ROOT_PREPARATION_HANDOFF.json` binds all local/remote descriptors and commands. Remote descriptors are planned destinations, with no staging receipt.

Before enable: independent delta review PASS, actual separately released fabricated QA and physical terminal, root runtime/resource admission accounting for TRAIN+VALID queries and private writes, then a separately enabled one-time heldout release. `heldout_gate.py` now requires the actual successor QA receipt with all 18 cases and literal 40 original / 52 stub entries. Candidates remain disabled and numerical result fields remain unknown.
''')
release_rows=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(R.iterdir()) if p.is_file() and p.name not in ('MANIFEST.json','SEAL.json','ROOT_PREPARATION_HANDOFF.json')]
write(R/'MANIFEST.json',dict(schema='ncnc-heldout-successor-disabled-preparation-manifest-v2',files=release_rows))
write(R/'SEAL.json',dict(schema='ncnc-heldout-successor-disabled-preparation-seal-v2',manifest_sha256=sha(R/'MANIFEST.json'),payloads=len(release_rows),execution_authorized=False,qualification_performed=False,remote_staging_performed=False))
hand=dict(status='LOCAL_SEALED_DISABLED_FROZEN_ALL25_HELDOUT_SUCCESSOR_PREPARATION_V2',source_manifest_sha256=source_sha,source_manifest_path=str(S/'MANIFEST.json'),source_seal=pin(S/'SEAL.json'),release_manifest=pin(R/'MANIFEST.json'),release_seal=pin(R/'SEAL.json'),remote_source=str(REMOTE_S),remote_release=str(REMOTE_R),remote_staging_complete=False,source_only=True,heldout_launch=False,qualification_executed=False,TEST_arrays_scores_opened=False,selected_checkpoint_tensors_loaded=False,GPU_work=False,original_v1_unchanged=True,original_v1_source_manifest_sha256=sha(OLD_S/'MANIFEST.json'),original_v1_handoff=pin(OLD_R/'ROOT_PREPARATION_HANDOFF.json'),original_independent_review=pin(P/'ncnc_all25_heldout_fresh_source_review_20261004_v1/REPORT.md'),disabled_release=pin(R/'ROOT_HELDOUT_RELEASE_DISABLED_CANDIDATE.json',REMOTE_R/'ROOT_HELDOUT_RELEASE_DISABLED_CANDIDATE.json'),disabled_qualification_release=pin(R/'ROOT_FABRICATED_QUALIFICATION_DISABLED_CANDIDATE.json',REMOTE_R/'ROOT_FABRICATED_QUALIFICATION_DISABLED_CANDIDATE.json'),supervisor=common['supervisor_source'],dispatcher=common['dispatcher_source'],accounting_source=pin(S/'heldout_accounting.py',REMOTE_S/'heldout_accounting.py'),qualification_entry=pin(S/'heldout_qualification.py',REMOTE_S/'heldout_qualification.py'),source_diff=pin(R/'SUCCESSOR_SOURCE_DIFF.patch'),report=pin(R/'REPORT.md'),frozen_lock=release['family_lock'],v4_audit_result=release['v4_audit_result'],planned_scorer_calls=40,official_metric_calls=25,served_cells=25,original_unique_fits=35,policy=policy,invocation=invocation,qualification_invocation=qa_invocation,qualification_case_count=18,qualification_planned_original_scorer_calls=40,qualification_planned_stub_scorer_entries=52,disabled_dispatch_command_path=str(R/'GPU77_DISABLED_DISPATCH_CANDIDATE.txt'),disabled_qualification_entry_command_path=str(R/'DISABLED_FABRICATED_QUALIFICATION_ENTRY_CANDIDATE.txt'),local_artifact_pins=[dict(local_path=str(R/r['path']),bytes=r['bytes'],sha256=r['sha256']) for r in release_rows],missing_before_enable=['Independent successor delta source review PASS','Actual separately enabled fabricated-only 18-case ordinary runtime QA and physical terminal','Root runtime/resource admission PASS for exact successor/invocation/bounds','Separately enabled one-time heldout root release with TEST authorization; actual TEST authentication only inside child'],direct_child_invocation_limit='Normal claim/dispatcher route is single-use; child does not independently authenticate a live parent or consume a child-entry token. No new direct invocation authority.',new_numerical_qualification=False,UTC=datetime.now(timezone.utc).isoformat())
write(R/'ROOT_PREPARATION_HANDOFF.json',hand)
print(json.dumps(dict(source_manifest_sha256=source_sha,release_manifest_sha256=sha(R/'MANIFEST.json'),handoff_sha256=sha(R/'ROOT_PREPARATION_HANDOFF.json'),supervisor_sha256=sha(R/'supervise_heldout_once.py'),dispatcher_sha256=sha(R/'dispatch_heldout_once.py'),source_files=len(rows),release_files=len(release_rows),executed=False)))
