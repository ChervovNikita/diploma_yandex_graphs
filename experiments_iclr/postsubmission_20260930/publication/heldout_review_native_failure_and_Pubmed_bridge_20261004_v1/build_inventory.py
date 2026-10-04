"""Explicit source/review/failure publication; no datasets or model states."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
paths = {PHASE / n for n in ('PUBLIC_STATUS.md','RESEARCH_STATE.md','research_ledger.json')}
paths.add(Path(__file__).resolve())

def packet(name):
    root = PHASE / name
    manifest = json.loads((root/'MANIFEST.json').read_text())
    paths.add(root/'MANIFEST.json')
    for row in manifest['files']:
        f = (root/row['path']).resolve()
        assert f.is_relative_to(PHASE)
        b = f.read_bytes()
        assert len(b) == row.get('bytes',row.get('size')) and hashlib.sha256(b).hexdigest() == row['sha256']
        paths.add(f)
    if (root/'SEAL.json').exists():
        paths.add(root/'SEAL.json')

for name in ('ncnc_all25_heldout_fresh_source_review_20261004_v1',
             'ncnc_frozen_all25_heldout_source_preparation_20261004_v1',
             'pooled_joint_native_batch_qualification_preparation_20261004_v1',
             'pubmed_shared4_extension_source_preparation_20261004_v1',
             'pubmed_shared4_bridge_qualification_source_20261004_v1'):
    packet(name)
root = PHASE/'ncnc_frozen_all25_heldout_release_preparation_20261004_v1'
for name in ('ROOT_PREPARATION_HANDOFF.json','LOCAL_PREPARATION_HANDOFF.json',
             'ROOT_HELDOUT_RELEASE_DISABLED_CANDIDATE.json','TEST_DATA_AUTHORITY_METADATA_CANDIDATE.json',
             'RUNTIME_RESOURCE_ADMISSION_DISABLED_CANDIDATE.json','INDEPENDENT_HELDOUT_SOURCE_REVIEW_DISABLED_CANDIDATE.json',
             'supervise_heldout_once.py','dispatch_heldout_once.py','SUPERVISOR_DISPATCH_DIFF.patch','REMOTE_STAGE_RECEIPT.json'):
    paths.add(root/name)
root = PHASE/'pooled_joint_native_batch_qualification_execution_root_20261004_v1'
for name in ('ROOT_SOURCE_REVIEW.json','SOURCE_STAGE_INVENTORY.json','SOURCE_RESOURCE_STAGE_RECEIPT.json',
             'ROOT_NATIVE_ADMISSION.json','ROOT_NATIVE_RELEASE.json','NATIVE_DETACHED_LAUNCH.json',
             'prepare_native_run.py','stage_missing_sources.py','stage_missing_sources_v2.py',
             'MISSING_SOURCE_STAGE_RECEIPTS_V2.json','SOURCE_STAGE_FAILURE_DIAGNOSTIC_TRANSPORT.json',
             'pooled_native_source_stage_20261004_v1_LOCAL_TRANSPORT.json'):
    paths.add(root/name)
paths.update(f for f in (root/'native_owned_monitor_v1').rglob('*') if f.is_file())
root = PHASE/'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_owned_monitoring_20261003_v1'
for sequence in (15,16):
    for kind in ('RESULT','SUMMARY','TRANSPORT'):
        paths.add(root/('MONITOR_%04d_%s.json'%(sequence,kind)))
files = []
for f in sorted(paths):
    assert f.resolve().is_relative_to(PHASE) and not f.is_symlink()
    b = f.read_bytes()
    assert len(b)<2000000
    files.append(dict(source=str(f.relative_to(PHASE)),target='experiments_iclr/postsubmission_20260930/'+str(f.relative_to(PHASE)),
                      bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
f = HERE/'README_MAIN.md'
b = f.read_bytes()
files.append(dict(source=str(f.relative_to(PHASE)),target='README.md',bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
value = dict(branch='codex/postsubmission-research-20260930',expected_head='93ca48a90f2b643c4a0c87c22c12064c34cf1fd2',
             message='Preserve heldout source review and native failure; prepare Pubmed shared4 bridge',remove=[],files=files)
with (HERE/'INVENTORY.json').open('x') as f:
    json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(dict(files=len(files),bytes=sum(f['bytes'] for f in files),large_data_models_included=False)))
