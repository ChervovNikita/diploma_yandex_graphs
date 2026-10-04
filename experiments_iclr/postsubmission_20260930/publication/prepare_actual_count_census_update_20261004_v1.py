"""Create an exact small publication inventory from completed local packets."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import json
import shlex
import subprocess

P = Path(__file__).resolve().parents[1]
OUT = P/'publication/actual_count_census_update_20261004_v1'
OUT.mkdir()
PREVIOUS = json.loads((P/'publication/status_count_prior_update_20261004_v1/COMMIT_RECEIPT.json').read_text())
HEAD = PREVIOUS['commit']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


# Read only the tracked README at the expected authorized repository/head.
code = '''from pathlib import Path
import base64,json,subprocess
r=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
head=subprocess.run(['git','rev-parse','HEAD'],cwd=r,capture_output=True,text=True,check=True).stdout.strip()
assert head==HEAD
b=subprocess.run(['git','show','HEAD:README.md'],cwd=r,capture_output=True,check=True).stdout
print(json.dumps(dict(head=head,data=base64.b64encode(b).decode())))
'''.replace('HEAD\n',repr(HEAD)+'\n')
ssh=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
     '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes',
     'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
r=subprocess.run([*ssh,shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code])],capture_output=True,text=True,timeout=45)
save(OUT/'README_FETCH_TRANSPORT.json',dict(exit_code=r.returncode,stderr=r.stderr,read_only=True,expected_head=HEAD))
assert r.returncode==0,r.stderr
old=base64.b64decode(json.loads(r.stdout)['data'],validate=True).decode()
(OUT/'README_BEFORE.md').write_text(old)
marker='## Post-submission research'
start=old.index(marker)
stop=old.find('\n## ',start+len(marker))
if stop<0:stop=len(old)
section='''## Post-submission research

Active research preserves every original paper score, failed experiment, review and decision. The five-seed official ogbl-collab comparison measured GNNM at67.2909% Hits@50 versus66.4426% for a single native model and67.6298% for an independent ensemble. The exploratory single-model gain is positive in all five seeds; superiority over the independent ensemble is not established. The primary private-vs-pooled contrast is inconclusive.

Current work evaluates a TRAIN-only count-conditioned neighbour-pattern auxiliary with separate-side mixture and capable structured-single controls. Actual mathematical/gradient checks passed; the full TRAIN census found both-side nonconstant support for5.8242% of positive queries and essentially none for sampled negatives. These checks establish neither predictive improvement nor methodological novelty. Full native-batch feasibility and prospective paired experiments remain required.

Amazon Polynormer training has completed4/15 fits; the original queue continues without partial score decisions. Six native Pubmed baselines are complete. The GNNM continuation diagnostic still fails its fixed numerical comparison; a native repeat control is under repair after an independently identified supervision defect. No new manuscript acceptance has been obtained.

See [current status](experiments_iclr/postsubmission_20260930/PUBLIC_STATUS.md), [research decisions](experiments_iclr/postsubmission_20260930/RESEARCH_STATE.md) and [complete ledger](experiments_iclr/postsubmission_20260930/research_ledger.json). Large raw evidence stays on authorized servers; small hash-linked records and all conclusions remain in Git. Science is confined to the authorized one-GPU allocation and18.77 project repositories. The seven-GPU route is forwarding only.
'''
(OUT/'README_MAIN.md').write_text(old[:start]+section+old[stop:])

roots=[
 'graph_count_conditioned_pattern_cpu_supervisor_fresh_review_20261004_v3',
 'graph_count_conditioned_pattern_cpu_qualification_execution_root_20261004_v1',
 'graph_count_conditioned_pattern_cpu_qualification_root_adoption_20261004_v1',
 'pubmed_shared4_owned_continuation_diagnostic_source_20261004_v3',
 'pubmed_shared4_owned_continuation_diagnostic_fresh_review_20261004_v3',
 'pubmed_shared4_owned_continuation_diagnostic_execution_root_20261004_v2',
 'pubmed_isolated_step_divergence_root_adoption_20261004_v1',
 'graph_count_conditioned_train_support_census_preparation_20261004_v3',
 'graph_count_conditioned_train_support_census_independent_source_review_20261004_v3',
 'graph_count_conditioned_train_support_census_execution_root_20261004_v2',
 'graph_count_conditioned_train_support_census_root_adoption_20261004_v1',
 'graph_count_conditioned_joint_identifiability_assessment_preparation_20261004_v1',
 'graph_count_conditioned_pattern_loss_prototype_preparation_20261004_v2',
 'pubmed_native_only_continuation_control_source_20261004_v1',
 'pubmed_native_only_continuation_control_fresh_source_review_20261004_v1',
 'pubmed_native_only_continuation_control_fresh_source_review_20261004_v2',
 'literature_raw_asset_offload_20261004_v1',
 'coordination_snapshots/20261004_actual_count_and_pubmed_results_before_state_v1',
]
# Add only complete sealed review packets, never live source preparation.
optional=[
 'graph_count_conditioned_train_support_census_fresh_evidence_audit_20261004_v1',
 'graph_count_conditioned_train_support_census_independent_evidence_audit_20261004_v1',
 'graph_count_conditioned_pattern_loss_prototype_fresh_source_review_20261004_v2',
]
for name in optional:
 if (P/name/'MANIFEST.json').is_file() and (P/name/'SEAL.json').is_file():roots.append(name)
files=[P/'PUBLIC_STATUS.md',P/'RESEARCH_STATE.md',P/'research_ledger.json',
 P/'record_actual_count_and_pubmed_progress_20261004_v1.py',Path(__file__).resolve(),OUT/'README_BEFORE.md',
 P/'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_owned_monitoring_20261003_v1/MONITOR_0021_RESULT.json',
 P/'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_owned_monitoring_20261003_v1/MONITOR_0021_TRANSPORT.json',
 P/'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_owned_monitoring_20261003_v1/MONITOR_0021_REMOTE_CODE.py.txt']
for name in roots:
 folder=P/name;assert folder.is_dir()
 manifest=folder/'MANIFEST.json'
 if manifest.is_file():
  for row in json.loads(manifest.read_text())['files']:
   f=folder/row['path'];assert f.is_file() and not f.is_symlink()
   assert f.stat().st_size==row.get('bytes',row.get('size')) and sha(f)==row['sha256']
 files.extend(sorted(f for f in folder.rglob('*') if f.is_file()))
inventory=[]
for f in dict.fromkeys(files):
 assert f.resolve().is_relative_to(P) and not f.is_symlink() and f.stat().st_size<2_000_000
 assert f.suffix in ('.py','.json','.md','.txt','.diff','.patch','.log','.csv','.sha256')
 inventory.append(dict(source=str(f.relative_to(P)),target='experiments_iclr/postsubmission_20260930/'+str(f.relative_to(P)),bytes=f.stat().st_size,sha256=sha(f)))
f=OUT/'README_MAIN.md';inventory.append(dict(source=str(f.relative_to(P)),target='README.md',bytes=f.stat().st_size,sha256=sha(f)))
save(OUT/'INVENTORY.json',dict(branch='codex/postsubmission-research-20260930',expected_head=HEAD,
 files=inventory,remove=[],message='Record actual count-law QA, full TRAIN support census, continuation failure, and review correction'))
print(json.dumps(dict(files=len(inventory),bytes=sum(row['bytes'] for row in inventory),expected_head=HEAD,raw_models_data_media_excluded=True)))
