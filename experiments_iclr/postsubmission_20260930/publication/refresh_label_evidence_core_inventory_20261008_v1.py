"""Complete the not-yet-published label-evidence inventory, without fitting models."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

P = Path(__file__).resolve().parents[1]
D = P / 'publication/label_evidence_scopes_and_required_controls_20261008_v1'
S = P / 'coordination_snapshots/label_core_and_progress_before_publication_20261008_v1'
S.mkdir(exist_ok=False)

def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def binding(relative):
    path = P / relative
    return {'path': relative, 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

for name in ('PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json'):
    (S / name).write_bytes((P / name).read_bytes())
(S / 'INVENTORY_BEFORE.json').write_bytes((D / 'INVENTORY.json').read_bytes())
(S / 'README_MAIN_BEFORE.md').write_bytes((D / 'README_MAIN.md').read_bytes())

monitor = json.loads((P / 'representative_full_family_progress_root_20261008_1600_v1/ALLOCATION_MONITOR.json').read_text())
assert monitor['exit_code'] == 0
observed = monitor['result']
assert observed['owner']['PID'] == 523400 and observed['owner']['start_ticks'] == 6019318952
assert observed['progress']['fits/P_7203/PROGRESS.json']['epoch'] == 45
failed = json.loads((P / 'gpu77_connection_recovery_v1/commands/graph_relation_full12_same_owner_progress_20261008_1600_v1/RECEIPT.json').read_text())
assert failed['exit_code'] == 4

state = (P / 'RESEARCH_STATE.md').read_text()
state = state.replace('Updated 8 October 2026. Graph progress verified 14:21 UTC; allocation progress verified 14:06 UTC.',
    'Updated 8 October 2026. Allocation progress verified 15:57 UTC. Graph progress last verified 14:21 UTC; the later SSH observation failed before authentication.')
state = state.replace('I_7203 at 54/100 epochs, 14:06 UTC', 'P_7203 at 45/100 epochs, 15:57 UTC; ten fits completed')
state = state.replace('allocation snapshot: `representative_full_family_progress_root_20261008_1410_v1/`.',
    'allocation snapshot: `representative_full_family_progress_root_20261008_1600_v1/`. Later graph monitoring failed before SSH authentication; no scientific process was restarted.')
state = state.replace('It is not implemented, admitted or novelty-cleared. Attention normalization and correction learning signal require explicit source choices.',
    'A disabled callable corrector core is source-prepared and root-inspected. Full native capture/trainer/restoration/selection integration and runtime qualification remain incomplete; no scientific launch or novelty clearance exists. All-observed-nonself-neighbor normalization is fixed, with sparse-anchor dilution and weak correction feedback retained as risks.')
state += '\n### Disabled label-only core preparation\n\n`label_only_private_corrector_core_source_20261008_v1/` now supplies the four-route operator, detached native input interface, common query mask, conditional value scaling, own-CE update and mean-probability serving. The complete core and three exact factor primitives were source-inspected in `label_only_private_corrector_core_root_source_review_20261008_v1/`. This provides no numerical validation or accuracy claim. A separate agent is preparing native integration; running graph12 and Mol18 sources and gates remain unchanged.\n'
(P / 'RESEARCH_STATE.md').write_text(state)

status = (P / 'PUBLIC_STATUS.md').read_text()
status = status.replace('The molecular eighteen-fit study continues on the allocation: nine fits were complete and I_7203 was at54/100 epochs at14:06 UTC.',
    'The molecular eighteen-fit study continues on the allocation: ten fits were complete and P_7203 was at 45/100 epochs at 15:57 UTC. A later 18.77 monitoring connection failed before SSH authentication; its current job state is unverified. No jobs were restarted.')
status = status.replace('A label-only graph-correction hypothesis is retained for future source preparation with close UniMP/C&S ancestry and strict common target exclusion.',
    'A disabled label-only graph-correction core is now source-prepared and root-inspected, with close UniMP/C&S ancestry and strict common target exclusion. Native driver integration and runtime qualification remain incomplete.')
(P / 'PUBLIC_STATUS.md').write_text(status)

readme = (D / 'README_MAIN.md').read_text()
readme = readme.replace('the current fit was at54/100 epochs at14:06 UTC.',
    'ten fits were complete and the current P_7203 fit was at 45/100 epochs at 15:57 UTC.')
readme = readme.replace('It is not implemented, launched or novelty-cleared.',
    'A disabled callable corrector core is source-prepared; native integration, runtime qualification, scientific execution and novelty clearance remain incomplete.')
readme = readme.replace('Later conditions remain queued;',
    'Later conditions remain queued; a later monitoring SSH connection failed before authentication and the current job state is unverified;')
(D / 'README_MAIN.md').write_text(readme)

ledger = json.loads((P / 'research_ledger.json').read_text())
key = 'Label_only_core_source_review_and_same_owner_progress_20261008_root_v1'
assert key not in ledger
ledger[key] = {
    'UTC': datetime.now(timezone.utc).isoformat(),
    'core': binding('label_only_private_corrector_core_source_20261008_v1/core.py'),
    'root_review': binding('label_only_private_corrector_core_root_source_review_20261008_v1/REVIEW.json'),
    'sampling_note': binding('common_label_query_mask_inverse_inclusion_assessment_20261008_v1/NOTE.md'),
    'core_source_prepared': True, 'full_driver_integrated': False,
    'numerical_qualification': False, 'scientific_launch_admitted': False,
    'Mol18_progress': {'observation_UTC': observed['UTC'], 'owner': observed['owner'],
        'complete_fits': 10, 'current': 'P_7203 epoch45/100'},
    'graph12_monitor_attempt': binding('gpu77_connection_recovery_v1/commands/graph_relation_full12_same_owner_progress_20261008_1600_v1/RECEIPT.json'),
    'graph12_current_state': 'Unknown after SSH transport failed before authentication; last verified 14:21 UTC',
    'running_sources_or_gates_changed': False, 'jobs_restarted': False,
    'partial_comparative_scores_read': False, 'new_paper_read_credit': 0,
    'accuracy_advantage_established': False, 'novelty_established': False,
    'original_scores_changed': False, 'manuscript_acceptance': False,
}
save(P / 'research_ledger.json', ledger)

inventory = json.loads((D / 'INVENTORY.json').read_text())
sources = {row['source']: row['target'] for row in inventory['files']}
folders = (
    'common_label_query_mask_inverse_inclusion_assessment_20261008_v1',
    'label_only_private_corrector_core_source_20261008_v1',
    'label_only_private_corrector_core_root_source_review_20261008_v1',
    'representative_full_family_progress_root_20261008_1600_v1',
    'gpu77_connection_recovery_v1/commands/graph_relation_full12_same_owner_progress_20261008_1600_v1',
    str(S.relative_to(P)),
)
for name in folders:
    for path in (P / name).rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            relative = str(path.relative_to(P))
            sources[relative] = 'experiments_iclr/postsubmission_20260930/' + relative
for relative in (
    'publication/refresh_label_evidence_core_inventory_20261008_v1.py',
    'gpu77_connection_recovery_v1/commands/graph_relation_full12_same_owner_progress_20261008_1600_v1.txt',
):
    sources[relative] = 'experiments_iclr/postsubmission_20260930/' + relative
rows = []
for relative, target in sorted(sources.items()):
    row = binding(relative)
    assert row['bytes'] < 2_000_000
    rows.append({'source': relative, 'target': target,
                 'bytes': row['bytes'], 'sha256': row['sha256']})
inventory['files'] = rows
inventory['message'] = 'Prepare label-evidence corrector core and required ensemble controls'
save(D / 'INVENTORY.json', inventory)
print(json.dumps({'files': len(rows), 'bytes': sum(row['bytes'] for row in rows),
                  'runtime_or_scientific_execution': False}))
