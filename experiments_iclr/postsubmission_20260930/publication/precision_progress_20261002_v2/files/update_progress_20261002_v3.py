"""Record verified training progress and distinct scoped reading; no scientific scoring."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

phase = Path(__file__).resolve().parent
now = datetime.now(timezone.utc).isoformat()
snapshot = phase / 'coordination_snapshots/20261002_precision_training_progress_v3'
snapshot.mkdir(exist_ok=False)
for name in ('PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json', 'REVIEW_RESPONSE_TRACKER.md'):
    shutil.copyfile(phase / name, snapshot / name)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

coordinator = phase / 'graph_init_precision_continuation_v3_cap_binding/coordinator_run_v3'
counts = {}
terminals = []
seconds = 0.0
remote_phase = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/'
for path in sorted(coordinator.glob('*COMPLETED.json')):
    key = path.name.removesuffix('_COMPLETED.json')
    completion = json.loads(path.read_text())
    claim = json.loads((coordinator / (key + '_LAUNCH_CLAIM.json')).read_text())['attempt']
    t = completion['phase_terminal']
    if not t['path'].startswith(remote_phase):
        raise RuntimeError('Unexpected phase terminal scope')
    local = phase / t['path'][len(remote_phase):]
    terminal = json.loads(local.read_text())
    if sha(local) != t['sha256'] or terminal['completed'] is not True or terminal['final_labels_read'] is not False:
        raise RuntimeError('Unverified completion or early final-label access')
    counts[claim['phase']] = counts.get(claim['phase'], 0) + 1
    seconds += completion['whole_supervised_seconds']
    terminals.append({'key': key, 'phase': claim['phase'], 'arm': claim.get('arm'),
                      'phase_terminal_sha256': t['sha256'], 'completion_sha256': sha(path)})
progress = {'UTC': now, 'counts': counts, 'whole_supervised_seconds_completed_phases': seconds,
            'ongoing_phase_seconds_excluded': True, 'final_labels_read': False,
            'predictive_advantage_claimed': False, 'terminals': terminals}
(snapshot / 'VERIFIED_PROGRESS.json').write_text(json.dumps(progress, indent=2) + '\n')

old_index = phase / 'literature_memory/index_v12/LITERATURE_INDEX.json'
index = json.loads(old_index.read_text())
index['predecessor_index_sha256'] = sha(old_index)
index['created_UTC'] = now
new_index = phase / 'literature_memory/index_v13'
new_index.mkdir(exist_ok=False)
round20_path = phase / 'distinct_mechanism_round20_v1/EVIDENCE.json'
round20 = json.loads(round20_path.read_text())
for item in round20['new_distinct_scoped_primary_papers']:
    conclusion = {k: item[k] for k in ('key', 'canonical_id', 'title', 'version_date', 'primary_url', 'read_scope', 'conclusion')}
    conclusion['read_status'] = 'primary_read_scoped'
    index['paper_records'].append({'canonical_id': item['canonical_id'],
        'conclusion_file': round20_path.relative_to(phase).as_posix(),
        'conclusion_file_sha256': sha(round20_path), 'conclusion': conclusion})
f1_path = phase / 'industrial_followup_small_task_scout_v2/PAPER_CONCLUSIONS.json'
f1 = json.loads(f1_path.read_text())
for item in f1['records']:
    index['paper_records'].append({'canonical_id': item['canonical_id'],
        'conclusion_file': f1_path.relative_to(phase).as_posix(),
        'conclusion_file_sha256': sha(f1_path), 'conclusion': item})
for p, kind in ((round20_path, 'scoped_primary_curvature_and_graph_boosting'),
                (f1_path, 'small_temporal_task_source_findings')):
    index['existing_packets'].append({'path': p.relative_to(phase).as_posix(), 'sha256': sha(p), 'kind': kind})
(new_index / 'LITERATURE_INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
(new_index / 'README.md').write_text(
    '# Literature memory\n\nConsult this index before fetching or rereading. '
    f'It preserves all earlier entries and contains {len(index["paper_records"])} conclusion records; '
    'this is not a count of full-paper reads. New entries bind scoped SSD/BGNN reading and '
    'RelGT/F1 protocol findings. Exact versions, passages, unresolved issues and execution limits '
    'remain in the linked source packets. Graph-filtered initialization has no established '
    'predictive or novelty claim.\n')

ledger_path = phase / 'research_ledger.json'
ledger = json.loads(ledger_path.read_text())
ledger['current_status_update_UTC'] = now
for key in ('last_updated_utc', 'updated_UTC', 'updated_utc'):
    ledger[key] = now
ledger['status'] = 'fixed_precision_study_training_industrial_mount_isolation_diagnosis'
ledger['graph_initializer_precision_execution_progress_v3'] = {
    'counts': counts, 'verified_progress': str((snapshot / 'VERIFIED_PROGRESS.json').relative_to(phase)),
    'verified_progress_sha256': sha(snapshot / 'VERIFIED_PROGRESS.json'),
    'whole_supervised_seconds_completed_phases': seconds, 'ongoing_seconds_excluded': True,
    'final_labels_read': False, 'predictive_advantage_claimed': False}
ledger['industrial_runtime_image_assembly_and_public_probe_v1'] = {
    'assembly_exit_code': 0, 'image_manifest_sha256': sha(phase / 'industrial_runtime_image_root_v1/runtime_image_v1/RUNTIME_IMAGE_MANIFEST.json'),
    'public_export_passed': True, 'isolation_probe_passed': False,
    'failure': 'unshare automatic root propagation denied; manual root-private propagation also denied',
    'failure_evidence': 'industrial_runtime_image_root_v1/public_probe_run_v1/ROOT_LAUNCH.json',
    'capability_evidence': 'industrial_runtime_image_root_v1/NAMESPACE_CAPABILITY_v1.json',
    'all_namespaces_without_automatic_propagation_passed': True,
    'label_decoding_performed': False, 'model_import_or_GPU_compute_performed': False,
    'dependent_industrial_science': 'not admitted while isolated mount route is unresolved'}
ledger['closest_graph_init_primary_round20_v1'] = {
    'report': 'distinct_mechanism_round20_v1/REPORT.md', 'evidence_sha256': sha(round20_path),
    'conclusion': 'Balanced-copy residual curvature and graph-dependent gradient ensemble fitting are close prior. Graph-to-optimization alignment remains an unproven empirical premise.'}
ledger['small_relational_task_scout_v2'] = {
    'report': 'industrial_followup_small_task_scout_v2/REPORT.md', 'conclusions_sha256': sha(f1_path),
    'native_readiness': False, 'reason': 'Forecast-time identity collisions, unfiltered fallback, unresolved configuration selection.',
    'next_work': 'Correct temporal sampler and strong readily available baseline feasibility; no execution yet.'}
ledger['latest_literature_memory'] = {'path': 'literature_memory/index_v13',
    'index_sha256': sha(new_index / 'LITERATURE_INDEX.json'), 'conclusion_records': len(index['paper_records'])}
ledger['latest_local_fetch'] = 160
ledger['active_parallel_work_current'] = [
    {'agent': '/root/post_failure_distinct_method_v1', 'task': 'Graph-band versus generic curvature selection; precise falsifiable contribution'},
    {'agent': '/root/heterogeneous_factor_gap_v1', 'task': 'Correct small temporal task baseline feasibility'},
    {'agent': '/root/graph_init_source_audit_v1', 'task': 'No-sudo namespace mount diagnosis without relaxing isolation'}]
ledger['current_priority'] = 'Complete live fixed graph-initialization cohort; resolve industrial isolated mounting; test precise distinct ideas against closest prior.'
ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')

text = f'''# Current research status

Updated: {now}. The research goal remains incomplete.

## Running experiment

The fixed graph-initialization study is running on the authorized one-GPU allocation. Verified completion counts are {counts.get('qualify', 0)}/6 complete-graph numerical checks, {counts.get('warm', 0)}/6 warm training trajectories, {counts.get('initialize', 0)}/30 initializations and {counts.get('fit', 0)}/30 continuation fits. The warm trajectories performed 900 optimizer updates. Completed phases consumed {seconds:.1f} whole supervised seconds; this excludes running phases and is not an ETA. Final labels remain closed. No predictive improvement is claimed.

The candidate uses graph-filtered training errors to initialize private BatchEnsemble factors after a shared warm start. Five fixed arms compare graph guidance, common descent, random directions, permuted topology and warm copying on PolyFormer-Mono/Squirrel and Polynormer-r/Photo, with seeds17/29/43 paired to splits0/1/2. The full cohort has72 phases. Its425,700-second cap sum is a maximum bound, not expected runtime. Do not restart the live coordinator on an observation timeout.

## Scientific assessment

SSD already selects balanced-copy residual curvature; BGNN already fits ensemble components to graph-dependent gradient updates. LoRA-GA and functional repulsive ensembles further constrain broad initialization claims. The unresolved premise is whether correctly aligned graph-error bands yield useful private-route learning beyond the controls. First-order band construction alone cannot guarantee favorable curvature or graph-frequency-restricted predictions. Distinct analysis checks whether a curvature-selected extension would add anything beyond generic curvature selection.

Literature conclusions and exact reading limits are saved in literature_memory/index_v13, with {len(index['paper_records'])} conclusion records rather than that many full-paper reads. The small RelBench F1 task has a recent RelGT trainer, but inspected forecast-time context collisions, unfiltered fallback sampling and unresolved selection prevent native execution readiness. A correct temporal baseline route is being assessed before dataset execution.

## Industrial graph preparation

The Tolokers2 source packet passed independent source review and the dedicated candidate runtime passed earlier package/CUDA imports. Its allowlist image was assembled successfully. The public-only extraction succeeded without opening target values. Actual worker isolation failed before model imports: unshare denied root filesystem propagation. User/mount/PID/network namespaces themselves work with automatic propagation disabled; manual root-private propagation still fails. Minimal safe mount capabilities are being diagnosed. Label export, full-graph model qualification and useful fits remain unadmitted until actual isolation passes. No sudo or unisolated fallback is used. GPU science will serialize after the current study.

## Preserved findings and publication

The earlier54-fit coordinate screen remains STAGE1_NO_GO. The original initialization study failed a Photo scalar finite-difference check before useful training. The exact-state diagnostic and reduction-only precision amendment, metadata registration failure, missing predecessor logs and prelaunch cap-field exceptions remain preserved. New numerical checks passing does not convert old failures into favorable outcomes. Original paper scores remain unchanged.

Latest verified pushed revision:3cb4b100e712c1b33848d3b5afcbba7662dd8fe0 on codex/postsubmission-research-20260930. New assembly/probe/literature progress is recorded locally pending its next source/evidence publication. Explicit fetches through160 are retained. Independent source audits are engineering reviews, not paper acceptance recommendations.

## Scope and outcome

Work remains inside the local project and authorized remote repository. Only anogena-2 on port2222 and GPU UUID GPU-44039938-fd82-41d2-fefd-de71514e2fac are authorized. Its physical minor is7; it is one device. Never access the seven-GPU account. The18.77 route remains unresolved. No sudo, PDF compilation, GENLINK, Desktop access or original-score recalculation.

No new predictive advantage, established novelty, revised manuscript or acceptance verdict is claimed. Fresh independent paper review requires a complete defensible manuscript/evidence packet and no author history or requested verdict.
'''
for name, heading in [('PUBLIC_STATUS.md', '# Current research status'), ('RESEARCH_STATE.md', '# GNNM post-submission research state')]:
    (phase / name).write_text(text.replace('# Current research status', heading, 1))
tracker = phase / 'REVIEW_RESPONSE_TRACKER.md'
s = tracker.read_text()
start = s.index('The current method candidate is ')
end = s.index('\n\n| Required conclusion', start)
s = s[:start] + ('The current candidate initializes private factors from graph-filtered training errors after a common warm start. '
    f'The fixed precision study has completed all six numerical checks, all six warm trajectories, all30 initializations and {counts.get("fit", 0)} of30 continuations. '
    'Its final labels remain closed. The earlier54-fit coordinate screen is STAGE1_NO_GO. '
    'Tolokers public extraction succeeded, but actual isolated mounting remains unresolved before labels/model execution. '
    'No new predictive advantage, established novelty or manuscript is claimed.') + s[end:]
s = s.replace('Precisely attribute TabM/BatchEnsemble, PreGS, graph residual propagation, gradient diversity and warm-copy ancestry.',
    'Precisely attribute TabM/BatchEnsemble, PreGS, graph residual propagation, LoRA-GA, functional repulsion, warm copying, SSD balanced-copy curvature and BGNN graph-gradient boosting.')
s = s.replace('The newly pushed source/evidence branch head is28d95833fb86f56e96ce07e64a79d49f2954994e.',
    'The latest verified pushed source/evidence branch head is3cb4b100e712c1b33848d3b5afcbba7662dd8fe0.')
tracker.write_text(s)
print(json.dumps({'counts': counts, 'completed_whole_seconds': round(seconds, 1),
                  'literature_records': len(index['paper_records']), 'snapshot': str(snapshot)}))
