"""Record verified completed phases without reading scientific binary payloads."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

PHASE = Path(__file__).resolve().parents[1]
REMOTE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
STUDY = PHASE / 'graph_init_precision_execution_root_v2/study_v2'
RUN = PHASE / 'graph_init_precision_continuation_v3_cap_binding/coordinator_run_v3'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(record):
    path = PHASE / Path(record['path']).relative_to(REMOTE)
    assert path.resolve().is_relative_to(PHASE)
    assert digest(path) == record['sha256'], str(path)
    return path


def main():
    now = datetime.now(timezone.utc).isoformat()
    registry_path = STUDY / 'GRAPH_INIT_ATTEMPT_REGISTRY.json'
    assert digest(registry_path) == '715c361c82b543d4c436a565ccaa86470a9204433998d410954f2f268300307f'
    registry = read(registry_path)
    assert len(registry['attempts']) == 72
    completed = []
    outstanding = []
    for row in registry['attempts']:
        key = row['key']
        done = RUN / (key + '_COMPLETED.json')
        claim = RUN / (key + '_LAUNCH_CLAIM.json')
        if not done.exists():
            if claim.exists():
                outstanding.append({'phase': row['phase'], 'arm': row['arm'],
                                    'output': row['output'], 'launch_UTC': read(claim)['UTC']})
            continue
        terminal_path = STUDY / 'terminals' / (key + '.json')
        terminal = read(terminal_path)
        assert terminal['completed'] is True and terminal['final_labels_read'] is False
        bound(terminal['claim'])
        freeze_path = bound(terminal['freeze'])
        freeze = read(freeze_path)
        for item in freeze['payload']:
            # Numeric tensors/checkpoints remain remote; verify every local text receipt.
            path = freeze_path.parent / item['path']
            if path.suffix in {'.json', '.jsonl', '.md', '.txt', '.csv'}:
                assert digest(path) == item['sha256'], str(path)
        receipt = read(done)
        bound(receipt['whole_terminal'])
        output = freeze_path.parent
        entry = {'key': key, 'phase': row['phase'], 'arm': row['arm'],
                 'context': output.name, 'terminal_sha256': digest(terminal_path),
                 'whole_supervised_seconds': receipt['whole_supervised_seconds']}
        if row['phase'] == 'qualify':
            q = read(output / 'QUALIFICATION.json')
            assert q['passed'] is True and q['scientific_result'] is False
            assert q['label_scope'] == ['train']
        elif row['phase'] == 'warm':
            w = read(output / 'WARM_RECEIPT.json')
            entry['actual_updates'] = w['metadata']['actual_updates']
            entry['warm_checkpoint_sha256'] = w['warm_checkpoint']['sha256']
        completed.append(entry)
    counts = dict(Counter(x['phase'] for x in completed))
    snapshot = PHASE / 'coordination_snapshots/20261002_precision_training_started_v1'
    snapshot.mkdir(exist_ok=False)
    for name in ['PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json']:
        shutil.copy2(PHASE / name, snapshot / name)
    observation = {'UTC': now, 'study_id': registry['study_id'],
        'registry_sha256': digest(registry_path), 'completed_counts': counts,
        'completed_phases': completed, 'outstanding_launches_in_local_mirror': outstanding,
        'whole_seconds_completed': sum(x['whole_supervised_seconds'] for x in completed),
        'numeric_tensors_recomputed': False, 'binary_payloads_fetched': False,
        'test_labels_read': False, 'predictive_improvement_established': False,
        'coordinator_terminal_present': (RUN / 'COMPLETED.json').exists(),
        'coordinator_failure_present': (RUN / 'FAILED.json').exists()}
    (snapshot / 'VERIFIED_EXECUTION_OBSERVATION.json').write_text(json.dumps(observation, indent=2) + '\n')
    intro = f'''Updated: {now}. The research goal remains incomplete.

## Experiment registration and training

“Registration succeeded” means the fixed experiment plan was saved and verified on the authorized server. It does not refer to conference registration or a successful result.

The precision study is now executing. Verified local receipts contain {counts.get('qualify', 0)} of six successful complete-graph numerical checks and {counts.get('warm', 0)} of six completed warm training trajectories. The completed warm trajectories have performed {sum(x.get('actual_updates', 0) for x in completed)} optimizer updates. These checkpoints are initial conditions for the five-arm comparison, not evidence of improvement. All six numerical checks passed using the disclosed precision measurement; the original failed study is preserved.

The fixed study compares graph-guided factor initialization with common descent, random directions, permuted topology, and warm copying on PolyFormer-Mono/Squirrel and Polynormer-r/Photo. Seeds 17, 29, and 43 are paired with source splits 0, 1, and 2. Its 72 phases comprise six numerical checks, six warm trajectories, 30 initializations, and 30 continuation fits. The coordinator stops on failure and does not retry. Real warm states must pass their own checks before initialization. Final labels remain closed during training.

The candidate builds private BatchEnsemble directions from graph-filtered training errors after a shared warm start. Their additional pooled-logit change cancels to first order at equal step length. Whether the directions improve subsequent learning is the question the controls test; utility and methodological novelty remain unproven.

## Parallel research

Tolokers2 is a separate industrial-graph candidate. A repo-owned Python 3.12 runtime passed package/CUDA import checks. The native runner source is undergoing an independent audit; its runtime image, isolation, model qualification, and useful fits remain pending. GPU work will be serialized with the current study. Relational retail prediction is being evaluated as another dataset lead using source and metadata only.

Distinct method analysis examines the initializer's falsifiable predictions and closest prior work. Literature conclusions are saved for reuse. Independent source checks are engineering reviews and do not count as paper acceptance recommendations.

## Preserved failures and earlier findings

The old graph-initialization study stopped at a Photo finite-difference check before useful training. An exact-state diagnostic reproduced the failure and isolated a loss-reduction precision issue. The new measurement promotes per-example FP32 losses for FP64 mean/difference calculations; it does not change the model, training gradient, initializer, fixed directions, step sizes, or tolerances.

The first new metadata registration failed because four predecessor supervisor logs were missing from the local inventory. A complete inventory wrapper then registered the current cohort. Two local prelaunch cap-field exceptions were retained before any scientific claim. A separately reviewed continuation amendment corrected the START/TERMINAL cap binding and launched the same registered cohort. Every source version and failure remains recorded.

The earlier 54-fit coordinate screen remains STAGE1_NO_GO; its larger continuation is not admitted. Unsuccessful ideas and wrong-allocation evidence are retained and are not promoted as successful results. Original paper scores remain unchanged.

## Resources and publication

Only anogena-2 on port 2222 and GPU UUID GPU-44039938-fd82-41d2-fefd-de71514e2fac are authorized here. The seven-GPU account must never be accessed. The 18.77 route remains unresolved. Work stays inside the local project and authorized remote repository. No sudo, PDF compilation, GENLINK, Desktop access, or original-score recalculation.

The study's 425,700-second cap sum is a maximum bound, not an ETA or measured use. The current receipt snapshot records {observation['whole_seconds_completed']:.1f} whole supervised seconds for completed phases; running work is excluded. Disk/memory forecasts remain provisional.

Latest verified public revision: a7e8867db8ff47e26a113e8b797aff1657a4b2a4 on codex/postsubmission-research-20260930. New launch/training evidence is being prepared for publication. Explicit fetches through 156 are retained; next unused is 157. The coordinator also preserves per-phase metadata fetch receipts.

## Outcome boundary

No new predictive improvement, established novelty, revised manuscript, or acceptance recommendation is claimed. Fresh independent paper review requires a complete defensible manuscript and evidence packet.
'''
    (PHASE / 'PUBLIC_STATUS.md').write_text('# Current research status\n\n' + intro)
    (PHASE / 'RESEARCH_STATE.md').write_text('# GNNM post-submission research state\n\n' + intro)
    ledger = read(PHASE / 'research_ledger.json')
    ledger['graph_initializer_precision_execution_started_v3'] = observation
    ledger['current_status_update_UTC'] = now
    ledger['current_priority'] = 'Complete fixed graph-initialization comparisons; prepare industrial qualification and distinct falsifiable method analysis in parallel.'
    ledger['active_parallel_work_current'] = ['graph_init_source_audit_v1: industrial native source audit',
        'heterogeneous_factor_gap_v1: representative new-dataset scout',
        'post_failure_distinct_method_v1: initializer scientific interpretation']
    (PHASE / 'research_ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps({'completed_counts': counts, 'completed_warm_updates': sum(x.get('actual_updates', 0) for x in completed),
                      'snapshot': str(snapshot), 'predictive_improvement_established': False}))


if __name__ == '__main__':
    main()
