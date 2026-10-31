"""Separately admitted evidence replay; save member and pooled logits once.

The unchanged primary test writes scalar results only. This exporter performs
one additional forward per locked cell and never changes training, selection,
the primary scoring function or any primary result. Dry-run is the default.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

import evaluate77 as preparation
from gate_contract import require, matches, runtime_mode_gate

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'root_predictions_v1'
ADMISSION = HERE / 'ROOT_PREDICTION_EXPORT_ADMISSION.json'


def export(gpu, admission_path):
    started = time.monotonic()
    launcher, _, guards, context, _, rows, identity, _, boundary = preparation.base_context(
        'export', admission_path, gpu, allow_completed_primary_evaluation=True)
    admission_path = launcher.confined(admission_path)
    require(admission_path == ADMISSION, 'Use this packet\'s separate root prediction-export admission')
    launcher.confined(OUTPUT)
    primary_path = preparation.EVAL_DIR / 'EVALUATION_RECEIPT.json'
    primary = launcher.read_json(primary_path)
    matches(primary, dict(schema='buddy77-postfamily-evaluation-receipt-v2', **identity,
                          status='all15_locked_cells_scored_once', family_lock_sha256=launcher.sha(preparation.LOCK),
                          lock_audit_sha256=launcher.sha(preparation.LOCK_AUDIT), scoring_exit_code=0,
                          scoring_argv=preparation.score_command(), test_payload_opened=True,
                          scoring_attempted=True, other_jobs_stopped=False), 'Completed primary evaluation')
    require(primary['evaluation_admission_sha256'] == launcher.sha(preparation.ADMISSION), 'Primary root admission changed')
    audit = launcher.read_json(preparation.LOCK_AUDIT)
    matches(audit, dict(schema='buddy77-production-family-lock-audit-v1', **identity,
                        status='all15_production_lock_and_selected_checkpoints_audited',
                        family_lock_sha256=launcher.sha(preparation.LOCK), all_selected_checkpoints_runtime_validated=True,
                        locked_runs=rows, exit_code=0), 'Production lock audit')
    lock, checkpoints = guards.verify_family_metadata(preparation.LOCK, context['cache_manifest_sha256'], guards.file_sha(preparation.SOURCE / 'CONFIG.json'))
    require(lock['runs'] == rows and len(checkpoints) == 15, 'Production lock changed before export')
    checkpoint_metadata = {str(path): metadata for path, metadata in checkpoints}
    test_manifest = launcher.read_json(preparation.CACHE / 'test_manifest.json')
    matches(test_manifest, dict(cache_manifest_sha256=context['cache_manifest_sha256'],
                                family_lock_sha256=launcher.sha(preparation.LOCK), graph_policy='training_only_all_splits'),
            'Primary final cache')
    matches(primary['final_cache_qualification'], dict(test_manifest_sha256=launcher.sha(preparation.CACHE / 'test_manifest.json'),
                                                       test_cache_sha256=test_manifest['test_sha256'],
                                                       full_candidate_order_and_shapes_qualified=True,
                                                       test_topology_added_to_graph=False), 'Primary cache qualification')
    require(len(primary.get('final_results', [])) == 15, 'Every primary result required before export')
    result_rows = {(item['arm'], item['seed']): item for item in primary['final_results']}
    require(len(result_rows) == 15 and set(result_rows) == {(row['arm'], row['seed']) for row in rows}, 'Primary result cohort differs')
    result_bindings = []
    for row in rows:
        path = Path(row['run_directory']) / 'final_test.json'
        digest = launcher.sha(path)
        require(digest == result_rows[(row['arm'], row['seed'])]['result_sha256'], 'Primary result bytes changed')
        result_bindings.append(dict(arm=row['arm'], seed=row['seed'], result_sha256=digest))
    export_identity = dict(**identity, family_lock_sha256=launcher.sha(preparation.LOCK),
                           lock_audit_sha256=launcher.sha(preparation.LOCK_AUDIT),
                           primary_evaluation_receipt_sha256=launcher.sha(primary_path),
                           primary_evaluation_admission_sha256=launcher.sha(preparation.ADMISSION),
                           test_manifest_sha256=launcher.sha(preparation.CACHE / 'test_manifest.json'),
                           test_cache_sha256=test_manifest['test_sha256'],
                           primary_result_bindings=result_bindings, replay_GPU_UUID=gpu)
    admission = launcher.read_json(admission_path)
    runtime_mode_gate(admission)
    matches(admission, dict(export_identity, schema='buddy77-prediction-export-admission-v2', decision='admitted',
                            family_cells=15, export_once=True, extra_forward_per_cell=1,
                            no_new_training=True, no_new_selection=True, other_jobs_stopped=False,
                            prior_partial_fits_excluded=True),
            'Separate root prediction-export admission')
    require(isinstance(admission.get('root_export_cost_decision'), str) and bool(admission['root_export_cost_decision'].strip()),
            'Root evidence-replay cost decision missing')
    require(not OUTPUT.exists(), 'Prediction export already attempted; explicit root recovery review required')
    OUTPUT.mkdir(exist_ok=False)
    record = dict(schema='buddy77-prediction-evidence-replay-v2', UTC=datetime.now(timezone.utc).isoformat(),
                  **export_identity, export_admission_sha256=launcher.sha(admission_path), status='in_progress',
                  primary_inference_timing_unchanged=True, no_new_training=True, no_new_selection=True,
                  scientific_advantage_claimed=False, execution_boundary=boundary, completed_exports=[])
    launcher.write_json(OUTPUT / 'EXPORT_CLAIM.json', record)
    try:
        _, runtime_paths = preparation.owned_runtime_environment(launcher, OUTPUT)
        record['repository_local_runtime_paths'] = runtime_paths
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
        os.environ.pop('TORCH_FORCE_WEIGHTS_ONLY_LOAD', None)
        sys.path.insert(0, str(preparation.SOURCE))
        sys.path.insert(1, str(preparation.REPO / '.gnnm_runtime/buddy_extra_v1/site'))
        import torch
        import run as source_run
        import cache_builder as builder
        from models import make_model, member_logits
        require(Path(source_run.__file__).resolve() == preparation.SOURCE / 'run.py', 'Unexpected primary run module')
        source_run.qualification(preparation.QUAL)
        torch.set_num_threads(4)
        begin = time.monotonic()
        builder.load_manifest(preparation.CACHE)
        builder.verify_family_lock(preparation.LOCK, context['cache_manifest_sha256'], builder.file_sha(preparation.SOURCE / 'CONFIG.json'))
        require(builder.file_sha(preparation.CACHE / 'test.pt') == export_identity['test_cache_sha256'], 'Final test cache bytes changed')
        common = torch.load(preparation.CACHE / 'common.pt', map_location='cpu')
        common.pop('hashes'); common.pop('cards')
        data = torch.load(preparation.CACHE / 'test.pt', map_location='cpu')
        config = builder.read_json(preparation.SOURCE / 'CONFIG.json')
        require(builder.tensor_sha(data['links']) == test_manifest['pair_order_sha256'] and
                data['n_positive'] == primary['official_test_qualification']['positive_rows'], 'Primary candidate ordering changed')
        order_path = OUTPUT / 'CANDIDATE_ORDER.pt'
        torch.save(dict(links=data['links'], n_positive=data['n_positive'],
                        metadata=dict(test_manifest_sha256=export_identity['test_manifest_sha256'],
                                      pair_order_sha256=test_manifest['pair_order_sha256'],
                                      ordering='official positives then official negatives')), order_path)
        record.update(candidate_order_file_sha256=builder.file_sha(order_path),
                      checkpoint_validation_cache_load_and_order_write_seconds=time.monotonic() - begin)
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        for row in rows:
            begin = time.monotonic()
            folder = Path(row['run_directory'])
            require(builder.file_sha(folder / 'selected.pt') == row['selected_checkpoint_sha256'], 'Selected checkpoint changed before replay')
            model = make_model(config, row['arm'], row['seed']).to(device)
            envelope = torch.load(folder / 'selected.pt', map_location=device, weights_only=True)
            require(envelope['metadata'] == checkpoint_metadata[str(folder / 'selected.pt')], 'Selected checkpoint envelope changed before replay')
            model.load_state_dict(envelope['model_state'], strict=True)
            model.eval()
            source_run.synchronize(device)
            setup_seconds = time.monotonic() - begin
            begin = time.monotonic()
            member_batches, pooled_batches = [], []
            with torch.no_grad():
                for indices in source_run.batch_rows(len(data['links']), config['eval_batch_size']):
                    inputs, _ = source_run.hydrated(common, data, indices, device)
                    values = member_logits(model, inputs)
                    pooled = values.mean(dim=1)  # exact unchanged primary device-side pooling
                    require(torch.isfinite(values).all() and torch.isfinite(pooled).all(), 'Nonfinite evidence replay logits')
                    member_batches.append(values.cpu())
                    pooled_batches.append(pooled.cpu())
            source_run.synchronize(device)
            replay_seconds = time.monotonic() - begin
            members, pools = torch.cat(member_batches), torch.cat(pooled_batches)
            require(members.shape == (len(data['links']), 4 if row['arm'] in {'factorized4', 'independent4'} else 1) and
                    pools.shape == (len(data['links']),), 'Exported member/pooled logit shape differs')
            primary_result = launcher.read_json(folder / 'final_test.json')
            require(launcher.sha(folder / 'final_test.json') == result_rows[(row['arm'], row['seed'])]['result_sha256'],
                    'Primary result changed during replay')
            require(source_run.hits50(pools, data['n_positive']) == primary_result['hits50'],
                    'Evidence replay Hits@50 disagrees with the saved unchanged primary result')
            metadata = dict(arm=row['arm'], seed=row['seed'], config_sha256=builder.file_sha(preparation.SOURCE / 'CONFIG.json'),
                            implementation_hashes=context['implementation_hashes'], torch_version=context['torch_version'],
                            checkpoint_sha256=row['selected_checkpoint_sha256'], family_lock_sha256=export_identity['family_lock_sha256'],
                            cache_manifest_sha256=context['cache_manifest_sha256'], test_manifest_sha256=export_identity['test_manifest_sha256'],
                            test_cache_sha256=export_identity['test_cache_sha256'], candidate_order_file_sha256=record['candidate_order_file_sha256'],
                            pair_order_sha256=test_manifest['pair_order_sha256'], n_positive=data['n_positive'],
                            primary_result_sha256=result_rows[(row['arm'], row['seed'])]['result_sha256'],
                            pooling='mean_raw_logits on the evaluation device before CPU transfer', replay_GPU_UUID=gpu)
            output = OUTPUT / f'{row["arm"]}_seed{row["seed"]}.pt'
            begin = time.monotonic()
            torch.save(dict(metadata=metadata, member_logits=members, pooled_logits=pools), output)
            record['completed_exports'].append(dict(arm=row['arm'], seed=row['seed'], file_sha256=builder.file_sha(output),
                                                    bytes=output.stat().st_size, model_checkpoint_setup_seconds=setup_seconds,
                                                    evidence_replay_forward_seconds=replay_seconds,
                                                    export_write_and_hash_seconds=time.monotonic() - begin,
                                                    unchanged_primary_Hits50_agrees=True))
            del model, envelope, member_batches, pooled_batches, members, pools
        require(launcher.sha(primary_path) == export_identity['primary_evaluation_receipt_sha256'] and
                launcher.sha(preparation.LOCK) == export_identity['family_lock_sha256'], 'Primary receipt/lock changed during export')
        for row in result_bindings:
            require(launcher.sha(preparation.RUNS / f'{row["arm"]}_seed{row["seed"]}' / 'final_test.json') == row['result_sha256'],
                    'Primary scalar result changed during export')
        for row in rows:
            require(builder.file_sha(Path(row['run_directory']) / 'selected.pt') == row['selected_checkpoint_sha256'],
                    'Selected checkpoint changed during evidence export')
        record.update(status='all15_prediction_evidence_replays_exported_once',
                      extra_forwards=15, summed_replay_forward_seconds=sum(row['evidence_replay_forward_seconds'] for row in record['completed_exports']),
                      isolated_speedup_established=False,
                      cost_scope='Additional evidence replay only. All checkpoint/setup, cache/order I/O, GPU forwards and export writes are charged in wrapper wall time; primary inference timing is preserved. Shared-host contention prevents isolated speed claims.')
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error), replay_requires_explicit_root_review=True)
        raise
    finally:
        record['total_export_wrapper_seconds'] = time.monotonic() - started
        launcher.write_json(OUTPUT / 'PREDICTIONS_RECEIPT.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--admission', default=str(ADMISSION))
    parser.add_argument('--gpu', choices=preparation.UUIDS, default=preparation.UUIDS[0])
    args = parser.parse_args()
    if not args.execute:
        import json
        print(json.dumps(dict(stage='post-primary prediction evidence replay', extra_forwards=15,
                              required_admission=str(ADMISSION), output=str(OUTPUT), primary_results_unchanged=True,
                              files='one authentic candidate-order tensor plus fifteen member/pooled logit envelopes',
                              no_training=True, no_selection=True, no_scientific_execution_in_preparation=True), indent=2))
        return
    export(args.gpu, args.admission)


if __name__ == '__main__':
    main()
