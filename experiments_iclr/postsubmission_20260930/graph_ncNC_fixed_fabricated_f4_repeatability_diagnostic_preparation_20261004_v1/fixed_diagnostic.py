#!/usr/bin/env python3
"""Four predeclared original-scorer calls on one existing fabricated F4 state."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

HERE = Path(__file__).resolve().parent
REMOTE_PHASE = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930')
FAILED = REMOTE_PHASE / 'ncnc_selected_replay_synthetic_execution_root_20261004_v1'
FIXTURE = FAILED / 'qualification/run01/inputs_baseline_all25'
UNIT = FIXTURE / 'factor_private4_3'
ARM = 'factorized_private_4'
SCHEDULE = (('False1', False), ('False2', False), ('True1', True), ('True2', True))
CUBLAS_CONFIG = ':4096:8'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def bound(pin, expected=None, *, decode=False):
    path = Path(pin['path'])
    require(path.is_absolute() and path.resolve() == path and path.is_file(), 'Noncanonical diagnostic input')
    require(expected is None or path == expected, 'Fixed fabricated input path differs')
    require(type(pin['bytes']) is int and pin['bytes'] > 0 and path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'], 'Diagnostic input bytes differ')
    return json.loads(path.read_text()) if decode else path


def atomic_json(path, value):
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def preflight(release_path, output):
    release_path, output = Path(release_path), Path(output)
    require(release_path.is_absolute() and release_path.resolve() == release_path, 'Canonical separate root release required')
    release = json.loads(release_path.read_text())
    require(release['schema'] == 'ncnc-fixed-fabricated-f4-repeatability-root-release-v1' and release['execution_enabled'] is True and release['root_authorization_reference'], 'Separate diagnostic root release is disabled or missing')
    require(release['authorized_stages'] == ['fixed_fabricated_f4_repeatability_diagnostic'], 'Only fixed fabricated diagnostic stage allowed')
    require(output.is_absolute() and output.resolve() == output and output.is_relative_to(REMOTE_PHASE), 'Canonical diagnostic output required')
    require(not output.exists() and not output.is_relative_to(FAILED) and not output.is_relative_to(HERE), 'Output must be fresh and outside preserved run/source')
    require(not any((p / 'MANIFEST.json').exists() for p in (output, *output.parents)), 'Output is inside sealed source')
    require(release['authorized_invocations'] == [dict(stage='fixed_fabricated_f4_repeatability_diagnostic', output_directory=str(output), cuda_visible_devices=release['cuda_visible_devices'])], 'Exactly one output/GPU invocation allowed')
    require(release['cuda_visible_devices'] == 'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998' and os.environ.get('CUDA_VISIBLE_DEVICES') == release['cuda_visible_devices'], 'Root-bound GPU0 required')
    require(release['forward_schedule'] == [dict(label=n, deterministic_algorithms=p, warn_only=False) for n, p in SCHEDULE], 'Fixed four-forward profile schedule differs')
    require(release['common_CUBLAS_WORKSPACE_CONFIG'] == CUBLAS_CONFIG and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == CUBLAS_CONFIG, 'Explicit common diagnostic cuBLAS setting required before numerical imports')
    require(release['scientific_execution_authorized'] is False and release['training_updates_authorized'] == 0 and release['old_qualification_PASS_claim_authorized'] is False, 'Diagnostic cannot admit fits/scientific replay or repair old qualification')
    require(not any(k in release for k in ('data_authority', 'dataset_root', 'unit_custody', 'scientific_family_lock', 'ordinary_runtime_synthetic_qualification')), 'Study inputs/production admission are forbidden')
    bindings = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    require(release['original_source_bindings'] == bindings['original_source_bindings'] and release['runtime_authority'] == bindings['runtime_authority'], 'Original source/runtime binding differs')
    sidecar = Path(bindings['original_sidecar_root'])
    require(sha(sidecar / 'MANIFEST.json') == bindings['original_sidecar_manifest_sha256'], 'Original sidecar source changed')
    sys.path.insert(0, str(sidecar))
    from replay_gate import verify_manifest
    verify_manifest(sidecar, bindings['original_sidecar_manifest_sha256'])
    verify_manifest(HERE, release['diagnostic_manifest_sha256'])
    paths = {}
    for item in bindings['original_source_bindings']:
        root = Path(item['root']); verify_manifest(root, item['manifest_sha256']); paths[item['key']] = root
    require(all(not output.is_relative_to(root) for root in paths.values()), 'Output belongs to original source')
    runtime = bound(release['runtime_authority'], decode=True)
    require(runtime['deterministic_algorithms'] is False and runtime['TF32'] is False and runtime['mixed_precision'] is False and runtime['ordinary_host_execution'] is True, 'Original admission must remain the frozen False profile')
    environment = {'PYTHONPATH': runtime['project_PYTHONPATH'], 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    require(release['original_process_environment'] == environment and all(os.environ.get(k) == v for k, v in environment.items()), 'Original process path/thread environment differs')
    authority = json.loads((sidecar / 'FABRICATED_AUTHORITY.json').read_text())
    require(authority['files'] == {} and authority['fabricated_inputs_only'] is True and authority['test_file_opened'] is False, 'Only fabricated authority allowed')
    qual = bound(release['failed_qualification'], FAILED / 'qualification/run01/QUALIFICATION.json', decode=True)
    baseline = bound(release['failed_baseline_result'], FAILED / 'qualification/run01/result_baseline_all25/REPLAY_RESULT.json', decode=True)
    private = bound(release['failed_baseline_details'], FAILED / 'qualification/run01/result_baseline_all25/PRIVATE_REPLAY_DETAILS.json', decode=True)
    require(qual['status'] == 'FAILED' and qual['fabricated_inputs_only'] is True and qual['study_lock_data_outcome_checkpoint_accessed'] is False and qual['TEST_opened'] is False and qual['scientific_fit_updates'] == 0, 'Preserved failed fabricated qualification required')
    require(baseline['status'] == 'FAILED_SELECTED_STATE_REPLAY' and baseline['fabricated_inputs_only'] is True and baseline['scientific_replay'] is False, 'Preserved failed baseline required')
    failed = [r for r in private['cells'] if r['status'] != 'PASS']
    require(len(failed) == 1 and (failed[0]['arm'], failed[0]['base_seed']) == (ARM, 3) and failed[0]['failure']['condition'] == 'Original per-model raw score digest mismatch', 'Fixed original failed cell differs')
    require(failed[0]['score_valid_calls'] == 1 and len(failed[0]['member_receipts']) == 1 and 'selected_hits50_equal' not in failed[0] and 'replayed_VALID_hits50' not in failed[0], 'Old failed metric must remain not reached')
    pins = release['fixed_fabricated_inputs']
    expected = {'fabricated_family_metadata': FIXTURE / 'lock/FAMILY_LOCK.json', 'complete_metadata': UNIT / 'COMPLETE.json', 'journal_metadata': UNIT / 'JOURNAL.json', 'journal_state': UNIT / 'STATE_SLOT_0.pt', 'selected_state': UNIT / ('SELECTED_' + ARM + '.pt'), 'fabricated_tensors': FIXTURE / 'FABRICATED_TENSORS.pt'}
    require(set(pins) == set(expected), 'Fixed fabricated input set differs')
    for key, path in expected.items():
        bound(pins[key], path)
    require(pins['fabricated_family_metadata'] == baseline['family_lock'], 'Fabricated lock metadata must match failed baseline custody')
    lock = bound(pins['fabricated_family_metadata'], decode=True)
    identity = lock['identity']
    require(identity['family_id'] == 'FABRICATED_NCNC_REPLAY_QUALIFICATION_ONLY' and identity['synthetic_only'] is True and identity['family_lock_output_directory'] == str(FIXTURE / 'lock') and lock['fabricated_metadata_only'] is True, 'A scientific family cannot enter this diagnostic')
    complete = bound(pins['complete_metadata'], decode=True); journal = bound(pins['journal_metadata'], decode=True)
    require(complete['identity'] == journal['identity'] == identity and complete['unit'] == journal['unit'] == 'factor_private4' and complete['base_seed'] == journal['seed'] == 3, 'Fixed fabricated unit identity differs')
    for key, item in [('journal_state', journal['state_file']), ('selected_state', complete['selected_checkpoints'][ARM])]:
        require({**pins[key], 'path': Path(pins[key]['path']).name} == item, 'Fixed authenticated payload pin differs')
    cell = next(c for c in lock['cells'] if (c['arm'], c['base_seed']) == (ARM, 3))
    require(cell['checkpoint'] == {**complete['selected_checkpoints'][ARM], 'output_directory': str(UNIT)}, 'Fixed fabricated checkpoint cell differs')
    custody = [release['runtime_authority'], release['failed_qualification'], release['failed_baseline_result'], release['failed_baseline_details'], *pins.values()]
    return dict(release=release, release_path=release_path, release_sha256=sha(release_path), output=output,
                paths=paths, runtime=runtime, authority=authority, bindings=bindings, pins=pins,
                identity=identity, complete=complete, journal=journal, cell={**cell, 'unit': 'factor_private4'},
                old_observed_score_digests=failed[0]['member_receipts'][0]['score_digests'], custody=custody)


def differences(expected, current):
    """FP32 finite vector differences; no threshold or acceptance tolerance."""
    import numpy as np
    a, b = expected.numpy(), current.numpy()
    require(a.dtype == b.dtype == np.float32 and a.shape == b.shape and a.ndim == 1 and np.isfinite(a).all() and np.isfinite(b).all(), 'Finite same-shape FP32 comparison required')
    x, y = a.astype(np.float64), b.astype(np.float64)
    delta = np.abs(x - y); denominator = np.maximum(np.abs(x), np.abs(y))
    relative = np.divide(delta, denominator, out=np.zeros_like(delta), where=denominator != 0)
    aa, bb = a.view(np.uint32).astype(np.int64), b.view(np.uint32).astype(np.int64)
    def rank(bits):
        return np.where((bits & 0x80000000) != 0, 0x80000000 - (bits & 0x7fffffff), 0x80000000 + bits)
    return dict(rows=len(a), numerical_changed_elements=int(np.count_nonzero(a != b)),
                numerical_changed_query_rows=int(np.count_nonzero(a != b)),
                bitwise_changed_elements=int(np.count_nonzero(aa != bb)), max_absolute_error=float(delta.max()),
                bitwise_changed_query_rows=int(np.count_nonzero(aa != bb)),
                max_relative_error=float(relative.max()), max_ULP_error=int(np.abs(rank(aa) - rank(bb)).max()),
                relative_error_definition='abs(a-b)/max(abs(a),abs(b)); both-zero error=0',
                ULP_definition='Distance between monotone IEEE754 FP32 ranks; +0 and -0 share one rank. Bitwise changes are separately retained.')


def execute(context, started):
    output = context['output']; output.mkdir(parents=True, mode=0o700, exist_ok=False)
    record = dict(schema='ncnc-fixed-fabricated-f4-repeatability-diagnostic-v1', status='IN_PROGRESS',
        UTC=datetime.now(timezone.utc).isoformat(), root_release=descriptor(context['release_path']),
        source_entry=descriptor(Path(__file__).resolve()), diagnostic_manifest_sha256=sha(HERE / 'MANIFEST.json'),
        actual_argv=list(sys.argv), fabricated_inputs_only=True, source_and_input_custody=context['custody'],
        fixed_arm=ARM, fixed_base_seed=3, old_qualification_status_preserved='FAILED',
        old_failed_cell_Hits50_comparison='unknown/not reached; this diagnostic does not amend it',
        old_observed_score_digests=context['old_observed_score_digests'],
        common_CUBLAS_WORKSPACE_CONFIG=CUBLAS_CONFIG,
        process_setting_scope='Explicit new diagnostic setting common to both profiles; original failed qualification did not bind this ambient value.',
        original_process_environment=context['release']['original_process_environment'],
        original_API_paths=context['bindings']['original_API_paths'],
        original_scorer_call_attempts=0, original_scorer_calls_returned=0,
        training_updates=0, calls=[], comparisons=[], scientific_execution=False,
        old_qualification_PASS_claim=False, production_replay_admission=False, no_retry=True)
    atomic_json(output / 'DIAGNOSTIC.json', record)
    torch = None
    previous_profile = None
    previous_warn_only = None
    arrays = {}
    try:
        from replay_numeric import original_modules, trusted_load
        from replay_contract import validate_journal_payload, validate_selected_payload, validate_valid_receipt
        api = original_modules(context)
        api['pilot_common'].runtime_stdlib(context)
        device, _ = api['pilot_model'].runtime(context)
        import torch as torch_module
        torch = torch_module
        previous_profile = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        require(previous_profile is False, 'Original runtime must enter False profile before the explicit contrast')
        record['admitted_original_profile'] = dict(deterministic_algorithms=False, warn_only=previous_warn_only,
            TF32_matmul=bool(torch.backends.cuda.matmul.allow_tf32), TF32_cudnn=bool(torch.backends.cudnn.allow_tf32),
            mixed_precision=bool(torch.is_autocast_enabled()), default_dtype=str(torch.get_default_dtype()))
        torch.cuda.reset_peak_memory_stats(0)
        mods = api['pilot_model'].modules(context)
        state_api, data_api, evaluate = (api[k] for k in ('pilot_state', 'pilot_data', 'pilot_evaluate'))
        def load(key):
            pin = context['pins'][key]; path = Path(pin['path'])
            return trusted_load(torch, path.parent, {**pin, 'path': path.name})
        payload = load('journal_state')
        state = validate_journal_payload(payload, context['journal'], context['complete'], context['identity'], 'factor_private4', 3, mods['design'].select_validation_candidate)
        selected = load('selected_state')
        snapshots, digests = validate_selected_payload(selected, context['cell'], state, context['identity'], state_api.state_digest)
        require(len(snapshots) == len(digests) == 1, 'One fixed F4 snapshot is required')
        saved, expected_digests = snapshots[0], digests[0]
        expected_snapshot_digest = state_api.state_digest(saved)
        expected_model_Adam_digest = state_api.state_digest({'models': saved['models'], 'optimizer': saved['optimizer']})
        expected_RNG_digest = state_api.rng_digest(saved['rng'])
        record.update(expected_reference_score_digests=expected_digests, expected_selected_Hits50=selected['selection']['hits50'],
                      fixed_snapshot_sha256=expected_snapshot_digest, fixed_model_Adam_sha256=expected_model_Adam_digest, fixed_RNG_sha256=expected_RNG_digest,
                      original_reference_array_comparison_available=False,
                      original_reference_array_limit='Only reference digests were saved; numeric/ULP errors are measured between the four new scored vectors, not against missing historical reference arrays.')
        tensor_payload = load('fabricated_tensors')
        require(tensor_payload['schema'] == 'ncnc-replay-owned-fabricated-tensors-v1' and tensor_payload['fabricated_inputs_only'] is True, 'Only existing fabricated tensors allowed')
        values = tensor_payload['tensors']
        shapes = {'x': (16, 128), 'pairs': (13, 2), 'raw_edge_index': (2, 26), 'valid_positive': (60084, 2), 'valid_negative': (100000, 2)}
        require(set(values) == set(shapes), 'Fixed fabricated tensor set differs')
        for key, shape in shapes.items():
            require(tuple(values[key].shape) == shape and values[key].dtype == (torch.float32 if key == 'x' else torch.long), 'Fixed fabricated shape/dtype differs')
        input_digests = {k: data_api.tensor_sha(v) for k, v in values.items()}
        data = {k: v.to(device) for k, v in values.items()}; data['fabricated_inputs_only'] = True
        metric = evaluate.evaluator(context)
        record['fabricated_tensor_digests'] = input_digests
        for label, profile in SCHEDULE:
            began = perf_counter(); current = dict(label=label, deterministic_algorithms=profile, warn_only=False, original_scorer_called=False, metric_comparison_reached=False, status='FAILED')
            instance = optimizer = None
            restored = False
            try:
                torch.use_deterministic_algorithms(profile, warn_only=False)
                require(torch.are_deterministic_algorithms_enabled() is profile and torch.is_deterministic_algorithms_warn_only_enabled() is False, 'Explicit strict diagnostic profile differs')
                instance, optimizer = api['pilot_model'].make_factorized(mods, 3, device)
                state_api.restore_snapshot(instance, optimizer, saved, restore_random=True)
                require(state_api.state_digest(state_api.snapshot(instance, optimizer)) == expected_snapshot_digest, 'Identical complete snapshot restoration failed')
                restored = True
                current.update(restored_snapshot_sha256=expected_snapshot_digest, before_RNG_sha256=state_api.rng_digest(state_api.rng_state()),
                               before_flags_sha256=state_api.state_digest(state_api.flags(instance)))
                record['original_scorer_call_attempts'] += 1; current['original_scorer_called'] = True
                positive, negative, receipt = evaluate.score_valid(instance, data, mods, mode='private')
                record['original_scorer_calls_returned'] += 1
                arrays[label] = {'positive': positive, 'negative': negative}
                validate_valid_receipt(receipt)
                actual_digests = {k: data_api.tensor_sha(v) for k, v in arrays[label].items()}
                require(actual_digests == receipt['score_digests'], 'Original raw-pool digest receipt differs from returned vectors')
                current.update(original_score_receipt=receipt, expected_score_digests=expected_digests,
                    current_score_digests=actual_digests, per_pool_exact_reference_digest_equal={k: actual_digests[k] == expected_digests[k] for k in ('positive', 'negative')},
                    status='SCORER_RETURNED')
                current['actual_Hits50'] = evaluate.hits50(metric, positive, negative)
                current['metric_comparison_reached'] = True
                current['exact_selected_Hits50_equal'] = current['actual_Hits50'] == selected['selection']['hits50']
                current['status'] = 'SCORER_RETURNED_AND_METRIC_REACHED'
            except Exception as error:
                current['failure'] = dict(exception_type=type(error).__name__, condition=str(error))
            finally:
                try:
                    if restored:
                        after = state_api.snapshot(instance, optimizer)
                        current.update(after_model_Adam_sha256=state_api.state_digest({'models': after['models'], 'optimizer': after['optimizer']}),
                            after_RNG_sha256=state_api.rng_digest(after['rng']), after_flags_sha256=state_api.state_digest(after['flags']),
                            profile_at_end=bool(torch.are_deterministic_algorithms_enabled()), warn_only_at_end=bool(torch.is_deterministic_algorithms_warn_only_enabled()))
                        current['model_Adam_unchanged'] = current['after_model_Adam_sha256'] == expected_model_Adam_digest
                        current['RNG_unchanged'] = current['after_RNG_sha256'] == expected_RNG_digest
                        current['fabricated_tensors_unchanged'] = {k: data_api.tensor_sha(data[k]) for k in input_digests} == input_digests
                        current['profile_unchanged'] = current['profile_at_end'] is profile and current['warn_only_at_end'] is False
                        current['scorer_flag_scope'] = 'Original score_valid sets all module evaluation flags; identical saved flags were restored before every call and flags before/after are separately hashed.'
                except Exception as error:
                    current.update(status='FAILED_CUSTODY_CAPTURE', custody_capture_failure=dict(exception_type=type(error).__name__, condition=str(error)))
                current['inclusive_call_wall_seconds'] = perf_counter() - began
                record['calls'].append(current)
                del instance, optimizer
                atomic_json(output / 'DIAGNOSTIC.json', record)
            require(restored and all(current.get(k) is True for k in ('model_Adam_unchanged', 'RNG_unchanged', 'fabricated_tensors_unchanged', 'profile_unchanged')), 'Snapshot/RNG/tensor/profile custody failed; no further calls allowed')
        for i, (expected_label, _) in enumerate(SCHEDULE):
            for current_label, _ in SCHEDULE[i + 1:]:
                comparison = dict(expected_call=expected_label, current_call=current_label, kind='within_profile' if expected_label[:-1] == current_label[:-1] else 'between_profiles')
                if expected_label in arrays and current_label in arrays:
                    comparison['pools'] = {k: dict(expected_digest=data_api.tensor_sha(arrays[expected_label][k]), current_digest=data_api.tensor_sha(arrays[current_label][k]), **differences(arrays[expected_label][k], arrays[current_label][k])) for k in ('positive', 'negative')}
                    for pool in comparison['pools'].values():
                        pool['exact_raw_digest_equal'] = pool['expected_digest'] == pool['current_digest']
                else:
                    comparison.update(pools=None, reason='At least one predeclared original scorer call did not return arrays; no fallback or additional call.')
                record['comparisons'].append(comparison)
        record['profile_results'] = []
        for profile_name in ('False', 'True'):
            rows = [c for c in record['calls'] if c['label'][:-1] == profile_name]
            comparison = next(c for c in record['comparisons'] if c['expected_call'] == profile_name + '1' and c['current_call'] == profile_name + '2')
            pools = comparison['pools']
            record['profile_results'].append(dict(deterministic_algorithms=profile_name == 'True', warn_only=False,
                calls=[dict(label=c['label'], status=c['status'], failure=c.get('failure'), current_score_digests=c.get('current_score_digests'),
                    actual_Hits50=c.get('actual_Hits50'), metric_comparison_reached=c['metric_comparison_reached']) for c in rows],
                within_profile_exact_raw_digest_equal={k: v['exact_raw_digest_equal'] for k, v in pools.items()} if pools else None,
                within_profile_comparison_available=pools is not None))
        require(sha(context['release_path']) == context['release_sha256'], 'Diagnostic release custody changed')
        for pin in context['custody']:
            bound(pin)
        from replay_gate import verify_manifest
        verify_manifest(HERE, context['release']['diagnostic_manifest_sha256'])
        verify_manifest(Path(context['bindings']['original_sidecar_root']), context['bindings']['original_sidecar_manifest_sha256'])
        for item in context['bindings']['original_source_bindings']:
            verify_manifest(Path(item['root']), item['manifest_sha256'])
        record['final_source_input_custody'] = 'UNCHANGED'
        record['status'] = 'DIAGNOSTIC_COMPLETE' if record['original_scorer_call_attempts'] == record['original_scorer_calls_returned'] == 4 and all(c['status'] == 'SCORER_RETURNED_AND_METRIC_REACHED' for c in record['calls']) else 'DIAGNOSTIC_INCOMPLETE_WITH_RETAINED_FAILURES'
    except Exception as error:
        record.update(status='DIAGNOSTIC_FAILED_CUSTODY_OR_SETUP', failure=dict(exception_type=type(error).__name__, condition=str(error)))
    finally:
        if torch is not None:
            try:
                torch.cuda.synchronize(0)
                record.update(CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
                if previous_profile is not None:
                    torch.use_deterministic_algorithms(previous_profile, warn_only=previous_warn_only)
            except Exception as error:
                record.update(status='DIAGNOSTIC_FAILED_ACCOUNTING', accounting_failure=dict(exception_type=type(error).__name__, condition=str(error)))
        record.update(UTC=datetime.now(timezone.utc).isoformat(), inclusive_diagnostic_wall_seconds=perf_counter() - started,
                      terminal_write_tail_measured=False, internal_call_intervals_not_added_to_total=True,
                      tensor_bodies_exported=False, old_qualification_PASS_claim=False, production_replay_admission=False)
        atomic_json(output / 'DIAGNOSTIC.json', record)
    print('FIXED_FABRICATED_DIAGNOSTIC_TERMINAL status=' + record['status'] + ' scorer_attempts=' + str(record['original_scorer_call_attempts']), flush=True)
    return 0 if record['status'] == 'DIAGNOSTIC_COMPLETE' else 1


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--root-release'); parser.add_argument('--output')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='SOURCE_ONLY_DISABLED_EXAMPLE_RELEASE', fixed_unit='fabricated factor_private4 seed3', original_scorer_calls=4,
            profiles=[dict(label=n, deterministic_algorithms=p, warn_only=False) for n, p in SCHEDULE], common_CUBLAS_WORKSPACE_CONFIG=CUBLAS_CONFIG,
            training_updates=0, production_replay_admission=False, old_qualification_status='FAILED'), indent=2))
        return 0
    require(args.root_release and args.output, 'Separate enabled root release and one fresh output required')
    started = perf_counter()
    return execute(preflight(args.root_release, args.output), started)


if __name__ == '__main__':
    raise SystemExit(main())
