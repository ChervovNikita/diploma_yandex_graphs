"""One separately admitted normal-host parity diagnostic; no fit/TEST stage.

The immutable diagnostic receives a scalar-only publisher. Its atomic writer
has no model/optimizer/data references or scientific imports/calls. Model flag
neutrality is established by this exact source boundary, not by a dynamic model
inspection or hook. Every callback, including its final call, verifies complete
RNG and runtime-profile neutrality after both scalar writes have returned.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
EXECUTION = PHASE / 'graph_ncNC_structural_pattern_parity_diagnostic_execution_root_20261004_v1'
DIAGNOSTIC = PHASE / 'graph_ncNC_structural_pattern_parity_diagnostic_source_preparation_20261004_v1'
DIAGNOSTIC_SHA = 'b665affd187593875a2be81e30dbe7a3c8e3ce231f7e344d791d67a800331ebe'
DIAGNOSTIC_REVIEW = PHASE / 'graph_ncNC_structural_pattern_parity_diagnostic_independent_source_review_20261004_v1'
DIAGNOSTIC_REVIEW_SHA = 'c09384131c4f378337591ba3f29655911598ce32a887160a58e180b818402bd2'
V4 = PHASE / 'graph_ncNC_structural_pattern_pilot_preparation_20261003_v4'
V4_SHA = '9021a598c7642bf428124de1230a078dfddbf3968095432354ee18a14541104c'
GPU_UUID = 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_packet(root, expected):
    require(root.is_relative_to(PHASE) and sha(root / 'MANIFEST.json') == expected, 'Runner source custody differs')
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for row in manifest['files']:
        path = (root / row['path']).resolve()
        require(path.is_relative_to(root) and path.stat().st_size == row.get('bytes', row.get('size'))
                and sha(path) == row['sha256'], 'Runner payload custody differs: ' + row['path'])


def scalar_receipt(value):
    # Exact built-in types exclude model objects, tensor/array subclasses,
    # custom encoders and arbitrary __str__/iteration callbacks.
    kind = type(value)
    if value is None or kind in (str, int, bool):
        return
    if kind is float:
        require(math.isfinite(value), 'Nonfinite scalar receipt')
        return
    if kind is list:
        for child in value:
            scalar_receipt(child)
        return
    if kind is dict:
        require(all(type(key) is str for key in value), 'Scalar receipt keys must be plain strings')
        for child in value.values():
            scalar_receipt(child)
        return
    raise TypeError('Publisher accepts only plain scalar/list/dict receipts')


def scalar_writer(path):
    """The returned writer's only input is a validated scalar receipt.

    Its closure contains only the exclusive owned output path. No random-name
    generator, tempfile, scientific module, model hook or receipt mutation is
    used. The runner's fresh output and flock establish sole write ownership.
    """
    path = Path(path)
    require(path.parent.resolve() == EXECUTION / 'diagnostic/run01'
            and not path.parent.is_symlink(), 'Publisher output must be owned and repo-local')
    def write(receipt):
        scalar_receipt(receipt)
        temporary = path.with_suffix(path.suffix + '.tmp')
        try:
            with temporary.open('x', encoding='utf-8') as handle:
                json.dump(receipt, handle, indent=2, allow_nan=False)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return write


def utc():
    return datetime.now(timezone.utc).isoformat()


def host_peak_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def receipt(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-admission', required=True, type=Path)
    parser.add_argument('--admission-sha256', required=True)
    args = parser.parse_args()
    admission_path = args.root_admission.resolve()
    require(Path.cwd().resolve() == REPO and os.environ.get('GNNM_SSH_DESTINATION') == 'shmelev@192.168.18.77',
            'Exact normal host route/repository required')
    require(admission_path == EXECUTION / 'ROOT_DIAGNOSTIC_ADMISSION.json'
            and sha(admission_path) == args.admission_sha256, 'Exact independent diagnostic admission required')
    admission = json.loads(admission_path.read_text())
    verify_packet(HERE, admission['normal_supervision_manifest_sha256'])
    verify_packet(DIAGNOSTIC, DIAGNOSTIC_SHA)
    verify_packet(DIAGNOSTIC_REVIEW, DIAGNOSTIC_REVIEW_SHA)
    review = json.loads((DIAGNOSTIC_REVIEW / 'REVIEW.json').read_text())
    require(review['verdict'] == 'PASS' and review['candidate_manifest_sha256'] == DIAGNOSTIC_SHA,
            'Exact independent diagnostic source PASS required')
    verify_packet(V4, V4_SHA)
    spec = importlib.util.spec_from_file_location('reviewed_parity_diagnostic', DIAGNOSTIC / 'parity_diagnostic.py')
    require(spec is not None and spec.loader is not None, 'Diagnostic source loader unavailable')
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)  # exact reviewed module, stdlib-only top level
    plan, diagnostic_admission = diagnostic.admission_gate(admission_path, args.admission_sha256)
    output = EXECUTION / 'diagnostic/run01'
    require(admission['diagnostic_output_directory'] == str(output)
            and admission['supervision_output_directory'] == str(EXECUTION / 'supervision/run01')
            and admission['runner_path'] == str(HERE / 'diagnostic_run.py')
            and admission['diagnostic_review_manifest_sha256'] == DIAGNOSTIC_REVIEW_SHA
            and os.environ.get('CUDA_VISIBLE_DEVICES') == GPU_UUID, 'Exact diagnostic-only runner paths/source differ')
    require(not output.exists(), 'Diagnostic output already exists; no retry/resume')
    output.mkdir(parents=True, mode=0o700)
    sys.path.insert(0, str(V4))
    from pilot_common import read_bound_json, runtime_stdlib, lock_output
    lock = lock_output(output)
    write_attempt = scalar_writer(output / 'ATTEMPT.json')
    write_final = scalar_writer(output / 'FINAL.json')
    write_failure = scalar_writer(output / 'FAILED.json')
    identity = {'admission_sha256': args.admission_sha256,
                'normal_supervision_manifest_sha256': admission['normal_supervision_manifest_sha256'],
                'diagnostic_manifest_sha256': DIAGNOSTIC_SHA,
                'diagnostic_review_manifest_sha256': DIAGNOSTIC_REVIEW_SHA,
                'data_authority_sha256': plan['data_authority_sha256'],
                'runtime_authority_sha256': plan['runtime_authority_sha256']}
    attempt = {'schema': 'ncnc-pattern-parity-diagnostic-inclusive-attempt-v1', 'identity': identity,
               'UTC_started': utc(), 'PID': os.getpid(), 'status': 'IN_PROGRESS', 'phase': 'qualified_runtime_admission',
               'executed_optimizer_updates': 0, 'executed_member_trajectory_updates': 0,
               'state_donor': False, 'TEST_opened': False, 'project_metric_computed': False,
               'qualification_or_fit_admission': False, 'automatic_retry_or_resume': False}
    cuda_started = False
    publisher_audit = []
    callback_returns = 0
    write_audit = scalar_writer(output / 'PUBLICATION_NEUTRALITY.json')
    def phase(name):
        attempt['phase'] = name
        attempt['runner_inclusive_wall_seconds'] = perf_counter() - started
        write_attempt(attempt)
    def peaks():
        values = {'peak_host_RSS_bytes': host_peak_bytes()}
        if cuda_started:
            import torch
            values.update(cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
                          cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)))
        return values
    try:
        phase('qualified_interpreter_distribution_source_binary_CUDA_admission')
        authority_pin, runtime_pin = plan['authority_files']
        authority = read_bound_json(PHASE / authority_pin['path'], authority_pin['sha256'])
        runtime_authority = read_bound_json(PHASE / runtime_pin['path'], runtime_pin['sha256'])
        context = {'authority': authority, 'runtime': runtime_authority,
                   'release': {'cuda_visible_devices': GPU_UUID},
                   'paths': {'prototype_root': PHASE / plan['prototype_root'], 'design_root': PHASE / plan['design_root']}}
        versions = runtime_stdlib(context)
        from pilot_model import runtime, modules
        device, unused_sampler = runtime(context)
        cuda_started = True
        mods = modules(context)
        phase('authenticated_complete_TRAIN_raw_VALID_load_no_VALID_forward')
        from pilot_data import load_data
        data = load_data(context, device)
        phase('complete_TRAIN_only_observation_teacher')
        from pattern_teacher import ObservationTeacher
        teacher = ObservationTeacher.from_train(data['pairs'], len(data['x']))
        bound = diagnostic.bound_modules(plan, mods, teacher)
        state = bound['pilot_state']
        import torch
        def profile():
            # TorchVersion is a str subclass; canonical JSON makes every value
            # a plain built-in scalar before it enters the strict writer.
            return json.loads(json.dumps(diagnostic.runtime_settings(torch), allow_nan=False))
        caller_rng = state.cpu_clone(state.rng_state())
        caller_profile = profile()
        write_diagnostic = scalar_writer(output / 'DIAGNOSTIC.json')
        def publish(payload):
            nonlocal callback_returns
            scalar_receipt(payload)
            before = state.cpu_clone(state.rng_state())
            profile_before = profile()
            require(profile_before == caller_profile, 'Publication entered with changed admitted runtime flags')
            write_diagnostic(payload)
            after_report = state.cpu_clone(state.rng_state())
            profile_after_report = profile()
            require(state.rng_digest(before) == state.rng_digest(after_report)
                    and profile_before == profile_after_report, 'Scalar diagnostic write changed RNG/runtime flags')
            row = {'call': len(publisher_audit) + 1, 'diagnostic_status': payload['status'],
                   'phase': payload['current_phase'], 'executed_optimizer_updates': payload['executed_optimizer_updates'],
                   'RNG_before_sha256': state.rng_digest(before), 'RNG_after_report_write_sha256': state.rng_digest(after_report),
                   'RNG_components': {key: {'before_sha256': state.state_digest(before[key]),
                                           'after_report_write_sha256': state.state_digest(after_report[key])} for key in before},
                   'runtime_profile_before': profile_before, 'runtime_profile_after_report_write': profile_after_report,
                   'report_write_RNG_runtime_exactly_neutral': True,
                   'caller_rng_restored_exactly_in_final_payload': payload.get('caller_rng_restored_exactly'),
                   'model_flag_boundary': 'static exact stdlib writer proof; no model/optimizer/data references or dynamic model inspection'}
            publisher_audit.append(row)
            write_audit({'schema': 'ncnc-pattern-scalar-publisher-neutrality-v1', 'identity': identity,
                         'model_training_flags_neutral_by_exact_source_boundary': True,
                         'dynamic_model_flags_measured': False, 'calls': publisher_audit})
            # Includes the audit write and the diagnostic's final publication.
            after_all = state.rng_state()
            require(state.rng_digest(before) == state.rng_digest(after_all)
                    and profile_before == profile(),
                    'Publication callback changed full RNG/runtime flags')
            callback_returns += 1
            attempt['executed_optimizer_updates'] = payload['executed_optimizer_updates']
            attempt['executed_member_trajectory_updates'] = payload['executed_member_trajectory_updates']
        phase('one_complete_graph_fixed_query_V3_repeat_and_V3_V4_diagnostic')
        result = diagnostic.run(mods, data, teacher, device,
                                admission_path=admission_path, admission_sha256=args.admission_sha256, publish=publish)
        torch.cuda.synchronize(0)
        require(result['status'] == 'DIAGNOSTIC_COMPLETE' and result['executed_optimizer_updates'] == 12
                and result['executed_member_trajectory_updates'] == 48
                and result['caller_rng_restored_exactly'] and result['runtime_flags_unchanged'], 'Diagnostic completion coverage differs')
        require(state.rng_digest(state.rng_state()) == state.rng_digest(caller_rng)
                and profile() == caller_profile,
                'Final callback or diagnostic donated caller RNG/runtime flags')
        require(callback_returns == len(publisher_audit) == 29
                and publisher_audit[-1]['diagnostic_status'] == 'DIAGNOSTIC_COMPLETE'
                and publisher_audit[-1]['caller_rng_restored_exactly_in_final_payload'] is True,
                'Final publication neutrality or callback coverage missing')
        phase('final_inclusive_scalar_accounting')
        final = {'schema': 'ncnc-pattern-normal-parity-diagnostic-run-terminal-v1', 'status': 'DIAGNOSTIC_COMPLETE',
                 'identity': identity, 'diagnostic_receipt': receipt(output / 'DIAGNOSTIC.json'),
                 'publication_neutrality_receipt': receipt(output / 'PUBLICATION_NEUTRALITY.json'),
                 'publisher_callbacks_returned_neutral': callback_returns, 'final_publication_neutrality_verified': True,
                 'final_caller_RNG_runtime_profile_exact': True, 'model_flags_static_boundary_only': True,
                 'executed_optimizer_updates': 12, 'executed_member_trajectory_updates': 48,
                 'versions': versions, 'data_digests': data['digests'], **peaks(),
                 'runner_inclusive_wall_seconds': perf_counter() - started,
                 'runner_wall_overlaps_supervisor_not_added': True, 'terminal_write_tail_measured': False,
                 'TEST_opened': False, 'state_donor': False, 'project_metric_computed': False,
                 'qualification_or_fit_admission': False, 'scientific_cap_verdict': None}
        # All scientific images/model/Adam objects stay in the diagnostic call;
        # this runner retains only loaded inputs and scalar receipts until exit.
        write_final(final)
        attempt.update(status='DIAGNOSTIC_COMPLETE', **peaks(), runner_inclusive_wall_seconds=perf_counter() - started)
        write_attempt(attempt)
    except BaseException as error:
        attempt.update(status='FAILED', failure={'type': type(error).__name__, 'condition': str(error)},
                       **peaks(), runner_inclusive_wall_seconds=perf_counter() - started)
        write_attempt(attempt)
        write_failure({'schema': 'ncnc-pattern-normal-parity-diagnostic-failure-v1', 'identity': identity,
                       'attempt': attempt, 'publisher_callbacks_returned_neutral': callback_returns,
                       'diagnostic_receipt': receipt(output / 'DIAGNOSTIC.json') if (output / 'DIAGNOSTIC.json').exists() else None,
                       **peaks(), 'TEST_opened': False, 'state_donor': False, 'qualification_or_fit_admission': False})
        raise
    finally:
        os.close(lock)


if __name__ == '__main__':
    main()
