"""One bounded, postclosure full15 reconstruction; source preparation is stdlib-only."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
STAGE = PHASE / 'query_conditioned_value_gate_full15_callable_source_20261009_v1'
OWNER = PHASE / 'query_value_gate_scientific_owner_source_20261009_v1'
FAMILY = PHASE / 'query_value_gate_full15_execution_root_20261009_v1'
ACTIVATION = PHASE / 'query_value_gate_full15_activation_root_20261009_v1'
OUTPUT = PHASE / 'query_value_gate_complete_readout_execution_root_20261009_v1'
STAGE_SHA = '67a6f621b23f77761a96954119bade6d3646021ebde4a0a89c6c74b76df8ced0'
OWNER_SHA = '240596e76ba583cef69808c3610bd22c600df8c611650a8e4effeac324665f91'
SEEDS = (6101, 6203, 6307)
ARMS = ('C4_gate', 'S_joint4head_gate', 'U4_sameB_full_untied_gate', 'S_one_path_gate', 'C4_gate_identity_erased')
TOLERANCE = 2e-5
WALL_SECONDS, CLEANUP_SECONDS = 300, 15


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sealed(root, digest):
    require(sha(root / 'MANIFEST.json') == digest and read(root / 'SEAL.json')['manifest_sha256'] == digest,
            'Changed sealed source manifest')
    for row in read(root / 'MANIFEST.json')['files']:
        path = (root / row['path']).resolve(strict=True)
        require(path.is_relative_to(root) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Changed sealed source payload')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def sources():
    sealed(STAGE, STAGE_SHA)
    sealed(OWNER, OWNER_SHA)
    stage = load(STAGE / 'stage.py', 'stage')
    owner = load(OWNER / 'owned.py', '_readout_sealed_owner')
    pins = stage.verify_bindings()
    runtime = read(PHASE / pins['runtime_path'])
    stage.route(runtime)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == runtime['GPU_uuid'], 'Frozen physical GPU route')
    return stage, owner, pins, runtime


def closed_owner(stage, owner):
    """Only work/custody receipts here; no selected metrics are deserialized."""
    records = stage.verify_closed_family(FAMILY)
    require(set(records) == {(s, a) for s in SEEDS for a in ARMS}, 'All3seed/all15bank closure')
    release_path = ACTIVATION / 'RELEASE.json'
    launch = read(ACTIVATION / 'LAUNCH.json')
    cfg, pins, phase = owner.release_config(release_path, sha(release_path), True)
    require(phase == PHASE and cfg['owner_manifest_sha256'] == OWNER_SHA
            and cfg['output_relative'] == FAMILY.name and pins['stage_manifest_sha256'] == STAGE_SHA,
            'Original exact owner release and family')
    parent = read(FAMILY / 'PARENT_OWNER.json')
    terminal = read(FAMILY / 'PARENT_TERMINAL.json')
    closure = read(FAMILY / 'FAMILY_CLOSURE.json')
    require(not (FAMILY / 'FAMILY_FAILURE.json').exists() and terminal['family_complete'] is True
            and terminal['error'] is None and terminal['completed_seed_count'] == 3
            and terminal['owner'] == parent['identity'] and owner.same(launch['parent'], parent['identity'])
            and owner.identity(parent['identity']['pid']) is None and not owner.members(parent['identity'])
            and launch['release_sha256'] == parent['release_sha256'] == sha(release_path)
            and closure['complete'] is True and closure['fixed_seeds'] == list(SEEDS)
            and closure['required_bank_records'] == 15 and closure['stage_source'] == stage.source_identity()
            and closure['source_manifest_sha256'] == OWNER_SHA
            and closure['execution_source_commit'] == parent['execution_source_commit'] == cfg['execution_source_commit'],
            'Successful original owner terminal and actual absent parent/group')
    require([row['seed'] for row in closure['completed']] == list(SEEDS), 'Exactly three completed owner entries')
    gpu_pids = {pid for pid, _ in owner.gpu_rows()}
    receipts = {'release': cfg, 'launch': launch, 'owner_source_bindings': pins,
                'parent_owner': parent, 'parent_terminal': terminal, 'family_closure': closure, 'seed_handles': {},
                'bank_completions': {str(s): {a: records[(s, a)][1] for a in ARMS} for s in SEEDS}}
    bound_paths = [release_path, ACTIVATION / 'LAUNCH.json', FAMILY / 'PARENT_OWNER.json',
                   FAMILY / 'PARENT_TERMINAL.json', FAMILY / 'FAMILY_CLOSURE.json']
    for seed, row in zip(SEEDS, closure['completed']):
        handle_path = FAMILY / 'handles' / ('seed' + str(seed) + '.json')
        exit_path = handle_path.with_name('seed' + str(seed) + '.EXIT.json')
        handle, exited = read(handle_path), read(exit_path)
        require(handle['seed'] == exited['seed'] == seed and handle['parent'] == parent['identity']
                and handle['child'] == exited['child'] and exited == row['exit_receipt']
                and exited['reaped'] is True and exited['exit_code'] == 0 and exited['reason'] is None
                and exited['group_absent'] is True and exited['no_owned_CUDA'] is True
                and owner.identity(exited['child']['pid']) is None and not owner.members(exited['child'])
                and all(owner.identity(item['pid']) is None for item in exited['witnessed_owned_members'])
                and not gpu_pids.intersection({parent['identity']['pid'], *(item['pid'] for item in exited['witnessed_owned_members'])})
                and sha(FAMILY / ('seed' + str(seed)) / 'COMPLETE.json') == row['completion_sha256']
                and sha(FAMILY / ('seed' + str(seed)) / 'WORKER_COST.json') == row['worker_cost_sha256'],
                'Original reaped child receipt and actual absent owned handles/CUDA')
        receipts['seed_handles'][str(seed)] = {'handle': handle, 'exit': exited,
            'block_completion': read(FAMILY / ('seed' + str(seed)) / 'COMPLETE.json'),
            'scientific_worker_cost': read(FAMILY / ('seed' + str(seed)) / 'WORKER_COST.json')}
        bound_paths.extend([handle_path, exit_path, FAMILY / ('seed' + str(seed)) / 'COMPLETE.json',
                            FAMILY / ('seed' + str(seed)) / 'WORKER_COST.json'])
    receipts['bound_files'] = {str(path.relative_to(PHASE)): sha(path) for path in bound_paths}
    return records, receipts


def compare(actual, frozen):
    require(actual.keys() == frozen.keys(), 'Unchanged readout metric inventory')
    differences, numeric_errors = {}, []
    exact = ('correctcount', 'member_correctcount', 'member_metric_semantics')
    for key, value in actual.items():
        old = frozen[key]
        if key in exact:
            matches = value == old
            detail = {'reconstructed': value, 'frozen': old, 'exact': True}
        else:
            left, right = (value, old) if isinstance(value, list) else ([value], [old])
            require(len(left) == len(right), 'Unchanged member metric count')
            delta = [abs(a - b) for a, b in zip(left, right)]
            numeric_errors.extend(delta)
            matches = all(math.isfinite(a) and math.isfinite(b) and d <= TOLERANCE
                          for a, b, d in zip(left, right, delta))
            detail = {'reconstructed': value, 'frozen': old, 'absolute_errors': delta, 'tolerance': TOLERANCE}
        if not matches:
            differences[key] = detail
    return {'matches': not differences, 'exact_correctcounts': all(actual[k] == frozen[k] for k in exact[:2]),
            'maximum_absolute_numeric_error': max(numeric_errors, default=0.), 'discrepancies': differences}


def paired_seed_intervals(metrics):
    tcrit = 4.302652729911275
    contrasts = {}
    for reference in ('S_joint4head_gate', 'U4_sameB_full_untied_gate', 'S_one_path_gate', 'C4_gate_identity_erased'):
        differences = dict(accuracy_gain_pp=[100*(metrics[str(s)]['C4_gate']['correctcount']
            - metrics[str(s)][reference]['correctcount'])/5274 for s in SEEDS])
        for field in ('served_NLL', 'Brier'):
            differences[field + '_delta'] = [metrics[str(s)]['C4_gate'][field]-metrics[str(s)][reference][field] for s in SEEDS]
        summaries = {}
        for name, values in differences.items():
            mean = sum(values)/3
            standard_error = math.sqrt(sum((value-mean)**2 for value in values)/2/3)
            summaries[name] = dict(all3_differences=values, mean=mean, standard_error=standard_error,
                                  descriptive_t_interval95=[mean-tcrit*standard_error, mean+tcrit*standard_error])
        contrasts[reference] = dict(role=('one-path secondary' if reference == 'S_one_path_gate' else
            'permanent value-erasure interpretation' if reference == 'C4_gate_identity_erased' else 'co-primary'), metrics=summaries)
    return dict(role='Descriptive paired-seed intervals only', n=3, df=2, tcrit=tcrit, seeds=list(SEEDS),
        difference_direction='C4_gate minus reference', contrasts=contrasts,
        scope='Fixed-split optimizer-seed variation conditional on the selected development checkpoints',
        generalization_certificate=False, selection_bias_certificate=False, new_gate=False, frozen_gate_changed=False)


def descriptive(torch, prediction, truth, native_prediction):
    pool = prediction['served_probabilities'].argmax(-1)
    members = prediction['member_probabilities'].argmax(-1)
    native_ok, pool_ok = native_prediction == truth, pool == truth
    member_ok = members == truth[None]
    any_correct = member_ok.any(0)
    common_wrong = (~any_correct) & (members == members[0]).all(0)
    count = lambda mask: int(mask.sum())
    return dict(native_correctcount=count(native_ok), served_correctcount=count(pool_ok),
        repairs=count((~native_ok) & pool_ok), harm=count(native_ok & (~pool_ok)),
        member_repairs=[count((~native_ok) & row) for row in member_ok],
        member_harm=[count(native_ok & (~row)) for row in member_ok],
        any_member_correct=count(any_correct), any_member_correct_on_native_errors=count((~native_ok) & any_correct),
        common_wrong_rival=count(common_wrong), common_wrong_rival_on_native_errors=count(common_wrong & (~native_ok)),
        common_wrong_rival_by_class=[count(common_wrong & (members[0] == k)) for k in range(10)],
        pooled_only_rescues=count(pool_ok & (~any_correct)),
        pooled_only_rescues_on_native_errors=count((~native_ok) & pool_ok & (~any_correct)),
        pool_wrong_with_any_member_correct=count((~pool_ok) & any_correct))


def descriptive_erasure_contrast(errors):
    """Subtract the already reported error counts; no additional metric or gate."""
    seeds = {}
    for seed in SEEDS:
        candidate = errors[str(seed)]['C4_gate']
        erased = errors[str(seed)]['C4_gate_identity_erased']
        seeds[str(seed)] = {field: ([a-b for a, b in zip(values, erased[field])]
            if isinstance(values, list) else values-erased[field]) for field, values in candidate.items()}
    return dict(role='Descriptive permanent value-erasure contrast only; cannot rescue the frozen gate',
        candidate='C4_gate', reference='C4_gate_identity_erased',
        difference_direction='C4_gate minus C4_gate_identity_erased',
        metric_inventory='Differences of the existing descriptive error counts and count vectors', seeds=seeds,
        correct_neighbor_semantics_proven=False, new_gate=False, frozen_gate_changed=False)


def worker(output):
    started = time.monotonic()
    with (output / 'WORKER_CLAIM.json').open('x') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), parent_pid=os.getppid(), readout_program_sha256=sha(__file__))) + '\n')
    stage = torch = live = None
    timings, validation, errors, bank_receipts = {}, {}, {}, {}
    native_calls = 0
    success = False
    try:
        then = time.monotonic()
        stage, owner, pins, runtime = sources()
        records, custody = closed_owner(stage, owner)
        timings['source_route_all15_owner_barrier_seconds'] = time.monotonic() - then
        write(output / 'CUSTODY.json', dict(source=stage.source_identity(), readout_program_sha256=sha(__file__),
            owner_manifest_sha256=OWNER_SHA, runtime=runtime, role_bindings=read(STAGE / 'INPUT_FILES.json'), receipts=custody))
        then = time.monotonic()
        collector = load(STAGE / 'collect.py', '_readout_original_collector')
        gate = collector.collect_family(family_root=FAMILY, output=output / 'FROZEN_GATE.json',
            later_execution_authorized=True, comparative_opening_authorized=True)
        write(output / 'DESCRIPTIVE_SEED_INTERVALS.json', paired_seed_intervals(gate['complete_metrics']))
        timings['original_frozen_gate_seconds'] = time.monotonic() - then
        then = time.monotonic()
        train, valid, origin = stage.load_roles(later_execution_authorized=True)
        timings['role_hash_loading_seconds'] = time.monotonic() - then
        (output / 'server_arrays').mkdir()
        for seed in SEEDS:
            validation[str(seed)], errors[str(seed)], bank_receipts[str(seed)] = {}, {}, {}
            for arm in ARMS:
                then = time.monotonic()
                path, endpoint = records[(seed, arm)]
                live = stage.reconstruct_selected_files(bank_directory=path, train_data=train, origin=origin,
                                                       later_execution_authorized=True)
                torch = live.native.torch
                setup_seconds = time.monotonic() - then
                native_calls += live.run['capture_calls_executed']
                require(live.purpose == 'selected_serving' and live.work['native_capture_calls'] == 0
                        and live.work['native_updates'] == 0 and live.work['learned_restores'] == 1
                        and live.run['native_training_executed'] == native_calls == 0, 'Cached-only reconstruction; zero native calls')
                bank = live.banks[arm]
                before = dict(bank.counters)
                then = time.monotonic()
                with torch.no_grad():
                    prediction = live.serve_ids(valid['ids'])[arm]
                    truth = valid['y'].to(live.native.device)
                    require(prediction['heldout_ids'].shape == (5274,)
                            and torch.equal(prediction['heldout_ids'].cpu(), valid['ids']), 'All5274 original ordered VALID IDs')
                    actual = live.posterior.readout(torch, prediction, truth)
                    native_p = torch.softmax(live.base[prediction['heldout_ids']], dim=-1)
                    errors[str(seed)][arm] = descriptive(torch, prediction, truth, native_p.argmax(-1))
                torch.cuda.synchronize(0)
                serving_readout_seconds = time.monotonic() - then
                delta = {key: bank.counters[key] - value for key, value in before.items()}
                require(delta['serving_calls'] == 1 and delta['serving_route_forwards']
                        == delta['serving_route_forward_attempts'] == stage.EXPECTED[arm][1]
                        and bank.steps == bank.counters['completed_training_steps'] == 0
                        and live.work['complete_VALID_events'] == 0, 'Actual serving work only; no evaluate/train/selector')
                validation[str(seed)][arm] = dict(selected_label_epoch=endpoint['selected_label_epoch'], metrics=actual,
                    **compare(actual, gate['complete_metrics'][str(seed)][arm]))
                then = time.monotonic()
                array_path = output / 'server_arrays' / ('seed' + str(seed) + '_' + arm + '.pt')
                torch.save(live.native._cpu_tree(dict(schema='query-value-gated-allVALID-selected-serving-v1',
                    server_only=True, seed=seed, arm=arm, source=live.run['source'], selected_sha256=endpoint['selected_sha256'],
                    fixed_capture_sha256=endpoint['fixed_capture_sha256'], selected_label_epoch=endpoint['selected_label_epoch'],
                    prediction=prediction, VALID_ids=valid['ids'], VALID_truth=truth,
                    native_probabilities=native_p)), array_path)
                array_sha = sha(array_path)
                bank_receipts[str(seed)][arm] = dict(reconstruction_seconds=setup_seconds,
                    serving_readout_error_counts_seconds=serving_readout_seconds,
                    array_export_save_hash_seconds=time.monotonic() - then, arrays_sha256=array_sha,
                    arrays_bytes=array_path.stat().st_size, arrays_server_path=str(array_path), actual_work=live.work,
                    bank_counter_delta=delta, bank_counts=dict(bank.counters), run=live.run, cuda_memory=live.memory())
                live.close()
                live = bank = prediction = truth = native_p = None
                gc.collect()
                write(output / 'PROGRESS.json', dict(completed_bank_count=sum(len(row) for row in validation.values()),
                    native_calls=native_calls, bank_receipts=bank_receipts, timings_seconds=timings, automatic_retry=False))
        write(output / 'RECONSTRUCTION_VALIDATION.json', dict(role='Source reconstruction validation only',
            all15_reconstructed=True, numerical_tolerance=TOLERANCE,
            all15_match=all(row['matches'] for block in validation.values() for row in block.values()), banks=validation,
            discrepancies_preserved=True, reselection_performed=False, frozen_gate_changed=False))
        write(output / 'DESCRIPTIVE_ERRORS.json', dict(role='Descriptive error counts only; cannot rescue the frozen gate',
            all3seeds_all15banks=True, population='Original ordered all5274 VALID nodes',
            repair_harm_reference='The same seed cached native argmax',
            common_wrong_rival_semantics='All actual served members choose the same incorrect class',
            pooled_only_rescue_semantics='Pool correct while every actual served member is wrong', seeds=errors,
            permanent_value_erasure_contrast=descriptive_erasure_contrast(errors)))
        success = True
    except BaseException as error:
        write(output / 'FAILURE.json', dict(error_type=type(error).__name__, error=str(error),
            completed_validation=validation, bank_receipts=bank_receipts, native_calls=native_calls,
            live_work=live.work if live else None,
            live_bank_counts={a: dict(b.counters) for a, b in live.banks.items()} if live else None,
            discrepancies_preserved=True, automatic_retry=False, resume=False))
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        memory = live.memory() if live else None
        if live is not None:
            live.close()
        write(output / 'WORKER_COST.json', dict(success=success, inclusive_seconds=time.monotonic()-started,
            CPU_user_seconds=usage.ru_utime, CPU_system_seconds=usage.ru_stime, peak_RSS_bytes=usage.ru_maxrss*1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0) if torch is not None else None,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0) if torch is not None else None,
            last_live_cuda_memory=memory, native_calls=native_calls, serving_bank_count=sum(len(x) for x in validation.values()),
            timings_seconds=timings, bank_receipts=bank_receipts, automatic_retry=False, resume=False,
            scope='Includes closure hashes, gate, roles, imports/restores, all serving/scoring, exports and receipt I/O; excludes this final write',
            historical_cost_scope='Original scientific worker/owner costs are bound in CUSTODY; original native1100 epochs per seed remain spent, with historical elapsed/memory join pending'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorized', action='store_true')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    require(args.authorized is True, 'Root review/publication and later execution authorization required')
    if args.worker is not None:
        require(args.worker.resolve(strict=True) == OUTPUT and read(OUTPUT / 'ATTEMPT.json')['parent_pid'] == os.getppid()
                and read(OUTPUT / 'ATTEMPT.json')['readout_program_sha256'] == sha(__file__), 'One reserved exact-source readout child')
        worker(OUTPUT)
        return
    started = time.monotonic()
    require(not OUTPUT.exists(), 'One fresh routine only; no retry/resume/overwrite')
    OUTPUT.mkdir()
    write(OUTPUT / 'ATTEMPT.json', dict(readout_program_sha256=sha(__file__), parent_pid=os.getpid(), wall_seconds=WALL_SECONDS,
        active_seconds=WALL_SECONDS-CLEANUP_SECONDS, cleanup_seconds=CLEANUP_SECONDS, attempts=1, automatic_retry=False))
    process = saved = owner = None
    error = None
    actions = []
    cleanup_error = None
    def interrupted(number, frame):
        raise TimeoutError('Bounded readout stop/deadline ' + str(number))
    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(.01, WALL_SECONDS-CLEANUP_SECONDS-(time.monotonic()-started)))
    try:
        _, owner, _, _ = sources()
        with (OUTPUT / 'WORKER.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--authorized', '--worker', str(OUTPUT)],
                cwd=Path.cwd(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            saved = owner.identity(process.pid)
            require(saved and saved['group'] == saved['session'] == process.pid, 'Exact newly owned readout worker')
            process.wait(timeout=max(.01, WALL_SECONDS-CLEANUP_SECONDS-(time.monotonic()-started)))
            require(process.returncode == 0, 'Readout worker failed; outputs and costs retained')
    except BaseException as caught:
        signal.setitimer(signal.ITIMER_REAL, 0)
        error = dict(type=type(caught).__name__, message=str(caught))
        if process is not None and process.poll() is None:
            try:
                require(owner.same(owner.identity(process.pid), saved), 'Exact owned readout cleanup custody')
                os.killpg(saved['group'], signal.SIGTERM)
                actions.append('SIGTERM exact owned readout group')
                process.wait(timeout=min(5, max(.01, WALL_SECONDS-(time.monotonic()-started))))
            except subprocess.TimeoutExpired:
                try:
                    require(owner.same(owner.identity(process.pid), saved), 'Exact owned readout kill custody')
                    os.killpg(saved['group'], signal.SIGKILL)
                    actions.append('SIGKILL exact owned readout group')
                    process.wait(timeout=max(.01, WALL_SECONDS-(time.monotonic()-started)-5))
                except BaseException as failed:
                    cleanup_error = dict(type=type(failed).__name__, message=str(failed))
            except BaseException as failed:
                cleanup_error = dict(type=type(failed).__name__, message=str(failed))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        absent = saved is not None and owner.identity(saved['pid']) is None and not owner.members(saved)
        no_cuda = None
        try:
            no_cuda = saved is not None and saved['pid'] not in {pid for pid, _ in owner.gpu_rows()}
        except BaseException as failed:
            cleanup_error = dict(type=type(failed).__name__, message=str(failed))
        complete = (error is None and cleanup_error is None and process is not None and process.returncode == 0
                    and absent and no_cuda and time.monotonic()-started <= WALL_SECONDS)
        write(OUTPUT / 'ROUTINE_TERMINAL.json', dict(complete=complete, error=error, worker=saved,
            cleanup_error=cleanup_error,
            exit_code=process.returncode if process else None, reaped=process is not None and process.returncode is not None,
            actual_group_absent=absent, actual_owned_CUDA_absent=no_cuda, signals=actions,
            inclusive_seconds=time.monotonic()-started, wall_limit_seconds=WALL_SECONDS,
            child_CPU_user_seconds=usage.ru_utime, child_CPU_system_seconds=usage.ru_stime, child_peak_RSS_bytes=usage.ru_maxrss*1024,
            attempts=1, automatic_retry=False, resume=False, partial_files_and_costs_retained=True,
            result_roles=['RECONSTRUCTION_VALIDATION.json', 'FROZEN_GATE.json', 'DESCRIPTIVE_ERRORS.json',
                          'DESCRIPTIVE_SEED_INTERVALS.json'],
            original_paper_scores_replaced=False, server_arrays_exported=False))
    require(complete, 'Postclosure routine failed; retain all receipts without retry')


if __name__ == '__main__':
    main()
