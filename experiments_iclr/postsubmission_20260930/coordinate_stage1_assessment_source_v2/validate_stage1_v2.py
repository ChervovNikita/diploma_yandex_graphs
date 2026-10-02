"""Prepare a validation-only Stage1 assessment from retained, indexed evidence.

Source preparation only. Uses the Python standard library; no framework/model
imports, tensor deserialization, remote access, prediction or metric recomputation.
Execution requires separately frozen root analysis adoption and evidence hashes.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pairs(values):
    result = {}
    for key, value in values:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode(value):
    return json.loads(value, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))


def read(path):
    return decode(path.read_text())


def number(value, lower=0, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite or untyped number')
    require(value > lower if positive else value >= lower, 'Numeric range differs')
    return value


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7)


def safe(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts and str(relative) != '.',
            'Require strict relative evidence paths')
    path = root
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), 'Symlink evidence forbidden')
    require(path.resolve().is_relative_to(root), 'Evidence escapes phase')
    return path


class Evidence:
    def __init__(self, phase, remote_phase, index):
        self.phase, self.remote = phase, Path(remote_phase)
        require(index['schema'] == 'coordinate-stage1-evidence-index-v1', 'Unknown evidence index')
        self.entries = {}
        for entry in index['files']:
            require(set(entry) == {'path', 'sha256', 'bytes'}, 'Evidence index entry shape differs')
            require(entry['path'] not in self.entries, 'Duplicate indexed evidence')
            self.entries[entry['path']] = entry
        self.checked = {}

    def relative(self, remote_path):
        return str(Path(remote_path).relative_to(self.remote))

    def file(self, relative):
        require(relative in self.entries, 'Missing indexed evidence: ' + relative)
        entry, path = self.entries[relative], safe(self.phase, relative)
        require(path.is_file() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                'Evidence hash/length differs: ' + relative)
        self.checked[relative] = entry
        return path

    def json(self, relative):
        return read(self.file(relative))

    def remote_json(self, remote_path):
        return self.json(self.relative(remote_path))

    def binding(self, binding):
        path = safe(self.phase, self.relative(binding['path']))
        require(path.is_file() and sha(path) == binding['sha256'], 'Frozen binding differs')
        return path


def validate_metrics(metrics, members):
    require(len(metrics['members']) == members, 'Actual member metric count differs')
    for row in [metrics['primary_probability_pool'], metrics['same_checkpoint_mean_logit_sensitivity'],
                *metrics['members']]:
        number(row['nll'])
        for field in ['accuracy', 'macro_f1']:
            require(0 <= number(row[field]) <= 1, 'Metric outside probability range')


def validate_curve(curve, report, recipe, members, strategy):
    total = report['completed_updates']
    require(type(total) is int and 1000 <= total <= 2000 and report['final_epoch'] == total,
            'Incomplete competent horizon')
    require(total % 10 == 0 and len(curve) == total, 'Curve/update count differs')
    best, best_epoch, selected, final, stop_epoch = math.inf, None, None, None, None
    step_times, validation_times = [], []
    for epoch, row in enumerate(curve, 1):
        require(type(row['epoch']) is int and row['epoch'] == epoch and
                row['complete_optimizer_update'] is True, 'Incomplete or misordered optimizer update')
        require(type(row['nonzero_gradient_tensors']) is int and row['nonzero_gradient_tensors'] > 0 and
                row['optimizer_strategy'] == strategy, 'Gradient/optimizer receipt differs')
        number(row['training_member_mean_ce'])
        step_times.append(number(row['train_step_seconds'], positive=True))
        scheduled = epoch % recipe['eval_every'] == 0
        require(('validation' in row) == scheduled, 'Validation schedule differs')
        if not scheduled:
            require('checkpoint_improved' not in row and 'validation_seconds' not in row,
                    'Unscheduled selection metadata')
            continue
        validate_metrics(row['validation'], members)
        validation_times.append(number(row['validation_seconds'], positive=True))
        current = row['validation']['primary_probability_pool']['nll']
        improved = current < best
        require(row['checkpoint_improved'] is improved, 'Strict earliest-tie selection trace differs')
        if improved:
            best, best_epoch, selected = current, epoch, row['validation']
        final = row['validation']
        if epoch >= recipe['min_epochs'] and epoch - best_epoch >= recipe['patience'] and stop_epoch is None:
            stop_epoch = epoch
    require(report['selected_epoch'] == best_epoch and report['selected_validation'] == selected and
            report['final_validation'] == final, 'Reported checkpoint/metrics differ from retained curve')
    expected_stop = 'validation_nll_patience' if stop_epoch is not None else 'max_epochs'
    require(report['stopping_reason'] == expected_stop and
            total == (stop_epoch if stop_epoch is not None else recipe['max_epochs']),
            'Fit did not stop at the first fixed stopping opportunity')
    convergence = report['convergence']
    require(convergence['selected_validation_nll'] == best and
            convergence['epochs_since_selected'] == total - best_epoch and
            convergence['minimum_epochs_satisfied'] is True and
            convergence['patience_exhausted'] is (stop_epoch is not None) and
            convergence['max_epochs_reached'] is (total == 2000), 'Convergence receipt differs')
    require(close(math.fsum(step_times), report['elapsed_seconds']['training_complete_updates']) and
            close(math.fsum(validation_times), report['elapsed_seconds']['validation']),
            'Complete update/evaluation timing sums differ')
    boundary = best_epoch == total and curve[-1]['checkpoint_improved'] is True
    return boundary


def validate_cell(evidence, cell, protocol, protocol_binding, sources, sources_binding, rules):
    dataset, arm, seed, recipe_name = (cell[k] for k in ('dataset', 'arm', 'seed', 'recipe'))
    recipe = protocol['recipes'][recipe_name]
    suffix = f'{dataset}__core0__seed{seed}__{arm}__{recipe_name}'
    root = evidence.relative(protocol['phases']['fit']['output_root']) + '/' + suffix
    manifest = evidence.json(root + '/artifacts.json')
    entries = manifest['files']
    require(manifest['schema_version'] == 1 and len({e['path'] for e in entries}) == len(entries),
            'Invalid cell artifact inventory')
    snapshots = [f'verified_sources/{i:03d}_{Path(item["path"]).name}'
                 for i, item in enumerate(sources['files'])]
    expected = {'cell_report.json', 'training_curve.jsonl', 'invocation.json', 'model_storage.json',
                'frozen_protocol.json', 'frozen_sources.json', 'frozen_data_manifest.json',
                'source_snapshots.json', 'initial_model_state.pt', 'selected_state.pt', 'final_state.pt',
                'selected_validation_member_logits.pt', 'final_validation_member_logits.pt',
                'selected_deployment_state.pt', 'wall_clock_receipt.json', *snapshots}
    require({item['path'] for item in entries} == expected, 'Unexpected/missing cell artifact payload')
    for item in entries:
        path = evidence.file(root + '/' + item['path'])
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                'Cell artifact manifest differs from trusted indexed payload')
    payloads = {item['path']: item for item in entries}
    for filename, binding in [('frozen_protocol.json', protocol_binding), ('frozen_sources.json', sources_binding)]:
        require(sha(evidence.file(root + '/' + filename)) == binding['sha256'], 'Frozen cell input differs')
    data = evidence.json(root + '/frozen_data_manifest.json')
    data_sha = protocol['data_bindings'][dataset]['data_manifest_sha256']
    require(sha(evidence.file(root + '/frozen_data_manifest.json')) == data_sha, 'Data snapshot differs')
    mapping = evidence.json(root + '/source_snapshots.json')
    require(mapping == [{**item, 'snapshot': snapshot} for item, snapshot in zip(sources['files'], snapshots)],
            'Verified source snapshot provenance differs')
    for source, snapshot in zip(sources['files'], snapshots):
        require(sha(evidence.file(root + '/' + snapshot)) == source['sha256'], 'Source snapshot differs')
    invocation = evidence.json(root + '/invocation.json')
    for key, value in {'mode': 'fit', 'dataset': dataset, 'arm': arm, 'seed': seed, 'split': 'core0',
                       'recipe': recipe_name, 'protocol_sha256': protocol_binding['sha256'],
                       'source_manifest_sha256': sources_binding['sha256'], 'data_manifest_sha256': data_sha}.items():
        require(invocation[key] == value, 'Invocation identity differs: ' + key)
    report = evidence.json(root + '/cell_report.json')
    require(report['schema_version'] == 1 and report['mode'] == 'fit' and report['status'] == 'FIT_COMPLETE' and
            report['test_labels_read'] is False and report['test_scores_or_logits_written'] is False,
            'No complete isolated fit receipt')
    require(report['checkpoint_serialized_bytes']['selected'] == payloads['selected_state.pt']['bytes'] and
            report['checkpoint_serialized_bytes']['final'] == payloads['final_state.pt']['bytes'] and
            report['deployment_model_only_checkpoint_bytes'] == payloads['selected_deployment_state.pt']['bytes'],
            'Serialized checkpoint lengths differ')
    fresh = report['fresh_checkpoint_deployment']
    require(fresh['checkpoint'] == 'selected' and fresh['checkpoint_sha256'] ==
            payloads['selected_deployment_state.pt']['sha256'] and fresh['checkpoint_bytes'] ==
            payloads['selected_deployment_state.pt']['bytes'] and fresh['new_process'] is False and
            fresh['process_import_initialization_included'] is False and fresh['cuda_context_initialization_included'] is False,
            'Fresh selected deployment scope/checkpoint binding differs')
    require(report['device'] == 'cuda:0' and report['gpu_name'] == 'NVIDIA A100-SXM4-80GB' and
            report['torch_version'] == protocol['runtime']['torch_version'], 'Runtime identity differs')
    members = 1 if arm in ('single', 'gt_sep_single') else 4
    strategy = ('independent_member_optimizers_sum_member_ce_backward' if arm == 'untied'
                else 'one_optimizer_mean_member_ce_backward')
    expected_provenance = {'protocol_sha256': protocol_binding['sha256'],
                          'source_manifest_sha256': sources_binding['sha256'],
                          'data_manifest_sha256': data_sha, 'cell': cell, 'recipe': recipe,
                          'optimizer': recipe['optimizer'], 'optimizer_strategy': strategy,
                          'optimizer_count': 4 if arm == 'untied' else 1,
                          'graph_artifact_sha256': data['graph']['sha256'],
                          'development_target_artifact_sha256': {
                              role: data['splits']['core0'][role]['sha256'] for role in ('train', 'validation')},
                          'mask_indices_sha256': {**data['splits']['core0']['mask_indices_sha256'],
                                                  'test_indices_only': data['test_indices_sha256']}}
    require(report['provenance'] == expected_provenance, 'Full report provenance differs')
    curve = [decode(line) for line in evidence.file(root + '/training_curve.jsonl').read_text().splitlines()]
    boundary = validate_curve(curve, report, recipe, members, strategy)
    storage = evidence.json(root + '/model_storage.json')
    gate, deployed = cell['storage_gate'], report['deployed_tensor_bytes']
    require(storage['total_model_tensor_bytes'] == storage['unique_storage_bytes'] ==
            gate['expected_model_tensor_bytes'] == deployed['model_parameters_and_all_buffers_including_indices'] and
            storage['index_buffer_bytes'] == gate['expected_index_buffer_bytes'], 'Deployed byte gate differs')
    require(storage['parameter_bytes'] + storage['buffer_bytes'] == storage['total_model_tensor_bytes'] and
            all(p['dtype'] == 'torch.float32' for p in storage['parameters']) and
            all(b['dtype'] == 'torch.int64' for b in storage['buffers'] if b['kind'] == 'index'),
            'Parameter/index accounting differs')
    spec = storage['specification']
    for key, value in {'arm': arm, 'seed': seed, 'permutation_seed': cell['permutation_seed'],
                       'actual_members': members, 'requested_members': 4, 'hidden_dim': recipe['width'],
                       'num_layers': recipe['num_layers'], 'input_dim': data['num_features'],
                       'output_dim': data['num_classes']}.items():
        require(spec[key] == value, 'Model storage specification differs: ' + key)
    if gate['equal_byte_reference'] is not None:
        reference = gate['equal_byte_reference']
        require(abs(storage['total_model_tensor_bytes'] / reference['bytes'] - 1) <= reference['relative_tolerance'],
                'Equal-byte control exceeds frozen tolerance')
    elapsed, latency = report['elapsed_seconds'], report['latency']
    for value in elapsed.values():
        number(value)
    require(latency['checkpoint'] == 'selected' and latency['nodes'] == 'validation' and
            latency['scope'] == protocol['profiling']['scope'] and latency['cuda_synchronization'] is True and
            latency['warmups'] == 5 and latency['repeats'] == 20 and len(latency['seconds']) == 20 and
            latency['input_transfer_included'] is False and latency['output_transfer_included'] is False,
            'Warm all-member inference scope differs')
    for value in latency['seconds']:
        number(value, positive=True)
    require(close(statistics.median(latency['seconds']), latency['median_seconds']) and
            close(statistics.mean(latency['seconds']), latency['mean_seconds']) and
            close(sorted(latency['seconds'])[18], latency['p95_seconds']), 'Warm latency summaries differ')
    require(report['gpu_peak_memory']['profiling_includes_optimizer_or_targets'] is False,
            'Prediction timing retained optimizer/targets')
    descriptive_sum = math.fsum(elapsed[key] for key in rules['full_training_components_descriptive'])
    excluded_profiling = math.fsum(elapsed[key] for key in rules['excluded_profiling_components'])
    training = elapsed['total_scientific_phase'] - excluded_profiling
    require(training > 0 and math.isfinite(training), 'Residual-inclusive full-training duration is invalid')
    wall = evidence.json(root + '/wall_clock_receipt.json')
    require(wall['includes_source_verification_input_snapshots_and_framework_imports'] is True and
            wall['excludes_final_artifact_hashing'] is True and
            number(wall['invocation_seconds_before_final_manifest'], positive=True) >= elapsed['total_scientific_phase'],
            'Invocation timing scope differs')
    selected = report['selected_validation']['primary_probability_pool']
    return {'identity': {k: cell[k] for k in ('dataset', 'split', 'seed', 'arm', 'recipe')},
            'status': 'INCONCLUSIVE_BOUNDARY_IMPROVING' if boundary else 'COMPLETE_COMPETENT',
            'boundary_improving': boundary, 'report_path': root + '/cell_report.json',
            'report_sha256': sha(evidence.file(root + '/cell_report.json')),
            'curve_sha256': sha(evidence.file(root + '/training_curve.jsonl')),
            'artifact_manifest_sha256': sha(evidence.file(root + '/artifacts.json')),
            'completed_updates': report['completed_updates'], 'selected_epoch': report['selected_epoch'],
            'selected_primary_nll': selected['nll'], 'selected_primary_accuracy': selected['accuracy'],
            'selected_metrics': report['selected_validation'], 'final_metrics': report['final_validation'],
            'full_training_seconds': number(training, positive=True),
            'six_component_training_seconds_descriptive': descriptive_sum,
            'unattributed_training_residual_seconds_descriptive': training - descriptive_sum,
            'excluded_profiling_seconds_descriptive': excluded_profiling,
            'warm_median_seconds': number(latency['median_seconds'], positive=True),
            'warm_p95_seconds': latency['p95_seconds'], 'complete_update_seconds': elapsed['training_complete_updates'],
            'total_scientific_seconds': elapsed['total_scientific_phase'],
            'invocation_seconds_before_final_manifest': wall['invocation_seconds_before_final_manifest'],
            'all_cost_components': elapsed, 'storage': storage, 'gpu_peak_memory': report['gpu_peak_memory'],
            'fresh_checkpoint_deployment': report['fresh_checkpoint_deployment'], 'issues': []}


def validate_batch(evidence, batch, request, queue):
    bound, inner, bridge = (batch[key] for key in ('remote_bound', 'remote_supervisor', 'remote_bridge'))
    bstart, bend = evidence.remote_json(bound + '/START.json'), evidence.remote_json(bound + '/TERMINAL.json')
    own = [batch['remote_argv'][0], queue['remote_phase'] + '/coordinate_ensemble_stage_entry_v4/stage_entry.py',
           '--request', batch['request']['path'], '--supervisor', inner]
    require(bstart['child_argv'] == own and bstart['root_request'] == {
        'path': evidence.relative(batch['request']['path']), 'sha256': batch['request']['sha256']} and
        bstart['whole_cap_seconds'] == 28000 and bstart['normal_child_budget_seconds'] == 27995 and
        bstart['termination_grace_seconds'] == 5 and bstart['ssh_destination'] == queue['ssh_destination'] and
        bstart['inner_supervisor_directory'] == inner and bstart['cwd'] == queue['remote_repository'] and
        bstart['phase'] == queue['remote_phase'] and bstart['shell'] is False and
        bstart['start_new_session'] is True and bstart['root_request_child_flag'] == '--request',
        'Bounded supervisor start differs')
    require(bstart['argv'] == batch['remote_argv'][batch['remote_argv'].index('--') + 1:],
            'Bounded nested argv differs')
    protected = {item['path']: item['sha256'] for item in request['protected_files']}
    for key, relative in [('source_sha256', 'protocols/bounded_run_v1.py'),
                          ('authorization_source_sha256', 'protocols/run_authorized_v2.py'),
                          ('logging_source_sha256', 'protocols/run_logged.py'),
                          ('wrapper_sha256', 'protocols/repo_env.sh'),
                          ('research_script_sha256', 'coordinate_ensemble_stage_entry_v4/stage_entry.py')]:
        require(bstart[key] == protected[queue['remote_phase'] + '/' + relative], 'Bounded source differs')
    require(bend['complete'] is True and bend['child_exit_code'] == 0 and bend['timed_out'] is False and
            bend['interruption'] is None and bend['within_whole_cap'] is True and
            bend['root_request_unchanged'] is True and bend['whole_supervised_seconds'] <= 28000 and
            bend['START_sha256'] == sha(evidence.file(evidence.relative(bound) + '/START.json')) and
            bend['stdout_stderr_sha256'] == sha(evidence.file(evidence.relative(bound) + '/stdout_stderr.log')),
            'No complete normally bounded batch')
    for name, item in bend['inner_evidence'].items():
        path = evidence.file(evidence.relative(inner) + '/' + name)
        require(item['sha256'] == sha(path) and item['bytes'] == path.stat().st_size, 'Inner evidence differs')
    command, environment = evidence.remote_json(inner + '/command.json'), evidence.remote_json(inner + '/environment.json')
    require(command == {'argv': own, 'cwd': queue['remote_repository'], 'shell': False}, 'Inner command differs')
    require(environment['repository'] == queue['remote_repository'] and environment['phase'] == queue['remote_phase'] and
            environment['python_bytecode_disabled'] is True and environment['CUDA_VISIBLE_DEVICES'] == '0' and
            environment['script_sha256'] == protected[own[1]] and
            environment['supervisor_sha256'] == protected[queue['remote_phase'] + '/protocols/run_logged.py'] and
            environment['wrapper_sha256'] == protected[queue['remote_phase'] + '/protocols/repo_env.sh'],
            'Inner runtime environment differs')
    for value in environment['cache_and_temp'].values():
        require(Path(value).is_relative_to(Path(queue['remote_phase'])), 'Cache/temp escapes phase')
    completion, authorization = evidence.remote_json(inner + '/completion.json'), evidence.remote_json(inner + '/authorization.json')
    require(completion['exit_code'] == 0 and completion['environment_sha256'] ==
            sha(evidence.file(evidence.relative(inner) + '/environment.json')) and
            completion['command_sha256'] == sha(evidence.file(evidence.relative(inner) + '/command.json')) and
            completion['log_sha256'] == sha(evidence.file(evidence.relative(inner) + '/stdout_stderr.log')),
            'Inner completion differs')
    require(authorization['ssh_destination'] == queue['ssh_destination'] and authorization['visible_gpu_count'] == 1 and
            authorization['gpu_uuid'] == queue['GPU_UUID'] and authorization['gpu_name'] == 'NVIDIA A100-SXM4-80GB' and
            authorization['idle_required'] is True and 0 <= authorization['initial_memory_MiB'] <= 100 and
            authorization['initial_utilization_percent'] == 0 and authorization['child_exit_code'] == 0 and
            authorization['inner_completion_sha256'] == sha(evidence.file(evidence.relative(inner) + '/completion.json')) and
            authorization['route_record_sha256'] == protected[queue['remote_phase'] + '/protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json'] and
            authorization['allocation_guard_sha256'] == protected[queue['remote_phase'] + '/protocols/run_authorized_v2.py'],
            'Authorized GPU/route completion differs')
    start, terminal = evidence.remote_json(bridge + '/START.json'), evidence.remote_json(bridge + '/TERMINAL.json')
    require(start['request'] == batch['request']['path'] and start['request_sha256'] == batch['request']['sha256'] and
            start['bound_start_sha256'] == sha(evidence.file(evidence.relative(bound) + '/START.json')) and
            start['bridge_sha256'] == protected[own[1]] and start['supervisor'] == inner and start['mode'] == 'fit' and
            start['operation_count'] == 18 and start['test_admitted'] is False and start['validation_fit_admitted'] is True and
            start['parent_CUDA_VISIBLE_DEVICES'] == start['child_CUDA_VISIBLE_DEVICES'] == '0', 'Bridge admission differs')
    require(terminal['complete'] is True and terminal['error'] is None and terminal['request_unchanged'] is True and
            terminal['completed_operation_count'] == terminal['requested_operation_count'] == len(terminal['records']) == 18 and
            terminal['test_admitted'] is False and terminal['validation_fit_admitted'] is True and
            terminal['START_sha256'] == sha(evidence.file(evidence.relative(bridge) + '/START.json')) and
            len(terminal['gpu_guards']) == 18, 'Bridge completion differs')
    for number_, (operation, record) in enumerate(zip(request['operations'], terminal['records'])):
        prefix = evidence.relative(bridge) + f'/operation_{number_:03d}'
        saved = evidence.json(prefix + '_TERMINAL.json')
        launched = evidence.json(prefix + '_START.json')
        guard = evidence.json(prefix + '_GPU_GUARD.json')
        require(saved == record and record['number'] == number_ and record['exit_code'] == 0 and
                record['argv'] == operation['argv'] and record['shell'] is False and
                record['child_CUDA_VISIBLE_DEVICES'] == '0' and record['seconds'] <= 1500 and
                record['log_sha256'] == sha(evidence.file(prefix + '.log')) and
                all(record[k] == v for k, v in launched.items()), 'Operation completion/argv differs')
        require(guard == terminal['gpu_guards'][number_] and guard['status'] == 'idle' and
                guard['maximum_settling_seconds'] == 15 and guard['elapsed_seconds'] <= 15 and
                record['gpu_guard_receipt_sha256'] == sha(evidence.file(prefix + '_GPU_GUARD.json')) and
                record['gpu_guard_receipt'] == bridge + f'/operation_{number_:03d}_GPU_GUARD.json',
                'Retained GPU guard differs')
        final_sample = guard['samples'][-1]
        require(len(final_sample['gpu_rows']) == 1 and final_sample['query_finished_seconds'] <= 15,
                'Final guard sample scope differs')
        metadata = final_sample['gpu_rows'][0]
        require(len(metadata) == 6 and metadata[0] == '0' and metadata[1] == queue['GPU_UUID'] and
                metadata[2] == 'NVIDIA A100-SXM4-80GB' and 0 <= int(metadata[3]) <= 100 and
                int(metadata[4]) == 81920 and int(metadata[5]) == 0, 'Final GPU metadata sample differs')
        gpu = record['gpu_before_child']
        require(gpu['uuid'] == queue['GPU_UUID'] and gpu['index'] == '0' and gpu['memory_total_MiB'] == 81920 and
                gpu['name'] == 'NVIDIA A100-SXM4-80GB' and 0 <= gpu['memory_used_MiB'] <= 100 and
                gpu['utilization_percent'] == 0, 'Operation GPU identity/idle state differs')
        cell = operation['cell']
        suffix = f'{operation["dataset"]}__{cell["split"]}__seed{cell["seed"]}__{cell["arm"]}__{cell["recipe"]}'
        protocol = read(evidence.binding(request['protocol']))
        require(record['runner_output'] == protocol['phases']['fit']['output_root'] + '/' + suffix,
                'Operation canonical output differs')
    return {'number': batch['number'], 'complete': True, 'start_UTC': bstart['start_UTC'],
            'terminal_UTC': bend['terminal_UTC'], 'whole_supervised_seconds': bend['whole_supervised_seconds']}


def validate_launcher(evidence, index, queue, spec):
    root = index['queue_launcher_receipt_root']
    start, terminal = evidence.json(root + '/START.json'), evidence.json(root + '/TERMINAL.json')
    require(start['manifest_sha256'] == spec['queue']['sha256'] and start['launcher_sha256'] ==
            spec['launcher']['sha256'] and start['ssh_destination'] == queue['ssh_destination'] and
            start['aggregate_reserved_cap_seconds'] == 84000 and start['shell_local'] is False and
            start['validation_only'] is True and start['test_scoring_admitted'] is False,
            'Local serial launcher admission differs')
    require(terminal['complete'] is True and terminal['error'] is None and terminal['manifest_unchanged'] is True and
            len(terminal['records']) == 3 and terminal['remote_completion_requires_retained_evidence_audit'] is True,
            'Local serial launcher incomplete')
    for batch, record in zip(queue['batches'], terminal['records']):
        n = batch['number']
        begun = evidence.json(root + f'/batch{n:02d}_START.json')
        ended = evidence.json(root + f'/batch{n:02d}_TERMINAL.json')
        require(ended == record and record['number'] == n and record['exit_code'] == 0 and
                begun['request'] == batch['request'] and begun['remote_argv'] == batch['remote_argv'] and
                record['log_sha256'] == sha(evidence.file(root + f'/batch{n:02d}_SSH.log')),
                'Local exact batch start/completion differs')


def aggregate(rows, rules):
    grouped = {(r['identity']['dataset'], r['identity']['arm'], r['identity']['seed']): r for r in rows}
    graphs, gates = {}, {}
    for dataset in ['AmazonPhoto', 'CoauthorCS']:
        def values(arm, field):
            return [grouped[dataset, arm, seed][field] for seed in [17, 29, 43]]
        def mean(arm, field):
            return statistics.mean(values(arm, field))
        def gain(control):
            denominator = mean(control, 'selected_primary_nll')
            require(denominator > 0, 'Relative NLL undefined for zero control mean')
            return 1 - mean('coordinate', 'selected_primary_nll') / denominator
        strongest = min(rules['byte_control_tie_order'], key=lambda arm: mean(arm, 'selected_primary_nll'))
        gains = {arm: gain(arm) for arm in ['factor', 'original', 'permutation', 'bias_only', strongest]}
        wins = sum(pf < f for pf, f in zip(values('coordinate', 'selected_primary_nll'), values('factor', 'selected_primary_nll')))
        accuracy_loss = {arm: 100 * (mean(arm, 'selected_primary_accuracy') - mean('coordinate', 'selected_primary_accuracy'))
                         for arm in ['factor', 'original', strongest]}
        costs = {}
        for field in ['full_training_seconds', 'warm_median_seconds']:
            paired = [a / b for a, b in zip(values('coordinate', field), values('factor', field))]
            costs[field] = {'ratio_of_seed_means': mean('coordinate', field) / mean('factor', field),
                            'paired_ratios': paired, 'mean_paired_ratio_descriptive': statistics.mean(paired)}
        parameter_match = all(grouped[dataset, 'coordinate', seed]['storage']['parameter_bytes'] ==
                              grouped[dataset, 'factor', seed]['storage']['parameter_bytes'] and
                              grouped[dataset, 'coordinate', seed]['storage']['buffer_bytes'] -
                              grouped[dataset, 'factor', seed]['storage']['buffer_bytes'] ==
                              grouped[dataset, 'coordinate', seed]['storage']['index_buffer_bytes'] -
                              grouped[dataset, 'factor', seed]['storage']['index_buffer_bytes'] for seed in [17, 29, 43])
        paired_gains = {arm: [1 - a / b if b > 0 else None for a, b in
                            zip(values('coordinate', 'selected_primary_nll'), values(arm, 'selected_primary_nll'))]
                        for arm in ['factor', 'original', 'permutation', 'bias_only', 'single', 'untied', 'heads', 'gt_sep_single']}
        graphs[dataset] = {'relative_nll_gain_ratio_of_means': gains, 'strongest_byte_control': strongest,
                           'PF_F_strict_wins': wins, 'accuracy_loss_percentage_points': accuracy_loss,
                           'costs': costs, 'PF_F_parameters_match_except_index_buffers': parameter_match,
                           'paired_relative_gains_descriptive': paired_gains,
                           'all_arm_mean_NLL': {arm: mean(arm, 'selected_primary_nll') for arm in rules['all_arms']}}
        gates[dataset + '_F_G0_NLL'] = gains['factor'] >= .02 and gains['original'] >= .02
        gates[dataset + '_PF_F_wins'] = wins >= 2
        gates[dataset + '_accuracy'] = all(loss <= .5 for loss in accuracy_loss.values())
        gates[dataset + '_parameter_match'] = parameter_match
        gates[dataset + '_full_training_cost'] = costs['full_training_seconds']['ratio_of_seed_means'] <= 1.25
        gates[dataset + '_warm_inference_cost'] = costs['warm_median_seconds']['ratio_of_seed_means'] <= 1.25
    pooled = {arm: statistics.mean(graphs[d]['relative_nll_gain_ratio_of_means'][arm] for d in graphs)
              for arm in ['permutation', 'bias_only']}
    gates['equal_graph_P0_B0_NLL'] = all(value >= .01 for value in pooled.values())
    byte_gains = [g['relative_nll_gain_ratio_of_means'][g['strongest_byte_control']] for g in graphs.values()]
    gates['strongest_byte_NLL'] = any(value >= .01 for value in byte_gains) and all(value >= -.01 for value in byte_gains)
    return {'graphs': graphs, 'equal_graph_relative_gains': pooled, 'gates': gates,
            'decision': 'STAGE1_PASS' if all(gates.values()) else 'STAGE1_NO_GO',
            'confirmation_or_continuation_execution_authorized': False,
            'inference_scope': 'Three model seeds on one core0 partition per graph; no graph-population significance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['specification', 'adoption', 'evidence_index', 'output']:
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    for name in ['specification_sha256', 'adoption_sha256', 'evidence_index_sha256']:
        parser.add_argument('--' + name.replace('_', '-'), required=True)
    args = parser.parse_args()
    phase = Path(__file__).resolve().parents[1]
    for path in [args.specification, args.adoption, args.evidence_index]:
        require(path.resolve().is_relative_to(phase), 'Assessment input escapes allowed phase')
    for name in ['specification', 'adoption', 'evidence_index']:
        require(sha(getattr(args, name)) == getattr(args, name + '_sha256'), 'Frozen assessment input differs')
    spec, adoption, index = read(args.specification), read(args.adoption), read(args.evidence_index)
    require(spec['schema'] == 'coordinate-stage1-assessment-specification-v2' and
            adoption['schema'] == 'coordinate-stage1-assessment-adoption-v2' and adoption['root_adopted'] is True and
            adoption['comparative_outcomes_inspected_before_adoption'] is False and
            adoption['specification_sha256'] == args.specification_sha256 and
            adoption['validator_sha256'] == sha(Path(__file__)), 'Assessment not prospectively adopted')
    rules = adoption['rules']
    require(rules == spec['proposed_rules_requiring_root_adoption'], 'Analysis precision differs from this source version')
    evidence = Evidence(phase, spec['remote_phase'], index)
    evidence.binding(spec['launcher'])
    queue = read(evidence.binding(spec['queue']))
    protocol = read(evidence.binding(queue['protocol']))
    require(protocol['cells'] == spec['expected_cells'] and protocol['recipes'] == spec['expected_recipes'],
            'Scientific packet differs from assessment freeze')
    requests = [read(evidence.binding(batch['request'])) for batch in queue['batches']]
    for request in requests:
        for binding in [request[k] for k in ['protocol', 'sources', 'runner', 'allocation_evidence']] + request['protected_files']:
            evidence.binding(binding)
    sources = read(evidence.binding(requests[0]['sources']))
    for item in sources['files']:
        evidence.binding({'path': item['path'], 'sha256': item['sha256']})
    fit_prefix = evidence.relative(protocol['phases']['fit']['output_root']) + '/'
    expected_suffixes = {f'{c["dataset"]}__core0__seed{c["seed"]}__{c["arm"]}__{c["recipe"]}'
                         for c in protocol['cells']}
    require(all(relative[len(fit_prefix):].split('/')[0] in expected_suffixes
                for relative in evidence.entries if relative.startswith(fit_prefix)),
            'Evidence includes unfrozen Stage1 cell outputs')
    operations = [operation for request in requests for operation in request['operations']]
    require(len(operations) == 54 and [(o['dataset'], *[o['cell'][k] for k in ['split', 'seed', 'arm', 'recipe']]) for o in operations] ==
            [(c['dataset'], *[c[k] for k in ['split', 'seed', 'arm', 'recipe']]) for c in protocol['cells']],
            'All54 exact operations/order not preserved')
    batch_records, global_issues, rows = [], [], []
    try:
        validate_launcher(evidence, index, queue, spec)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        global_issues.append({'reason': 'Serial launcher evidence: ' + str(exc)})
    for batch, request in zip(queue['batches'], requests):
        try:
            batch_records.append(validate_batch(evidence, batch, request, queue))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            global_issues.append({'batch': batch['number'], 'reason': str(exc)})
    if len(batch_records) == 3:
        for left, right in zip(batch_records, batch_records[1:]):
            if datetime.fromisoformat(left['terminal_UTC']) > datetime.fromisoformat(right['start_UTC']):
                global_issues.append({'reason': 'Serial bounded batch intervals overlap'})
    for cell in protocol['cells']:
        try:
            row = validate_cell(evidence, cell, protocol, queue['protocol'], sources,
                                requests[0]['sources'], rules)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            row = {'identity': {k: cell[k] for k in ['dataset', 'split', 'seed', 'arm', 'recipe']},
                   'status': 'INCONCLUSIVE_EVIDENCE_OR_FIT', 'issues': [str(exc)]}
            suffix = f'{cell["dataset"]}__core0__seed{cell["seed"]}__{cell["arm"]}__{cell["recipe"]}'
            retained = evidence.relative(protocol['phases']['fit']['output_root']) + '/' + suffix + '/cell_report.json'
            try:
                row['retained_report_diagnostic_only'] = evidence.json(retained)
                row['report_sha256'] = sha(evidence.file(retained))
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                row['retained_report_diagnostic_only'] = None
            try:
                row['retained_failure_diagnostic_only'] = evidence.json(retained.rsplit('/', 1)[0] + '/failure.json')
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                row['retained_failure_diagnostic_only'] = None
        rows.append(row)
    complete = not global_issues and len(rows) == 54 and all(r['status'] == 'COMPLETE_COMPETENT' for r in rows)
    assessment = {'decision': 'STAGE1_INCONCLUSIVE', 'gates_evaluated': False}
    if complete:
        try:
            assessment = aggregate(rows, rules)
            assessment['gates_evaluated'] = True
        except (ValueError, ZeroDivisionError) as exc:
            global_issues.append({'reason': str(exc)})
    output = args.output.resolve()
    require(output.is_relative_to(phase) and output != phase and not output.exists(), 'Require new confined assessment output')
    output.mkdir(parents=True)
    report = {'schema': 'coordinate-stage1-assessment-v2', 'specification_sha256': args.specification_sha256,
              'adoption_sha256': args.adoption_sha256, 'evidence_index_sha256': args.evidence_index_sha256,
              'validator_sha256': sha(Path(__file__)), 'all_expected_rows_retained': True, 'cells': rows,
              'batches': batch_records, 'global_issues': global_issues, 'assessment': assessment,
              'test_labels_read': False, 'predictions_recomputed': False, 'binary_tensor_semantics_verified': False,
              'exploratory_intervals_computed': False, 'scientific_novelty_or_paper_verdict': None}
    with (output / 'ASSESSMENT.json').open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    columns = ['dataset', 'split', 'seed', 'arm', 'recipe', 'status', 'selected_primary_nll',
               'selected_primary_accuracy', 'full_training_seconds', 'warm_median_seconds', 'issues']
    with (output / 'ALL54_CELLS.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row['identity'], **{k: row.get(k) for k in columns[5:] if k != 'issues'},
                             'issues': '; '.join(row['issues'])})
    with (output / 'VERIFIED_EVIDENCE.json').open('x') as stream:
        json.dump({'files': list(evidence.checked.values())}, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'decision': assessment['decision'], 'rows': len(rows), 'output': str(output)}))


if __name__ == '__main__':
    main()
