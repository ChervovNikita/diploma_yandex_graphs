#!/usr/bin/env python3
"""Owned epoch5 restore diagnosis: exactly two full native epoch6 continuations."""
import argparse
import ast
import math
import json
from pathlib import Path
import time
from common import (HERE, PHASE, EXECUTION, require, sha, write as strict_write, utc, gate,
                    runtime, load_module, Progress, monitor, inventory)


def finite_json(value):
    """Preserve nonfinite telemetry as explicit tagged metadata, not invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'finite': False, 'repr': repr(value)}
    if isinstance(value, dict):
        return {key: finite_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(child) for child in value]
    return value


def write(path, value):
    strict_write(path, finite_json(value))


def exact(left, right, torch, label):
    require(type(left) is type(right), label + ': type differs')
    if isinstance(left, torch.Tensor):
        require(left.shape == right.shape and left.dtype == right.dtype and left.layout == right.layout
                and torch.equal(left, right), label + ': tensor differs')
    elif isinstance(left, dict):
        require(left.keys() == right.keys(), label + ': keys differ')
        for key in left:
            exact(left[key], right[key], torch, label + '/' + str(key))
    elif isinstance(left, (list, tuple)):
        require(len(left) == len(right), label + ': length differs')
        for index, (a, b) in enumerate(zip(left, right)):
            exact(a, b, torch, label + '/' + str(index))
    else:
        require(left == right, label + ': value differs')


def differences(candidate, reference, torch, atol, rtol):
    """Scalar diagnostics for every block/tensor; reference remains on the right."""
    report = {'exact': True, 'within_fixed_numeric_rule': True, 'tensor_count': 0,
              'element_count': 0, 'maximum_absolute_difference': 0.0,
              'maximum_fraction_of_allowed_error': 0.0, 'items': []}
    def visit(a, b, name):
        row = {'path': name, 'exact': True, 'within_fixed_numeric_rule': True}
        if type(a) is not type(b):
            row.update(exact=False, within_fixed_numeric_rule=False, issue='type differs')
        elif isinstance(a, torch.Tensor):
            report['tensor_count'] += 1
            report['element_count'] += a.numel()
            row.update(shape=list(a.shape), dtype=str(a.dtype), layout=str(a.layout))
            if a.shape != b.shape or a.dtype != b.dtype or a.layout != b.layout:
                row.update(exact=False, within_fixed_numeric_rule=False, issue='tensor schema differs')
            else:
                row['exact'] = bool(torch.equal(a, b))
                if a.is_floating_point():
                    finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
                    row['finite'] = finite
                    if finite and a.numel():
                        delta = (a.to(torch.float64) - b.to(torch.float64)).abs()
                        allowed = atol + rtol * b.to(torch.float64).abs()
                        row.update(maximum_absolute_difference=float(delta.max()),
                                   maximum_fraction_of_allowed_error=float((delta / allowed.clamp_min(1e-300)).max()),
                                   elements_outside_fixed_numeric_rule=int((delta > allowed).sum()),
                                   within_fixed_numeric_rule=bool((delta <= allowed).all()))
                        report['maximum_absolute_difference'] = max(report['maximum_absolute_difference'], row['maximum_absolute_difference'])
                        report['maximum_fraction_of_allowed_error'] = max(report['maximum_fraction_of_allowed_error'], row['maximum_fraction_of_allowed_error'])
                    elif not finite:
                        row.update(within_fixed_numeric_rule=False, issue='nonfinite tensor')
                else:
                    row.update(within_fixed_numeric_rule=row['exact'], unequal_elements=int((a != b).sum()))
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                row.update(exact=False, within_fixed_numeric_rule=False, issue='keys differ')
            else:
                for key in a:
                    visit(a[key], b[key], name + '/' + str(key))
                return
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                row.update(exact=False, within_fixed_numeric_rule=False, issue='length differs')
            else:
                for index, (x, y) in enumerate(zip(a, b)):
                    visit(x, y, name + '/' + str(index))
                return
        else:
            row.update(exact=a == b, within_fixed_numeric_rule=a == b)
        report['exact'] = report['exact'] and row['exact']
        report['within_fixed_numeric_rule'] = report['within_fixed_numeric_rule'] and row['within_fixed_numeric_rule']
        report['items'].append(row)
    visit(candidate, reference, '')
    return report


def step_aliases(unit, saved, torch):
    """Only CPU Adam step scalars and storage addresses are exported as values."""
    live = unit[2].state_dict()['state']
    source = saved['Adam']['state']
    require(live.keys() == source.keys(), 'Live/saved Adam parameter IDs differ')
    rows = []
    for key in source:
        if 'step' not in source[key]:
            continue
        a, b = live[key]['step'], source[key]['step']
        require(isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.numel() == b.numel() == 1,
                'Qualified Adam step scalar schema differs')
        require(a.device.type == b.device.type == 'cpu', 'Non-capturable native Adam CPU steps required')
        rows.append({'parameter_id': key, 'live_step': float(a.item()), 'saved_step': float(b.item()),
                     'live_device': str(a.device), 'saved_device': str(b.device),
                     'live_data_ptr': a.data_ptr(), 'saved_data_ptr': b.data_ptr(),
                     'same_tensor_object': a is b, 'same_storage_pointer': a.data_ptr() == b.data_ptr()})
    require(rows, 'No owned Adam step scalars found')
    return rows


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True, type=Path)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    release, plan = gate(args.root_release, args.release_sha256, 'update_inclusive')
    output = EXECUTION / 'update_inclusive/run01'
    require(not output.exists(), 'Fresh bounded diagnostic output required; no retry/resume')
    output.mkdir(parents=True)
    progress = Progress(output)
    progress.update(state_file_loads=0, feature_weights_only_loads=0, diagnostic_optimizer_pre_observations=0, diagnostic_optimizer_post_observations=0,
                    diagnostic_reference_payload_bytes=0, diagnostic_reference_peak_payload_bytes=0)
    stopped = observer = sample = None
    evidence = {}
    fixed_reports, fixed_verdicts = [], []
    try:
        native, loaded, bridge, prototype, graph_ops = runtime(plan)
        torch, np, bodies, x, train, inspection, identities = loaded
        progress.add(feature_weights_only_loads=1, weights_only_loads=1)
        stopped, observer, sample = monitor(output, plan['stages']['update_inclusive']['caps'], progress, torch)
        observed = load_module('pubmed_shared4_owned_diagnostic_native_observer',
                               PHASE / plan['observer_source_path'])
        write(output / 'TRAIN_INSPECTION.json', inspection)
        authority = plan['state_authority']
        identity_path = PHASE / authority['identity_path']
        identity = json.loads(identity_path.read_text())
        require(sha(identity_path) == release['owned_serialized_identity_sha256'] and all(identity['input_files'][name] == row for name, row in identities.items()),
                'Owned engineering state input/identity provenance differs')
        state_path = PHASE / authority['state_path']
        require(not state_path.is_symlink() and sha(state_path) == identity['state']['sha256']
                and state_path.stat().st_size == identity['state']['bytes'], 'Just-owned engineering state bytes differ')
        saved = torch.load(state_path, map_location='cpu', weights_only=True)
        progress.add(state_file_loads=1, weights_only_loads=1)
        for key in ('schema','mode','seed','selected_epoch','source_manifest_sha256','root_release_sha256',
                    'input_files','engineering_only','scientific_donor_allowed','factory_initialization'):
            require(saved[key] == identity[key], 'Owned selected metadata differs: ' + key)
        require(saved['scores_sha256'] == identity['scores']['sha256'], 'Owned selected score hash metadata differs')
        pristine = native.clone(saved['after_VALID'], torch)
        saved_digest = observed.state_identity(saved, torch)
        units = [bridge.make_shared4(PHASE, bodies, prototype, graph_ops, 0, 'private', x, train) for _ in range(2)]
        progress.add(factory_calls=2)
        tolerance = plan['engineering_tolerance']
        # The original comparison FunctionDef is reused unchanged after admission.
        comparator_path = PHASE / plan['comparator_source_path']
        tree = ast.parse(comparator_path.read_text())
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'compare']
        require(len(definitions) == 1, 'Original fixed comparator absent')
        scope = {'require': require}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(comparator_path), 'exec'), scope)
        numeric_compare = scope['compare']
        def record_fixed(candidate, reference, label, exact_rule=False):
            verdict = {'label': label, 'comparison_collected': True, 'passed_original_fixed_predicate': True}
            try:
                numeric_compare(candidate, reference, torch, 0 if exact_rule else tolerance['atol'],
                                0 if exact_rule else tolerance['rtol'], label, fixed_reports)
            except Exception as error:
                verdict.update(passed_original_fixed_predicate=False, type=type(error).__name__, condition=str(error))
            fixed_verdicts.append(verdict)
        from step_observer import StepObserver, tensor_bytes
        from update_observer import UpdateChannels
        steps = StepObserver(native, observed, torch, np, require, exact, differences,
                             write, output, progress, tolerance, plan['diagnostic_observer_limits'])
        channels = UpdateChannels(native, observed, torch, np, exact, differences, numeric_compare, tolerance, plan['diagnostic_observer_limits'])
        retained = {'saved': saved, 'pristine': pristine}
        def retained_bound(name, value):
            retained[name] = value
            size = sum(tensor_bytes(child, torch) for child in retained.values())
            require(size <= plan['diagnostic_observer_limits']['fixed_retained_state_tensor_payload_bytes'],
                    'Diagnostic fixed retained state payload cap exceeded')
            evidence['fixed_retained_state_tensor_payload_bytes'] = size
        retained_bound('pristine', pristine)
        neutrality = {'checks': 0}
        def neutral(a, b, label, reports):
            exact(a, b, torch, label)
            neutrality['checks'] += 1
        def epoch(unit, role, label):
            progress.update(phase=label, current_arm='private', current_epoch=6)
            before_checks = neutrality['checks']
            remove_work = observed.observe_work(unit, prototype, progress)
            remove_channels = channels.install(unit, role)
            remove_steps = steps.install(unit, role)
            try:
                loss, stream = observed.observed_epoch(native, bodies, unit, train, torch, np, progress, neutral, None)
            finally:
                remove_steps()
                remove_channels()
                remove_work()
            require(neutrality['checks'] - before_checks == 38, 'Native observation RNG-neutrality coverage differs')
            return loss, stream
        # Separate complete clones immediately before each unchanged restore.
        first_input = native.clone(saved['after_VALID'], torch)
        retained_bound('first_restore_input', first_input)
        native.restore(units[0], first_input, torch, np)
        original_before = native.capture(units[0], torch, np)
        retained_bound('original_before', original_before)
        evidence.update(saved_tree_sha256_before_first_epoch=saved_digest,
                        first_pre_epoch_full_state_sha256=observed.state_identity(original_before, torch),
                        first_pre_epoch_vs_pristine=differences(original_before, pristine, torch, 0, 0),
                        first_restore_CPU_Adam_step_aliases_to_saved=step_aliases(units[0], saved['after_VALID'], torch),
                        first_restore_CPU_Adam_step_aliases_to_own_clone=step_aliases(units[0], first_input, torch))
        write(output / 'RESTORE_ALIAS_PROGRESS.json', evidence)
        original_loss, original_stream = epoch(units[0], 'reference', 'first_isolated_restored_copy_native_epoch6')
        original_end = native.capture(units[0], torch, np)
        retained_bound('original_end', original_end)
        evidence.update(saved_tree_sha256_after_first_epoch=observed.state_identity(saved, torch),
                        first_post_epoch_full_state_sha256=observed.state_identity(original_end, torch),
                        saved_tree_mutated_after_first_epoch=observed.state_identity(saved, torch) != saved_digest,
                        saved_after_VALID_mutation_after_first_epoch=differences(saved['after_VALID'], pristine, torch, 0, 0),
                        first_copy_CPU_Adam_step_aliases_to_saved_after_first_epoch=step_aliases(units[0], saved['after_VALID'], torch),
                        first_copy_CPU_Adam_step_aliases_to_own_clone_after_first_epoch=step_aliases(units[0], first_input, torch))
        write(output / 'RESTORE_ALIAS_PROGRESS.json', evidence)
        second_input = native.clone(saved['after_VALID'], torch)
        retained_bound('second_restore_input', second_input)
        native.restore(units[1], second_input, torch, np)
        restored_before = native.capture(units[1], torch, np)
        retained_bound('restored_before', restored_before)
        evidence.update(second_restore_CPU_Adam_step_aliases_to_saved=step_aliases(units[1], saved['after_VALID'], torch),
                        second_restore_CPU_Adam_step_aliases_to_own_clone=step_aliases(units[1], second_input, torch),
                        second_vs_first_live_CPU_Adam_step_aliases=step_aliases(units[1], {'Adam': units[0][2].state_dict()}, torch),
                        second_pre_epoch_full_state_sha256=observed.state_identity(restored_before, torch),
                        second_pre_epoch_vs_first_pre_epoch=differences(restored_before, original_before, torch, 0, 0),
                        second_pre_epoch_vs_pristine=differences(restored_before, pristine, torch, 0, 0),
                        second_pre_epoch_vs_current_saved_tree=differences(restored_before, saved['after_VALID'], torch, 0, 0))
        write(output / 'RESTORE_ALIAS_PROGRESS.json', evidence)
        # Numerical mismatches are recorded; this native epoch is never shortened.
        restored_loss, restored_stream = epoch(units[1], 'candidate', 'second_isolated_reconstructed_copy_native_epoch6')
        restored_end = native.capture(units[1], torch, np)
        retained_bound('restored_end', restored_end)
        original_finite, restored_finite = math.isfinite(original_loss), math.isfinite(restored_loss)
        finite = original_finite and restored_finite
        delta = abs(restored_loss - original_loss) if finite else None
        allowed = tolerance['atol'] + tolerance['rtol'] * abs(original_loss) if original_finite else None
        loss_report = {'candidate_restored_loss': restored_loss if restored_finite else None,
                       'reference_original_loss': original_loss if original_finite else None,
                       'candidate_restored_loss_repr': repr(restored_loss), 'reference_original_loss_repr': repr(original_loss),
                       'candidate_finite': restored_finite, 'reference_finite': original_finite,
                       'absolute_difference': delta, 'allowed_error': allowed,
                       'fraction_of_allowed_error': delta / allowed if finite else None,
                       'finite_and_within_fixed_numeric_rule': finite and delta <= allowed,
                       'atol': tolerance['atol'], 'rtol': tolerance['rtol'], 'reference_on_right': 'original_loss',
                       'finiteness_telemetry_changes_original_comparator': False}
        record_fixed(restored_loss, original_loss, 'private/serialized_complete_next_epoch_loss')
        blocks = {key: differences(restored_end[key], original_end[key], torch, tolerance['atol'], tolerance['rtol'])
                  for key in ('encoder','predictor','Adam','gradients')}
        exact_blocks = {key: differences(restored_end[key], original_end[key], torch, 0, 0) for key in ('RNG','flags','invest')}
        for key in ('encoder','predictor','Adam','gradients'):
            record_fixed(restored_end[key], original_end[key], 'private/serialized_next_epoch_' + key)
        for key in ('RNG','flags','invest'):
            record_fixed(restored_end[key], original_end[key], 'private/exact_serialized_next_epoch_' + key, True)
        record_fixed(original_before, pristine, 'first_isolated_restore_exact_pre_epoch', True)
        record_fixed(restored_before, original_before, 'second_isolated_restore_exact_pre_epoch', True)
        stream_keys = ('negative_calls','iterator_calls','batches','start_RNG_sha256','end_RNG_sha256')
        streams_equal = {key: original_stream[key] for key in stream_keys} == {key: restored_stream[key] for key in stream_keys}
        idle_first = native.capture(units[0], torch, np)
        retained_bound('idle_first', idle_first)
        first_idle_blocks = {key: differences(idle_first[key], original_end[key], torch, 0, 0)
                             for key in ('encoder','predictor','Adam','gradients','flags','invest')}
        evidence.update(loss=loss_report, next_state_blocks=blocks, next_state_exact_blocks=exact_blocks,
                        first_idle_unit_state_blocks=first_idle_blocks,
                        idle_unit_RNG_note='RNG is global; idle-unit mutation comparison excludes global RNG and compares all unit-owned state blocks.',
                        second_post_epoch_full_state_sha256=observed.state_identity(restored_end, torch),
                        saved_tree_sha256_after_second_epoch=observed.state_identity(saved, torch),
                        saved_tree_mutated_after_second_epoch=observed.state_identity(saved, torch) != saved_digest,
                        saved_after_VALID_mutation_after_second_epoch=differences(saved['after_VALID'], pristine, torch, 0, 0),
                        first_copy_CPU_Adam_step_aliases_to_saved_after_second_epoch=step_aliases(units[0], saved['after_VALID'], torch),
                        second_copy_CPU_Adam_step_aliases_to_saved_after_second_epoch=step_aliases(units[1], saved['after_VALID'], torch),
                        first_copy_CPU_Adam_step_aliases_to_own_clone_after_second_epoch=step_aliases(units[0], first_input, torch),
                        second_copy_CPU_Adam_step_aliases_to_own_clone_after_second_epoch=step_aliases(units[1], second_input, torch),
                        native_streams_exact=streams_equal, native_streams={'first':original_stream,'second':restored_stream},
                        RNG_neutral_observations=neutrality['checks'], per_step_observer=steps.finish(),
                        unchanged_original_comparator_reports=fixed_reports, unchanged_original_comparator_verdicts=fixed_verdicts)
        # These are diagnostic observations, never qualification/science admission.
        final_parity = (loss_report['finite_and_within_fixed_numeric_rule'] and streams_equal
                        and all(row['within_fixed_numeric_rule'] for row in blocks.values())
                        and all(row['exact'] for row in exact_blocks.values())
                        and all(row['passed_original_fixed_predicate'] for row in fixed_verdicts))
        evidence['observed_final_continuation_parity_within_fixed_rule'] = bool(final_parity)
        evidence['selected_update_channels'] = channels.finish()
        channel_pass = all(v['passed_original_fixed_predicate'] for r in evidence['selected_update_channels']['comparisons']
                           for v in r['original_fixed_comparator_verdicts'])
        transitions = [row for role in ('reference', 'candidate') for event in steps.ledger[role]
                       for row in event.get('moment_step_transitions', [])]
        transition_pass = all(row['transition_within_original_rule'] for row in transitions)
        selected_state_pass = all(report['within_fixed_numeric_rule']
                                  for event in steps.ledger['candidate'] if event['step'] <= 2
                                  for name, report in event['blocks'].items()
                                  if name in ('encoder', 'predictor', 'Adam', 'gradients'))
        controls = (streams_equal and not evidence['saved_tree_mutated_after_first_epoch']
                    and not evidence['saved_tree_mutated_after_second_epoch']
                    and evidence['first_pre_epoch_vs_pristine']['exact']
                    and evidence['second_pre_epoch_vs_first_pre_epoch']['exact']
                    and all(row['exact'] for row in first_idle_blocks.values())
                    and all(row['state_fully_inspected'] and row['native_inputs_exact']
                            for row in evidence['selected_update_channels']['comparisons'])
                    and evidence['selected_update_channels']['comparisons'][0]['same_state_interpretation_allowed'])
        evidence['decision'] = {
            'preserved_old_qualification_status': 'FAILED', 'automatic_retry': False,
            'complete_initial_state_stream_saved_and_idle_controls_valid': bool(controls),
            'selected_update_original_rule_channels_pass': bool(channel_pass),
            'selected_pre_post_Adam_state_original_rule_pass': bool(selected_state_pass),
            'first_two_native_moment_step_transition_checks_pass': bool(transition_pass),
            'status': ('STATE_CONTROL_FAILURE_REPAIR_REQUIRED' if not controls else
                       'UPDATE_TRANSITION_INCONSISTENCY_REQUIRES_INSPECTION' if not transition_pass else
                       'UPDATE_INCLUSIVE_PARITY_OBSERVED' if final_parity and channel_pass and selected_state_pass else
                       'STOP_ENGINEERING_BRANCH_NO_DEMONSTRATED_REPAIR'),
            'reason': ('Inspect the concrete failing state/update path; no causal kernel attribution.'
                       if not controls or not transition_pass else
                       'Complete original-rule parity observed; complete remaining inherited qualification before fresh predictive fits.'
                       if final_parity and channel_pass and selected_state_pass else
                       'The preserved numerical blocker remains without a demonstrated state or moment-transition defect. Stop ad hoc repeats and return the admission decision to root.'),
            'scientific_fit_admitted': False, 'tolerance_changed': False}

        final = progress.snapshot()
        require(final['Adam_started'] == final['Adam_completed'] == 72 and final['native_epochs_started'] == final['native_epochs_completed'] == 2
                and final['VALID_started'] == final['VALID_completed'] == 0 and final['weights_only_loads'] == 2
                and final['factory_calls'] == 2 and neutrality['checks'] == 76
                and final['diagnostic_optimizer_pre_observations'] == final['diagnostic_optimizer_post_observations'] == 72,
                'Exact bounded diagnostic work differs')
        torch.cuda.synchronize(0)
        stopped.set()
        observer.join()
        sample()
        write(output / 'DIAGNOSTIC.json', {'schema':'pubmed-shared4-update-inclusive-continuation-diagnostic-v1','status':'COMPLETE','UTC':utc(),
              'diagnostic_evidence':evidence,'progress':final,'inclusive_child_wall_seconds':time.monotonic()-started,
              'source_manifest_sha256':release['source_manifest_sha256'],'root_release_sha256':args.release_sha256,
              'owned_engineering_state_sha256':identity['state']['sha256'],'owned_serialized_identity_sha256':sha(identity_path),
              'preserved_failed_qualification_terminal_sha256':release['failed_qualifier_terminal_sha256'],
              'engineering_qualification_PASS':False,'scientific_fit_admitted':False,'state_donor_allowed':False,'TEST_supported':False,
              'numeric_rule_changed':False,'restore_helper_changed':False,
              'VALID_files_opened':False,'TEST_files_opened':False,'score_files_opened':False,'kernel_causality_established':False,
              'diagnostic_completed_independently_of_observed_numerical_parity':True,
              'interpretation':'Collected isolated-restore/per-step continuation evidence only. Completion is not parity or qualification PASS. First observed divergence does not establish kernel causation.'})
    except BaseException as error:
        write(output / 'FAILURE.json', {'status':'FAILED','type':type(error).__name__,'condition':str(error),
              'progress':progress.snapshot(),'diagnostic_evidence':evidence,'inclusive_child_wall_seconds':time.monotonic()-started,'automatic_retry':False})
        raise
    finally:
        if stopped is not None:
            stopped.set()
            observer.join()
        write(output / 'FINAL_CUSTODY.json', {'files':inventory(output),'stage':'update_inclusive','completed':(output/'DIAGNOSTIC.json').exists()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
