#!/usr/bin/env python3
"""Owned epoch5 restore diagnosis: exactly two full native epoch6 continuations."""
import argparse
import json
from pathlib import Path
import time
from common import (HERE, PHASE, EXECUTION, require, sha, write, utc, gate,
                    runtime, load_module, Progress, monitor, inventory)


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
    release, plan = gate(args.root_release, args.release_sha256, 'continuation_diagnostic')
    output = EXECUTION / 'continuation_diagnostic/run01'
    require(not output.exists(), 'Fresh bounded diagnostic output required; no retry/resume')
    output.mkdir(parents=True)
    progress = Progress(output)
    stopped = observer = sample = None
    evidence = {}
    try:
        native, loaded, bridge, prototype, graph_ops = runtime(plan)
        torch, np, bodies, _, _, inspection, x, train, _, _, identities = loaded
        stopped, observer, sample = monitor(output, plan['stages']['continuation_diagnostic']['caps'], progress, torch)
        observed = load_module('pubmed_shared4_owned_diagnostic_native_observer',
                               PHASE / 'pubmed_shared4_bridge_qualification_source_20261004_v2/native_observer.py')
        write(output / 'AVAILABLE_INSPECTION.json', inspection)
        authority = plan['state_authority']
        identity_path = PHASE / authority['identity_path']
        identity = json.loads(identity_path.read_text())
        require(sha(identity_path) == release['owned_serialized_identity_sha256'] and identity['input_files'] == identities,
                'Owned engineering state input/identity provenance differs')
        state_path = PHASE / authority['state_path']
        require(not state_path.is_symlink() and sha(state_path) == identity['state']['sha256']
                and state_path.stat().st_size == identity['state']['bytes'], 'Just-owned engineering state bytes differ')
        saved = torch.load(state_path, map_location='cpu', weights_only=True)
        progress.add(weights_only_loads=1)
        for key in ('schema','mode','seed','selected_epoch','source_manifest_sha256','root_release_sha256',
                    'input_files','engineering_only','scientific_donor_allowed','factory_initialization'):
            require(saved[key] == identity[key], 'Owned selected metadata differs: ' + key)
        require(saved['scores_sha256'] == identity['scores']['sha256'], 'Owned selected score hash metadata differs')
        # This immutable CPU observation is never passed to restore. Both actual
        # restores use the original just-loaded tree and unchanged native helper.
        pristine = native.clone(saved['after_VALID'], torch)
        saved_digest = observed.state_identity(saved, torch)
        units = [bridge.make_shared4(PHASE, bodies, prototype, graph_ops, 0, 'private', x, train) for _ in range(2)]
        progress.add(factory_calls=2)
        neutrality = {'checks': 0}
        def neutral(a, b, label, reports):
            exact(a, b, torch, label)
            neutrality['checks'] += 1
        def epoch(unit, label):
            progress.update(phase=label, current_arm='private', current_epoch=6)
            before_checks = neutrality['checks']
            remove = observed.observe_work(unit, prototype, progress)
            try:
                loss, stream = observed.observed_epoch(native, bodies, unit, train, torch, np, progress, neutral, None)
            finally:
                remove()
            require(neutrality['checks'] - before_checks == 38, 'Native observation RNG-neutrality coverage differs')
            return loss, stream
        native.restore(units[0], saved['after_VALID'], torch, np)
        original_before = native.capture(units[0], torch, np)
        exact(original_before, pristine, torch, 'first_unmodified_restore_exact_pre_epoch')
        evidence.update(saved_tree_sha256_before_first_epoch=saved_digest,
                        first_pre_epoch_full_state_sha256=observed.state_identity(original_before, torch),
                        first_restore_CPU_Adam_step_aliases=step_aliases(units[0], saved['after_VALID'], torch))
        write(output / 'RESTORE_ALIAS_PROGRESS.json', evidence)
        original_loss, original_stream = epoch(units[0], 'first_restored_copy_native_epoch6')
        original_end = native.capture(units[0], torch, np)
        evidence.update(saved_tree_sha256_after_first_epoch=observed.state_identity(saved, torch),
                        first_post_epoch_full_state_sha256=observed.state_identity(original_end, torch),
                        saved_tree_mutated_after_first_epoch=observed.state_identity(saved, torch) != saved_digest,
                        saved_after_VALID_mutation_after_first_epoch=differences(saved['after_VALID'], pristine, torch, 0, 0),
                        first_copy_CPU_Adam_step_aliases_after_first_epoch=step_aliases(units[0], saved['after_VALID'], torch))
        write(output / 'RESTORE_ALIAS_PROGRESS.json', evidence)
        # Deliberately retain the failed source's unmodified shared-tree restore
        # order. Report its pre-epoch differences before any second continuation.
        native.restore(units[1], saved['after_VALID'], torch, np)
        restored_before = native.capture(units[1], torch, np)
        evidence.update(second_restore_CPU_Adam_step_aliases=step_aliases(units[1], saved['after_VALID'], torch),
                        second_pre_epoch_full_state_sha256=observed.state_identity(restored_before, torch),
                        second_pre_epoch_vs_first_pre_epoch=differences(restored_before, original_before, torch, 0, 0),
                        second_pre_epoch_vs_current_saved_tree=differences(restored_before, saved['after_VALID'], torch, 0, 0))
        exact(restored_before, saved['after_VALID'], torch, 'second_unmodified_restore_matches_current_saved_tree')
        write(output / 'RESTORE_ALIAS_PROGRESS.json', evidence)
        restored_loss, restored_stream = epoch(units[1], 'second_reconstructed_copy_native_epoch6')
        restored_end = native.capture(units[1], torch, np)
        tolerance = plan['engineering_tolerance']
        delta = abs(restored_loss - original_loss)
        allowed = tolerance['atol'] + tolerance['rtol'] * abs(original_loss)
        loss_report = {'candidate_restored_loss': restored_loss, 'reference_original_loss': original_loss,
                       'absolute_difference': delta, 'allowed_error': allowed,
                       'fraction_of_allowed_error': delta / allowed, 'within_fixed_numeric_rule': delta <= allowed,
                       'atol': tolerance['atol'], 'rtol': tolerance['rtol'], 'reference_on_right': 'original_loss'}
        blocks = {key: differences(restored_end[key], original_end[key], torch, tolerance['atol'], tolerance['rtol'])
                  for key in ('encoder','predictor','Adam','gradients')}
        exact_blocks = {key: differences(restored_end[key], original_end[key], torch, 0, 0) for key in ('RNG','flags','invest')}
        stream_keys = ('negative_calls','iterator_calls','batches','start_RNG_sha256','end_RNG_sha256')
        streams_equal = {key: original_stream[key] for key in stream_keys} == {key: restored_stream[key] for key in stream_keys}
        evidence.update(loss=loss_report, next_state_blocks=blocks, next_state_exact_blocks=exact_blocks,
                        second_post_epoch_full_state_sha256=observed.state_identity(restored_end, torch),
                        saved_tree_sha256_after_second_epoch=observed.state_identity(saved, torch),
                        saved_after_VALID_mutation_after_second_epoch=differences(saved['after_VALID'], pristine, torch, 0, 0),
                        first_copy_CPU_Adam_step_aliases_after_second_epoch=step_aliases(units[0], saved['after_VALID'], torch),
                        second_copy_CPU_Adam_step_aliases_after_second_epoch=step_aliases(units[1], saved['after_VALID'], torch),
                        first_live_state_mutation_without_first_copy_updates=differences(native.capture(units[0], torch, np)['Adam'], original_end['Adam'], torch, 0, 0),
                        native_streams_exact=streams_equal, native_streams={'first':original_stream,'second':restored_stream},
                        RNG_neutral_observations=neutrality['checks'])
        final = progress.snapshot()
        require(final['Adam_started'] == final['Adam_completed'] == 72 and final['native_epochs_started'] == final['native_epochs_completed'] == 2
                and final['VALID_started'] == final['VALID_completed'] == 0 and final['weights_only_loads'] == 1
                and final['factory_calls'] == 2 and neutrality['checks'] == 76, 'Exact bounded diagnostic work differs')
        torch.cuda.synchronize(0)
        stopped.set()
        observer.join()
        sample()
        write(output / 'DIAGNOSTIC.json', {'schema':'pubmed-shared4-owned-continuation-diagnostic-v1','status':'COMPLETE','UTC':utc(),
              'diagnostic_evidence':evidence,'progress':final,'inclusive_child_wall_seconds':time.monotonic()-started,
              'source_manifest_sha256':release['source_manifest_sha256'],'root_release_sha256':args.release_sha256,
              'owned_engineering_state_sha256':identity['state']['sha256'],'owned_serialized_identity_sha256':sha(identity_path),
              'preserved_failed_qualification_terminal_sha256':release['failed_qualifier_terminal_sha256'],
              'engineering_qualification_PASS':False,'scientific_fit_admitted':False,'state_donor_allowed':False,'TEST_supported':False,
              'numeric_rule_changed':False,'restore_helper_changed':False,
              'interpretation':'Collected owned continuation diagnosis only. Pre-epoch state mutation/alias evidence precedes numerical interpretation. No scientific or qualification pass.'})
    except BaseException as error:
        write(output / 'FAILURE.json', {'status':'FAILED','type':type(error).__name__,'condition':str(error),
              'progress':progress.snapshot(),'diagnostic_evidence':evidence,'inclusive_child_wall_seconds':time.monotonic()-started,'automatic_retry':False})
        raise
    finally:
        if stopped is not None:
            stopped.set()
            observer.join()
        write(output / 'FINAL_CUSTODY.json', {'files':inventory(output),'stage':'continuation_diagnostic','completed':(output/'DIAGNOSTIC.json').exists()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
