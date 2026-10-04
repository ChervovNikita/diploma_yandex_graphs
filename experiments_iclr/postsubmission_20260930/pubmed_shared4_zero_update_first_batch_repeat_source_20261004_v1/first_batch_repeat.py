#!/usr/bin/env python3
"""Exactly four zero-update native first-batch prefixes in failed shared4 state."""
import argparse
import ast
import json
import time
from pathlib import Path
from common import (HERE, PHASE, EXECUTION, require, sha, utc, gate, runtime,
                    load_module, Progress, monitor, inventory)
from diagnostic_helpers import write, differences, step_aliases, tensor_bytes
from prefix_observer import prefix, full_state


OWNED = ('encoder', 'predictor', 'Adam', 'gradients', 'flags', 'invest', 'unregistered_module_state')


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True, type=Path)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    release, plan = gate(args.root_release, args.release_sha256, 'first_batch_repeat')
    output = EXECUTION / 'first_batch_repeat/run01'
    require(not output.exists(), 'Fresh sole zero-update diagnostic required')
    output.mkdir(parents=True)
    progress = Progress(output)
    progress.update(prefixes_started=0, prefixes_completed=0, forbidden_optimizer_step_calls=0,
                    state_file_loads=0, feature_weights_only_loads=0, retained_tensor_payload_peak_bytes=0)
    stopped = watcher = sample = None
    guards, evidence, frames, retained = [], {}, {}, {}
    try:
        native, loaded, bridge, prototype, graph_ops = runtime(plan)
        torch, np, bodies, x, train, inspection, identities = loaded
        progress.add(feature_weights_only_loads=1, weights_only_loads=1)
        stopped, watcher, sample = monitor(output, plan['caps'], progress, torch)
        observed = load_module('pubmed_zero_update_native_observer', PHASE / plan['observer_source_path'])
        write(output / 'TRAIN_INSPECTION.json', inspection)
        authority = plan['state_authority']
        identity_path, state_path = PHASE / authority['identity_path'], PHASE / authority['state_path']
        identity = json.loads(identity_path.read_text())
        require(sha(identity_path) == release['owned_serialized_identity_sha256']
                and all(identity['input_files'][name] == row for name, row in identities.items()), 'Failed-state input/identity differs')
        require(not state_path.is_symlink() and sha(state_path) == identity['state']['sha256']
                and state_path.stat().st_size == identity['state']['bytes'], 'Owned engineering state bytes differ')
        saved = torch.load(state_path, map_location='cpu', weights_only=True)
        progress.add(state_file_loads=1, weights_only_loads=1)
        for key in ('schema', 'mode', 'seed', 'selected_epoch', 'source_manifest_sha256', 'root_release_sha256',
                    'input_files', 'engineering_only', 'scientific_donor_allowed', 'factory_initialization'):
            require(saved[key] == identity[key], 'Owned engineering metadata differs: ' + key)
        require(saved['scores_sha256'] == identity['scores']['sha256'], 'Owned score hash metadata differs; scores are not opened')
        pristine = native.clone(saved['after_VALID'], torch)
        saved_digest = observed.state_identity(saved, torch)
        units = [bridge.make_shared4(PHASE, bodies, prototype, graph_ops, 0, 'private', x, train) for _ in range(2)]
        progress.add(factory_calls=2)
        profile = plan['execution_profile']
        require(torch.backends.cudnn.deterministic == profile['cudnn_deterministic']
                and torch.backends.cudnn.benchmark == profile['cudnn_benchmark'], 'Factory changed qualified cuDNN profile')
        def forbid_step(optimizer, args, kwargs):
            progress.add(forbidden_optimizer_step_calls=1)
            raise RuntimeError('Optimizer.step call forbidden; stop must occur after backward before this call')
        guards = [unit[2].register_step_pre_hook(forbid_step) for unit in units]
        retained.update(saved=saved, pristine=pristine)
        def bound(scratch=None):
            count = tensor_bytes(retained, torch) + tensor_bytes(frames, torch) + tensor_bytes(scratch, torch)
            require(count <= plan['limits']['retained_tensor_payload_bytes'], 'Retained tensor payload cap exceeded')
            progress.update(retained_tensor_payload_peak_bytes=max(count, progress.snapshot()['retained_tensor_payload_peak_bytes']))
        tolerance = plan['engineering_tolerance']
        source = PHASE / plan['comparator_source_path']
        definitions = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'compare']
        require(len(definitions) == 1, 'Original fixed comparator missing')
        scope = {'require': require}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), scope)
        compare = scope['compare']
        evidence.update(initial_restores={}, prefixes={}, frozen_comparisons=plan['frozen_comparisons'],
                        no_cache_or_model_Adam_tensor_reload_between_repeats=True, saved_tree_sha256_before=saved_digest)
        for unit_index, unit in enumerate(units):
            name = 'A' if unit_index == 0 else 'B'
            restore_input = native.clone(saved['after_VALID'], torch)
            retained[name + '_restore_input'] = restore_input
            native.restore(unit, restore_input, torch, np)
            initial = native.capture(unit, torch, np)
            retained[name + '_initial_native_state'] = initial
            evidence['initial_restores'][name] = {
                'exact_native_checkpoint_state': differences(initial, pristine, torch, 0, 0),
                'CPU_Adam_steps_to_saved': step_aliases(unit, saved['after_VALID'], torch),
                'CPU_Adam_steps_to_own_clone': step_aliases(unit, restore_input, torch)}
            if unit_index:
                evidence['cross_unit_CPU_Adam_steps'] = step_aliases(unit, {'Adam': units[0][2].state_dict()}, torch)
            for repeat in (1, 2):
                label = name + str(repeat)
                progress.update(phase=label, current_arm='private', current_epoch=6)
                # No parameter/Adam/buffer/cache reload here. Original body
                # establishes train flags and gradient None before encoder.
                native.restore_rng(pristine['RNG'], torch, np, True)
                idle_before = full_state(units[1 - unit_index], native, torch, np)
                channels, prestate, stream = prefix(native, bodies, observed, unit, train, torch, np, progress)
                after = full_state(unit, native, torch, np)
                idle_after = full_state(units[1 - unit_index], native, torch, np)
                idle = differences({k: idle_after[k] for k in OWNED}, {k: idle_before[k] for k in OWNED}, torch, 0, 0)
                mutation = differences({k: after[k] for k in OWNED if k != 'gradients'},
                                       {k: prestate[k] for k in OWNED if k != 'gradients'}, torch, 0, 0)
                saved_unchanged = observed.state_identity(saved, torch) == saved_digest
                none_grads = all(value is None for root in prestate['gradients'] for value in root.values())
                train_flags = all(value is True for root in prestate['flags'] for value in root.values())
                frames[label] = {'channels': channels, 'prestate': prestate, 'stream': stream}
                evidence['prefixes'][label] = {
                    'pre_forward_state_sha256': observed.state_identity(prestate, torch),
                    'pre_forward_gradients_None': none_grads, 'pre_forward_all_training_flags_true': train_flags,
                    'unregistered_module_state_fully_inspected': prestate['unregistered_module_state']['fully_inspected'],
                    'unsupported_module_state': prestate['unregistered_module_state']['unsupported'],
                    'post_forward_backward_owned_state_mutation_excluding_gradients': mutation,
                    'idle_unit_owned_state': idle, 'saved_tree_unchanged': saved_unchanged,
                    'CPU_Adam_steps_to_saved': step_aliases(unit, saved['after_VALID'], torch),
                    'CPU_Adam_steps_to_own_clone': step_aliases(unit, restore_input, torch),
                    'native_stream': stream,
                    'channel_identities': {k: observed.state_identity(v, torch) for k, v in channels.items()}}
                bound({'idle_before': idle_before, 'idle_after': idle_after, 'active_after': after})
                write(output / 'REPLAY_PROGRESS.json', evidence)
        records = []
        for pair in plan['frozen_comparisons']:
            candidate, reference = frames[pair['candidate']], frames[pair['reference']]
            state = differences(candidate['prestate'], reference['prestate'], torch, 0, 0)
            streams_exact = candidate['stream'] == reference['stream']
            operands = [evidence['prefixes'][pair[key]] for key in ('candidate', 'reference')]
            initial_restores_exact = all(row['exact_native_checkpoint_state']['exact'] for row in evidence['initial_restores'].values())
            saved_and_idle_exact = all(row['saved_tree_unchanged'] and row['idle_unit_owned_state']['exact'] for row in operands)
            alias_isolation_exact = all(not row['same_storage_pointer'] and not row['same_tensor_object']
                                        and row['live_step'] == row['saved_step']
                                        for operand in operands for row in operand['CPU_Adam_steps_to_saved'])
            alias_isolation_exact = alias_isolation_exact and all(not row['same_storage_pointer'] and not row['same_tensor_object']
                                        and row['live_step'] == row['saved_step']
                                        for row in evidence['cross_unit_CPU_Adam_steps'])
            alias_isolation_exact = alias_isolation_exact and all(row['same_storage_pointer'] and row['same_tensor_object']
                                        and row['live_step'] == row['saved_step']
                                        for operand in operands for row in operand['CPU_Adam_steps_to_own_clone'])
            valid = (state['exact'] and streams_exact and initial_restores_exact and saved_and_idle_exact and alias_isolation_exact
                     and all(row['pre_forward_gradients_None'] and row['pre_forward_all_training_flags_true']
                             and row['unregistered_module_state_fully_inspected'] for row in operands))
            record = dict(pair, complete_pre_forward_state=state, native_streams_exact=streams_exact,
                          initial_restores_exact=initial_restores_exact, saved_and_idle_exact=saved_and_idle_exact,
                          saved_and_cross_unit_step_alias_isolation_exact=alias_isolation_exact,
                          state_matching_valid=valid, numerical_comparison_collected=valid,
                          status='MATCHED_COMPARISON' if valid else 'STATE_CONTROL_FAILURE_ISOLATED',
                          original_fixed_comparator_reports=[], original_fixed_comparator_verdicts=[])
            if valid:
                record['channels'] = differences(candidate['channels'], reference['channels'], torch,
                                                 tolerance['atol'], tolerance['rtol'])
                for channel in candidate['channels']:
                    verdict = {'channel': channel, 'passed_original_fixed_predicate': True}
                    try:
                        compare(candidate['channels'][channel], reference['channels'][channel], torch,
                                tolerance['atol'], tolerance['rtol'], pair['label'] + '/' + channel,
                                record['original_fixed_comparator_reports'])
                    except Exception as error:
                        verdict.update(passed_original_fixed_predicate=False, type=type(error).__name__, condition=str(error))
                    record['original_fixed_comparator_verdicts'].append(verdict)
            else:
                record['interpretation'] = 'Frozen pair isolated before numerical comparison: no complete same-state claim or causal attribution.'
            records.append(record)
        evidence['comparisons'] = records
        evidence['all_four_state_matching_valid'] = all(row['state_matching_valid'] for row in records)
        evidence['saved_tree_sha256_after'] = observed.state_identity(saved, torch)
        evidence['saved_tree_unchanged'] = evidence['saved_tree_sha256_after'] == saved_digest
        final = progress.snapshot()
        require(final['prefixes_started'] == final['prefixes_completed'] == final['gradient_backward_calls'] == 4
                and final['encoder_started'] == final['encoder_completed'] == 4
                and final['root_decoder_started'] == final['root_decoder_completed'] == 8
                and final['Adam_started'] == final['Adam_completed'] == final['forbidden_optimizer_step_calls'] == 0
                and final['factory_calls'] == 2 and final['state_file_loads'] == final['feature_weights_only_loads'] == 1
                and final['weights_only_loads'] == 2 and final['VALID_started'] == final['VALID_completed'] == 0
                and final['native_epochs_started'] == final['native_epochs_completed'] == final['serializations'] == 0,
                'Exact frozen zero-update work differs')
        torch.cuda.synchronize(0)
        stopped.set(); watcher.join(); sample()
        write(output / 'DIAGNOSTIC.json', {'schema': 'pubmed-shared4-zero-update-first-batch-repeat-v1', 'status': 'COMPLETE',
              'UTC': utc(), 'diagnostic_evidence': evidence, 'progress': final,
              'source_manifest_sha256': release['source_manifest_sha256'], 'root_release_sha256': args.release_sha256,
              'owned_engineering_state_sha256': identity['state']['sha256'], 'owned_serialized_identity_sha256': sha(identity_path),
              'inclusive_child_wall_seconds': time.monotonic() - started,
              'engineering_qualification_PASS': False, 'scientific_fit_admitted': False, 'state_donor_allowed': False,
              'VALID_files_opened': False, 'TEST_files_opened': False, 'score_files_opened': False,
              'numeric_rule_changed': False, 'restore_helper_changed': False, 'kernel_causality_established': False,
              'interpretation': plan['interpretation']})
    except BaseException as error:
        write(output / 'FAILURE.json', {'status': 'FAILED', 'type': type(error).__name__, 'condition': str(error),
              'progress': progress.snapshot(), 'diagnostic_evidence': evidence, 'automatic_retry': False})
        raise
    finally:
        for handle in guards:
            handle.remove()
        if stopped is not None:
            stopped.set(); watcher.join()
        write(output / 'FINAL_CUSTODY.json', {'files': inventory(output), 'stage': 'first_batch_repeat',
              'completed': (output / 'DIAGNOSTIC.json').exists()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
