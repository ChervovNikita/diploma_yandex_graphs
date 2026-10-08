"""Root-authorized four-update, full-input engineering qualification; no scores."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import resource
import signal
import socket
import subprocess
import sys
import time

P = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
R = P.parents[1]
SOURCE = P / 'query_conditioned_value_gate_full15_callable_source_20261009_v1'
OUT = P / 'query_value_gate_complete_input_qualification_execution_root_20261009_v1'
SOURCE_MANIFEST_SHA256 = '67a6f621b23f77761a96954119bade6d3646021ebde4a0a89c6c74b76df8ced0'
ARMS = ('C4_gate', 'S_joint4head_gate', 'U4_sameB_full_untied_gate', 'S_one_path_gate', 'C4_gate_identity_erased')
TOLERANCE = 2e-6


def require(value, message):
    if not value:
        raise ValueError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorized', action='store_true')
    args = parser.parse_args()
    require(args.authorized, 'Separate root engineering qualification authorization required')
    require(socket.gethostname() == 'anogena-2-0' and Path.cwd() == R, 'Normal original host/repository only')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
            text=True, timeout=5).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Sole original GPU UUID')
    require(hashlib.sha256((SOURCE / 'MANIFEST.json').read_bytes()).hexdigest() == SOURCE_MANIFEST_SHA256,
            'Exact sealed full15 source')
    OUT.mkdir(exist_ok=False)
    began = time.monotonic()
    result = dict(complete=False, scientific_fit=False, native_training_updates=0,
        development_scores_computed=False, development_truth_consumed_by_qualifier=False, TEST_access=False,
        discarded_updates=0, new_native_captures=0, source_manifest_sha256=SOURCE_MANIFEST_SHA256,
        arms=list(ARMS), reconstruction_tolerance=TOLERANCE, automatic_retry=False)
    stage = cached = torch = None
    hooks = []
    restorations = []
    phase = ['setup']
    observed = dict(permanent_erasure_training_calls=0, permanent_erasure_serving_calls=0,
                    gate_zero_message_rows_training=0, gate_zero_message_rows_serving=0)
    timing = {}
    signal.signal(signal.SIGALRM, lambda number, frame: (_ for _ in ()).throw(TimeoutError('Qualification180s budget')))
    signal.setitimer(signal.ITIMER_REAL, 180)
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(.2, 0)
        torch.cuda.reset_peak_memory_stats(0)
        mark = time.monotonic()
        m = load_module(SOURCE / 'stage.py', '_root_query_value_gate_qualification')
        require(m.verify_packet() == SOURCE_MANIFEST_SHA256 and m.ARMS == ARMS, 'Exact sealed five-arm roster')
        train, valid, origin = m.load_roles(later_execution_authorized=True)
        ids = valid['ids']
        del valid  # No development truth is indexed, scored or passed to a qualification operation.
        timing['source_route_role_loading_seconds'] = time.monotonic() - mark
        mark = time.monotonic()
        stage = m.make_stage(train_data=train, origin=origin, seed=6101,
                             later_execution_authorized=True, purpose='engineering_qualification')
        timing['full_input_stage_constructor_seconds'] = time.monotonic() - mark
        require(tuple(stage.banks) == ARMS, 'Every required bank constructed')
        result['new_native_captures'] = stage.work['native_capture_calls']
        result['role_shapes'] = {k: list(v.shape) for k, v in train.items() if torch.is_tensor(v)}
        result['development_id_shape'] = list(ids.shape)
        result['constructor_parameter_counts'] = {a: sum(p.numel() for p in bank.parameters()) for a, bank in stage.banks.items()}
        require(result['role_shapes'] == dict(x=[11701,300], edge_index=[2,442907], ids=[580], y=[580])
                and list(ids.shape) == [5274], 'Actual complete WikiCS TRAIN and development IDs')

        def gate_items(obj):
            values = {}
            for arm, bank in obj.banks.items():
                if arm == 'U4_sameB_full_untied_gate':
                    values.update({arm + '/' + str(i): route['query_value_gate'] for i, route in enumerate(bank.routes)})
                else:
                    values[arm] = bank.query_value_gate
            require(len(values) == 8, 'One shared gate for each shared/single bank; four private U4 gates')
            return values

        gates = gate_items(stage)
        helper = load_module(SOURCE / 'query_value_gate_hook.py', '_qualification_identity_gate_helper')
        rng = (torch.get_rng_state().clone(), [state.clone() for state in torch.cuda.get_rng_state_all()])
        identity_gate = helper.make_query_value_gate(torch=torch, later_execution_authorized=True)
        require(torch.equal(torch.get_rng_state(), rng[0])
                and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(), rng[1])),
                'Zero-initialized gate constructor consumes no CPU/CUDA RNG')
        del identity_gate
        masks_before = {a: bank.mask_generator.get_state().clone() for a, bank in stage.banks.items()}
        with torch.no_grad():
            for gate in gates.values():
                require(torch.equal(gate.A, torch.zeros_like(gate.A)) and torch.equal(gate.b, torch.zeros_like(gate.b)),
                        'Every actual bank gate starts at A=b=0')
                message = stage.H[:8, :64]
                require(torch.equal(gate(message, stage.H[:8]), message), 'Exact initial gate identity')
        require(torch.equal(torch.get_rng_state(), rng[0])
                and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(), rng[1]))
                and all(torch.equal(bank.mask_generator.get_state(), masks_before[a]) for a,bank in stage.banks.items()),
                'Initial identity check preserves ambient and live-mask RNG')

        def ownership(obj, require_moments=False):
            bank = obj.banks['U4_sameB_full_untied_gate']
            require(len(bank.routes) == len(bank.optimizers) == 4 and len({id(x) for x in bank.optimizers}) == 4,
                    'Four independent U4 Adam instances')
            all_parameters = []
            moments = []
            for i, route in enumerate(bank.routes):
                optimizer = bank.optimizers[i]
                owned = [p for group in optimizer.param_groups for p in group['params']]
                require(len(owned) == len({id(p) for p in owned})
                        and {id(p) for p in owned} == {id(p) for p in route.parameters()}, 'Exact private complete route/gate owner')
                for parameter in route['query_value_gate'].parameters():
                    all_parameters.append(parameter)
                    require(sum(any(parameter is p for group in other.param_groups for p in group['params'])
                            for other in bank.optimizers) == 1, 'Each U4 gate coordinate has exactly its own Adam')
                    if require_moments:
                        state = optimizer.state[parameter]
                        require('exp_avg' in state and 'exp_avg_sq' in state, 'Actual private gate Adam moments exist')
                        moments.extend((state['exp_avg'].data_ptr(), state['exp_avg_sq'].data_ptr()))
            require(len({id(p) for p in all_parameters}) == len(all_parameters)
                    and len({p.data_ptr() for p in all_parameters}) == len(all_parameters), 'Four U4 gates have disjoint parameters/storage')
            require(not require_moments or len(set(moments)) == len(moments), 'Private U4 gate Adam moments do not alias')

        def observe(obj):
            erased = obj.banks['C4_gate_identity_erased']
            require(erased.erase_value_class_identity is True and erased._fixed_value_erasure_policy is True,
                    'Construction-bound permanent erasure')
            function = erased._route_delta.__func__
            closure = dict(zip(function.__code__.co_freevars, (cell.cell_contents for cell in function.__closure__)))
            erase_helper = closure['gates']
            original = erase_helper.lookup_permitted_label_embedding
            def lookup(embedding, **kwargs):
                require(embedding is erased.label_embedding and kwargs['erase_class_identity'] is True
                        and kwargs['visible_labels'] is None, 'Actual erased value lookup never receives class identities')
                output = original(embedding, **kwargs)
                require(torch.equal(output, embedding.weight.mean(0, keepdim=True).expand(kwargs['visible_count'], -1)),
                        'Actual values equal the fixed uniform-code/row-mean lookup')
                key = 'permanent_erasure_training_calls' if phase[0] == 'training' else 'permanent_erasure_serving_calls'
                observed[key] += 1
                return output
            erase_helper.lookup_permitted_label_embedding = lookup
            restorations.append((erase_helper, original))
            def gate_observer(gate, inputs, output):
                message, query_H = inputs
                require(not query_H.requires_grad, 'Actual target H remains detached')
                zero = (message == 0).all(-1)
                require(torch.equal(output[zero], torch.zeros_like(output[zero])), 'Actual zero messages stay exactly zero through the learned gate')
                key = 'gate_zero_message_rows_training' if phase[0] == 'training' else 'gate_zero_message_rows_serving'
                observed[key] += int(zero.sum())
            for gate in gate_items(obj).values():
                hooks.append(gate.register_forward_hook(gate_observer))

        def serve(obj):
            phase[0] = 'serving'
            predictions = obj.serve_ids(ids)
            native = torch.softmax(obj.base[ids.to(obj.native.device)], -1)
            empty = ~obj.reach[ids.to(obj.native.device)]
            require(bool(empty.any()), 'Full roles contain actual no-anchor serving rows')
            for value in predictions.values():
                require(torch.equal(value['served_probabilities'][empty], native[empty])
                        and torch.equal(value['label_member_logits'][:,empty], torch.zeros_like(value['label_member_logits'][:,empty])),
                        'Every actual bank has exact zero no-anchor correction and native fallback')
            obj.fixed()
            return predictions

        observe(stage)
        ownership(stage)
        gradient_rows = []
        mark = time.monotonic()
        for epoch in (1,2,3,4):
            if epoch == 3:
                live = {a:(bank.mask_generator.get_state().clone(),bank.steps,copy.deepcopy(bank.counters))
                        for a,bank in stage.banks.items()}
                live_work = copy.deepcopy(stage.work)
                stage.restore_learned(first_snapshot)
                require(stage.label_epoch == 2 and stage.parameter_label_epoch == 1
                        and stage.work == dict(live_work, learned_restores=live_work['learned_restores']+1), 'Restore preserves completed stage work')
                for a,bank in stage.banks.items():
                    require(torch.equal(bank.mask_generator.get_state(),live[a][0]) and bank.steps == live[a][1]
                            and bank.counters == live[a][2], 'Restore preserves each live mask/step/work stream')
                restored = serve(stage)
                restore_diff = {a:float((restored[a]['served_probabilities']-first[a]).abs().max()) for a in ARMS}
                require(max(restore_diff.values()) < TOLERANCE, 'Learned gate/Adam restore serving within practical tolerance')
            before = {key:tuple(p.detach().clone() for p in gate.parameters()) for key,gate in gates.items()}
            masks = stage.draw_common_masks()
            mask = masks['C4_gate']
            require(len(mask.query_positions) == 290 and all((x.query_positions,x.query_ids,x.draw_id) ==
                    (mask.query_positions,mask.query_ids,mask.draw_id) for x in masks.values()), 'Actual identical common290-query masks')
            phase[0] = 'training'
            stage.train_step(label_epoch=epoch, masks=masks)
            result['discarded_updates'] += 1
            if epoch >= 2:
                for key,gate in gates.items():
                    grads = [float(p.grad.abs().max()) if p.grad is not None else 0. for p in gate.parameters()]
                    change = max(float((p.detach()-old).abs().max()) for p,old in zip(gate.parameters(),before[key]))
                    require(all(value > 0 for value in grads) and change > 0, 'Actual common-mask loss gives nonzero A/b gradients and gate parameter change')
                    gradient_rows.append(dict(label_epoch=epoch,gate=key,A_gradient_max_abs=grads[0],b_gradient_max_abs=grads[1],parameter_change_max_abs=change))
            ownership(stage, require_moments=True)
            predictions = serve(stage)
            if epoch == 1:
                first = {a:v['served_probabilities'].detach().clone() for a,v in predictions.items()}
                first_snapshot = stage.snapshot()
            if epoch == 3:
                after3 = {a:v['served_probabilities'].detach().clone() for a,v in predictions.items()}
                snapshot3 = stage.snapshot()
                cache = stage.capture_state()
                io_mark = time.monotonic()
                torch.save(cache, OUT/'QUALIFICATION_CAPTURE.pt')
                torch.save(snapshot3, OUT/'QUALIFICATION_STATE.pt')
                cache = torch.load(OUT/'QUALIFICATION_CAPTURE.pt',map_location='cpu',weights_only=False)
                snapshot3 = torch.load(OUT/'QUALIFICATION_STATE.pt',map_location='cpu',weights_only=False)
                cached = m.make_stage(train_data=train,origin=origin,seed=6101,later_execution_authorized=True,
                                      purpose='engineering_qualification',capture_cache=cache)
                observe(cached)
                require(cached.work['native_capture_calls'] == 0, 'Cached reconstruction performs no further native forward/capture')
                cached.restore_learned(snapshot3)
                ownership(cached, require_moments=True)
                reconstructed = serve(cached)
                cache_diff = {a:float((reconstructed[a]['served_probabilities']-after3[a]).abs().max()) for a in ARMS}
                require(max(cache_diff.values()) < TOLERANCE, 'Cached five-bank gate/Adam serving within practical tolerance')
                timing['engineering_state_cache_save_load_reconstruction_seconds'] = time.monotonic()-io_mark
        timing['four_updates_observers_serving_restore_and_cache_seconds'] = time.monotonic()-mark
        stage.fixed()
        cached.fixed()
        require(stage.work['all_bank_update_attempts'] == stage.work['all_bank_update_completions']
                == stage.work['common_Q_checks'] == 4 and stage.work['complete_VALID_events'] == stage.work['native_updates'] == 0
                and stage.work['native_capture_calls'] == 1 and result['discarded_updates'] == 4,
                'Exactly four discarded complete-five-bank updates, one native capture, no scoring/native updates')
        require(observed['permanent_erasure_training_calls'] == 16 and observed['permanent_erasure_serving_calls'] > 0
                and observed['gate_zero_message_rows_training'] > 0 and observed['gate_zero_message_rows_serving'] > 0,
                'Actual training and serving erasure/zero-message observations completed')
        result.update(complete=True, initial_identity_gate_and_no_RNG_consumption=True,
            actual_gate_gradient_and_change_checks=gradient_rows, four_U4_private_gate_Adam_owners_checked=True,
            permanent_erasure_and_exact_zero_message_checks=observed, exact_native_fallback_checked=True,
            common_query_and_frozen_native_guards_passed=True, live_masks_and_work_retained_on_restore=True,
            cache_reconstructed_without_native_forward=True, restore_probability_max_abs=restore_diff,
            cache_probability_max_abs=cache_diff, constructor_setup=stage.run['new_setup_timings_seconds'],
            native_capture_identity=stage.capture_identity, work=stage.work,
            bank_counts={a:dict(bank.counters) for a,bank in stage.banks.items()},
            selected_serving_and_gate_validation_deferred_until_all15_science_closed=True)
    except BaseException as error:
        result['error'] = dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for hook in hooks:
            hook.remove()
        for helper, original in restorations:
            helper.lookup_permitted_label_embedding = original
        for obj in (stage,cached):
            if obj is not None:
                obj.close()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(seconds=time.monotonic()-began, timings_seconds=timing,
            CPU_user_seconds=usage.ru_utime, CPU_system_seconds=usage.ru_stime, peak_RSS_bytes=usage.ru_maxrss*1024,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(0) if torch is not None and torch.cuda.is_initialized() else None,
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(0) if torch is not None and torch.cuda.is_initialized() else None,
            qualification_states_and_costs_preserved=True, quality_or_promotion_claim=False,
            partial_work=stage.work if stage is not None else None,
            partial_bank_counts={a:dict(bank.counters) for a,bank in stage.banks.items()} if stage is not None else None)
        (OUT/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=result['complete'],seconds=result['seconds'],discarded_updates=result['discarded_updates'],
                         source_manifest_sha256=SOURCE_MANIFEST_SHA256)))


if __name__ == '__main__':
    main()
