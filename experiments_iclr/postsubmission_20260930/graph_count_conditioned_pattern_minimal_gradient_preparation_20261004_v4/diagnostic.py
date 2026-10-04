"""One TRAIN-only native forward graph and three same-state reverse evaluations."""
import argparse
import ast
import codecs
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from common import (DATA_FILES, HERE, PHASE, final_file_custody, gate, load_module,
                    module_custody, pin, require, sha, write)


def load_train(authority, device, tensor_sha):
    # Narrow adaptation of qualified pilot_data.load_data: exactly TRAIN/raw.
    import numpy as np
    import pandas as pd
    import torch
    root = Path(authority['dataset_root']).resolve()
    require(root.is_relative_to(PHASE), 'TRAIN dataset outside project')
    for name in DATA_FILES:
        path = root/name
        require(path.resolve().is_relative_to(root), 'TRAIN file outside dataset')
        pin(path, authority['files'][name])
    allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.int64)), codecs.encode]
    with torch.serialization.safe_globals(allowed):
        stored = torch.load(root/'split/time/train.pt', map_location='cpu', weights_only=True)
    specs = authority['expected_arrays']['train']
    require(type(stored) is dict and set(stored) == set(specs) == {'edge','weight','year'}, 'Complete TRAIN schema differs')
    for name, spec in specs.items():
        value = stored[name]
        require(type(value) is np.ndarray and str(value.dtype) == spec['dtype'] and list(value.shape) == spec['shape']
                and tensor_sha(torch.from_numpy(value)) == spec['sha256'], 'TRAIN array identity differs')
    pairs = torch.from_numpy(stored['edge'])
    require(pairs.shape == (1179052, 2) and pairs.dtype == torch.long and int(stored['year'].max()) == 2017,
            'Full native TRAIN record contract differs')
    x = torch.from_numpy(pd.read_csv(root/'raw/node-feat.csv.gz', compression='gzip', header=None).values.astype(np.float32))
    raw = pd.read_csv(root/'raw/edge.csv.gz', compression='gzip', header=None).values.T.astype(np.int64)
    expanded = np.repeat(raw, 2, axis=1)
    expanded[0,1::2] = expanded[1,0::2]; expanded[1,1::2] = expanded[0,0::2]
    edges = torch.from_numpy(expanded)
    require(x.shape == (235868,128) and edges.shape == (2,2358104) and bool(torch.isfinite(x).all()), 'Features/raw geometry differs')
    digests = dict(train_records=tensor_sha(pairs), raw_features=tensor_sha(x), ordered_raw_graph=tensor_sha(edges))
    require(digests == authority['train_raw_tensor_digests'], 'Full TRAIN/raw tensor identity differs')
    for values in (pairs, edges):
        require(int(values.min()) >= 0 and int(values.max()) < len(x), 'TRAIN/raw endpoint out of range')
    def keys(values):
        lo, hi = values.min(1).values, values.max(1).values
        return (lo*len(x)+hi).sort().values
    require(torch.equal(keys(torch.from_numpy(raw.T.copy())), keys(pairs)), 'Raw/TRAIN record multiset differs')
    return root, dict(x=x.to(device), pairs=pairs.to(device), raw_edge_index=edges.to(device), digests=digests)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    plan, runtime, execution = gate(args.release, args.release_sha256)
    release = json.loads(args.release.read_text())
    output = execution/'run01'
    require(not output.exists(), 'One fresh diagnostic only; no restart')
    output.mkdir(parents=True)
    data_root = None; finished = False; monitor_stop = threading.Event(); monitor = None
    resources = dict(sampled_CUDA_allocated_peak_bytes=0, sampled_CUDA_reserved_peak_bytes=0, samples=0)
    resource_lock = threading.Lock()
    try:
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        paths = {row['module']: PHASE/row['path'] for row in plan['imported_modules']}
        for name in ('pilot_common','pilot_data','pilot_state','pilot_model'):
            load_module(name, paths[name])
        import pilot_data, pilot_state, pilot_model
        authority = json.loads((PHASE/plan['data_authority']).read_text())
        context = dict(release=release, runtime=runtime, authority=authority)
        device, sampler = pilot_model.ordinary_runtime(context)
        require(torch.cuda.is_initialized() and torch.cuda.current_device() == 0,
                "Authenticated CUDA context must precede memory-stat reset")
        torch.cuda.reset_peak_memory_stats(0)
        before = pilot_model.runtime_settings(); before_rng = pilot_state.rng_digest(pilot_state.rng_state())
        require(before['deterministic_algorithms'] is False and before['deterministic_warn_only'] is False
                and before['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8' and before['autocast_CPU'] is False,
                'Qualified pre-transition profile differs')
        torch.use_deterministic_algorithms(True, warn_only=False)
        profile = pilot_model.runtime_settings(); after_rng = pilot_state.rng_digest(pilot_state.rng_state())
        require(profile == dict(before, deterministic_algorithms=True, deterministic_warn_only=False)
                and after_rng == before_rng, 'Qualified explicit transition/profile/RNG differs')
        transition = dict(status='TRANSITION_VERIFIED', ordinary_runtime_authority_sha256=sha(PHASE/plan['runtime_authority']),
                          actual_profile_before=before, actual_profile_after=profile, RNG_exactly_unchanged=True)
        def checkpoint_resource(stage):
            with resource_lock:
                allocated = torch.cuda.max_memory_allocated(0); reserved = torch.cuda.max_memory_reserved(0)
                resources['sampled_CUDA_allocated_peak_bytes'] = max(resources['sampled_CUDA_allocated_peak_bytes'], allocated)
                resources['sampled_CUDA_reserved_peak_bytes'] = max(resources['sampled_CUDA_reserved_peak_bytes'], reserved)
                resources['samples'] += 1
                require(allocated <= plan['caps']['cuda_allocated_bytes'] and reserved <= plan['caps']['cuda_reserved_bytes'], 'Observed CUDA cap exceeded')
                require(time.monotonic()-started <= plan['caps']['wall_seconds'], 'Inclusive child wall exceeded')
                write(output/'PROGRESS.json', dict(stage=stage, elapsed_seconds=time.monotonic()-started, **resources))
        def sampled_cuda():
            while not monitor_stop.wait(.25):
                try:
                    checkpoint_resource('sampled_resource_observation')
                except BaseException as error:
                    write(output/'RESOURCE_STOP.json', dict(type=type(error).__name__, condition=str(error), **resources))
                    os._exit(2)
        monitor = threading.Thread(target=sampled_cuda, name='owned_cuda_observer', daemon=True)
        monitor.start()
        # Original qualified imports only; no fit/qualification/serving helpers called.
        for name in ('graph_ops','prototype','pattern_model','pattern_teacher','conditional_loss','_diagnostic_design','_census_metadata'):
            load_module(name, paths[name])
        import graph_ops, pattern_model, pattern_teacher, conditional_loss as core
        from metrics import magnitudes, norms, parameter_reports, slot_reports
        design = sys.modules['_diagnostic_design']; census = sys.modules['_census_metadata']
        tensor_sha = pilot_data.tensor_sha
        data_root, data = load_train(authority, device, tensor_sha)
        times = dict(setup_and_packing=time.monotonic()-started)
        rng = {}
        def snapshot_rng():
            value = pilot_state.rng_state()
            return dict(total_sha256=pilot_state.rng_digest(value),
                        component_sha256={key:pilot_state.state_digest(item) for key,item in value.items()})
        # Exact qualified make_pattern initialization order; omit its optimizer construction.
        pilot_model.seed_all(plan['seed'])
        rng['before_factory'] = snapshot_rng()
        model = pattern_model.PatternTwin().to(device)
        sign_seed = design.factor_sign_seed(plan['seed'])
        generator = torch.Generator(device='cpu').manual_seed(sign_seed)
        sign_rng_before = tensor_sha(generator.get_state())
        with torch.no_grad():
            for name, parameter in sorted(model.named_parameters()):
                if name.endswith('.r') or name.endswith('.s'):
                    signs = 2*torch.randint(0,2,parameter.shape,dtype=torch.int64,generator=generator,device='cpu')-1
                    parameter.copy_(signs.to(device=device,dtype=parameter.dtype))
        require(model.recipe.member_count == 4 and sum(p.numel() for p in model.parameters()) == 43790, 'Qualified F4 family differs')
        sign_rng_after = tensor_sha(generator.get_state())
        model.train(); named = list(model.named_parameters()); parameters = tuple(p for _,p in named)
        initial_flags = pilot_state.flags(model)
        initial_state = pilot_state.state_digest(model.state_dict()); rng['after_factory'] = snapshot_rng()
        teacher = pattern_teacher.ObservationTeacher.from_train(data['pairs'], len(data['x']))
        # Exact native PermIterator definition, as in the successful census source.
        iterator_path = PHASE/plan['iterator_source']
        definitions = [node for node in ast.parse(iterator_path.read_text()).body if isinstance(node,ast.ClassDef) and node.name == 'PermIterator']
        require(len(definitions) == 1, 'Exact native iterator absent')
        scope = dict(torch=torch)
        exec(compile(ast.Module(body=definitions,type_ignores=[]),str(iterator_path),'exec'),scope)
        mods = dict(native_utils=SimpleNamespace(PermIterator=scope['PermIterator']))
        negative_pairs, iterator, stream = pilot_data.epoch_stream(data, mods, sampler)
        rng['after_sampler_and_iterator'] = snapshot_rng()
        record_ids = next(iter(iterator))
        require(record_ids.shape == (65536,), 'Predeclared first full native batch differs')
        graph = graph_ops.Graph.mask_train_batch(data['pairs'], record_ids, len(data['x']))
        stream.update(record_ids_sha256=tensor_sha(record_ids), masked_graph_sha256=tensor_sha(torch.stack((graph.row,graph.col))),
                      mask_rowptr_sha256=tensor_sha(graph.rowptr), zero_based_batch=0, seed=plan['seed'], sign_seed=sign_seed,
                      factor_sign_generator_before_sha256=sign_rng_before, factor_sign_generator_after_sha256=sign_rng_after,
                      exact_replay_of_census_or_fit_claimed=False)
        checkpoint_resource('setup_and_mask_complete')
        times['setup_and_packing'] = time.monotonic()-started
        # Count actual qualified calls, including checkpoint recomputation.
        dispatch = dict(encoder_calls=0, population_forward_calls=0, depth_zero_calls={},
                        conditional_side_calls=0, genuine_ESP_group_calls=0,
                        genuine_ESP_group_slot_loops=0, reverse_evaluations=0)
        phase = ['forward']
        original_depth_zero = model.decoder.depth_zero
        def depth_zero(*values, **keywords):
            key = phase[0]; dispatch['depth_zero_calls'][key] = dispatch['depth_zero_calls'].get(key,0)+1
            return original_depth_zero(*values, **keywords)
        model.decoder.depth_zero = depth_zero
        original_esp = core._group_log_esp_reachable
        def esp(centered, degree):
            dispatch['genuine_ESP_group_calls'] += 1
            dispatch['genuine_ESP_group_slot_loops'] += centered.shape[-1]
            return original_esp(centered, degree)
        core._group_log_esp_reachable = esp
        original_side = core._grouped_side_nll
        def conditional_side(*values):
            dispatch['conditional_side_calls'] += 1
            return original_side(*values)
        core._grouped_side_nll = conditional_side
        forward_started = time.monotonic(); rng['before_forward'] = snapshot_rng()
        h = model.encoder(data['x'], graph); dispatch['encoder_calls'] += 1
        records = []
        for population, query in (('positive',data['pairs'][record_ids]),('negative',negative_pairs[record_ids])):
            logits, detail = model.decoder.pattern_forward(h,graph,query,auxiliary_grad=True)
            dispatch['population_forward_calls'] += 1
            rows, bits = teacher.labels(query, detail['neighbors'])
            left_rows, _ = detail['neighbors'].left; right_rows, _ = detail['neighbors'].right
            require(torch.equal(rows,torch.cat((left_rows,right_rows))), 'Teacher/native slot order differs')
            labels = core.TrainPatternLabels(bits.to(detail['t'].dtype), 'complete_TRAIN_observation_membership')
            values = core.training_pattern_losses(detail['t'][:,:len(left_rows)],detail['t'][:,len(left_rows):],
                                                  left_rows,right_rows,labels,len(query))
            n = torch.tensor(values['support_counts'],device=device,dtype=torch.long).T
            k = torch.tensor(values['teacher_counts'],device=device,dtype=torch.long).T
            records.append(dict(population=population,query=query,logits=logits,detail=detail,rows=(left_rows,right_rows),
                                labels=bits,values=values,n=n,k=k))
        torch.cuda.synchronize(0); times['native_and_auxiliary_forward'] = time.monotonic()-forward_started
        rng['after_forward'] = snapshot_rng()
        import torch.nn.functional as F
        objectives = dict(base=-F.logsigmoid(records[0]['logits']).mean()-F.logsigmoid(-records[1]['logits']).mean())
        for key in ('J_K','J_K_sep'):
            objectives[key] = sum(record['values'][key].mean() for record in records)
        require(all(bool(torch.isfinite(value)) for value in objectives.values()), 'Nonfinite diagnostic objective')
        checkpoint_resource('forward_complete')
        gradients = {}; slots = {}; slots_targets = tuple(record['detail']['t'] for record in records)
        for key in ('base','J_K','J_K_sep'):
            phase[0] = key; reverse_started = time.monotonic()
            targets = parameters if key == 'base' else parameters+slots_targets
            result = torch.autograd.grad(objectives[key], targets, retain_graph=key!='J_K_sep',
                                         create_graph=False, allow_unused=True, materialize_grads=False)
            dispatch['reverse_evaluations'] += 1
            gradients[key] = result[:len(parameters)]
            if key != 'base':
                require(all(value is not None for value in result[len(parameters):]), 'Conditional slot gradient disconnected')
                slots[key] = result[len(parameters):]
            torch.cuda.synchronize(0); times['reverse_'+key] = time.monotonic()-reverse_started
            rng['after_reverse_'+key] = snapshot_rng()
            require(rng['after_reverse_'+key] == rng['after_forward'], 'Checkpoint reverse changed retained forward RNG')
            checkpoint_resource('reverse_'+key+'_complete')
        metadata_started = time.monotonic()
        require(dispatch['encoder_calls'] == 1 and dispatch['population_forward_calls'] == 2
                and dispatch['reverse_evaluations'] == 3, 'Native workload dispatch differs')
        require(initial_state == pilot_state.state_digest(model.state_dict()) and all(p.grad is None for p in parameters)
                and initial_flags == pilot_state.flags(model),
                'Parameters/buffers or .grad changed; zero-update diagnostic violated')
        populations = {}
        for index, record in enumerate(records):
            count = census.summarize(torch,record['query'],record['detail']['neighbors'],teacher.keys,teacher.nodes)
            require(count['support_count_digest'] == tensor_sha(torch.cat((record['n'],record['k']),dim=1)), 'Core/teacher support counts differ')
            group_details = []
            for side in range(2):
                n = record['n'][:,side]; k = record['k'][:,side]; r = torch.minimum(k,n-k)
                keys, frequency = torch.unique(torch.stack((n,r),dim=1),dim=0,return_counts=True)
                group_details.append(dict(n_r_groups=[pair+[freq] for pair,freq in zip(keys.tolist(),frequency.tolist())],
                    genuine_groups=int((keys[:,1]>1).sum()), genuine_group_slot_loops=int(keys[keys[:,1]>1,0].sum())))
            populations[record['population']] = dict(census=count, support_coordinates_sha256=teacher.support_digest(record['query'],record['detail']['neighbors'],record['labels']),
                ordered_slot_logits_sha256=tensor_sha(record['detail']['t']), native_target_logits=magnitudes(record['logits']),
                auxiliary_logits=magnitudes(record['detail']['t']), aux_losses={key: magnitudes(record['values'][key]) for key in ('J_K','J_K_sep')},
                actual_side_group_dispatch=group_details,
                slot_gradients=slot_reports(record,slots['J_K'][index],slots['J_K_sep'][index],**plan['fixed_gradient_agreement_rule']))
        blocks = parameter_reports(named, gradients)
        observed_groups = sum(side['genuine_groups'] for value in populations.values() for side in value['actual_side_group_dispatch'])
        observed_loops = sum(side['genuine_group_slot_loops'] for value in populations.values() for side in value['actual_side_group_dispatch'])
        require(dispatch['conditional_side_calls'] == 4 and dispatch['genuine_ESP_group_calls'] == observed_groups
                and dispatch['genuine_ESP_group_slot_loops'] == observed_loops, 'Actual conditional dispatch/count census differs')
        module_custody(plan)
        require(pilot_model.runtime_settings() == profile, 'Actual runtime profile changed')
        times['metadata'] = time.monotonic()-metadata_started
        checkpoint_resource('metadata_complete')
        monitor_stop.set(); monitor.join(timeout=1); require(not monitor.is_alive(), 'Resource monitor did not close')
        write(output/'DIAGNOSTIC.json',dict(schema='native-count-conditioned-gradient-diagnostic-v1',status='COMPLETE_DIAGNOSTIC_ONLY',
            UTC=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=sha(HERE/'MANIFEST.json'),release_sha256=args.release_sha256,
            core_manifest_sha256=plan['core_manifest_sha256'],runtime_profile_transition=transition,workload=plan['workload'],
            data_files_opened=list(DATA_FILES),data_tensor_digests=data['digests'],teacher=teacher.receipt(),stream=stream,RNG=rng,
            initial_and_final_state_sha256=initial_state,parameter_blocks=blocks,populations=populations,
            losses={key:float(value.detach()) for key,value in objectives.items()},times_seconds=times,
            dispatch=dispatch,inclusive_child_wall_seconds=time.monotonic()-started,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            sampled_resources=resources,optimizer_constructed=False,optimizer_updates=0,fits=0,VALID_TEST_reads=False,
            no_predictive_or_novelty_claim=True,native_target_serving_count_free=True,
            Single_evaluated=False,Single_sorted_candidate_ID_order_dependence_acknowledged=True,
            interpretation='Gradient opportunity at this initialization/mask only. Natural zero is valid. Three reverse passes do not qualify an ordinary update or VALID schedule.'))
        finished = True
    except BaseException as error:
        write(output/'FAILURE.json',dict(status='FAILED',type=type(error).__name__,condition=str(error),automatic_retry=False,
                                       inclusive_child_wall_seconds=time.monotonic()-started))
        raise
    finally:
        monitor_stop.set()
        if monitor is not None: monitor.join(timeout=1)
        try:
            custody = final_file_custody(plan,runtime,args.release,args.release_sha256,data_root)
        except BaseException as error:
            custody = dict(schema='native-gradient-final-file-custody-v1',status='FAILED',type=type(error).__name__,condition=str(error))
        write(output/'FILE_CUSTODY.json',custody)
        write(output/'FINAL_CUSTODY.json',dict(completed=finished and custody['status']=='MATCH',file_custody_sha256=sha(output/'FILE_CUSTODY.json'),
            inclusive_child_wall_seconds_through_custody=time.monotonic()-started,
            files=[dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(output.iterdir()) if path.is_file() and not path.name.endswith('.tmp')]))


if __name__ == '__main__':
    main()
