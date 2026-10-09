"""Inactive full-input actual geometry bank/gradient qualifier; no VALID metrics."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import resource
import time

from bank import make_bank
from shared_fit import optimizer, train_update
from support import HERE, admission, borrowed, read, require, sha, source_checks, topology_bytes

VERSIONS = ('torch', 'numpy', 'scipy', 'torch-geometric', 'torch-sparse', 'torch-scatter', 'torch-householder', 'scikit-learn')
TOPOLOGY_NAMES = ('edge_index', 'time_range', 'full_left_right_idx', 'left_right_idx', 'vertex_tril_idx',
                  'diag_indices', 'tril_indices', 'fixed_diag_indices', 'fixed_tril_indices', 'deg')


def storage_key(value):
    return (str(value.device), value.untyped_storage().data_ptr())


def parameter_groups(model):
    private_names = {path + suffix for path in model.factor_paths for suffix in ('.r', '.s')}
    groups = {}
    for name, value in model.named_parameters():
        local = name.split('.', 2)[2]
        group = ('private_member_' + name.split('.')[1] if local in private_names
                 else 'shared_incidence' if 'sheaf_learners' in name else 'shared_other')
        groups[name] = group
    return groups


def snapshot_gradients(torch, model):
    return {name: None if value.grad is None else value.grad.detach().cpu().clone()
            for name, value in model.named_parameters()}


def differences(torch, first, second, groups):
    """Record numerical differences by group, without numerical pass thresholds."""
    require(set(first) == set(second) == set(groups), 'Matching comparison parameter schema')
    totals = {}
    for name in first:
        a, b, group = first[name], second[name], groups[name]
        row = totals.setdefault(group, dict(parameter_objects=0, elements=0, max_abs=0.0, sum_abs=0.0, sum_square=0.0,
                                           first_sum_square=0.0, second_sum_square=0.0))
        require((a is None) == (b is None), 'Matching gradient availability: ' + name)
        row['parameter_objects'] += 1
        if a is None: continue
        require(a.shape == b.shape and a.dtype == b.dtype and torch.isfinite(a).all().item()
                and torch.isfinite(b).all().item(), 'Finite matching comparison tensor: ' + name)
        delta = a.double() - b.double()
        row['elements'] += a.numel()
        row['max_abs'] = max(row['max_abs'], float(delta.abs().max().item()))
        row['sum_abs'] += float(delta.abs().sum().item())
        row['sum_square'] += float(delta.square().sum().item())
        row['first_sum_square'] += float(a.double().square().sum().item())
        row['second_sum_square'] += float(b.double().square().sum().item())
    for row in totals.values():
        count = row['elements']
        row['mean_abs'] = row.pop('sum_abs') / count if count else 0.0
        row['RMS_difference'] = (row.pop('sum_square') / count)**0.5 if count else 0.0
        row['first_L2_norm'] = row.pop('first_sum_square')**0.5
        row['second_L2_norm'] = row.pop('second_sum_square')**0.5
    return totals


class GradientObserver:
    def __init__(self, torch, model, persist):
        self.torch, self.persist, self.records = torch, persist, []
        self.private = {path + suffix for path in model.factor_paths for suffix in ('.r', '.s')}
        self.previous_private, self.previous_shared = {}, {}

    def __call__(self, index, model):
        torch, names = self.torch, [dict(member.named_parameters()) for member in model.members]
        require(index == len(self.records), 'Fixed observer member order')
        finite_private_norms, gradient_keys = {}, []
        for member, values in enumerate(names):
            for name in sorted(self.private):
                value, key = values[name], (member, name)
                if member > index:
                    require(value.grad is None, 'Future member private gradient absent')
                    continue
                require(value.grad is not None and torch.isfinite(value.grad).all().item(), 'Active/completed private finite gradient present')
                current = value.grad.detach().cpu().clone()
                if member < index:
                    require(torch.equal(current, self.previous_private[key]), 'Other member backward leaves completed private gradient exactly unchanged')
                self.previous_private[key] = current
                finite_private_norms[str(member) + ':' + name] = float(current.double().norm().item())
                gradient_keys.append(storage_key(value.grad))
        require(len(set(gradient_keys)) == len(gradient_keys), 'Disjoint private gradient storage')
        shared_deltas, shared_keys = {}, []
        for name in sorted(set(names[0]) - self.private):
            value = names[0][name]
            require(value.grad is not None and torch.isfinite(value.grad).all().item(), 'Shared native finite gradient present after each backward')
            current = value.grad.detach().cpu().clone()
            previous = self.previous_shared.get(name)
            delta = current if previous is None else current - previous
            shared_deltas[name] = dict(accumulated_L2_norm=float(current.double().norm().item()),
                                      this_backward_accumulator_delta_L2_norm=float(delta.double().norm().item()))
            self.previous_shared[name] = current
            shared_keys.append(storage_key(value.grad))
        require(not set(gradient_keys).intersection(shared_keys), 'Private and shared gradient storage disjoint')
        row = dict(member=index, private_active_and_completed_objects=len(gradient_keys), future_private_gradients_absent=True,
                   prior_private_gradients_exactly_unchanged=True, private_gradient_storage_disjoint=True,
                   private_gradient_L2_norms=finite_private_norms, shared_accumulator=shared_deltas,
                   nonzero_gradient_required=False, numerical_parity_threshold=None)
        self.records.append(row)
        self.persist()


def topology_tensors(torch, model):
    for member, path in enumerate(model.members):
        for owner_name, owner in (('model', path), ('builder', path.laplacian_builder)):
            for name in TOPOLOGY_NAMES:
                value = getattr(owner, name, None)
                if isinstance(value, torch.Tensor): yield str(member) + ':' + owner_name + ':' + name, value


def topology_snapshot(torch, model):
    references, copies = {}, {}
    for name, value in topology_tensors(torch, model):
        references[name] = (id(value), value._version)
        if id(value) not in copies: copies[id(value)] = value.detach().cpu().clone()
    return references, copies


def verify_topology(torch, model, snapshot):
    references, copies = snapshot
    current = dict(topology_tensors(torch, model))
    require(set(current) == set(references), 'Same complete native topology attribute schema')
    for name, value in current.items():
        require((id(value), value._version) == references[name]
                and torch.equal(value.detach().cpu(), copies[id(value)]), 'Native topology identity/version/values unchanged: ' + name)
    return dict(attribute_references=len(references), unique_tensors=len(copies), identity_version_values_unchanged=True,
                CPU_snapshot_bytes=sum(value.numel()*value.element_size() for value in copies.values()))


def equivalent_topology(torch, model, snapshot):
    references, copies = snapshot
    current = dict(topology_tensors(torch, model))
    require(set(current) == set(references) and all(torch.equal(value.detach().cpu(), copies[references[name][0]])
            for name, value in current.items()), 'Fresh bank native topology exactly equals original construction')
    return True


def cache_ownership(torch, model, data, role_meta):
    learners, keys, rows = [], [], []
    parameters = {storage_key(value) for value in model.parameters()}
    shape = (role_meta['support_counts']['canonical_undirected'], model.members[0].d, model.members[0].d)
    for member, path in enumerate(model.members):
        for layer, learner in enumerate(path.sheaf_learners):
            learners.append(id(learner))
            cache = learner.L
            require(isinstance(cache, torch.Tensor) and tuple(cache.shape) == shape and cache.device == data['x'].device
                    and cache.dtype == data['x'].dtype and not cache.requires_grad and cache.grad_fn is None
                    and torch.isfinite(cache).all().item(), 'Native private detached full transport cache')
            keys.append(storage_key(cache))
            rows.append(dict(member=member, layer=layer, shape=list(cache.shape), dtype=str(cache.dtype), device=str(cache.device),
                             bytes=cache.untyped_storage().nbytes(), detached=True))
    require(len(set(learners)) == len(learners) == 4*model.members[0].layers
            and len(set(keys)) == len(keys) and not set(keys).intersection(parameters), 'Distinct learners/live caches, no cache/parameter alias')
    return dict(distinct_native_builders=4, distinct_layer_learners=len(learners), distinct_live_transport_cache_storages=len(keys),
                native_cache_kind='pre-normalization detached transport maps from original set_L', caches=rows,
                total_cache_storage_bytes=sum(row['bytes'] for row in rows), numerical_difference_required=False)


def structural(torch, model, opt, data, role_meta, initial=False):
    ownership = model.assert_ownership()
    private = {path + suffix for path in model.factor_paths for suffix in ('.r', '.s')}
    names = [dict(member.named_parameters()) for member in model.members]
    slow = [value for name, value in names[0].items() if name not in private]
    fast = [values[name] for values in names for name in sorted(private)]
    require(not {storage_key(value) for value in slow}.intersection(storage_key(value) for value in fast), 'Private factor/native shared storage disjoint')
    if initial: require(all(torch.equal(value, torch.ones_like(value)) for value in fast), 'All fast factors initialize at one')
    entries = [value for group in opt.param_groups for value in group['params']]
    require(len(entries) == len({id(value) for value in entries}) == len(list(model.parameters()))
            and {id(value) for value in entries} == {id(value) for value in model.parameters()}, 'Actual Adam one entry per deduplicated parameter')
    return dict(ownership=ownership, optimizer_deduplicated=True, optimizer_parameter_entries=len(entries),
                private_shared_storage_disjoint=True, unit_initial_factors_checked=initial,
                static_topology_unique_bytes=topology_bytes(torch, model.members))


def label_free_serving(np, torch, helpers, old, model, data, seed, counters, prefix):
    """Four full native eval paths, no labels, metric function or scientific gate."""
    before = helpers.capture_rng(np, torch)
    model.eval(); model.assert_ownership()
    try:
        helpers.seed_all(np, torch, seed+2000003+1)
        outputs = []
        with torch.no_grad():
            for member in model.members:
                counters[prefix+'_forward_attempts'] += 1
                logp = member(data['x'])
                require(logp.shape == (data['x'].shape[0], 2) and torch.isfinite(logp).all().item(), 'Finite complete binary label-free serving')
                outputs.append(logp.detach().cpu())
                counters[prefix+'_forwards_completed'] += 1
                del logp
        pooled = torch.log(torch.stack([value.exp() for value in outputs]).mean(0))
        require(torch.isfinite(pooled).all().item(), 'Finite probability-mean label-free serving')
        return dict(pooled=pooled, members=outputs)
    finally:
        helpers.restore_rng(np, torch, before)
        require(old.exact(torch, helpers.capture_rng(np, torch), before), 'Exact label-free serving RNG restoration')


def serving_materiality(torch, first, second):
    rows = {}
    pairs = [('pooled', first['pooled'], second['pooled'])] + [('member_'+str(i), first['members'][i], second['members'][i]) for i in range(4)]
    for name, a, b in pairs:
        require(a.shape == b.shape and torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), 'Finite matching label-free outputs')
        rows[name] = dict(max_abs_logp=float((a-b).abs().max().item()), mean_abs_logp=float((a-b).abs().mean().item()),
                         max_abs_probability=float((a.exp()-b.exp()).abs().max().item()),
                         mean_abs_probability=float((a.exp()-b.exp()).abs().mean().item()),
                         decision_change_count=int((a.argmax(-1) != b.argmax(-1)).sum().item()))
    return rows


def joint_update(torch, model, opt, data, counters):
    model.train(); model.assert_ownership(); opt.zero_grad(set_to_none=True)
    versions = [(value, value._version) for value in model.parameters()]
    losses = []
    for member in model.members:
        counters['joint_train_forward_attempts'] += 1
        logp = member(data['x'])
        require(logp.shape == (data['x'].shape[0],2) and torch.isfinite(logp).all().item(), 'Finite complete joint reference TRAIN output')
        counters['joint_train_forwards_completed'] += 1
        loss = torch.nn.functional.nll_loss(logp[data['train_index']], data['train_y'])
        require(torch.isfinite(loss).item(), 'Finite joint own all-TRAIN NLL')
        losses.append(loss)
        del logp
    member = None
    values = [float(loss.item()) for loss in losses]
    counters['joint_backward_attempts'] += 1
    torch.stack(losses).mean().backward()
    counters['joint_backwards_completed'] += 1
    del losses, loss
    require(all(value._version == version for value, version in versions), 'Joint mean uses unchanged old parameters')
    require(all(value.grad is not None and torch.isfinite(value.grad).all().item() for value in model.parameters()), 'Finite joint reference gradients')
    counters['joint_Adam_attempts'] += 1
    opt.step()
    counters['joint_Adam_steps_completed'] += 1
    return values


def run(args):
    pins, protocol = source_checks(), read(HERE/'PROTOCOL.json')
    release, receipt, config, output = admission(args, pins, protocol)
    require(release['action'] == 'geometry_engineering_qualify' and type(release['compare_joint_reference']) is bool
            and release['qualifier_seed'] == protocol['engineering_qualifier']['seed'] == 7409,
            'Prospective qualifier action/seed/joint-reference choice')
    require(set(release['expected_runtime_versions']) == set(VERSIONS), 'Complete existing runtime versions')
    started, usage0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    old, common, helpers, placement = borrowed(pins)
    output.mkdir(parents=True, exist_ok=False)
    counters = {key: 0 for key in ('train_forward_attempts','train_forwards_completed','backward_attempts','backwards_completed',
        'Adam_attempts','Adam_steps_completed','joint_train_forward_attempts','joint_train_forwards_completed',
        'joint_backward_attempts','joint_backwards_completed','joint_Adam_attempts','joint_Adam_steps_completed',
        'original_serving_forward_attempts','original_serving_forwards_completed','restored_serving_forward_attempts',
        'restored_serving_forwards_completed','original_constructor_attempts','original_constructors_completed')}
    record = dict(status='started', qualification_passed=False, source_only_preparation=False,
        qualification_kind='actual_full_input_sharing_private_gradient_and_streamed_update', full_graph=True, all_TRAIN_rows=True,
        validation_metric_access=False, scientific_metric_function_called=False, TEST_truth_present=False, automatic_retry=False,
        compare_joint_reference=release['compare_joint_reference'], joint_choice_declared_before_execution=True,
        tiny_float_gate=False, context_regularizer=0.0, scientific_full_fits_or_VALID_selected_checkpoints_created=False,
        identity=dict(source_seal_sha256=sha(HERE/'SEAL.json'), manifest_sha256=read(HERE/'SEAL.json')['manifest_sha256'],
            roles_sha256=release['roles_sha256'], role_metadata_sha256=release['role_metadata_sha256'], configuration_id=config['id'],
            configuration=config, optimizer=receipt['optimizer'], admission_receipt_sha256=sha(args.admission),
            root_release_sha256=sha(args.release), execution_source_commit=release['execution_source_commit'], base_seed=7409))
    observer = None
    def persist():
        common.write_json(output/'QUALIFICATION.json', dict(record, counters=counters,
            per_backward_gradient_observations=[] if observer is None else observer.records))
    persist()
    stage, model, opt, device, torch = 'runtime', None, None, None, None
    try:
        import numpy as np
        import torch
        versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        require(versions == release['expected_runtime_versions'] and str(torch.__version__) == '2.1.2+cu118'
                and np.__version__ == '1.26.4', 'Exact root-qualified existing runtime')
        device = torch.device(release['device'])
        require(device == torch.device('cuda:0') and torch.cuda.device_count() == 1, 'One root-owned visible cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(release['deterministic_algorithms'])
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        record['identity']['runtime_versions'] = versions
        record['identity'].update(device=str(device), deterministic_algorithms=release['deterministic_algorithms'], allow_tf32=False)
        stage = 'full_input_roles'
        arrays, role_meta, role_sha = common.read_roles(np, args.roles)
        require(role_meta.get('exposure_classification') == 'original_paper_benchmark_exploratory'
                and role_sha == release['role_metadata_sha256'], 'Same complete original benchmark input')
        # The original six-key loader validates roles. VALID labels/indices are dropped before any tensor/model path.
        del arrays['valid_y'], arrays['valid_index']
        data = {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}
        data['cpu_edge_index'] = torch.from_numpy(arrays['edge_index'])
        common.write_json(output/'ROLE_METADATA.json', role_meta)
        record.update(validation_arrays_dropped_before_model_path=True, input_role_counts=role_meta['role_counts'],
                      actual_support_counts=role_meta['support_counts'])
        adapter = common.load_adapter()
        cfg, seed = receipt['optimizer'], release['qualifier_seed']
        native_args = dict(config['native_args'], graph_size=data['x'].shape[0], input_dim=data['x'].shape[1], output_dim=2, device=str(device))
        factory = placement.make_native_placed_factory(torch, adapter, data['cpu_edge_index'], data['edge_index'], native_args)
        def observed_factory():
            counters['original_constructor_attempts'] += 1
            value = factory()
            counters['original_constructors_completed'] += 1
            return value
        def fresh_bank(): return make_bank(torch, adapter, observed_factory)
        stage = 'streamed_bank_construction'
        helpers.seed_all(np, torch, seed)
        tick = time.perf_counter()
        model = fresh_bank(); opt = optimizer(torch, model, cfg)
        helpers.synchronize(torch, device)
        record.update(native_args=native_args, streamed_construction_seconds=time.perf_counter()-tick,
                      initial_structure=structural(torch, model, opt, data, role_meta, initial=True))
        groups = parameter_groups(model)
        initial_state = old.cpu_tree(torch, model.state_dict())
        initialization_end_rng = helpers.capture_rng(np, torch)
        topology = topology_snapshot(torch, model)
        original_topology = topology
        stage = 'actual_streamed_four_backward_one_Adam_step'
        observer = GradientObserver(torch, model, persist)
        tick = time.perf_counter()
        record['streamed_own_TRAIN_NLL'] = train_update(torch, model, opt, data, counters, observer)
        helpers.synchronize(torch, device)
        record.update(streamed_update_seconds=time.perf_counter()-tick, after_streamed_structure=structural(torch, model, opt, data, role_meta),
                      streamed_cache_ownership=cache_ownership(torch, model, data, role_meta), streamed_topology=verify_topology(torch, model, topology))
        streamed_gradients = snapshot_gradients(torch, model)
        streamed_state = old.cpu_tree(torch, model.state_dict())
        streamed_optimizer = old.cpu_tree(torch, opt.state_dict())
        streamed_end_rng = helpers.capture_rng(np, torch)
        stage = 'original_label_free_four_path_serving'
        tick = time.perf_counter()
        original_serving = label_free_serving(np, torch, helpers, old, model, data, seed, counters, 'original_serving')
        helpers.synchronize(torch, device)
        record.update(original_serving_seconds=time.perf_counter()-tick, original_serving_topology=verify_topology(torch, model, topology),
                      original_serving_cache_ownership=cache_ownership(torch, model, data, role_meta))
        persist()
        model = opt = None
        gc.collect(); torch.cuda.empty_cache()
        if release['compare_joint_reference']:
            stage = 'joint_reference_bank_construction'
            tick = time.perf_counter()
            model = fresh_bank(); model.load_state_dict(initial_state, strict=True)
            require(old.exact(torch, old.cpu_tree(torch, model.state_dict()), initial_state), 'Joint reference exact initial same-bank state')
            record['joint_topology_exactly_matches_original'] = equivalent_topology(torch, model, original_topology)
            opt = optimizer(torch, model, cfg)
            helpers.restore_rng(np, torch, initialization_end_rng)
            require(old.exact(torch, helpers.capture_rng(np, torch), initialization_end_rng), 'Synchronized joint forward RNG start')
            topology = topology_snapshot(torch, model)
            helpers.synchronize(torch, device)
            record.update(joint_construction_seconds=time.perf_counter()-tick, joint_initial_structure=structural(torch, model, opt, data, role_meta, initial=True))
            stage = 'same_bank_joint_mean_own_NLL_backward'
            tick = time.perf_counter()
            record['joint_own_TRAIN_NLL'] = joint_update(torch, model, opt, data, counters)
            helpers.synchronize(torch, device)
            record.update(joint_update_seconds=time.perf_counter()-tick, joint_cache_ownership=cache_ownership(torch, model, data, role_meta),
                joint_after_structure=structural(torch, model, opt, data, role_meta),
                joint_topology=verify_topology(torch, model, topology),
                gradient_differences=differences(torch, streamed_gradients, snapshot_gradients(torch, model), groups),
                parameter_update_differences=differences(torch, {name:streamed_state[name].double()-initial_state[name].double() for name in groups},
                    {name:value.detach().cpu().double()-initial_state[name].double() for name,value in model.named_parameters()}, groups),
                own_NLL_signed_differences=[b-a for a,b in zip(record['streamed_own_TRAIN_NLL'],record['joint_own_TRAIN_NLL'])],
                joint_end_RNG_exact=old.exact(torch, helpers.capture_rng(np, torch), streamed_end_rng),
                joint_numeric_difference_threshold=None)
            require(record['joint_end_RNG_exact'], 'Streamed/joint complete forwards consume identical owned RNG')
            record['joint_reference_completed'] = True
            persist()
            model = opt = None
            gc.collect(); torch.cuda.empty_cache()
        else:
            record.update(joint_reference_completed=False, joint_reference_prospectively_disabled=True,
                          joint_reference_limit='joint autograd retains four complete full-graph graphs; root prospectively disabled under resource custody')
        stage = 'serialize_owned_streamed_engineering_state'
        tick = time.perf_counter()
        state_path = output/'ENGINEERING_STATE.pt'
        state = dict(schema='geometry-only-one-update-engineering-state-v2', identity=record['identity'], model_state=streamed_state,
                     optimizer_state=streamed_optimizer, training_rng=streamed_end_rng, label_free_serving=original_serving,
                     TEST_truth_saved=False, VALID_truth_saved=False, scientific_checkpoint=False)
        temporary = state_path.with_suffix('.tmp'); torch.save(state, temporary); temporary.replace(state_path)
        record.update(engineering_state_write_seconds=time.perf_counter()-tick, engineering_state_sha256=sha(state_path),
                      engineering_state_bytes=state_path.stat().st_size, server_only_engineering_state=True)
        del state
        stage = 'fresh_exact_streamed_state_reconstruction'
        tick = time.perf_counter()
        saved = torch.load(state_path, map_location='cpu', weights_only=True)
        require(saved['identity'] == record['identity'] and saved['TEST_truth_saved'] is False and saved['VALID_truth_saved'] is False, 'Exact engineering checkpoint identity')
        model = fresh_bank(); model.load_state_dict(saved['model_state'], strict=True)
        require(old.exact(torch, old.cpu_tree(torch, model.state_dict()), saved['model_state']), 'Exact reconstructed engineering model parameter/buffer state')
        record['reconstructed_topology_exactly_matches_original'] = equivalent_topology(torch, model, original_topology)
        opt = optimizer(torch, model, cfg); opt.load_state_dict(saved['optimizer_state'])
        require(old.exact(torch, old.cpu_tree(torch, opt.state_dict()), saved['optimizer_state']), 'Exact reconstructed engineering Adam state')
        helpers.restore_rng(np, torch, saved['training_rng'])
        require(old.exact(torch, helpers.capture_rng(np, torch), saved['training_rng']), 'Exact reconstructed engineering training RNG')
        topology = topology_snapshot(torch, model)
        record.update(restored_structure=structural(torch, model, opt, data, role_meta), model_parameter_buffer_restore_exact=True,
                      optimizer_restore_exact=True, training_RNG_restore_exact=True, reconstruction_seconds=time.perf_counter()-tick)
        stage = 'restored_label_free_four_path_serving'
        tick = time.perf_counter()
        serving = label_free_serving(np, torch, helpers, old, model, data, seed, counters, 'restored_serving')
        helpers.synchronize(torch, device)
        record.update(restored_serving_seconds=time.perf_counter()-tick, label_free_serving_materiality=serving_materiality(torch, saved['label_free_serving'], serving),
                      restored_cache_ownership=cache_ownership(torch, model, data, role_meta), restored_topology=verify_topology(torch, model, topology),
                      finite_reconstructed_serving=True, status='complete', qualification_passed=True)
        require(counters['train_forwards_completed'] == counters['backwards_completed'] == 4 and counters['Adam_steps_completed'] == 1,
                'Exactly four actual streamed native forwards/backwards then one Adam step')
        require(counters['joint_train_forwards_completed'] == (4 if release['compare_joint_reference'] else 0)
                and counters['joint_backwards_completed'] == (1 if release['compare_joint_reference'] else 0)
                and counters['joint_Adam_steps_completed'] == (1 if release['compare_joint_reference'] else 0), 'Declared joint work counted exactly')
        require(counters['original_serving_forwards_completed'] == counters['restored_serving_forwards_completed'] == 4, 'All original/reconstructed native serving paths counted')
    except Exception as error:
        record.update(status='failed', qualification_passed=False, failure=common.failure_record(error, stage))
    except BaseException as error:
        record.update(status='interrupted', qualification_passed=False, failure=common.failure_record(error, stage))
        raise
    finally:
        try:
            if torch is not None and device is not None:
                helpers.synchronize(torch, device)
                record.update(peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(device))
                model = opt = None
                gc.collect(); torch.cuda.empty_cache()
        except Exception as error:
            record.update(status='failed', qualification_passed=False, cost_or_cleanup_failure=common.failure_record(error, 'finalize'))
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record.update(complete_attempt_seconds=time.perf_counter()-started, CPU_user_seconds=usage.ru_utime-usage0.ru_utime,
                      CPU_system_seconds=usage.ru_stime-usage0.ru_stime, process_RSS_high_water_cumulative_bytes=common.process_peak_rss_bytes(),
                      actual_TRAIN_forwards=counters['train_forwards_completed']+counters['joint_train_forwards_completed'],
                      all_native_forwards=sum(counters[key] for key in ('train_forwards_completed','joint_train_forwards_completed',
                          'original_serving_forwards_completed','restored_serving_forwards_completed')),
                      cost_includes_initial_and_joint_and_restored_constructors_clones_all_forwards_backwards_steps_serialization_and_failures=True,
                      member_graphs_streamed_with_four_persistent_private_native_caches=True, comparative_opening_authorized=False)
        persist()
    require(record['qualification_passed'], 'Actual full-input geometry engineering qualification failed; retained once-only attempt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    for name in ('release','admission','roles','output'): parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(inactive=True, numeric_model_or_role_import=False, protocol=read(HERE/'PROTOCOL.json')['engineering_qualifier'])))
        return
    require(args.release and args.admission and args.roles and args.output, 'Explicit root release/admission/roles/fresh output required')
    run(args)


if __name__ == '__main__': main()
