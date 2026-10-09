"""Inactive full-input authentic BSNN engineering qualifier; exactly one update."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
MODULES = {'torch': 'torch', 'numpy': 'numpy', 'scipy': 'scipy', 'torch-geometric': 'torch_geometric',
           'torch-sparse': 'torch_sparse', 'torch-scatter': 'torch_scatter',
           'torch-householder': 'torch_householder', 'scikit-learn': 'sklearn'}


def require(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def source_checks():
    seal = read(HERE / 'SEAL.json')
    require(seal['source_only'] is True and seal['execution_enabled'] is False
            and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Exact inactive source seal')
    for row in read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed qualifier source')
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    for row in pins['files']:
        path = (PHASE / row['path']).resolve(strict=True)
        require(path.is_relative_to(PHASE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed sealed wrapper/V2 source')
    return pins


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def wrapper(pins):
    root = PHASE / pins['wrapper_directory']
    support = module(root / 'support.py', '_bsnn_qualifier_sealed_support')
    original_pins = support.source_checks()
    absent = object()
    old = sys.modules.get('support', absent)
    try:
        sys.modules['support'] = support
        runner = module(root / 'runner.py', '_bsnn_qualifier_sealed_runner')
    finally:
        if old is absent: sys.modules.pop('support', None)
        else: sys.modules['support'] = old
    common, helpers = support.borrowed(original_pins)
    return support, runner.optimizer, common, helpers, original_pins


class Costs:
    """Observation only; synchronized timings include observer overhead."""
    def __init__(self, common, output):
        self.common, self.output, self.totals = common, output, {}
        self.torch = self.device = None

    @contextmanager
    def measure(self, name, gpu=False):
        row = dict(scope=name, status='started')
        tick, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
        start = end = None
        error = final_error = None
        try:
            if gpu:
                self.torch.cuda.synchronize(self.device)
                start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
                start.record()
            yield row
            row['status'] = 'complete'
        except BaseException as caught:
            error = caught
            row.update(status='failed', failure=self.common.failure_record(caught, name))
            raise
        finally:
            if gpu and start is not None:
                try:
                    end.record()
                    end.synchronize()
                    row['CUDA_stream_elapsed_ms'] = float(start.elapsed_time(end))
                except Exception as caught:
                    final_error = caught
                    row.update(status='failed', cost_failure=self.common.failure_record(caught, name))
            final_usage = resource.getrusage(resource.RUSAGE_SELF)
            row.update(wall_seconds=time.perf_counter()-tick,
                CPU_user_seconds=final_usage.ru_utime-usage.ru_utime,
                CPU_system_seconds=final_usage.ru_stime-usage.ru_stime)
            self.common.append_jsonl(self.output / 'COST_EVENTS.jsonl', row)
            total = self.totals.setdefault(name, dict(attempts=0, completed=0, wall_seconds=0.0,
                CPU_user_seconds=0.0, CPU_system_seconds=0.0, CUDA_stream_elapsed_ms=0.0))
            total['attempts'] += 1
            total['completed'] += int(row['status'] == 'complete')
            for key in ('wall_seconds', 'CPU_user_seconds', 'CPU_system_seconds', 'CUDA_stream_elapsed_ms'):
                total[key] += row.get(key, 0.0)
            if final_error is not None and error is None: raise final_error


@contextmanager
def observe_native_sampling(np, torch, model, costs, counters, stage):
    """Call each original API once, preserve values/gradients/RNG, restore hooks."""
    sampler = model.laplacian_builder.random_so
    original_rvs, original_tensor = sampler.rvs, torch.tensor
    def rvs(*args, **kwargs):
        counters['native_SO_sampler_attempts'] += 1
        with costs.measure(stage + ':SciPy_SO_CPU_sample') as row:
            value = original_rvs(*args, **kwargs)
            row.update(shape=list(value.shape), dtype=str(value.dtype), returned_CPU_bytes=int(value.nbytes),
                       requested_size=int(kwargs['size']), RNG_type=type(sampler.random_state).__name__)
            counters['native_SO_sampler_calls_completed'] += 1
            return value
    def tensor(value, *args, **kwargs):
        tracked = isinstance(value, np.ndarray) and kwargs.get('device') is not None and torch.device(kwargs['device']) == costs.device
        if not tracked: return original_tensor(value, *args, **kwargs)
        counters['native_sampler_array_transfer_attempts'] += 1
        with costs.measure(stage + ':native_numpy_sample_to_GPU_tensor', gpu=True) as row:
            result = original_tensor(value, *args, **kwargs)
            row.update(CPU_source_shape=list(value.shape), CPU_source_dtype=str(value.dtype), CPU_source_bytes=int(value.nbytes),
                       destination_shape=list(result.shape), destination_dtype=str(result.dtype),
                       destination_device=str(result.device), destination_tensor_bytes=result.numel()*result.element_size())
            require(result.dtype == torch.float32 and result.device == costs.device, 'Actual native float32 sampler transfer')
            counters['native_sampler_array_transfers_completed'] += 1
            return result
    sampler.rvs, torch.tensor = rvs, tensor
    try: yield
    finally: sampler.rvs, torch.tensor = original_rvs, original_tensor


def four_draws(np, torch, support, helpers, model, data, seed, costs, counters, stage, persist):
    model.eval()
    before = support.capture(np, torch, helpers, model)
    try:
        support.seed_streams(np, torch, helpers, model, seed + 2000003 + 1)
        with observe_native_sampling(np, torch, model, costs, counters, stage), torch.no_grad():
            probabilities = []
            for draw in range(4):
                counters[stage + '_forward_attempts'] += 1
                with costs.measure(stage + ':full_native_forward', gpu=True):
                    logp, kl = model(data['x'])
                    require(logp.shape == (data['x'].shape[0], 2) and torch.isfinite(logp).all().item()
                            and kl.numel() == 1 and torch.isfinite(kl).item(), 'Finite full-graph native evaluation')
                    probabilities.append(logp.exp())
                counters[stage + '_forwards_completed'] += 1
                persist()
            pooled = torch.log(torch.mean(torch.stack(probabilities), 0))
            require(torch.isfinite(pooled).all().item(), 'Finite full-graph probability mean')
            spread = float((probabilities[0]-probabilities[-1]).abs().max().item())
            return pooled.detach().cpu(), spread
    finally:
        support.restore(np, torch, helpers, model, before)
        require(support.exact(torch, support.capture(np, torch, helpers, model), before), 'Exact evaluation stream restoration')


def run(args):
    pins = source_checks()
    protocol, release = read(HERE / 'PROTOCOL.json'), read(args.release)
    require(release['enabled'] is True and release['source_seal_sha256'] == sha(HERE / 'SEAL.json')
            and release['wrapper_seal_sha256'] == pins['wrapper_seal_sha256'], 'Exact explicit root qualifier release')
    require(type(release['deterministic_algorithms']) is bool and set(release['expected_runtime_versions']) == set(MODULES)
            and all(isinstance(v, str) and v for v in release['expected_runtime_versions'].values()), 'Complete recorded provider versions and deterministic policy')
    require(len(release['execution_source_commit']) == 40 and all(c in '0123456789abcdef' for c in release['execution_source_commit']), 'Pinned committed execution source identity')
    require(sha(args.roles) == release['roles_sha256'] and sha(args.roles.parent / 'ROLE.json') == release['role_metadata_sha256'], 'Exact original full role archive and metadata')
    output = args.output.resolve()
    require(str(output) == release['output_directory'] and not output.exists() and output.is_relative_to(PHASE)
            and not output.is_relative_to(HERE) and not output.is_relative_to(PHASE / pins['wrapper_directory']), 'Fresh bound normal phase output')
    started, usage0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    support, make_optimizer, common, helpers, original_pins = wrapper(pins)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status='started', qualification_passed=False, protocol=protocol, release=release,
                  engineering_only=True, VALID_metrics=False, TEST_truth=False, automatic_retry=False)
    counters = {key: 0 for key in ('TRAIN_forward_attempts', 'TRAIN_forwards_completed', 'backward_attempts',
        'backwards_completed', 'Adam_attempts', 'Adam_steps_completed', 'evaluation_forward_attempts',
        'evaluation_forwards_completed', 'reconstruction_forward_attempts', 'reconstruction_forwards_completed',
        'native_SO_sampler_attempts', 'native_SO_sampler_calls_completed', 'native_sampler_array_transfer_attempts',
        'native_sampler_array_transfers_completed', 'original_constructor_attempts', 'original_constructors_completed')}
    costs = Costs(common, output)
    model = opt = restored = restored_opt = torch = None
    def persist(): common.write_json(output / 'RESULT.json', dict(record, counters=counters, cost_scopes=costs.totals))
    stage = 'runtime_imports'
    persist()
    try:
        with costs.measure(stage):
            providers = {name: importlib.import_module(module_name) for name, module_name in MODULES.items()}
            np, torch = providers['numpy'], providers['torch']
            versions = {name: importlib.metadata.version(name) for name in MODULES}
            require(versions == release['expected_runtime_versions'] and str(torch.__version__) == '2.1.2+cu118'
                    and np.__version__ == '1.26.4', 'Exact recorded existing runtime')
            record['actual_imported_providers'] = {name: dict(version=versions[name], module_path=str(Path(value.__file__).resolve()),
                module_file_sha256=sha(value.__file__)) for name, value in providers.items()}
            device = torch.device(release['device'])
            require(device == torch.device('cuda:0') and torch.cuda.device_count() == 1, 'One root-owned visible cuda:0')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(release['deterministic_algorithms'])
            torch.cuda.reset_peak_memory_stats(device)
            costs.torch, costs.device = torch, device
        stage = 'official_role_load'
        with costs.measure(stage):
            arrays, role_meta, role_sha = common.read_roles(np, args.roles)
            require(role_meta.get('exposure_classification') == 'original_paper_benchmark_exploratory', 'Original exploratory Tolokers role classification')
            record.update(role_metadata_sha256=role_sha, full_TRAIN_count=role_meta['role_counts']['train'],
                actual_directed_support=role_meta['support_counts']['canonical_directed'], all_TRAIN_labels_used=True,
                official_six_key_loader_VALID_rows_validated=True, VALID_rows_transferred_or_scored=False)
            del arrays['valid_index'], arrays['valid_y']
        stage = 'role_arrays_CPU_to_GPU'
        with costs.measure(stage, gpu=True) as row:
            data = {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}
            data['cpu_edge_index'] = torch.from_numpy(arrays['edge_index'])
            row['actual_input_tensor_bytes'] = {key: value.numel()*value.element_size() for key, value in data.items() if key != 'cpu_edge_index'}
        stage = 'authentic_native_import'
        with costs.measure(stage): cls = support.native_class(original_pins)
        native_args = dict(read(PHASE / pins['wrapper_directory'] / 'PROTOCOL.json')['native_args'])
        seed = protocol['seed']
        helpers.seed_all(np, torch, seed)
        construction_tail = {}
        def original_constructor(edge_index, native_arguments):
            counters['original_constructor_attempts'] += 1
            with costs.measure('original_CPU_constructor'):
                value = cls(edge_index, native_arguments)
            counters['original_constructors_completed'] += 1
            construction_tail['tick'] = time.perf_counter()
            construction_tail['event'] = torch.cuda.Event(enable_timing=True)
            construction_tail['event'].record()
            return value
        with costs.measure('sealed_factory_setup_and_topology_identity_check', gpu=True) as row:
            row['actual_topology_GPU_to_CPU_guard_tensor_bytes'] = data['edge_index'].numel()*data['edge_index'].element_size()
            make = support.factory(torch, original_constructor, data['cpu_edge_index'], data['edge_index'], native_args)
        def construct(label):
            with costs.measure(label + ':original_constructor_and_placement', gpu=True) as row:
                value = make()
                helpers.synchronize(torch, device)
                end = torch.cuda.Event(enable_timing=True)
                end.record(); end.synchronize()
                row.update(placement_and_checks_after_constructor_wall_seconds=time.perf_counter()-construction_tail['tick'],
                    placement_and_checks_after_constructor_CUDA_stream_ms=float(construction_tail['event'].elapsed_time(end)))
                return value
        stage = 'initial_model_construction'
        model = construct('initial')
        support.seed_scipy(np, model, seed)
        opt = make_optimizer(torch, model, read(PHASE / pins['wrapper_directory'] / 'PROTOCOL.json')['optimizer'])
        record.update(native_args=native_args, parameters=helpers.parameter_counts(model),
            model_builder_topology_bytes=helpers.static_topology_bytes(model),
            edge_weight_index_storage_bytes=model.weight_learner.full_left_right_idx.untyped_storage().nbytes(),
            sampler_RNG_type=type(support.scipy_rng(model)).__name__,
            sampler_RNG_aliases_global_NumPy=support.scipy_rng(model) is np.random.mtrand._rand)
        initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
        stage = 'sole_native_TRAIN_update'
        record['stage'] = stage; persist()
        model.train(); opt.zero_grad(set_to_none=True)
        with observe_native_sampling(np, torch, model, costs, counters, 'TRAIN'):
            counters['TRAIN_forward_attempts'] += 1
            with costs.measure('TRAIN:full_native_forward_and_loss', gpu=True):
                logp, kl = model(data['x'])
                require(logp.shape == (data['x'].shape[0], 2) and torch.isfinite(logp).all().item()
                        and kl.numel() == 1 and torch.isfinite(kl).item(), 'Finite native full-input TRAIN call')
                beta = torch.sigmoid(torch.tensor(((1 - 1) % 40) / 2 - 10))
                nll = torch.nn.functional.nll_loss(logp[data['train_index']], data['train_y'])
                loss = nll + beta * kl
                require(torch.isfinite(loss).item(), 'Finite mean all-TRAIN NLL plus native annealed KL')
                record['TRAIN_objective'] = dict(nll=float(nll.item()), native_KL=float(kl.item()), beta=float(beta.item()), objective=float(loss.item()))
            counters['TRAIN_forwards_completed'] += 1; persist()
            counters['backward_attempts'] += 1
            with costs.measure('TRAIN:backward', gpu=True):
                loss.backward()
                require(all(value.grad is None or torch.isfinite(value.grad).all().item() for value in model.parameters()), 'Finite native gradients')
            counters['backwards_completed'] += 1; persist()
            counters['Adam_attempts'] += 1
            with costs.measure('TRAIN:Adam_step', gpu=True): opt.step()
            counters['Adam_steps_completed'] += 1
        record['changed_parameter_objects'] = sum(not torch.equal(value.detach().cpu(), initial[name]) for name, value in model.named_parameters())
        require(record['changed_parameter_objects'] > 0, 'One actual parameter-changing Adam update')
        del initial, logp, kl, nll, loss
        stage = 'four_native_evaluation_draws'
        record['stage'] = stage; persist()
        pooled, spread = four_draws(np, torch, support, helpers, model, data, seed, costs, counters, 'evaluation', persist)
        record['evaluation_first_last_probability_difference'] = spread
        stage = 'owned_post_update_state_checkpoint'
        with costs.measure(stage, gpu=True):
            saved = dict(schema='owned-sole-update-engineering-state-v1', wrapper_seal_sha256=pins['wrapper_seal_sha256'],
                qualifier_seal_sha256=sha(HERE / 'SEAL.json'), roles_sha256=release['roles_sha256'], seed=seed, epoch=1,
                model_state=support.cpu_tree(torch, model.state_dict()), optimizer_state=support.cpu_tree(torch, opt.state_dict()),
                training_rng=support.capture(np, torch, helpers, model), pooled_full_graph_logp=pooled,
                evaluation_draws=4, evaluation_seed=seed+2000003+1, VALID_metrics=False, TEST_truth=False)
            checkpoint = output / 'OWNED_POST_UPDATE_STATE.pt'
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(saved, temporary); temporary.replace(checkpoint)
            record['checkpoint_sha256'] = sha(checkpoint)
        del model, opt, saved
        model = opt = None
        gc.collect(); torch.cuda.empty_cache()
        stage = 'fresh_parameter_optimizer_stream_reconstruction'
        with costs.measure(stage, gpu=True):
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            require(saved['wrapper_seal_sha256'] == pins['wrapper_seal_sha256'] and saved['qualifier_seal_sha256'] == sha(HERE / 'SEAL.json')
                    and saved['roles_sha256'] == release['roles_sha256'] and saved['seed'] == seed and saved['epoch'] == 1
                    and saved['evaluation_draws'] == 4, 'Owned sole-update state identity')
            restored = construct('reconstruction')
            restored.load_state_dict(saved['model_state'], strict=True)
            require(support.exact(torch, support.cpu_tree(torch, restored.state_dict()), saved['model_state']), 'Exact reconstructed parameter/buffer state')
            restored_opt = make_optimizer(torch, restored, read(PHASE / pins['wrapper_directory'] / 'PROTOCOL.json')['optimizer'])
            restored_opt.load_state_dict(saved['optimizer_state'])
            require(support.exact(torch, support.cpu_tree(torch, restored_opt.state_dict()), saved['optimizer_state']), 'Exact reconstructed optimizer state')
            support.restore(np, torch, helpers, restored, saved['training_rng'])
            require(support.exact(torch, support.capture(np, torch, helpers, restored), saved['training_rng']), 'Exact reconstructed owned streams')
        stage = 'four_native_reconstruction_draws'
        record['stage'] = stage; persist()
        replay, replay_spread = four_draws(np, torch, support, helpers, restored, data, seed, costs, counters, 'reconstruction', persist)
        diagnostics = dict(max_abs_full_graph_logp=float((replay-saved['pooled_full_graph_logp']).abs().max().item()),
            max_abs_full_graph_probability=float((replay.exp()-saved['pooled_full_graph_logp'].exp()).abs().max().item()),
            full_graph_prediction_changes=int((replay.argmax(-1) != saved['pooled_full_graph_logp'].argmax(-1)).sum().item()),
            reconstruction_first_last_probability_difference=replay_spread, bitwise_output_required=False)
        record['reconstruction_diagnostics'] = diagnostics
        require(diagnostics['max_abs_full_graph_logp'] <= protocol['gross_max_abs_logp']
                and diagnostics['max_abs_full_graph_probability'] <= protocol['gross_max_abs_probability'], 'Practical gross reconstruction drift only')
        require(counters['TRAIN_forwards_completed'] == counters['backwards_completed'] == counters['Adam_steps_completed'] == 1
                and counters['evaluation_forwards_completed'] == counters['reconstruction_forwards_completed'] == 4
                and counters['original_constructors_completed'] == 2
                and counters['native_SO_sampler_calls_completed'] == counters['native_sampler_array_transfers_completed'] == 18,
                'One update, two constructors, eight eval calls, eighteen actual native per-layer samples/transfers')
        record.update(status='complete', qualification_passed=True, model_parameter_buffer_state_exact=True,
            optimizer_state_exact=True, selected_streams_exact=True, VALID_metrics=False, TEST_truth=False,
            baseline_screen_authorized=False, competence_claim=False, repeated_update_or_retry=False)
    except BaseException as error:
        record.update(status='failed', qualification_passed=False, failure=common.failure_record(error, stage))
        common.append_jsonl(output / 'FAILURES.jsonl', record)
        raise
    finally:
        try:
            if torch is not None and costs.device is not None:
                helpers.synchronize(torch, costs.device)
                record.update(peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(costs.device),
                    peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(costs.device))
                model = opt = restored = restored_opt = None
                gc.collect(); torch.cuda.empty_cache()
        except Exception as error:
            record.update(status='failed', qualification_passed=False, cleanup_or_cost_failure=common.failure_record(error, 'finalize'))
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record.update(complete_attempt_seconds=time.perf_counter()-started,
            CPU_user_seconds=usage.ru_utime-usage0.ru_utime, CPU_system_seconds=usage.ru_stime-usage0.ru_stime,
            process_RSS_high_water_cumulative_bytes=common.process_peak_rss_bytes(),
            cost_includes_imports_role_load_constructor_transfers_all_native_draws_backward_Adam_checkpoint_reconstruction=True,
            timed_scopes_overlap=True, observer_synchronization_overhead_included=True,
            baseline_timing_estimate=False, root_subprocess_resource_custody_required=True)
        persist()
        common.write_json(output / 'COMPLETE.json', dict(qualification_passed=record['qualification_passed'], status=record['status'],
            result_file='RESULT.json', counters=counters, VALID_metrics=False, TEST_truth=False,
            baseline_screen_authorized=False, automatic_retry=False))
    require(record['qualification_passed'], 'Engineering qualification failed; costs and failure retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--roles', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(inactive=True, protocol=read(HERE / 'PROTOCOL.json'), numeric_import_or_role_load=False)))
        return
    require(args.release and args.roles and args.output, 'Explicit root release, role archive and fresh output required')
    run(args)


if __name__ == '__main__': main()
