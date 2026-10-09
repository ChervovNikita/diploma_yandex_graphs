"""Inactive one-opportunity full native M4/source-correction integration witness."""
from contextlib import contextmanager
import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
ASSIGNMENTS = ('actor', 'director', 'keyword', None)
MEMBER_SEEDS = (1001, 1002, 1003, 1004)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def binding(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def bound(row, name):
    require(Path(row['path']).name == name and not Path(row['path']).is_symlink()
            and binding(row['path']) == row, 'Exact root metadata receipt: ' + name)
    return Path(row['path'])


def module(name, path):
    require(name not in sys.modules, 'Fresh private qualifier module')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def gates(args):
    seal, sources = read(HERE / 'SEAL.json'), read(HERE / 'SOURCE_BINDINGS.json')
    require(seal['runtime_disabled'] is True and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Inactive source seal')
    for row in read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Unchanged qualifier source')
    for row in sources['files']:
        path = (HERE.parent / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE.parent) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Unchanged reviewed native/view/bank/helper source')
    release = read(args.release)
    require(release['enabled'] is True and release['root_source_review_approved'] is True
            and release['bank_view_source_review_approved'] is True
            and release['source_seal_sha256'] == sha(HERE / 'SEAL.json')
            and release['projection_backend_identity_sha256'] == sources['projection_backend_identity_sha256']
            and release['action'] == 'qualify_one_full_input_M4_own_and_source_opportunity'
            and release['native_seed'] == 1 and tuple(release['assignments']) == ASSIGNMENTS
            and tuple(release['member_rng_seeds']) == MEMBER_SEEDS, 'One prospectively fixed root integration witness')
    project = Path(release['project_root']).resolve(strict=True)
    require(str(args.input_root.resolve()) == release['input_root'] and args.input_root.resolve().is_relative_to(project)
            and str(args.output.resolve()) == release['output_directory'] and args.output.resolve().is_relative_to(project)
            and not args.output.exists(), 'Normal host project-only inputs and fresh output')
    require(release['resource_budget'] == dict(host_RSS_bytes=32 * 1024**3, device_bytes=24 * 1024**3, seconds=3600)
            and release['root_external_resource_monitor'] is True, 'Fixed full-input caps and root monitor custody')
    qualification = read(bound(release['native_v2_qualification'], 'COHORT_REPORT.json'))
    native_release = read(bound(release['native_runtime_release'], Path(qualification['release']['path']).name))
    require(release['native_runtime_release'] == qualification['release']
            and qualification['status'] == 'complete' and qualification['complete'] is True
            and qualification['native_backbone_numerically_qualified'] is True
            and qualification['mode'] == 'qualification' and qualification['seeds'] == [1]
            and qualification['source_seal_sha256'] == sources['native_v2_seal_sha256']
            and native_release['source_seal_sha256'] == sources['native_v2_seal_sha256']
            and native_release['enabled'] is True
            and native_release['input_root'] == release['input_root'], 'Successful exact native v2 serialization qualifier before bank integration')
    require(str(Path(sys.executable).absolute()) == qualification['runtime']['python_executable'], 'Qualified native interpreter invocation')
    role = qualification['roles']['1']
    bound(role, 'seed1.json')
    return release, sources, qualification, native_release, role


def tensors_snapshot(torch, session):
    return [[(name, value, value.detach().cpu().clone()) for name, value in member.named_buffers(remove_duplicate=False)]
            for member in session.bank.members]


def buffers_exact(torch, session, before):
    return all([(name, id(value)) for name, value in member.named_buffers(remove_duplicate=False)]
               == [(name, id(value)) for name, value, _ in records]
               and all(torch.equal(value.detach().cpu(), old) for _, value, old in records)
               for member, records in zip(session.bank.members, before))


def site_statistics(torch, names, values, channels):
    result = []
    for name, value in zip(names, values):
        require(torch.isfinite(value).all().item(), 'Finite actual helper weighted VJP contribution')
        row = dict(name=name, shape=list(value.shape), L2=float(value.double().norm().item()),
                   nonzero_entries=int(torch.count_nonzero(value).item()))
        if value.ndim == 2 and value.shape[0] == len(channels):
            norms = value.double().square().sum(dim=1).sqrt().tolist()
        elif name == 'fc_after_concat.input_factor' and value.numel() == 512 * len(channels):
            norms = value.double().reshape(512, len(channels)).square().sum(dim=0).sqrt().tolist()
        else:
            norms = None
        if norms is not None:
            row['native_channel_L2'] = [dict(namespace=c.namespace, key=c.key, row=c.row, L2=float(norm)) for c, norm in zip(channels, norms)]
        result.append(row)
    return result


@contextmanager
def observe_real_correction(session, result):
    """Delegate unchanged helper once; observe existing J accumulations, no extra VJP."""
    helper, torch = session.helper, session.rt['torch']
    make, replay = helper.make_credit_plan, helper.accumulate_replay
    def observed_make(*args, **kwargs):
        plan = make(*args, **kwargs)
        differences = []
        for member, family in enumerate(session.config.assignments):
            if family is None:
                continue
            full = plan.outputs[helper.OutputKey('train_full', member)]
            probe = plan.outputs[helper.OutputKey('train_probe', member, family)]
            differences.append(dict(member=member, source=family, rows=int(full.shape[0]),
                max_abs_logit=float((full-probe).abs().max().item()),
                mean_abs_logit=float((full-probe).abs().mean().item()),
                max_abs_probability=float((torch.sigmoid(full)-torch.sigmoid(probe)).abs().max().item())))
        result['source_visible_output_differences'] = differences
        require(all(math.isfinite(value) for value in plan.values.values()), 'Finite actual reference J and native risks')
        result['risk_names'] = list(plan.risks)
        return plan
    def observed_replay(plan, key, output):
        observe = key.member < 3 and key.kind in ('train_full', 'train_probe') and key in plan.cotangents['J']
        if not observe:
            return replay(plan, key, output)
        with session.costs.measure('observe_existing_helper_J_VJP_' + key.kind + '_member' + str(key.member), gpu=True):
            parameters = plan.ownership.private[key.member]
            old = plan.gradients.get('J', {}).get(key.member)
            before = None if old is None else [value.detach().cpu().clone() for value in old]
            value = replay(plan, key, output)
            current = plan.gradients['J'][key.member]
            delta = [value.detach().cpu().clone() for value in current]
            if before is not None:
                delta = [now-previous for now, previous in zip(delta, before)]
            require(len(parameters) == len(session.bank.private_names) == len(delta), 'Actual declared private block coverage')
            cotangent = plan.cotangents['J'][key]
            result.setdefault('actual_weighted_J_path_contributions', []).append(dict(member=key.member,
                source=key.source, path=key.kind, output_cotangent_L2=float(cotangent.double().norm().item()),
                sites=site_statistics(torch, session.bank.private_names, delta, session.adapter.channel_order(session.bank.members[key.member])),
                measurement='delta of actual FP32 helper J accumulation; addition/subtraction roundoff descriptive',
                full_Jacobian_inferred=False))
            return value
    helper.make_credit_plan, helper.accumulate_replay = observed_make, observed_replay
    try:
        yield
    finally:
        helper.make_credit_plan, helper.accumulate_replay = make, replay


def serve_known(session):
    """Factual-only known-role outputs; no accuracy or VALID quality scoring."""
    torch = session.rt['torch']
    with session.costs.measure('qualifier_complete_factual_known_role_serving', gpu=True):
        tokens = [session.token(member, None, 'eval', 'KNOWN') for member in range(4)]
        probabilities, logits = session.helper.serve_full_input(session.helper_config, session.ownership,
            session.native_forward, tokens, full_view_binding=session.config.full_view_binding)
        require(probabilities.shape == (session.ctx.train_count + session.ctx.valid_count, 5)
                and torch.isfinite(probabilities).all().item(), 'Complete finite native all-full probability mean')
        return {role: dict(probabilities=probabilities[start:end].detach().cpu().clone(),
                           member_logits=tuple(value[start:end].detach().cpu().clone() for value in logits))
                for role, start, end in [('TRAIN', 0, session.ctx.train_count),
                                        ('VALID', session.ctx.train_count, session.ctx.train_count + session.ctx.valid_count)]}


def witness(rt, engine, adapter, helper, views_module, bank_module, data, role, role_binding, release, identity, folder, report):
    torch = rt['torch']
    costs = engine.Costs(folder, torch, rt['device'])
    torch.cuda.reset_peak_memory_stats(rt['device'])
    static = engine.prepare_static(rt, data, costs)
    scalar = torch.cuda.amp.GradScaler()  # Original master lifecycle, before seed setup.
    ctx = engine.prepare_seed(rt, static, role, 1, costs)
    full_binding = bank_module.digest(dict(native_qualification_binding=identity['native_qualification_binding'],
        role_binding=role_binding, input_files=data.input_bindings, seed=1,
        feature_shapes=static.feature_shapes, label_shapes={k: list(v.shape) for k, v in ctx.label_feats.items()}))
    sources = read(HERE / 'SOURCE_BINDINGS.json')
    views = views_module.build_family_views(rt, static, ctx, costs, views_module.ViewConfig(enabled=True,
        root_source_review_approved=True, native_qualification_binding=identity['native_qualification_binding'],
        full_view_binding=full_binding, role_binding=role_binding, source_seal_sha256=sources['view_seal_sha256']))
    config = bank_module.BankConfig(enabled=True, root_source_review_approved=True,
        source_seal_sha256=sources['bank_seal_sha256'], native_qualification_binding=identity['native_qualification_binding'],
        full_view_binding=full_binding, role_binding=role_binding, member_rng_seeds=MEMBER_SEEDS,
        assignments=ASSIGNMENTS, source_corrections=True)
    prototype = engine.make_model(rt, ctx, costs)
    require(engine.parameter_counts(prototype)['total'] == 83659532, 'Whole literal native prototype')
    native_parameters = dict(prototype.named_parameters())
    session = bank_module.BankSession(rt, engine, adapter, helper, ctx, views, prototype, scalar, costs, config)
    require(session.projection_backend == sources['projection_backend'], 'Bank uses the explicitly reviewed compressed-Gram backend identity')
    report['projection_backend'] = session.projection_backend
    require(set(native_parameters) == set(session.bank.slow_names)
            and all(dict(member.named_parameters())[name] is value for member in session.bank.members for name, value in native_parameters.items()),
            'Every actual native prototype slow Parameter is the same object in all four members')
    native_parameters = None
    prototype = None
    session.bank.verify_ownership()
    report['ownership'] = dict(shared_native_parameters=sum(p.numel() for p in session.ownership.shared),
        private_per_member=[sum(p.numel() for p in block) for block in session.ownership.private],
        actual_same_slow_parameter_objects=True, deduplicated_Adam=True)
    require(report['ownership']['shared_native_parameters'] == 83659532
            and report['ownership']['private_per_member'] == [97536] * 4, 'Complete native slow and six private factor sites')
    buffers = tensors_snapshot(torch, session)
    objects = [[id(value) for _, value, _ in records] for records in buffers]
    require(len({value for block in objects for value in block}) == sum(map(len, objects)), 'Every registered buffer separate across M4')
    own = session.train_epoch()
    require(session.counters['own_epochs'] == session.counters['actual_Adam_steps'] == 1
            and session.counters['own_member_forwards'] == session.counters['own_backwards'] == 4, 'One complete M4 native AMP own update, not a skipped step')
    report['own_update'] = {key: value for key, value in own.items() if key != 'own_displacements'}
    report['own_update']['private_L2_displacements'] = [math.sqrt(sum(float(value.double().square().sum().item()) for value in own['own_displacements'][m])) for m in range(4)]
    report['counters'] = dict(session.counters)
    report['BN_own_counters'] = []
    for member, records in enumerate(buffers):
        for name, value, old in records:
            if name.endswith('num_batches_tracked'):
                require(int(value.item()) == int(old.item()) + 1, 'Native member BN evolves exactly once on own update')
                report['BN_own_counters'].append(dict(member=member, name=name, before=int(old.item()), after=int(value.item())))
    slow_before = [p.detach().cpu().clone() for p in session.ownership.shared]
    buffers = tensors_snapshot(torch, session)
    modes = [[module.training for module in member.modules()] for member in session.bank.members]
    caller = engine.capture_rng(rt['numpy'], torch)
    observation = {}
    report['source_dependence'] = observation
    try:
        with observe_real_correction(session, observation):
            correction = session.correct(own.pop('own_displacements'))
    finally:
        report['counters'] = dict(session.counters)
    require(correction['status'] in ('zero', 'rejected', 'accepted')
            and all(math.isfinite(v) for v in correction['reference_values'].values()), 'Actual finite correction opportunity with recorded zero/rejection/acceptance')
    require(all(all(math.isfinite(value) for value in trial['values'].values()) for trial in correction['trials']), 'All reported native trial risk values finite')
    require(all(torch.equal(p.detach().cpu(), old) for p, old in zip(session.ownership.shared, slow_before)), 'Actual slow values unchanged by private correction')
    require(buffers_exact(torch, session, buffers) and modes == [[module.training for module in member.modules()] for member in session.bank.members]
            and engine.exact(torch, engine.capture_rng(rt['numpy'], torch), caller), 'Real correction preserves BN objects/values, modes and caller streams')
    report.update(correction=correction, source_dependence=observation, counters=dict(session.counters),
                  slow_values_unchanged_in_correction=True, BN_modes_RNG_preserved=True,
                  view_bindings={family: view.binding for family, view in views.items()})
    slow_before = buffers = None
    outputs = serve_known(session)
    identity = bank_module.plain_metadata(identity)
    saved = session.snapshot(0, {'selection': 'fixed single post-opportunity state; no quality scoring'}, outputs, identity)
    saved['qualification_selected_modes'] = [[module.training for module in member.modules()] for member in session.bank.members]
    checkpoint = folder / 'SELECTED_BANK_STATE.pt'
    with costs.measure('qualifier_owned_checkpoint_write', gpu=True):
        torch.save(saved, checkpoint)
    master_state = engine.cpu_tree(torch, scalar.state_dict())
    caller = engine.capture_rng(rt['numpy'], torch)
    session = saved = outputs = None
    gc.collect()
    torch.cuda.empty_cache()
    fresh = None
    try:
        with costs.measure('qualifier_safe_load_and_fresh_complete_native_bank', gpu=True):
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            prototype = engine.make_model(rt, ctx, costs)
            fresh = bank_module.BankSession(rt, engine, adapter, helper, ctx, views, prototype,
                torch.cuda.amp.GradScaler(), costs, config)
            prototype = None
            fresh.restore_selected(saved, identity)
            require(saved['qualification_selected_modes'] == [[module.training for module in member.modules()] for member in fresh.bank.members], 'Actual fresh witness restores native selected module modes')
        current = serve_known(fresh)
        report['fresh_selected'] = dict(exact_owned_state_restored=True, output_bitwise_gate=False,
            descriptive_drift={role: dict(max_abs_probability=float((current[role]['probabilities']-saved['outputs'][role]['probabilities']).abs().max().item()),
                max_abs_member_logit=max(float((x-y).abs().max().item()) for x, y in zip(current[role]['member_logits'], saved['outputs'][role]['member_logits'])),
                prediction_changes=int(((current[role]['probabilities']>.5)!=(saved['outputs'][role]['probabilities']>.5)).sum().item())) for role in ('TRAIN', 'VALID')})
        report['checkpoint_sha256'] = sha(checkpoint)
    finally:
        fresh = prototype = saved = None
        gc.collect()
        torch.cuda.empty_cache()
        engine.restore_rng(rt['numpy'], torch, caller)
        require(engine.exact(torch, engine.capture_rng(rt['numpy'], torch), caller)
                and engine.exact(torch, scalar.state_dict(), master_state), 'Fresh restore preserves caller streams and active master scaler')
    report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(rt['device'])
    report['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(rt['device'])
    require(report['peak_cuda_reserved_bytes'] <= release['resource_budget']['device_bytes'], 'Observed whole witness exceeds fixed device cap')


def execute(args):
    release, sources, qualified, native_release, role_binding = gates(args)
    args.output.mkdir(parents=True, exist_ok=False)
    start, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    report = dict(status='started', complete=False, source_seal_sha256=release['source_seal_sha256'],
        native_qualification_binding=release['native_v2_qualification']['sha256'], release=binding(args.release),
        packet_preparation_was_source_only=True, enabled_root_runtime_execution=True,
        scientific_admission=False, quality_scoring=False, TEST_file_access=False,
        native_seed=1, member_rng_seeds=list(MEMBER_SEEDS), assignments=list(ASSIGNMENTS))
    report['projection_backend'] = sources['projection_backend']
    report['projection_backend_identity_sha256'] = sources['projection_backend_identity_sha256']
    rt = None
    write(args.output / 'QUALIFICATION_REPORT.json', report)
    try:
        native = module('_owned_native_v2_qualifier_entry', HERE.parent / sources['native_v2_directory'] / 'runner.py')
        native.source_gate()
        rt, engine, report['runtime'] = native.runtime(native_release)
        require(report['runtime'] == qualified['runtime'], 'Complete runtime exactly matches admitted native v2 qualifier')
        adapter = module('_owned_M4_adapter_qualifier', HERE.parent / sources['adapter_directory'] / 'grouped_member_factors.py')
        helper = module('_owned_source_helper_qualifier', HERE.parent / sources['helper_directory'] / 'source_supply.py')
        views_module = module('_owned_raw_views_qualifier', HERE.parent / sources['view_directory'] / 'native_family_views.py')
        bank_module = module('_owned_bank_callback_qualifier', HERE.parent / sources['bank_directory'] / 'bank_training.py')
        bank_module.source_gate()
        seam = HERE.parent / sources['schema_directory']
        loader = module('_owned_roles_integration_qualifier', seam / 'role_loader.py')
        loader.source_gate()
        costs = engine.Costs(args.output, rt['torch'], rt['device'])
        with costs.measure('qualifier_once_loaded_real_allowed_development_inputs'):
            data = loader.load(args.input_root, Path(role_binding['path']), read(seam / 'SOURCE_EXPECTATIONS.json'))
            require(data.input_bindings == qualified['input_files'], 'Exact actual full native qualification inputs')
        identity = dict(source_seal_sha256=release['source_seal_sha256'], native_qualification_binding=report['native_qualification_binding'],
            projection_backend=sources['projection_backend'], projection_backend_identity_sha256=sources['projection_backend_identity_sha256'],
            role_binding=role_binding['sha256'], release_sha256=report['release']['sha256'], input_files=data.input_bindings,
            runtime=report['runtime'], assignments=list(ASSIGNMENTS), member_rng_seeds=list(MEMBER_SEEDS))
        witness(rt, engine, adapter, helper, views_module, bank_module, data, read(role_binding['path']), role_binding['sha256'],
                release, identity, args.output, report)
        require(loader.bindings(loader.permitted_files(args.input_root)) == qualified['input_files'], 'Inputs unchanged through entire witness')
        report.update(status='complete', complete=True, representative_integration_eligible=True,
                      native_pilot_or_predictive_success_claimed=False, qualification_retries=0)
    except BaseException as error:
        report.update(status='failed', complete=False, representative_integration_eligible=False,
                      concrete_runtime_obstacle={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        end = resource.getrusage(resource.RUSAGE_SELF)
        report.update(seconds=time.perf_counter()-start, CPU_user_seconds=end.ru_utime-usage.ru_utime,
            CPU_system_seconds=end.ru_stime-usage.ru_stime,
            cumulative_RSS_peak_bytes=int(end.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)))
        if report['seconds'] > 3600 or report['cumulative_RSS_peak_bytes'] > 32 * 1024**3:
            report.update(status='failed', complete=False, representative_integration_eligible=False,
                          concrete_runtime_obstacle={'type': 'ResourceCap', 'message': 'Fixed whole-witness time or RSS cap exceeded'})
        if rt is not None:
            try:
                rt['torch'].cuda.synchronize(rt['device'])
                report['peak_cuda_allocated_bytes'] = rt['torch'].cuda.max_memory_allocated(rt['device'])
                report['peak_cuda_reserved_bytes'] = rt['torch'].cuda.max_memory_reserved(rt['device'])
                rt['torch'].cuda.empty_cache()
            except BaseException as error:
                report.update(status='failed', complete=False, representative_integration_eligible=False,
                    cleanup_obstacle={'type': type(error).__name__, 'message': str(error)})
        write(args.output / 'QUALIFICATION_REPORT.json', report)
        write(args.output / 'COMPLETE.json', dict(status=report['status'], complete=report['complete'],
            representative_integration_eligible=report.get('representative_integration_eligible', False), scientific_admission=False, TEST_file_access=False))
        if not report['complete'] and 'concrete_runtime_obstacle' in report and report['concrete_runtime_obstacle']['type'] == 'ResourceCap':
            raise RuntimeError(report['concrete_runtime_obstacle']['message'])
        if not report['complete'] and 'cleanup_obstacle' in report and 'concrete_runtime_obstacle' not in report:
            raise RuntimeError('Witness cleanup failed; final receipts preserve the obstacle')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.qualify:
        print(json.dumps({'inactive': True, 'data_provider_model_or_fit_execution': False}))
        return
    require(args.release and args.input_root and args.output, 'Explicit root qualification release and bound paths')
    execute(args)


if __name__ == '__main__':
    main()
