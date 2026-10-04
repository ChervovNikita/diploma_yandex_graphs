"""Source-only install witnesses; no numerical imports or execution on import.

The runtime helper bindings are instrumented explicitly. Original selection and
installation functions are called once, and both hooks are restored in finally.
"""
import copy
import hashlib
import inspect
import json
import sys
import time
import weakref


def require(value, message):
    if not value:
        raise ValueError(message)


def tensor_fingerprint(tensor):
    """Logical tensor bytes only; no evaluation, gradient or RNG operation."""
    value = tensor.detach().cpu().contiguous()
    raw = value.numpy().tobytes(order='C')
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                bytes=len(raw), byteorder=sys.byteorder,
                sha256_raw_logical_bytes=hashlib.sha256(raw).hexdigest())


def state_fingerprint(value, torch):
    def describe(item):
        if torch.is_tensor(item):
            return dict(kind='tensor', **tensor_fingerprint(item))
        if isinstance(item, dict):
            keys = sorted(item, key=lambda key: (type(key).__name__, repr(key)))
            return dict(kind=type(item).__name__, items=[
                [dict(kind=type(key).__name__, value=key), describe(item[key])] for key in keys])
        if isinstance(item, (tuple, list)):
            return dict(kind=type(item).__name__, items=[describe(part) for part in item])
        return dict(kind=type(item).__name__, value=item)
    descriptor = describe(value)
    encoded = json.dumps(descriptor, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return dict(sha256_logical_descriptor=hashlib.sha256(encoded).hexdigest(),
                descriptor=descriptor)


def input_preprocessing_fingerprints(x, edges, labels, graph, binding):
    return dict(schema='graph-curvature-fresh-input-preprocessing-fingerprints-v1',
        verified_input_descriptors=copy.deepcopy(binding),
        raw_features=tensor_fingerprint(x), canonical_edges=tensor_fingerprint(edges),
        compact_source_roles={name: dict(nodes=tensor_fingerprint(pack.nodes),
                                         labels=tensor_fingerprint(pack.labels))
                              for name, pack in labels.items()},
        teacher_input=tensor_fingerprint(graph.teacher_input),
        teacher_edge_index=(tensor_fingerprint(graph.teacher_edge_index)
                            if hasattr(graph, 'teacher_edge_index') else None),
        computed_after_original_preprocessing_before_original_warm=True,
        no_new_forward_trial_RNG_draw_or_label_role=True)


def fresh_warm_fingerprints(checkpoint, checkpoint_path, torch, input_fingerprints):
    raw = checkpoint_path.read_bytes()
    return dict(schema='graph-curvature-fresh-warm-fingerprints-v1',
        checkpoint_file=dict(path=checkpoint_path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
        logical_model=state_fingerprint(checkpoint['model'], torch),
        logical_optimizer=state_fingerprint(checkpoint['optimizer'], torch),
        logical_RNG=state_fingerprint(checkpoint['rng'], torch),
        specification=copy.deepcopy(checkpoint['specification']),
        global_stage=checkpoint['global_stage'], model_training=checkpoint['model_training'],
        warm_metadata=copy.deepcopy(checkpoint['metadata']),
        input_preprocessing_fingerprints=input_fingerprints,
        file_serialization_hash_is_distinct_from_logical_state_fingerprints=True,
        repeatability_or_qualification_claim=False)


def capture_actual_preprocessing(graph, output_path, torch, input_fingerprints):
    """Save the already produced inputs exactly; never recompute preprocessing."""
    teacher_input = graph.teacher_input.detach().cpu().clone()
    teacher_edges = graph.teacher_edge_index.detach().cpu().clone()
    input_identity, edge_identity = tensor_fingerprint(teacher_input), tensor_fingerprint(teacher_edges)
    require(input_identity == input_fingerprints['teacher_input']
            and edge_identity == input_fingerprints['teacher_edge_index'],
            'Preprocessing snapshot differs from the preceding actual-input fingerprint')
    with output_path.open('xb') as stream:
        torch.save(dict(schema='graph-curvature-actual-preprocessed-input-v1',
            teacher_input=teacher_input, teacher_edge_index=teacher_edges,
            teacher_backbone=graph.teacher_backbone,
            preprocessing=copy.deepcopy(graph.preprocessing)), stream)
    digest, size = hashlib.sha256(), 0
    with output_path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            size += len(chunk)
            digest.update(chunk)
    return dict(schema='graph-curvature-preprocessing-snapshot-receipt-v1',
        path=output_path.name, storage_bytes=size, sha256_file=digest.hexdigest(),
        teacher_input=input_identity, teacher_edge_index=edge_identity,
        actual_input_saved_before_prescribed_warm=True,
        preprocessing_recomputed=False, numerical_policy_changed=False,
        scientific_reuse_or_replay_qualification_claim=False)


class InstallWitnessRecorder:
    def __init__(self, driver, selector, adapter):
        self.driver, self.selector, self.adapter = driver, selector, adapter
        self.original_select = driver.select_initializations
        self.original_install = selector.install_head
        require(self.original_select is selector.select_initializations,
                'Driver must import the original bound selector function')
        self.frozen = None
        self.rows = []
        self.selector_calls = 0
        self.capture_wall_seconds = 0.0
        self.capture_process_cpu_seconds = 0.0
        self.restored = False

    def __enter__(self):
        recorder = self

        def observed_select(*args, **kwargs):
            bound = inspect.signature(recorder.original_select).bind(*args, **kwargs)
            values = bound.arguments
            recorder.selector_calls += 1
            require(recorder.selector_calls == 1, 'Exactly one selector call is admitted')
            wall, cpu = time.perf_counter(), time.process_time()
            prototype = values['prototype']
            recorder.frozen = dict(
                prototype_state=recorder.adapter.cpu_copy(prototype.state_dict()),
                optimizer=recorder.adapter.cpu_copy(values['frozen_optimizer']),
                rng=recorder.adapter.cpu_copy(values['native_rng']),
                modes={name: module.training for name, module in prototype.named_modules()},
                backbone=values['backbone'], prototype_object_id=id(prototype))
            recorder.capture_wall_seconds += time.perf_counter()-wall
            recorder.capture_process_cpu_seconds += time.process_time()-cpu
            return recorder.original_select(*args, **kwargs)

        def observed_install(model, slices, backbone):
            # Call the original helper exactly once before observing its result.
            result = recorder.original_install(model, slices, backbone)
            wall, cpu = time.perf_counter(), time.process_time()
            require(recorder.frozen is not None, 'Install must occur inside the observed selector')
            recorder.rows.append(dict(
                serial=len(recorder.rows)+1, model_ref=weakref.ref(model),
                model_object_id=id(model), backbone=backbone,
                head=recorder.selector.HEADS[backbone][0],
                intended_slices=recorder.adapter.cpu_copy(slices),
                installed_model_state=recorder.adapter.cpu_copy(model.state_dict()),
                installed_modes={name: module.training for name, module in model.named_modules()},
                matched_returned_arm=None))
            recorder.capture_wall_seconds += time.perf_counter()-wall
            recorder.capture_process_cpu_seconds += time.process_time()-cpu
            return result

        self.driver.select_initializations = observed_select
        self.selector.install_head = observed_install
        return self

    def __exit__(self, error_type, error, trace):
        # Restore both imported/global bindings even when original execution fails.
        self.driver.select_initializations = self.original_select
        self.selector.install_head = self.original_install
        self.restored = (self.driver.select_initializations is self.original_select
                         and self.selector.install_head is self.original_install)
        return False

    def match(self, model, arm):
        matches = [row for row in self.rows if row['model_ref']() is model]
        require(len(matches) == 1, 'Returned model must match exactly one live install witness')
        row = matches[0]
        require(row['matched_returned_arm'] in (None, arm), 'A witness was reused by another arm')
        row['matched_returned_arm'] = arm
        return row

    def summary(self):
        return dict(schema='graph-curvature-install-witness-capture-v1',
            runtime_helpers_instrumented=['driver.select_initializations', 'selector.install_head'],
            selector_calls=self.selector_calls, installs=len(self.rows),
            hooks_restored=self.restored,
            object_identity='monotonic witness serial plus live weakref identity; never bare id matching',
            no_model_or_optimizer_strong_reference_retained_by_witness=True,
            selector_source_sha256=getattr(self.selector, '__executed_sha256__', None),
            driver_source_sha256=getattr(self.driver, '__executed_sha256__', None),
            original_selector_called_once=self.selector_calls == 1,
            original_installer_called_once_per_install=True,
            added_model_forwards=0, added_Adam_trials=0, added_RNG_draws=0,
            capture_wall_seconds=self.capture_wall_seconds,
            capture_process_cpu_seconds=self.capture_process_cpu_seconds,
            snapshots='CPU detached copies after original installation; actual prototype captured before selection',
            numerical_or_qualification_claim=False)

    def archive(self):
        return dict(schema='graph-curvature-install-witness-tensors-v1', frozen=self.frozen,
            installations=[dict({key: value for key, value in row.items() if key != 'model_ref'},
                object_alive_at_archive=row['model_ref']() is not None) for row in self.rows])


def collect_check(rows, name, function):
    try:
        function()
        row = dict(check=name, passed=True)
    except Exception as error:
        row = dict(check=name, passed=False, error_type=type(error).__name__, error=str(error))
        if hasattr(error, 'details'):
            row['exact_state_difference'] = copy.deepcopy(error.details)
    rows.append(row)
    return row['passed']


def source_custody(outputs, receipt, native, checkpoint, selector, adapter, torch,
                   exact_equal, tensor_storage, recorder):
    require(recorder.frozen is not None and recorder.restored, 'Complete restored instrumentation is required')
    frozen = recorder.frozen
    head = selector.HEADS[frozen['backbone']][0]
    rows, prior_storage = [], []
    donor_storage = tensor_storage(native.state_dict(), torch) | tensor_storage(checkpoint, torch)
    enumeration = []
    collect_check(enumeration, 'original_five_arm_and_trial_enumeration', lambda: require(
        tuple(outputs) == selector.ARMS and receipt['trial_maps_started'] <= 10
        and all(len(slots) == 3 for slots in receipt['candidates'].values()),
        'Original five-arm/nine-slot/trial enumeration differs'))
    for arm in selector.ARMS:
        checks = []
        row = dict(arm=arm, checks=checks, selection=receipt['selection'].get(arm))
        rows.append(row)
        if arm not in outputs:
            collect_check(checks, 'returned_arm_present', lambda: require(False, 'Returned arm is absent'))
            row['passed'] = False
            continue
        output = outputs[arm]
        witness = None
        try:
            witness = recorder.match(output['model'], arm)
            row['witness_serial'] = witness['serial']
            checks.append(dict(check='live_object_install_witness_identity', passed=True))
        except Exception as error:
            checks.append(dict(check='live_object_install_witness_identity', passed=False,
                               error_type=type(error).__name__, error=str(error)))
        state = output['model'].state_dict()
        if witness is not None:
            collect_check(checks, 'installed_head_equals_intended_source_slices', lambda: exact_equal(
                witness['intended_slices'], witness['installed_model_state'][head], torch,
                path=f"source_custody[{arm!r}].installed_head_vs_intended"))
            collect_check(checks, 'returned_head_equals_own_intended_source_slices', lambda: exact_equal(
                witness['intended_slices'], state[head], torch,
                path=f"source_custody[{arm!r}].returned_head_vs_intended"))
            collect_check(checks, 'returned_full_state_equals_own_post_install_witness', lambda: exact_equal(
                witness['installed_model_state'], state, torch,
                path=f"source_custody[{arm!r}].returned_state_vs_post_install"))
        collect_check(checks, 'non_head_state_equals_actual_frozen_prototype', lambda: exact_equal(
            {key: value for key, value in frozen['prototype_state'].items() if key != head},
            {key: value for key, value in state.items() if key != head}, torch,
            path=f"source_custody[{arm!r}].non_head_vs_frozen_prototype"))
        collect_check(checks, 'optimizer_equals_actual_frozen_selector_optimizer', lambda: exact_equal(
            frozen['optimizer'], adapter.named_optimizer_snapshot(output['model'], output['optimizer']), torch,
            path=f"source_custody[{arm!r}].optimizer"))
        collect_check(checks, 'RNG_equals_actual_selector_input', lambda: exact_equal(
            frozen['rng'], output['rng'], torch, path=f"source_custody[{arm!r}].rng_vs_selector_input"))
        collect_check(checks, 'RNG_equals_original_warm_checkpoint', lambda: exact_equal(
            checkpoint['rng'], output['rng'], torch, path=f"source_custody[{arm!r}].rng_vs_checkpoint"))
        collect_check(checks, 'modes_equal_actual_frozen_prototype', lambda: require(
            {name: module.training for name, module in output['model'].named_modules()} == frozen['modes'],
            'Returned module modes differ from actual frozen prototype'))
        pointer = (tensor_storage(state, torch)
                   | tensor_storage(list(output['optimizer'].state.values()), torch)
                   | tensor_storage(output['rng'], torch))
        collect_check(checks, 'storage_disjoint_from_native_and_checkpoint', lambda: require(
            not pointer & donor_storage, 'Returned state aliases native donor/checkpoint storage'))
        collect_check(checks, 'storage_disjoint_from_every_previous_arm', lambda: require(
            all(not pointer & prior for prior in prior_storage), 'Returned state aliases another arm'))
        prior_storage.append(pointer)
        row['passed'] = all(check['passed'] for check in checks)
    return dict(schema='graph-curvature-source-custody-diagnostics-v1',
        diagnostic_only=True, qualification_promoted=False,
        passed=all(check['passed'] for check in enumeration) and all(row['passed'] for row in rows),
        enumeration=enumeration, arms=rows, completed_arms=len(rows),
        capture_summary=recorder.summary())


def independent_reconstruction(outputs, receipt, common, bases, before, frozen,
        modes, args, logits_fn, selector, adapter, constants, torch, exact_equal,
        captured_prototype):
    head = selector.HEADS[captured_prototype['backbone']][0]
    center = common[0]
    rows, expected_heads = [], {}
    prototype_checks = []
    collect_check(prototype_checks, 'actual_vs_independent_identity_prototype', lambda: exact_equal(
        captured_prototype['prototype_state'], before, torch, path='independent_prototype.model'))
    collect_check(prototype_checks, 'actual_vs_independent_frozen_optimizer', lambda: exact_equal(
        captured_prototype['optimizer'], frozen, torch, path='independent_prototype.optimizer'))
    collect_check(prototype_checks, 'actual_vs_independent_prototype_modes', lambda: require(
        captured_prototype['modes'] == modes, 'Actual/independent prototype modes differ'))
    for arm in selector.ARMS:
        checks = []
        row = dict(arm=arm, checks=checks, selection=receipt['selection'].get(arm))
        rows.append(row)
        if arm not in outputs:
            collect_check(checks, 'returned_arm_present', lambda: require(False, 'Returned arm is absent'))
            row['passed'] = False
            continue
        output = outputs[arm]
        expected = common
        pair = receipt['selection'][arm]['pair']
        if pair is not None:
            span, index = pair
            radius = receipt['candidates'][span][index]['radius']
            a, b = [bases[span][j].to(center.dtype)*radius for j in selector.PAIRS[index]]
            expected = center[None, :]+torch.stack((a, -a, b, -b))
        expected_heads[arm] = adapter.cpu_copy(expected)
        for name, value in output['model'].state_dict().items():
            collect_check(checks, 'original_exact_model_tensor:'+name,
                lambda name=name, value=value: exact_equal(
                    expected if name == head else before[name], value, torch,
                    path=f"independent_reconstruction[{arm!r}].model[{name!r}]"))
        collect_check(checks, 'original_exact_optimizer', lambda: exact_equal(
            frozen, adapter.named_optimizer_snapshot(output['model'], output['optimizer']), torch,
            path=f"independent_reconstruction[{arm!r}].optimizer"))
        collect_check(checks, 'original_modes', lambda: require(
            {name: module.training for name, module in output['model'].named_modules()} == modes,
            'Returned pretrial module modes differ'))
        # This witness diagnostic adds no per-arm model evaluation. The original
        # mean-logit guard belongs to complete qualification and is not claimed.
        row['not_executed_checks'] = ['original_mean_logit_guard']
        row['passed'] = all(check['passed'] for check in checks)
    return (dict(schema='graph-curvature-independent-reconstruction-diagnostics-v1',
        diagnostic_only=True, qualification_promoted=False,
        passed=all(check['passed'] for check in prototype_checks) and all(row['passed'] for row in rows),
        prototype_checks=prototype_checks, arms=rows, completed_arms=len(rows),
        strict_predicate='Original dtype/shape and torch.equal; all mismatches remain failures',
        original_mean_logit_guard_executed=False,
        complete_original_qualification_check_set_executed=False,
        added_per_arm_model_forwards=0), expected_heads)
