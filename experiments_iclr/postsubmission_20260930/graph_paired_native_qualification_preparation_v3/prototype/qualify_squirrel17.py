"""Prepared only; author ran no native, remote, array or GPU operation.

Exact frozen Squirrel17 source/AD/paired-installation qualification. Original
inputs are read-only; output is one fresh local run directory. No training,
continuation, old driver phase, scheduler, optimizer step or checkpoint save.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
PACKET = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(record):
    path = Path(record['path'])
    require(path.is_file() and sha(path) == record['sha256'], 'Fingerprint mismatch: '+str(path))
    if 'bytes' in record:
        require(path.stat().st_size == record['bytes'], 'Byte length mismatch: '+str(path))
    return path


def verify_originals(records):
    receipts = []
    for row in records:
        path = Path(row['path'])
        if path.suffix in ('.npy', '.npz', '.pt', '.pth'):
            require(row.get('kind') in ('feature_descriptor', 'canonical_edge_descriptor',
                    'compact_TRAIN_label_descriptor', 'unlabeled_role_ID_array', 'remote_warm_checkpoint'),
                    'Unbound numeric input category')
        if '/labels/' in str(path):
            require(row.get('kind') == 'compact_TRAIN_label_descriptor'
                    and path.name == 'seed17_split0_train.npz', 'Only exact TRAIN label bytes may be verified')
        if row.get('kind') == 'unlabeled_role_ID_array':
            require(path.name in ('train_nodes.npy', 'validation_nodes.npy', 'pool_nodes.npy'),
                    'Only the three declared unlabeled role-ID arrays may be verified')
        receipts.append(dict(path=str(verify(row)), sha256=row['sha256'], preserved=True))
    return receipts


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resource_decision(free_bytes, evidence, visible_matches):
    reasons = []
    if not visible_matches:
        reasons.append('process_visible_GPU_UUID_does_not_match_fixed_UUID')
    if free_bytes < evidence['required_free_bytes']:
        reasons.append('insufficient_current_free_device_memory')
    if evidence['planned_wall_seconds'] > evidence['wall_budget_seconds']:
        reasons.append('planning_estimate_exceeds_qualification_budget')
    return dict(status='resource_deferred' if reasons else 'resource_preflight_passed',
                reasons=reasons, free_bytes=free_bytes,
                required_free_bytes=evidence['required_free_bytes'],
                planned_wall_seconds=evidence['planned_wall_seconds'],
                wall_budget_seconds=evidence['wall_budget_seconds'],
                scientific_merit_assessed=False)


def resource_failure_in_chain(error):
    """Find allocator failures through explicit/implicit causes without Torch."""
    pending = [(error, 'root')]
    seen = set()
    while pending:
        current, link = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        reason = None
        if isinstance(current, MemoryError):
            reason = 'MemoryError'
        elif any(cls.__name__ == 'OutOfMemoryError' for cls in type(current).__mro__):
            reason = 'OutOfMemoryError'
        elif 'cuda out of memory' in str(current).lower():
            reason = 'CUDA_out_of_memory_message'
        if reason is not None:
            return dict(reason=reason, exception_type=type(current).__name__,
                        exception_message=str(current), chain_link=link)
        # Cause is examined first; context is also retained when explicitly suppressed.
        if current.__context__ is not None:
            pending.append((current.__context__, link+'.__context__'))
        if current.__cause__ is not None:
            pending.append((current.__cause__, link+'.__cause__'))
    return None


def observed_forward(closure, theta, counts, sink, resource_failures):
    counts['calls_started'] += 1
    sink('closure_forward_started', dict(call=counts['calls_started']))
    try:
        value = closure(theta)
    except Exception as error:
        counts['calls_failed'] += 1
        sink('closure_forward_failed', dict(call=counts['calls_started'],
            error_type=type(error).__name__, error_message=str(error)))
        failure = resource_failure_in_chain(error)
        if failure is not None:
            resource_failures.append(dict(call=counts['calls_started'], exception=failure))
        raise
    counts['calls_completed'] += 1
    return value


def paired_return_status(slices, resource_failures):
    if resource_failures:
        return 'resource_deferred'
    return 'joint_failure' if slices is None else 'joint_accepted'


def resource_preflight(bound, evidence):
    uuid = bound['expected_gpu_uuid']
    command = ['nvidia-smi', '--id='+uuid,
               '--query-gpu=uuid,name,memory.total,memory.free', '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
        rows = list(csv.reader(result.stdout.strip().splitlines()))
        require(len(rows) == 1 and len(rows[0]) == 4, 'Need one fixed-GPU resource row')
        actual_uuid, name, total_mib, free_mib = [item.strip() for item in rows[0]]
        require(actual_uuid == uuid, 'GPU UUID mismatch')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        decision = resource_decision(int(free_mib)*2**20, evidence, visible == uuid)
        decision.update(uuid=actual_uuid, name=name, total_bytes=int(total_mib)*2**20,
                        cuda_visible_devices=visible, command=command)
        return decision
    except Exception as error:
        return dict(status='resource_deferred', reasons=['fixed_GPU_resource_probe_unavailable'],
                    uuid=uuid, command=command, error_type=type(error).__name__, error_message=str(error),
                    scientific_merit_assessed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    require(args.run_name not in ('', '.', '..') and Path(args.run_name).name == args.run_name,
            'Use a simple new run name')
    out = PACKET/'runs'/args.run_name
    out.mkdir(parents=True, exist_ok=False)
    bound = json.loads((PACKET/'BOUND_INPUTS.json').read_text())
    evidence = json.loads((PACKET/'RESOURCE_EVIDENCE.json').read_text())
    result = dict(schema='Squirrel17-native-paired-source-qualification-v3',
                  status='preparing', source_only_qualification=True, native_invocation_performed=False,
                  validation_or_test_labels_loaded=False, validation_or_test_scores_computed=False,
                  training_or_continuation_performed=False, optimizer_steps=0,
                  checkpoint_or_RNG_files_written=False, scientific_merit_assessed=False,
                  VJP_JVP_counters_mean='attempted primitive calls; completed counts require successful receipt',
                  precision_AD_maximum_operations=dict(vjp_forwards=1, vjp_calls=4, jvp_calls=3,
                                                       finite_difference_forward_calls=12),
                  original_preservation_before=None, original_preservation_after=None)
    started = time.perf_counter()
    driver = rt = ledger = None
    closure_counts = dict(calls_started=0, calls_completed=0, calls_failed=0)
    observed_resource_failures = []
    outside_closure_calls = dict(containers_started=0, containers_completed=0,
                                 scheduled_member_forwards=0)
    stdlib_costs = []

    def stdlib_stage(name, callback):
        begin = time.perf_counter()
        cost = dict(operation=name, status='failed', GPU_memory_measured=False)
        try:
            value = callback()
            cost['status'] = 'completed'
            return value
        except Exception as error:
            cost.update(error_type=type(error).__name__, error_message=str(error))
            raise
        finally:
            cost.update(seconds=time.perf_counter()-begin,
                        process_maxrss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            stdlib_costs.append(cost)

    exit_code = 0
    try:
        manifest_bytes = (PACKET/'MANIFEST.json').read_bytes()
        manifest, seal = json.loads(manifest_bytes), json.loads((PACKET/'SEAL.json').read_text())
        require(hashlib.sha256(manifest_bytes).hexdigest() == seal['manifest_sha256'], 'Preparation seal mismatch')
        for row in manifest['payload']:
            verify(dict(row, path=str(PACKET/row['path'])))
        result['original_preservation_before'] = stdlib_stage('original_fingerprints_before', lambda:
            verify_originals(bound['original_records']))
        result['resource_preflight'] = stdlib_stage('fixed_UUID_outcome_free_resource_preflight', lambda:
            resource_preflight(bound, evidence))
        if result['resource_preflight']['status'] == 'resource_deferred':
            result['status'] = 'resource_deferred'
        elif args.preflight_only:
            result['status'] = 'resource_preflight_complete'
        else:
            root = Path(bound['canonical_research_root'])
            active = root/'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v3_precision'
            driver = load('native_assay_bound_R17_precision_driver', active/'prototype/graph_init_driver.py')
            ledger = driver.Ledger(out)
            context = bound['context']
            source_directory = stdlib_stage('pinned_v3_precision_source_guard', lambda: driver.source_guard(context))
            rt = stdlib_stage('pinned_v3_precision_runtime', lambda: driver.load_runtime(context, source_directory))

            def measured(name, callback):
                return ledger.measured(rt, name, callback)

            def model_forward(name, callback, members, containers=1):
                outside_closure_calls['containers_started'] += containers
                outside_closure_calls['scheduled_member_forwards'] += members
                ledger.sink('model_forward_started', dict(name=name, scheduled_members=members))
                value = measured(name, callback)
                outside_closure_calls['containers_completed'] += containers
                ledger.sink('model_forward_completed', dict(name=name, scheduled_members=members))
                return value

            graph, edges, train, validation = measured('bound_source_inputs_TRAIN_only', lambda:
                driver.source_inputs(rt, context, ledger, validation=False))
            require(validation is None, 'Validation labels unexpectedly loaded')
            require(graph.preprocessing == bound['expected_preprocessing'], 'Warm preprocessing changed')
            require(graph.teacher_input.shape[0] == bound['nodes'] and graph.classes == bound['classes']
                    and graph.teacher_input.dtype == rt.torch.float32, 'Full homogeneous FP32 graph mismatch')
            checkpoint = measured('read_only_bound_warm_checkpoint_load', lambda: rt.torch.load(
                verify(bound['warm_checkpoint']), map_location=rt.device, weights_only=True))
            require(checkpoint['specification'] == bound['expected_specification'], 'Warm specification mismatch')
            require(all(not value.is_floating_point() or value.dtype == rt.torch.float32
                        for value in checkpoint['model'].values()), 'Native warm weights must remain FP32')
            result['native_invocation_performed'] = True
            native, optimizer = measured('pinned_native_restore_no_updates', lambda:
                rt.integration.restore_native(rt.adapter, checkpoint, rt.device))
            del optimizer, checkpoint
            native.eval()
            donor_snapshot = measured('in_memory_warm_donor_snapshot', lambda:
                {name: value.detach().clone() for name, value in native.state_dict().items()})
            k1 = measured('fresh_K1_common_clone', lambda: rt.integration.clone_boundary(native, rt.boundary, 1))
            k4_identity = measured('fresh_K4_identity_clone', lambda:
                rt.integration.clone_boundary(native, rt.boundary, 4))
            # Pinned audit performs native once, K1 once, and four K4 member passes.
            result['identity_audit'] = model_forward('native_K1_K4_identity_audit', lambda:
                rt.integration.identity_logits_audit(native, k1, k4_identity, graph), 6, containers=3)
            del k4_identity
            k1.eval()
            theta0, closure, binding = measured('pinned_full_output_common_binding', lambda:
                rt.method.bind_common_model(k1, 'PolyFormer-Mono',
                    rt.integration.raw_arguments(graph, 'polyformer_mono'), lambda value: value[0]))
            require(theta0.dtype == rt.torch.float32 and theta0.numel() == bound['private_dimensions'],
                    'Expected exact FP32 512-coordinate private slice')

            def observed(theta):
                return observed_forward(closure, theta, closure_counts, ledger.sink, observed_resource_failures)

            with rt.torch.no_grad():
                common_logits = measured('full_output_order_shape_finiteness', lambda: observed(theta0))
            require(tuple(common_logits.shape) == (bound['nodes'], bound['classes'])
                    and common_logits.dtype == rt.torch.float32
                    and bool(rt.torch.isfinite(common_logits).all()), 'Homogeneous full-node output mismatch')
            targets = rt.torch.arange(bound['nodes'], dtype=rt.torch.int64, device=rt.device)
            result['output_contract'] = dict(shape=list(common_logits.shape), canonical_node_order=True,
                exact_target_coverage=True, TRAIN_rows_sha256=hashlib.sha256(
                    train.nodes.detach().cpu().numpy().tobytes()).hexdigest(), TRAIN_count=train.nodes.numel(),
                factor_binding=binding, model_logits_gradients_dtype='FP32')
            precision = load('native_assay_bound_precision_qualification', active/'prototype/precision_qualification.py')
            before = closure_counts['calls_started']
            result['active_precision_AD'] = measured('active_actual_warm_precision_AD', lambda:
                precision.qualify_gradient_interface(observed, theta0, train.nodes, train.labels,
                                                       bound['preflight_seed']))
            require(closure_counts['calls_started']-before == 16, 'Active AD closure call accounting mismatch')
            require(result['active_precision_AD']['passed'], 'Actual full-output VJP/JVP qualification failed')
            S = measured('canonical_normalized_S', lambda: rt.method.symmetric_normalized_adjacency(
                bound['nodes'], edges, theta0.dtype, theta0.device))
            S_permuted, permutation = measured('frozen_topology_permutation', lambda:
                rt.method.permute_topology_nodes(S, bound['topology_permutation_seed']))
            expected = rt.torch.sparse_coo_tensor(permutation[S.indices()], S.values(), S.shape,
                                                dtype=S.dtype, device=S.device).coalesce()
            require(rt.torch.equal(permutation.sort().values, targets)
                    and rt.torch.equal(expected.indices(), S_permuted.indices())
                    and rt.torch.equal(expected.values(), S_permuted.values()), 'Pi S Pi^T equality failed')
            result['topology_control'] = dict(seed=bound['topology_permutation_seed'], genuine_Pi_S_PiT=True,
                feature_label_order_unchanged=True, permutation_sha256=hashlib.sha256(
                    permutation.detach().cpu().numpy().tobytes()).hexdigest(),
                permutation=permutation.detach().cpu().tolist())
            paired = load('native_assay_sealed_paired_helper', root/
                'graph_full_node_cotangent_paired_alpha_v1/prototype/paired_shared_alpha_initializer.py')
            before = closure_counts['calls_started']
            slices, paired_report = measured('sealed_actual_four_arm_shared_alpha', lambda:
                paired.initialize_paired_four_arms(observed, theta0, S, S_permuted, targets,
                    train.nodes, train.labels, homogeneous_full_node_outputs=True))
            result['paired'] = paired_report
            require(closure_counts['calls_started']-before == paired_report['vjp_forwards']
                    +paired_report['jvp_calls']+paired_report['line_search_forward_calls'],
                    'Actual paired closure call accounting mismatch')
            result['paired_graph_sparse_products_semantics'] = 'scheduled products; completed on successful geometry'
            result['installation'] = {}
            paired_status = paired_return_status(slices, observed_resource_failures)
            if paired_status == 'resource_deferred':
                exit_code = 1
                result.update(status='resource_deferred', resource_failure_during_qualification=True,
                              resource_failure_exception=observed_resource_failures[0]['exception'])
            elif slices is None:
                result['status'] = 'joint_failure'
            else:
                for arm in paired.ARM_NAMES:
                    fresh = measured('fresh_K4_clone_'+arm, lambda:
                        rt.integration.clone_boundary(native, rt.boundary, 4)).eval()
                    with rt.torch.no_grad():
                        warm_logits = model_forward('fresh_K4_warm_equality_'+arm, lambda:
                            fresh(*rt.integration.raw_arguments(graph, 'polyformer_mono')), 4)
                        warm_equal = rt.integration.difference(warm_logits, common_logits[None].repeat(4, 1, 1),
                                                               1e-6, 1e-5)
                        require(warm_equal['passed'], 'Fresh K4 common warm equality failed')
                        measured('private_slice_install_'+arm, lambda: rt.method.install_factor_slices(
                            fresh, slices[arm], 'PolyFormer-Mono'))
                        actual = model_forward('installed_K4_forward_'+arm, lambda:
                            fresh(*rt.integration.raw_arguments(graph, 'polyformer_mono')), 4)
                        expected_logits = measured('four_installed_closure_references_'+arm, lambda:
                            rt.torch.stack([observed(row) for row in slices[arm]]))
                        equality = rt.integration.difference(actual, expected_logits, 1e-6, 1e-5)
                    result['installation'][arm] = dict(fresh_warm_equality=warm_equal, installed_equality=equality)
                    ledger.sink('arm_installation_equality', dict(arm=arm, receipt=result['installation'][arm]))
                    require(equality['passed'], 'Installed arm differs from exact common closure')
                    del fresh, warm_logits, actual, expected_logits
                result['status'] = 'native_paired_source_qualified'
            require(measured('in_memory_warm_donor_preservation', lambda:
                all(rt.torch.equal(value, donor_snapshot[name]) for name, value in native.state_dict().items())),
                'In-memory native warm donor changed')
            result['warm_donor_state_unchanged'] = True
    except Exception as error:
        exit_code = 1
        result.update(status='qualification_failed', error_type=type(error).__name__,
                      error_message=str(error), traceback=traceback.format_exc())
        resource_failure = resource_failure_in_chain(error)
        if resource_failure is not None:
            result.update(status='resource_deferred', resource_failure_during_qualification=True,
                          resource_failure_exception=resource_failure)
        if hasattr(error, 'report'):
            result['paired_geometry_abort'] = error.report
            result['paired_graph_sparse_products_semantics'] = (
                'scheduled products only; completed product count unknown for an aborted bank')
    finally:
        try:
            result['original_preservation_after'] = stdlib_stage('original_fingerprints_after', lambda:
                verify_originals(bound['original_records']))
        except Exception as error:
            exit_code = 1
            result.update(status='original_fingerprint_verification_failed', preservation_error=str(error))
        result.update(closure_forward_calls=closure_counts, outside_closure_forward_calls=outside_closure_calls,
                      observed_resource_failures=observed_resource_failures,
                      whole_process_wall_seconds=time.perf_counter()-started,
                      process_maxrss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      scheduled_member_counter_on_container_abort_is_not_completed_pass_count=True)
        result['stdlib_operation_costs'] = stdlib_costs
        if ledger is not None:
            result['operation_costs'] = ledger.costs
        # The source driver's serializer handles nonfinite diagnostics, never arrays.
        if driver is not None:
            driver.write_json(out/'QUALIFICATION.json', result)
        else:
            with (out/'QUALIFICATION.json').open('x') as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
                handle.write('\n')
    print(json.dumps(dict(status=result['status'], output=str(out), exit_code=exit_code)))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
