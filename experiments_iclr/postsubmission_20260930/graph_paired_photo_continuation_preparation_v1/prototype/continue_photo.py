"""Prepared Photo paired follow-up; author ran synthetic stdlib interfaces only."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
PACKET = Path(__file__).resolve().parents[1]
ARMS = ('common_only', 'train_remasked', 'full_node', 'full_node_permuted')
CONTROLS = ('train_remasked', 'common_only', 'full_node_permuted')
PAIRS = ((17, 0), (29, 1), (43, 2))


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


def verify_inputs(records, allow_validation=False):
    receipts = []
    train_names = {f'seed{seed}_split{split}_train.npz' for seed, split in PAIRS}
    val_names = {f'seed{seed}_split{split}_validation.npz' for seed, split in PAIRS}
    for row in records:
        path, kind = Path(row['path']), row.get('kind')
        if '/labels/' in str(path):
            require((kind == 'TRAIN_labels' and path.name in train_names) or
                    (allow_validation and kind == 'validation_labels' and path.name in val_names),
                    'Only declared TRAIN or admitted continuation-validation label bytes')
        if path.suffix in ('.npy', '.npz', '.pt', '.pth'):
            require(kind in ('features', 'canonical_edges', 'role_ID', 'warm_checkpoint', 'TRAIN_labels')
                    or (allow_validation and kind == 'validation_labels'), 'Unbound numeric input')
        if kind == 'role_ID':
            require(path.name in ('train_nodes.npy', 'validation_nodes.npy', 'pool_nodes.npy'), 'Role IDs only')
        receipts.append(dict(path=str(verify(row)), sha256=row['sha256'], preserved=True))
    return receipts


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def study_guard(frozen):
    require(tuple(frozen['arms']) == ARMS and tuple(map(tuple, frozen['seed_split_pairs'])) == PAIRS,
            'Frozen arms or paired blocks differ')
    require(frozen['fit_slots'] == 12 and frozen['final_labels_closed'] is True
            and frozen['Squirrel_runs_authorized_by_this_driver'] is False, 'Photo follow-up scope differs')
    require(frozen['continuation'] == dict(global_cap=950, patience=None, midpoint=450), 'Continuation schedule differs')
    require((frozen['graph'], frozen['backbone'], frozen['configuration'], frozen['nodes'],
             frozen['features'], frozen['classes'], frozen['private_dimensions']) ==
            ('Photo', 'polynormer_r', 0, 7650, 745, 8, 1024), 'Exact Photo dimensions differ')
    require(frozen['selector'] == 'epoch0 and every update; source-validation predictor NLL; earliest strict minimum'
            and frozen['primary_pooling'] == 'softmax(mean raw member logits)', 'Selector/pooling differs')
    require(frozen['transport'] == 'named_alias_adam_private_bias_coordinate_transport_v1', 'Transport differs')
    require(frozen['mode'] == 'staged_validation_gate' and frozen['root_execution_admission_required'] is True,
            'Immutable root mode/admission differs')
    require(frozen['training_loss'] == 'arithmetic mean member CE'
            and frozen['same_post_warm_RNG_restored_after_initialization'] is True,
            'Frozen training/RNG contract differs')
    gate = frozen['followup_gate_custody']
    require(gate['root_confirmed_complete'] is True and gate['author_inspected_Squirrel_outcomes'] is False
            and gate['gate_is_quality_evidence'] is False
            and gate['requires_root_execution_release_reassertion'] is True, 'Follow-up gate custody differs')
    require([(row['seed'], row['source_split_index']) for row in frozen['cells']] == list(PAIRS),
            'Exactly three ordered paired blocks required')
    for cell in frozen['cells']:
        ctx = cell['context']
        require((ctx['graph'], ctx['backbone'], ctx['config'], ctx['seed'], ctx['source_split_index']) ==
                ('Photo', 'polynormer_r', 0, cell['seed'], cell['source_split_index']), 'Cell differs')
        require(set(ctx['source_labels']) == {'train', 'validation'}, 'Exact compact label descriptors required')


def admission_guard(frozen, release, manifest_sha256, run_name, execute=True):
    choice = json.loads(verify(frozen['mode_donor_freeze']).read_text())
    require(choice['mode'] == frozen['mode'] == 'staged_validation_gate'
            and choice['no_choice_changes_after_first_prospective_study_operation'] is True,
            'Root choice cutoff differs')
    require(choice['donors'] == frozen['all_six_donors'] and len(choice['donors']) == 6,
            'Six immutable donor identities differ')
    require(tuple(choice['paired_arm_order']) == ARMS and choice['continuation']['Photo'] == frozen['continuation']
            and tuple(choice['comparison_arm_order']) == ('full_node',)+CONTROLS
            and choice['continuation']['selector'] == frozen['selector']
            and choice['continuation']['training_loss'] == frozen['training_loss']
            and choice['continuation']['primary_pooling'] == frozen['primary_pooling']
            and choice['continuation']['optimizer_transport'] == frozen['transport']
            and choice['continuation']['same_post_warm_RNG_restored_after_initialization'] is True,
            'Root arm/trajectory/selector/training/RNG differs')
    binding = next(row for row in frozen['original_records'] if row['path'].endswith('/'+choice['binding_source']))
    require(binding['sha256'] == choice['binding_sha256'], 'Root six-donor binding source differs')
    for cell in frozen['cells']:
        donor = next(row for row in choice['donors'] if row['graph']=='Photo' and row['seed']==cell['seed'])
        require(donor['source_split_index'] == cell['source_split_index']
                and donor['warm_checkpoint']['sha256'] == cell['warm_checkpoint']['sha256']
                and Path(cell['warm_checkpoint']['path']) == Path(frozen['canonical_research_root'])/donor['warm_checkpoint']['path'],
                'Execution donor differs from root cutoff')
    require(release['graph'] == 'Photo' and release['run_name'] == run_name
            and release['prepared_manifest_sha256'] == manifest_sha256
            and release['mode_donor_freeze_sha256'] == frozen['mode_donor_freeze']['sha256'],
            'Root must independently review and admit the sealed source and frozen choices')
    require(not execute or release['execution_authorized'] is True, 'Native execution release required')
    gate = release['Squirrel_gate_assertion']
    require(gate['complete'] is True and gate['all3_blocks_complete'] is True
            and gate['all12_arm_terminals'] is True and gate['all12_selected_fits'] is True
            and gate['development_trigger_satisfied'] is True and gate['is_quality_evidence'] is False,
            'Root must reassert the completed full Squirrel gate; it is not quality evidence')
    evidence = release['Photo_resource_evidence']
    require(set(evidence) == {'required_free_bytes', 'planned_wall_seconds', 'wall_budget_seconds'}
            and all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                    for value in evidence.values()), 'Root Photo resource plan must contain positive integers')
    uuid = release['expected_gpu_uuid']
    require(isinstance(uuid, str) and uuid.startswith('GPU-') and len(uuid) > 4,
            'Root must bind one fixed GPU UUID before probing resources')
    return dict(mode=choice['mode'], six_donors=choice['donors'], prepared_manifest_sha256=manifest_sha256,
                mode_donor_freeze_sha256=frozen['mode_donor_freeze']['sha256'], run_name=run_name,
                Squirrel_gate_assertion=gate, expected_gpu_uuid=uuid, Photo_resource_evidence=evidence,
                gate_is_not_quality_evidence=True, author_inspected_Squirrel_outcomes=False)


def followup_comparison(blocks):
    require([(row['seed'], row['source_split_index']) for row in blocks] == list(PAIRS), 'Incomplete block list')
    terminal = all(set(row['fits']) == set(ARMS) and all(fit['status'] != 'pending'
                   for fit in row['fits'].values()) for row in blocks)
    selected = terminal and all(row['status'] == 'completed' and
                   all(fit['status'] == 'selected' for fit in row['fits'].values()) for row in blocks)
    if not selected:
        return dict(status='unevaluable', all12_arm_terminals=terminal, all12_selected_fits=False,
                    successful_subset_scored=False, final_labels_closed=True)
    means = {arm: sum(row['fits'][arm]['selection']['primary_validation_nll'] for row in blocks)/3
             for arm in ARMS}
    require(all(math.isfinite(value) for value in means.values()), 'Finite validation predictor NLL required')
    deltas = {arm: means['full_node']-means[arm] for arm in CONTROLS}
    return dict(status='Photo_followup_comparison_complete', all12_arm_terminals=True, all12_selected_fits=True,
                validation_macro_NLL=means, full_minus_control_NLL=deltas,
                successful_subset_scored=False, comparison_is_not_superiority=True, final_labels_closed=True,
                independent_heldout_confirmation=False)


def blocked_fits(status, reason):
    return {arm: dict(status=status, attempted=False, reason=reason) for arm in ARMS}


def run_block(driver, rt, shared, paired, precision, cell, frozen, out, validation_opened):
    out.mkdir(exist_ok=False)
    ledger = driver.Ledger(out)
    result = dict(seed=cell['seed'], source_split_index=cell['source_split_index'], status='preparing',
                  fits=blocked_fits('pending', 'not_attempted'), initializations={}, validation_labels_loaded=False,
                  final_labels_loaded=False, donor_unchanged=False)
    counts = dict(calls_started=0, calls_completed=0, calls_failed=0)
    resources = []
    block_started = time.perf_counter()
    native = checkpoint = donor = None
    try:
        ctx = cell['context']
        train_context = dict(ctx, source_labels={'train': ctx['source_labels']['train']})
        graph, edges, train, validation = ledger.measured(rt, 'TRAIN_only_inputs', lambda:
            driver.source_inputs(rt, train_context, ledger, validation=False))
        require(validation is None and graph.preprocessing['identity'] == cell['preprocessing_identity'],
                'TRAIN-only/preprocessing contract differs')
        checkpoint = ledger.measured(rt, 'read_only_exact_native_warm', lambda:
            rt.torch.load(verify(cell['warm_checkpoint']), map_location=rt.device, weights_only=True))
        require(checkpoint['specification'] == rt.adapter.specification('polynormer_r', 'single_author', 0,
                cell['seed']), 'Exact warm specification differs')
        require(checkpoint['global_stage'] is True, 'Photo donor must already be in global stage')
        require(all(not value.is_floating_point() or value.dtype == rt.torch.float32
                    for value in checkpoint['model'].values()), 'Warm weights must be FP32')
        native, restored_optimizer = ledger.measured(rt, 'restore_native_donor', lambda:
            rt.integration.restore_native(rt.adapter, checkpoint, rt.device))
        del restored_optimizer
        native.eval()
        donor = {name: value.detach().clone() for name, value in native.state_dict().items()}
        result['optimizer_equivalence'] = ledger.measured(rt, 'actual_warm_disposable_Adam_equivalence', lambda:
            rt.integration.optimizer_equivalence_audit(native, checkpoint['optimizer'], rt.boundary,
                                                       graph, train, ledger.sink))
        result['optimizer_equivalence_scope'] = frozen['optimizer_equivalence_scope']
        k1 = ledger.measured(rt, 'fresh_K1', lambda: rt.integration.clone_boundary(native, rt.boundary, 1)).eval()
        k4 = ledger.measured(rt, 'fresh_identity_K4', lambda: rt.integration.clone_boundary(native, rt.boundary, 4)).eval()
        result['identity'] = ledger.measured(rt, 'native_K1_K4_identity', lambda:
            rt.integration.identity_logits_audit(native, k1, k4, graph))
        del k4
        theta0, closure, binding = ledger.measured(rt, 'full_output_binding', lambda:
            rt.method.bind_common_model(k1, 'Polynormer-r',
                rt.integration.raw_arguments(graph, 'polynormer_r'), lambda value: value[0]))
        require(theta0.dtype == rt.torch.float32 and theta0.numel() == frozen['private_dimensions'],
                'Exact FP32 private slice differs')
        def observed(theta):
            return shared.observed_forward(closure, theta, counts, ledger.sink, resources)
        with rt.torch.no_grad():
            common = ledger.measured(rt, 'full_output_contract', lambda: observed(theta0))
        require(tuple(common.shape) == (frozen['nodes'], frozen['classes']) and common.dtype == rt.torch.float32
                and bool(rt.torch.isfinite(common).all()), 'Full output contract differs')
        targets = rt.torch.arange(frozen['nodes'], dtype=rt.torch.int64, device=rt.device)
        before = counts['calls_started']
        result['AD'] = ledger.measured(rt, 'actual_warm_precision_AD', lambda:
            precision.qualify_gradient_interface(observed, theta0, train.nodes, train.labels, cell['seed']+90000))
        require(result['AD']['passed'] and counts['calls_started']-before == 16, 'Actual AD/count mismatch')
        S = ledger.measured(rt, 'canonical_normalized_S', lambda:
            rt.method.symmetric_normalized_adjacency(frozen['nodes'], edges, theta0.dtype, theta0.device))
        permuted, permutation = ledger.measured(rt, 'frozen_permutation', lambda:
            rt.method.permute_topology_nodes(S, cell['seed']+80000))
        expected = rt.torch.sparse_coo_tensor(permutation[S.indices()], S.values(), S.shape,
                                            dtype=S.dtype, device=S.device).coalesce()
        require(rt.torch.equal(permutation.sort().values, targets) and
                rt.torch.equal(expected.indices(), permuted.indices()) and
                rt.torch.equal(expected.values(), permuted.values()), 'Exact Pi S PiT differs')
        result['topology'] = dict(seed=cell['seed']+80000, permutation=permutation.detach().cpu().tolist(),
                                 genuine_Pi_S_PiT=True, feature_label_order_unchanged=True)
        result['binding'] = binding
        before = counts['calls_started']
        slices, report = ledger.measured(rt, 'sealed_paired_shared_alpha', lambda:
            paired.initialize_paired_four_arms(observed, theta0, S, permuted, targets, train.nodes,
                train.labels, homogeneous_full_node_outputs=True))
        result['paired'] = report
        require(counts['calls_started']-before == report['vjp_forwards']+report['jvp_calls']+
                report['line_search_forward_calls'], 'Paired actual closure count mismatch')
        status = shared.paired_return_status(slices, resources)
        if status != 'joint_accepted':
            result.update(status=status, fits=blocked_fits(status, 'joint initialization did not admit continuation'))
        else:
            # Qualify every installed arm before loading validation or starting a fit.
            for arm in ARMS:
                arm_out = out/arm; arm_out.mkdir(exist_ok=False)
                arm_ledger = driver.Ledger(arm_out)
                raw = optimizer = None
                try:
                    raw = arm_ledger.measured(rt, 'fresh_K4', lambda:
                        rt.integration.clone_boundary(native, rt.boundary, 4)).eval()
                    with rt.torch.no_grad():
                        warm = arm_ledger.measured(rt, 'common_warm_output', lambda:
                            raw(*rt.integration.raw_arguments(graph, 'polynormer_r')))
                        equality = rt.integration.difference(warm, common[None].repeat(4, 1, 1), 1e-6, 1e-5)
                    require(equality['passed'], 'Fresh common warm equality failed')
                    optimizer, transport = arm_ledger.measured(rt, 'pinned_Adam_transport', lambda:
                        rt.integration.transport_optimizer(native, checkpoint['optimizer'], raw))
                    arm_ledger.measured(rt, 'admitted_private_slice_install', lambda:
                        rt.method.install_factor_slices(raw, slices[arm], 'Polynormer-r'))
                    with rt.torch.no_grad():
                        actual = arm_ledger.measured(rt, 'installed_member_outputs', lambda:
                            raw(*rt.integration.raw_arguments(graph, 'polynormer_r')))
                        reference = arm_ledger.measured(rt, 'installed_closure_references', lambda:
                            rt.torch.stack([observed(row) for row in slices[arm]]))
                        installed = rt.integration.difference(actual, reference, 1e-6, 1e-5)
                    require(installed['passed'], 'Installed member/closure mismatch')
                    rt.integration.rng_restore(checkpoint['rng'])
                    rt.torch.save(dict(schema='paired-Photo-initialized-v1', arm=arm, warm=cell['warm_checkpoint'],
                        state=rt.integration.cpu_copy(raw.state_dict()),
                        optimizer=rt.integration.named_optimizer_snapshot(raw, optimizer),
                        rng=rt.integration.rng_snapshot(), binding=binding, transport=rt.integration.TRANSPORT),
                        arm_out/'initialized_checkpoint.pt')
                    initial = dict(transport=transport, common_warm_equality=equality, installed_equality=installed,
                        checkpoint=driver.descriptor(arm_out/'initialized_checkpoint.pt'), costs=arm_ledger.costs)
                    result['initializations'][arm] = initial
                    driver.write_json(arm_out/'INITIALIZATION.json', initial)
                finally:
                    del raw, optimizer
            # The only validation-byte admission point follows all four installed checks.
            require(set(result['initializations']) == set(ARMS), 'All four installed arms must qualify')
            verify_inputs([cell['validation_labels']], allow_validation=True)
            validation_opened.append(cell['validation_labels'])
            val_ids = rt.np.load(verify(cell['validation_role_ID']), allow_pickle=False)
            validation = ledger.measured(rt, 'admitted_continuation_validation_labels', lambda:
                driver.load_labels(rt, ctx['source_labels']['validation'], val_ids, graph.classes))
            result['validation_labels_loaded'] = True
            for arm in ARMS:
                arm_out = out/arm
                arm_ledger = driver.Ledger(arm_out)
                raw = optimizer = best = None
                try:
                    initial = rt.torch.load(verify(result['initializations'][arm]['checkpoint']),
                                            map_location=rt.device, weights_only=True)
                    require(initial['schema']=='paired-Photo-initialized-v1' and initial['arm']==arm
                            and initial['warm']==cell['warm_checkpoint']
                            and initial['transport']==rt.integration.TRANSPORT, 'Initialized state custody differs')
                    raw = arm_ledger.measured(rt, 'fresh_K4_continuation_restore', lambda:
                        rt.integration.clone_boundary(native, rt.boundary, 4))
                    raw.load_state_dict(initial['state'])
                    optimizer = rt.integration.restore_named_optimizer(raw, initial['optimizer'])
                    require(all(parameter.requires_grad for name, parameter in raw.named_parameters()
                                if name.endswith('.R') or name.endswith('.S')), 'Actual continuation must learn R/S')
                    rt.integration.rng_restore(initial['rng'])
                    del initial
                    best, selection = arm_ledger.measured(rt, 'pinned_complete_continuation', lambda:
                        rt.integration.continuation(raw, optimizer, graph, train, validation,
                            lambda row: driver.append_trace(arm_out/'continuation_trace.jsonl', row),
                            lambda name, logits: driver.save_logits(rt, arm_out, name, logits)))
                    require(selection['update_cap'] == selection['continuation_updates_completed'] == 950
                            and selection['patience'] is None and selection['global_only'] is True
                            and selection['native_midpoint_continuation_epoch'] == 450
                            and selection['native_midpoint_saved'] is True,
                            'Actual Photo continuation did not finish the frozen global schedule')
                    rt.torch.save(dict(schema='paired-Photo-selected-v1', arm=arm,
                        warm=cell['warm_checkpoint'], **best), arm_out/'selected_checkpoint.pt')
                    fit = dict(status='selected', attempted=True, selection=selection,
                        initialization=result['initializations'][arm],
                        selected_checkpoint=driver.descriptor(arm_out/'selected_checkpoint.pt'),
                        selected_logits=driver.descriptor(arm_out/'selected_member_logits.npy'),
                        label_scope=['train','validation'], final_labels_loaded=False, costs=arm_ledger.costs)
                    result['fits'][arm] = fit
                    driver.write_json(arm_out/'SELECTION.json', fit)
                except Exception as error:
                    failure = shared.resource_failure_in_chain(error)
                    result['fits'][arm] = dict(status='resource_deferred' if failure else 'fit_failed',
                        attempted=True, error_type=type(error).__name__, error_message=str(error),
                        traceback=traceback.format_exc(), resource_failure_exception=failure, costs=arm_ledger.costs)
                    driver.write_json(arm_out/'FAILURE.json', result['fits'][arm])
                finally:
                    del raw, optimizer, best
            result['status'] = ('completed' if all(row['status']=='selected' for row in result['fits'].values()) else
                                'resource_deferred' if any(row['status']=='resource_deferred'
                                for row in result['fits'].values()) else 'failed_fits')
        require(all(rt.torch.equal(value, donor[name]) for name, value in native.state_dict().items()),
                'Native warm donor changed')
        result['donor_unchanged'] = True
    except Exception as error:
        failure = shared.resource_failure_in_chain(error)
        result.update(status='resource_deferred' if failure or resources else 'qualification_failed',
            error_type=type(error).__name__, error_message=str(error), traceback=traceback.format_exc(),
            resource_failure_exception=failure or (resources[0]['exception'] if resources else None))
        if hasattr(error, 'report'):
            result['paired_geometry_abort'] = error.report
        for arm in ARMS:
            if result['fits'][arm]['status']=='pending':
                result['fits'][arm] = dict(status=result['status'], attempted=False, reason='block prerequisite failed')
    finally:
        result.update(closure_calls=counts, observed_resource_failures=resources, shared_costs=ledger.costs,
            whole_block_wall_seconds=time.perf_counter()-block_started,
            sparse_products_on_bank_abort='scheduled; completed count unknown')
        driver.write_json(out/'BLOCK.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--admission', help='Root-authored execution release after sealed-source review')
    args = parser.parse_args(argv)
    require(args.run_name not in ('','.', '..') and Path(args.run_name).name == args.run_name, 'Simple fresh run name')
    out = PACKET/'runs'/args.run_name; out.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((PACKET/'FROZEN_STUDY.json').read_text())
    result = dict(schema='paired-Photo-followup-result-v1', status='preparing', blocks=[],
                  final_labels_loaded=False, old_cohort_modified=False, Squirrel_outcomes_loaded=False)
    validation_opened = []
    driver = None
    started = time.perf_counter()
    code = 0
    try:
        data = (PACKET/'MANIFEST.json').read_bytes()
        require(hashlib.sha256(data).hexdigest() == json.loads((PACKET/'SEAL.json').read_text())['manifest_sha256'],
                'Preparation seal mismatch')
        for row in json.loads(data)['payload']:
            verify(dict(row, path=str(PACKET/row['path'])))
        study_guard(frozen)
        result['originals_before'] = verify_inputs(frozen['original_records'])
        shared = load('paired_Photo_native_v3_shared', verify(frozen['shared_source']))
        require(args.admission is not None, 'Root Photo gate assertion and resource release required')
        release_path = Path(args.admission)
        require(release_path.suffix == '.json', 'Execution admission must be JSON metadata')
        release_bytes = release_path.read_bytes()
        admission = admission_guard(frozen, json.loads(release_bytes), hashlib.sha256(data).hexdigest(),
                                    args.run_name, execute=not args.preflight_only)
        admission['root_execution_release_sha256'] = hashlib.sha256(release_bytes).hexdigest()
        result['root_release'] = admission
        result['resource_preflight'] = shared.resource_preflight(
            dict(frozen, expected_gpu_uuid=admission['expected_gpu_uuid']), admission['Photo_resource_evidence'])
        if result['resource_preflight']['status'] == 'resource_deferred':
            result['status'] = 'resource_deferred'
        elif args.preflight_only:
            result['status'] = 'resource_preflight_complete'
        else:
            require(not list((PACKET/'runs').glob('*/PRE_EXECUTION_ADMISSION.json')), 'No restart or replacement execution')
            with (out/'PRE_EXECUTION_ADMISSION.json').open('x') as handle:
                json.dump(admission, handle, indent=2); handle.write('\n')
            result['immutable_pre_execution_admission'] = admission
            driver = load('paired_Photo_pinned_driver', verify(frozen['driver_source']))
            source = driver.source_guard(frozen['cells'][0]['context'])
            rt = driver.load_runtime(frozen['cells'][0]['context'], source)
            paired = load('paired_Photo_sealed_helper', verify(frozen['paired_source']))
            precision = load('paired_Photo_precision', verify(frozen['precision_source']))
            require(tuple(paired.ARM_NAMES) == ARMS, 'Paired arm source differs')
            for cell in frozen['cells']:
                require(cell['context']['environment'] == rt.environment, 'Block runtime differs')
                result['blocks'].append(run_block(driver, rt, shared, paired, precision, cell, frozen,
                    out/f"seed{cell['seed']}_split{cell['source_split_index']}", validation_opened))
            result['comparison'] = followup_comparison(result['blocks'])
            result['status'] = ('resource_deferred' if any(row['status']=='resource_deferred' for row in result['blocks'])
                                else result['comparison']['status'])
            code = 0 if result['comparison']['all12_selected_fits'] else 1
    except Exception as error:
        code = 1
        failure = shared.resource_failure_in_chain(error) if 'shared' in locals() else None
        result.update(status='resource_deferred' if failure else 'study_failed',
                      error_type=type(error).__name__, error_message=str(error), traceback=traceback.format_exc())
    finally:
        try:
            result['originals_after'] = verify_inputs(frozen['original_records'])
            result['validation_originals_after'] = verify_inputs(validation_opened, allow_validation=True)
        except Exception as error:
            code = 1
            result.update(status='original_preservation_failed', preservation_error=str(error), comparison=None)
        result.update(wall_seconds=time.perf_counter()-started, validation_blocks_loaded=len(validation_opened),
                      scientific_merit_decided_by_resource_availability=False, final_labels_closed=True)
        if driver is not None:
            driver.write_json(out/'FOLLOWUP.json', result)
        else:
            with (out/'FOLLOWUP.json').open('x') as handle:
                json.dump(result, handle, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps(dict(status=result['status'], output=str(out), exit_code=code)))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
