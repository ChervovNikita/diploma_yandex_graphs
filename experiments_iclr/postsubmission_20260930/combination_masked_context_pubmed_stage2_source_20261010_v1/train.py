"""Prospective full Stage2 fit. No run without a separate exact root release."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time
from admission import admit
from source import bind, fingerprint, load_v3, sha, write
from stage_plan import description, forecast, spec as condition_spec


VERIFY_ATOL, VERIFY_RTOL = 2e-6, 2e-6


def prediction_signatures(logits, probabilities, ids, labels):
    pooled = probabilities[ids].argmax(-1)
    members = logits[:, ids].argmax(-1)
    return dict(pooled_predictions=fingerprint(pooled.detach().cpu().numpy()),
                member_predictions=fingerprint(members.detach().cpu().numpy()),
                pooled_correct_mask=fingerprint((pooled == labels).detach().cpu().numpy()),
                member_correct_masks=fingerprint((members == labels[None,:]).detach().cpu().numpy()))


def verify_selected_metrics(actual, expected):
    """Only FP32-derived floats tolerate tiny repeats; integer counts stay exact."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected): raise ValueError('Selected metric fields changed')
        for key in expected: verify_selected_metrics(actual[key], expected[key])
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected): raise ValueError('Selected metric shape changed')
        for a,e in zip(actual,expected): verify_selected_metrics(a,e)
    elif isinstance(expected, float):
        if not isinstance(actual, (int,float)) or not math.isclose(actual,expected,abs_tol=VERIFY_ATOL,rel_tol=VERIFY_RTOL):
            raise ValueError('Selected FP32-derived metric changed beyond documented verification tolerance')
    elif actual != expected or type(actual) != type(expected):
        raise ValueError('Selected exact count/identity changed')


def checkpoint(session, epoch, metrics, signatures, ownership):
    return dict(schema='masked-context-selected-predictor-v1', epoch=epoch, selected_VALID=metrics,
                condition=session.name, seed=session.seed,
                selected_prediction_signatures=signatures, selected_last_ownership=ownership,
                bodies=[{k:v.detach().cpu().clone() for k,v in body.state_dict().items()} for body in session.bodies],
                decoder=None if session.decoder is None else {k:v.detach().cpu().clone() for k,v in session.decoder.state_dict().items()},
                selected_next_negative_states=[generator.getstate() for generator in session.negative_generators],
                serving='factual mean of four member softmax class probabilities',
                resumable=False, optimizer_history_included=False, labels_included=False)


def fit(spec, output, process_started):
    session = None
    output.mkdir(parents=True, exist_ok=False)
    timings = dict(numerical_imports_and_input_seconds=0., construction_seconds=0., update_seconds=0.,
                   regular_VALID_seconds=0., checkpoint_write_seconds=0., restore_seconds=0.,
                   selected_factual_seconds=0., selected_mask_diagnostics_seconds=0., logit_write_seconds=0.)
    checkpoint_writes = 0
    try:
        tick = time.monotonic()
        import numpy as np
        import torch
        from metrics import classification, mask_diagnostics
        v3 = load_v3(numerical=True)
        method = v3['method']
        providers = dict(torch=str(torch.__version__), numpy=str(np.__version__))
        for name in ('torch-geometric', 'scipy', 'torch-scatter', 'torch-sparse'):
            try: providers[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: providers[name] = None
        if providers != spec['frozen_providers']: raise ValueError('Exact qualification providers required')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(0)
        with np.load(bind(spec['train_bundle']), allow_pickle=False) as archive:
            if set(archive.files) != {'x', 'edge_index', 'train_ids', 'train_y'}: raise ValueError('Only four TRAIN-only arrays admitted')
            arrays = {key:archive[key].copy() for key in archive.files}
        with np.load(bind(spec['valid_bundle']), allow_pickle=False) as archive:
            if set(archive.files) != {'valid_ids', 'valid_y'}: raise ValueError('Only two projected VALID arrays admitted')
            validation = {key:archive[key].copy() for key in archive.files}
        if fingerprint_map(arrays) != spec['_train_custody']['array_fingerprints'] or fingerprint_map(validation) != spec['_validation_custody']['array_fingerprints']:
            raise ValueError('Loaded arrays differ from exact data-custodian fingerprints')
        if arrays['x'].dtype != np.float32 or arrays['x'].shape != (19717,500): raise ValueError('Complete float32 graph features required')
        for key in ('edge_index', 'train_ids', 'train_y'):
            if arrays[key].dtype != np.int64: raise ValueError('Exact int64 TRAIN arrays required')
        for key in validation:
            if validation[key].dtype != np.int64: raise ValueError('Exact int64 VALID projection required')
        if list(arrays['edge_index'].shape) != spec['edge_shape']: raise ValueError('Complete ordered factual graph required')
        for prefix, source, expected, counts in (('train', arrays, 11829, [2461,4643,4725]), ('valid', validation, 3942, [820,1547,1575])):
            ids, labels = source[prefix+'_ids'], source[prefix+'_y']
            if ids.shape != (expected,) or labels.shape != ids.shape or len(np.unique(ids)) != expected or ids.min() < 0 or ids.max() >= 19717:
                raise ValueError('Complete unique frozen role identities required')
            if labels.min() < 0 or labels.max() >= 3 or np.bincount(labels, minlength=3).tolist() != counts:
                raise ValueError('Frozen role class populations required')
        if np.intersect1d(arrays['train_ids'], validation['valid_ids']).size: raise ValueError('TRAIN and VALID overlap')
        tensor = {key:torch.from_numpy(value) for key,value in arrays.items()}
        valid_ids, valid_y = (torch.from_numpy(validation[key]).to('cuda:0') for key in ('valid_ids','valid_y'))
        timings['numerical_imports_and_input_seconds'] = time.monotonic()-tick
        tick = time.monotonic()
        auxiliary_edges = None
        if spec.get('auxiliary_graph') is not None:
            with np.load(bind(spec['auxiliary_graph']), allow_pickle=False) as archive:
                if archive.files != ['edge_index']: raise ValueError('Only complete public auxiliary edge_index admitted')
                edge_array = archive['edge_index'].copy()
                if edge_array.dtype != np.int64 or fingerprint(edge_array) != spec['_rewire_custody']['graph_fingerprint']:
                    raise ValueError('Frozen complete rewire graph fingerprint differs')
                auxiliary_edges = torch.from_numpy(edge_array)
        session = method.Session(spec['condition'], spec['seed'], tensor['x'], tensor['edge_index'], tensor['train_ids'], tensor['train_y'], device='cuda:0', auxiliary_edges=auxiliary_edges)
        assert_independence(session, torch)
        torch.cuda.synchronize()
        timings['construction_seconds'] = time.monotonic()-tick
        best_correct, best_epoch, best_metrics, best_signatures, epochs = -1, None, None, None, 0
        selected_path = output/'SELECTED_PREDICTOR.pt'
        active_cap = spec['limits']['external_active_seconds']
        with (output/'EPOCHS.jsonl').open('x') as trace:
            for epoch in range(1, 2001):
                if time.monotonic()-process_started >= active_cap: raise TimeoutError('Frozen scientific active-time cap')
                tick = time.monotonic(); train_metrics = session.train_step(audit=False)
                expected_decoder_divisor=4 if spec['condition']=='untied4_shared_decoder_core' else 1
                if train_metrics['decoder_data_gradient_divisor']!=expected_decoder_divisor:
                    raise ValueError('Exact V3 common-decoder route-mean data gradient required')
                torch.cuda.synchronize(); update_seconds = time.monotonic()-tick
                timings['update_seconds'] += update_seconds
                tick = time.monotonic(); probabilities, logits = session.factual_probabilities()
                valid_metrics = classification(logits, probabilities, valid_ids, valid_y)
                torch.cuda.synchronize(); serving_seconds = time.monotonic()-tick
                timings['regular_VALID_seconds'] += serving_seconds
                correct = valid_metrics['pooled']['correct']
                improved = correct > best_correct
                if improved:
                    best_correct, best_epoch, best_metrics = correct, epoch, valid_metrics
                    best_signatures = prediction_signatures(logits, probabilities, valid_ids, valid_y)
                    tick = time.monotonic(); temporary = output/'SELECTED_PREDICTOR.partial.pt'
                    torch.save(checkpoint(session, epoch, valid_metrics, best_signatures, train_metrics['ownership']), temporary)
                    os.replace(temporary, selected_path); checkpoint_writes += 1
                    timings['checkpoint_write_seconds'] += time.monotonic()-tick
                epochs = epoch
                trace.write(json.dumps(dict(epoch=epoch, TRAIN=train_metrics, VALID=valid_metrics,
                                           selected=improved, best_epoch=best_epoch,
                                           update_seconds=update_seconds, serving_seconds=serving_seconds), allow_nan=False)+'\n')
                trace.flush()
                write(output/'PROGRESS.json', dict(complete=False, record_id=spec['record_id'], epoch=epoch,
                    max_epochs=2000, counters=session.counters, inclusive_seconds=time.monotonic()-process_started))
                del probabilities, logits
                if epoch-best_epoch >= 250: break
        tick = time.monotonic(); selected = torch.load(selected_path, map_location='cpu')
        if selected['epoch'] != best_epoch or selected['condition'] != spec['condition'] or selected['seed'] != spec['seed']:
            raise ValueError('Whole coherent selected predictor mismatch')
        if len(selected['bodies']) != len(session.bodies) or len(selected['selected_next_negative_states']) != 4:
            raise ValueError('Complete selected body and auxiliary-generator snapshot required')
        if selected['selected_prediction_signatures'] != best_signatures:
            raise ValueError('Selected prediction/correct-mask identity changed')
        for body, state in zip(session.bodies, selected['bodies']): body.load_state_dict(state, strict=True)
        if session.decoder is not None: session.decoder.load_state_dict(selected['decoder'], strict=True)
        for generator, state in zip(session.negative_generators, selected['selected_next_negative_states']): generator.setstate(state)
        selected_ownership = selected['selected_last_ownership']
        del selected; torch.cuda.synchronize(); timings['restore_seconds'] = time.monotonic()-tick
        tick = time.monotonic(); probabilities, logits = session.factual_probabilities()
        selected_metrics = classification(logits, probabilities, valid_ids, valid_y)
        selected_signatures = prediction_signatures(logits, probabilities, valid_ids, valid_y)
        if selected_signatures != best_signatures: raise ValueError('Restored selected pooled/member predictions or correct masks changed')
        verify_selected_metrics(selected_metrics, best_metrics)
        torch.cuda.synchronize(); timings['selected_factual_seconds'] = time.monotonic()-tick
        tick = time.monotonic()
        mask_report, masked_logits = mask_diagnostics(session, method, dict(TRAIN=(session.train_ids, session.train_y), VALID=(valid_ids, valid_y)), selected_ownership)
        torch.cuda.synchronize(); timings['selected_mask_diagnostics_seconds'] = time.monotonic()-tick
        payload = dict(factual_member_logits=logits.detach().cpu().numpy())
        if masked_logits is not None: payload['owned_masked_member_logits'] = masked_logits.detach().cpu().numpy()
        tick = time.monotonic(); np.savez_compressed(output/'SELECTED_MEMBER_LOGITS.npz', **payload)
        timings['logit_write_seconds'] = time.monotonic()-tick
        expected = condition_spec(spec['condition'])
        expected_masks = epochs*expected['masked_views']+expected['masked_views']
        if session.counters['updates'] != epochs or session.counters['backwards'] != epochs*expected['backwards'] or session.counters['optimizer_steps'] != epochs*expected['optimizer_steps'] or session.counters['factual_forwards'] != epochs*2*expected['members']+expected['members'] or session.counters['serving_forwards'] != (epochs+1)*expected['members'] or session.counters['masked_forwards'] != expected_masks:
            raise ValueError('Complete declared training/evaluation/diagnostic work counts differ')
        if time.monotonic()-process_started >= active_cap: raise TimeoutError('Frozen scientific cap before completion')
        result = dict(schema='masked-context-PubMed-stage2-complete-v1', complete=True,
            record_id=spec['record_id'], condition=spec['condition'], seed=spec['seed'],
            source_manifest_sha256=spec['source_manifest_sha256'], release_sha256=spec['_release_sha256'],
            split_identity=spec['split_identity'], split_seed=190111, providers=providers,
            max_epochs=2000, patience=250, epochs_executed=epochs, selected_epoch=best_epoch,
            stopped_by='patience250' if epochs-best_epoch >= 250 else 'max2000', selector=spec['selector'],
            selected_VALID=selected_metrics, selected_mask_diagnostics=mask_report,
            selected_prediction_signatures=selected_signatures,
            selected_state_verification=dict(predictions_and_correct_masks_exact=True, counts_and_epoch_exact=True,
                float_absolute_tolerance=VERIFY_ATOL, float_relative_tolerance=VERIFY_RTOL,
                tolerance_is_restore_verification_only=True, continuation_gate_tolerance_added=False),
            selected_mask_diagnostics_epoch=best_epoch, selected_checkpoint_writes=checkpoint_writes,
            counters=session.counters, timings=timings, preprocessing_seconds=session.preparation_seconds,
            inclusive_seconds=time.monotonic()-process_started,
            CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
            CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_CUDA_bytes=dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved()),
            predictor_parameters=sum(p.numel() for body in session.bodies for p in body.parameters()),
            decoder_parameters=0 if session.decoder is None else sum(p.numel() for p in session.decoder.parameters()),
            complete_saved_member_logits=dict(path='SELECTED_MEMBER_LOGITS.npz', sha256=sha(output/'SELECTED_MEMBER_LOGITS.npz'), bytes=(output/'SELECTED_MEMBER_LOGITS.npz').stat().st_size,
                keys=sorted(payload), factual_shape=[expected['members'],19717,3], masked_shape=None if masked_logits is None else [4,19717,3], contains_labels_or_role_ids=False, server_only=True),
            selected_predictor=dict(path='SELECTED_PREDICTOR.pt', sha256=sha(selected_path), bytes=selected_path.stat().st_size, server_only=True, resumable=False),
            train_bundle_sha256=spec['train_bundle']['sha256'], valid_bundle_sha256=spec['valid_bundle']['sha256'],
            TEST_labels_or_id_inputs=False, TEST_scored=False, automatic_retry=False,
            competence_established_by_completion=False, whole_27_comparison_required=True, retained_Stage1_records=9, selected_ownership=selected_ownership,
            owner_success_not_inferred=True, independent_primary_common_bank_selector_limit_retained=True)
        write(output/'COMPLETE.json', result)
        return result
    except BaseException as error:
        write(output/'FAILURE.json', dict(complete=False, record_id=spec['record_id'],
            error_type=type(error).__name__, error=str(error), inclusive_seconds=time.monotonic()-process_started,
            counters=None if session is None else session.counters, timings=timings,
            automatic_retry=False, TEST_access=False, partial_family_comparison_allowed=False))
        raise


def fingerprint_map(arrays):
    return {key:fingerprint(value) for key,value in arrays.items()}


def assert_independence(session, torch):
    if session.name != 'untied4_shared_decoder_core': return
    parameters = [set(id(p) for p in body.parameters()) for body in session.bodies]
    if len(parameters) != 4 or any(parameters[a] & parameters[b] for a in range(4) for b in range(a)):
        raise ValueError('Untied4 control requires four disjoint full native bodies')
    if len(session.optimizers) != 5 or session.decoder is None:
        raise ValueError('Four private body Adams plus one common decoder Adam required')
    factors = session.factors.FactorLinear
    if any(isinstance(module, factors) for body in session.bodies for module in body.modules()):
        raise ValueError('Independent native bodies cannot contain BE factor wrappers')
    for member, optimizer in enumerate(session.optimizers[:4]):
        optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
        if optimized != parameters[member]: raise ValueError('Private optimizer/body mismatch')
    decoder_params = {id(p) for p in session.decoder.parameters()}
    if decoder_params != {id(p) for g in session.optimizers[4].param_groups for p in g['params']} or any(decoder_params & params for params in parameters):
        raise ValueError('One disjoint common decoder optimizer required')


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('describe','run'), default='describe')
    parser.add_argument('--release', type=Path); parser.add_argument('--release-sha256')
    args = parser.parse_args()
    if args.mode == 'describe':
        print(json.dumps(dict(plan=description(), cost=forecast(load_v3()['plan'].resource_forecast())), indent=2, sort_keys=True)); return
    if args.release is None or args.release_sha256 is None: parser.error('Separate exact root scientific release required')
    spec, output = admit(args.release, args.release_sha256)
    result = fit(spec, output, started)
    print(json.dumps(dict(complete=result['complete'], record_id=result['record_id'], output=str(output), owner_success_not_inferred=True)))


if __name__ == '__main__': main()
