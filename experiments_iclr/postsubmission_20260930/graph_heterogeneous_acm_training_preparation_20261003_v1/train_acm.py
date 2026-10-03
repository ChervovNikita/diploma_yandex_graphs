"""Prepared HGB-ACM confirmation. Root freeze/release required; development only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True
PACKET = Path(__file__).resolve().parent
SEEDS = (131, 137, 139, 149, 151)
ARMS = ('native_HGT', 'global_BE', 'shared_relation', 'CP', 'unrestricted', 'untied_HGT', 'wider_BE')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def freeze_guard(frozen, release, frozen_sha, manifest_sha, provenance):
    require(frozen['dataset'] == 'HGB-ACM' and frozen['scope'] == 'complete_release_development_only',
            'Complete ACM development scope required')
    require(frozen['seeds'] == list(SEEDS) and frozen['arms'] == list(ARMS), 'All35 prospectively paired ACM configurations required')
    for key, expected in provenance['confirmation_bindings'].items():
        require(frozen[key] == expected, 'Pinned ACM binding differs: ' + key)
    require(frozen['test_labels_closed'] is True and frozen['study_adopted_by_root'] is True,
            'Root must adopt the source-bound confirmation before execution')
    require(release['execution_authorized'] is True and release['study_freeze_sha256'] == frozen_sha
            and release['prepared_manifest_sha256'] == manifest_sha
            and release['implementation_manifest_sha256'] == frozen['implementation_manifest_sha256']
            and release['scientific_design_sha256'] == frozen['scientific_design_sha256']
            and release['DBLP_continuation_gate_passed'] is True
            and release['ACM_full_graph_resource_check_passed'] is True,
            'Exact root release, DBLP continuation decision and ACM full-graph resource check required')
    require(isinstance(frozen['GPU_uuid'], str) and frozen['GPU_uuid'].startswith('GPU-'), 'Prospective root GPU binding required')


def tensor_fingerprint(value):
    """Fingerprint exact dtype/shape/CPU bytes, without randomness or mutation."""
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(dtype=str(value.dtype), shape=list(value.shape)), sort_keys=True).encode())
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor_fingerprint(value).encode())
    return digest.hexdigest()


def epoch0_diagnostic(torch, existing, model, graph, features, split, validation_labels, seed, arm, path):
    """Ineligible eval-only initialization record around the immutable fit.

    Exact pre/post model and CPU/device RNG equality is required. Restore every
    module's original training flag, preserving the next fit's state transition.
    """
    device = next(model.parameters()).device
    modes = [(submodule, submodule.training) for submodule in model.modules()]
    cpu_rng = torch.get_rng_state().clone()
    device_rng = None if device.type == 'cpu' else torch.cuda.get_rng_state(device).clone()
    before = model_fingerprint(model)
    ids = torch.tensor(split['validation_ids'], dtype=torch.long, device=device)
    labels = torch.tensor(validation_labels, dtype=torch.long, device=device)
    try:
        model.eval()
        with torch.no_grad():
            logits = model(graph, features, '0')
            served = logits.mean(0)[ids]
            require(logits.dtype == torch.float32 and served.dtype == torch.float32, 'Frozen FP32 epoch0 predictor required')
            served_metrics = existing.metrics(torch, logits, ids, labels, 3)
            member_metrics = [existing.metrics(torch, logits[m:m + 1], ids, labels, 3)
                              for m in range(logits.shape[0])]
            prediction = served.argmax(-1)
    finally:
        for submodule, training in modes:
            submodule.training = training
    after = model_fingerprint(model)
    cpu_after = torch.get_rng_state()
    device_after = None if device.type == 'cpu' else torch.cuda.get_rng_state(device)
    unchanged = (before == after and torch.equal(cpu_rng, cpu_after)
                 and (device_rng is None or torch.equal(device_rng, device_after)))
    receipt = dict(schema='ACM_ineligible_epoch0_initialization_diagnostic_v1', seed=seed, arm=arm,
        epoch=0, eligible_for_selection=False, heldout=False, calibrated=False,
        interpretation='Descriptive initialization-confounding evidence; no causal attribution or tuning',
        served_validation=served_metrics, member_validation=member_metrics,
        served_FP32_validation_logits_sha256=tensor_fingerprint(served),
        served_validation_prediction_sha256=tensor_fingerprint(prediction),
        validation_ID_sha256=tensor_fingerprint(ids), model_state_sha256_before=before,
        model_state_sha256_after=after, CPU_RNG_sha256_before=tensor_fingerprint(cpu_rng),
        CPU_RNG_sha256_after=tensor_fingerprint(cpu_after),
        device_RNG_sha256_before=None if device_rng is None else tensor_fingerprint(device_rng),
        device_RNG_sha256_after=None if device_after is None else tensor_fingerprint(device_after),
        model_and_RNG_unchanged=unchanged, module_training_flags_restored=True)
    write(path, receipt)
    require(unchanged, 'Epoch0 diagnostic changed model or RNG; exact fit not launched')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--admission', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    require(args.run_name not in ('', '.', '..') and Path(args.run_name).name == args.run_name, 'Fresh simple run name required')
    manifest_bytes = (PACKET / 'MANIFEST.json').read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    require(json.loads((PACKET / 'SEAL.json').read_text())['manifest_sha256'] == manifest_sha, 'Packet seal mismatch')
    inputs = module('prepared_ACM_inputs', PACKET / 'acm_inputs.py')
    for row in json.loads(manifest_bytes)['payload']:
        inputs.verified(dict(row, path=str(PACKET / row['path'])))
    provenance = json.loads((PACKET / 'PROVENANCE.json').read_text())
    for row in provenance['inputs']:
        inputs.verified(dict(row, path=str(PACKET.parent / row['path'])))
    frozen_bytes = Path(args.freeze).read_bytes()
    frozen_sha = hashlib.sha256(frozen_bytes).hexdigest()
    frozen = json.loads(frozen_bytes)
    release_bytes = Path(args.admission).read_bytes()
    release = json.loads(release_bytes)
    freeze_guard(frozen, release, frozen_sha, manifest_sha, provenance)
    require(release['run_name'] == args.run_name and release['device'] == args.device, 'Root run/device release differs')
    require(not list((PACKET / 'runs').glob('*/STUDY_STARTED.json')), 'No automatic restart/replacement study')
    out = PACKET / 'runs' / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    admission = dict(study_freeze_sha256=frozen_sha, prepared_manifest_sha256=manifest_sha,
        root_release_sha256=hashlib.sha256(release_bytes).hexdigest(), test_labels_closed=True)
    write(out / 'STUDY_STARTED.json', admission)
    rows, summary, before = [], None, []
    study_started, preservation_error = time.perf_counter(), None
    # These stdlib functions are imported verbatim. The immutable fit handles
    # optimizer, latest-tie selection, complete state/RNG and selected replay.
    existing_driver = module('immutable_HGB_training_for_ACM', PACKET.parent / provenance['training_source'])
    try:
        before = [frozen['archive'], frozen['development_labels']] + [row['descriptor'] for row in frozen['splits']]
        for row in before:
            inputs.verified(row)
        schema, attributes, edges = inputs.stream_schema(frozen['archive'], frozen['members'])
        require(schema['member_sha256'] == frozen['member_sha256'], 'Byte-bound ACM node/link payload differs')
        require(schema['node_counts'] == frozen['expected_node_counts'], 'Complete released ACM type counts differ')
        require(schema['input_dims'] == frozen['expected_input_dims'], 'Native ACM feats0 dimensions differ')
        require(schema['provided_attribute_widths'] == frozen['expected_feature_widths'], 'Provided ACM type attributes differ')
        require({str(row['raw_id']): [row['source'], row['target']] for row in schema['relations']}
                == frozen['expected_relations'], 'Distinct raw directed ACM relation scope differs')
        statistics = {str(row['raw_id']): {key: row[key] for key in ('raw_records', 'support_edges', 'raw_self_records', 'duplicates_coalesced')}
                      for row in schema['relations']}
        require(statistics == frozen['expected_relation_statistics'], 'Released ACM relation/self support differs')
        development = inputs.read_development_labels(frozen['development_labels'], frozen['archive']['sha256'])
        require(development['source_member'] == 'ACM/label.dat'
                and development['source_member_sha256'] == frozen['source_label_member_sha256'],
                'Canonical development input must bind original ACM label.dat bytes')
        require(len(development['node_ids']) == 907 and development['train_class_schema'] == [0, 1, 2]
                and max(development['node_ids']) < schema['node_counts']['0'], 'Exact ACM development pool/classes required')
        import torch
        require(torch.__version__.split('+')[0] == '2.1.2', 'Pinned Torch runtime differs')
        device = torch.device(args.device)
        require(device.type in ('cpu', 'cuda'), 'Pinned CPU/CUDA backend only')
        if device.type == 'cuda':
            require(os.environ.get('CUDA_VISIBLE_DEVICES') == frozen['GPU_uuid'], 'Root must expose exactly the frozen GPU UUID')
            require(torch.cuda.device_count() == 1 and device.index in (None, 0), 'One visible GPU required')
            device = torch.device('cuda:0')
        implementation = module('qualified_HGB_HGT_private_for_ACM', PACKET.parent / provenance['implementation_source'])
        existing_families = module('immutable_HGB_families_for_ACM', PACKET.parent / provenance['families_source'])
        families = module('prepared_ACM_families', PACKET / 'families_acm.py')
        onecycle = module('immutable_HGB_portable_OneCycle_for_ACM', PACKET.parent / provenance['onecycle_source'])
        graph, features, schema = inputs.materialize(schema, attributes, edges, implementation, frozen['relation_row_order'], device)
        write(out / 'GRAPH_SCHEMA.json', schema)
        del attributes, edges
        bindings = dict(admission, archive=frozen['archive'], development_labels=frozen['development_labels'],
            implementation_manifest_sha256=frozen['implementation_manifest_sha256'], graph_schema=schema)
        for split_record in frozen['splits']:
            seed = split_record['seed']
            split, train_labels, val_labels = inputs.verify_split(split_record['descriptor'], development, seed)
            require(split['development_descriptor'] == frozen['development_labels'], 'Split development source binding differs')
            require(len(train_labels) == 726 and len(val_labels) == 181, 'Full ACM TRAIN/VAL sizes differ')
            models = families.build(implementation, existing_families, graph, schema['input_dims'], 3,
                seed, seed + 900001, device, frozen['arms'])
            for arm in frozen['arms']:
                case = out / f'seed{seed}' / arm
                case.parent.mkdir(exist_ok=True)
                arm_started = time.perf_counter()
                try:
                    epoch0_diagnostic(torch, existing_driver, models[arm], graph, features, split,
                        val_labels, seed, arm, case.parent / f'{arm}__EPOCH0.json')
                    rows.append(existing_driver.fit(torch, implementation, models[arm], graph, features, split,
                        train_labels, val_labels, seed, arm, case, dict(bindings, split=split_record), onecycle))
                except Exception as error:
                    case.mkdir(exist_ok=True)
                    status = 'resource_deferred' if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)) else 'failed'
                    row = dict(status=status, seed=seed, arm=arm, error_type=type(error).__name__, error_message=str(error),
                        paid_wall_seconds=time.perf_counter() - arm_started, traceback=traceback.format_exc(),
                        final_labels_closed=True, successful_subset_scored=False)
                    write(case / 'FAILURE.json', row)
                    rows.append(row)
                finally:
                    del models[arm]
        summary = existing_driver.paired_development(rows, frozen['arms'])
    except Exception as error:
        error_record = dict(error_type=type(error).__name__, error_message=str(error), traceback=traceback.format_exc())
        write(out / 'STUDY_FAILURE.json', error_record)
        for seed in SEEDS:
            for arm in frozen['arms']:
                if not any(row['seed'] == seed and row['arm'] == arm for row in rows):
                    rows.append(dict(seed=seed, arm=arm, status='blocked', attempted=False, reason='study prerequisite failed', **error_record))
        summary = existing_driver.paired_development(rows, frozen['arms'])
    finally:
        try:
            for row in before:
                inputs.verified(row)
            for row in provenance['inputs']:
                inputs.verified(dict(row, path=str(PACKET.parent / row['path'])))
            for row in json.loads(manifest_bytes)['payload']:
                inputs.verified(dict(row, path=str(PACKET / row['path'])))
            require(hashlib.sha256(Path(args.freeze).read_bytes()).hexdigest() == frozen_sha
                    and Path(args.admission).read_bytes() == release_bytes, 'Root freeze/release changed')
        except Exception as error:
            preservation_error, summary = str(error), None
        write(out / 'STUDY.json', dict(schema='HGB_ACM_development_confirmation_v1', rows=rows,
            summary=summary, wall_seconds=time.perf_counter() - study_started,
            original_inputs_verified_unchanged=preservation_error is None, preservation_error=preservation_error,
            final_labels_closed=True, admission=admission,
            comparison_scope='Frozen HGT confirmation family; external native challengers and final evaluation are separate'))
    print(json.dumps(dict(output=str(out), summary=summary)))
    return 0 if summary is not None and summary['status'] == 'complete_development_summary' else 1


if __name__ == '__main__':
    raise SystemExit(main())
