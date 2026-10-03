"""Closed selected15 native CPU custody/replay audit; exact root release required."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SEEDS = [131, 137, 139, 149, 151]
ARMS = ['native_GAT', 'native_Simple_HGN', 'native_SeHGNN']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def verify(record):
    path = Path(record['path'])
    if not path.is_absolute():
        path = PHASE/path
    require(digest(path) == record['sha256'] and path.stat().st_size == record['bytes'], 'Custody differs: '+str(path))
    return path


def descriptor(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_guard():
    manifest_sha = digest(HERE/'MANIFEST.json')
    require(json.loads((HERE/'SEAL.json').read_text())['manifest_sha256'] == manifest_sha, 'Auditor seal differs')
    for row in json.loads((HERE/'MANIFEST.json').read_text())['payload']:
        verify(dict(row, path=str(HERE/row['path'])))
    provenance = json.loads((HERE/'PROVENANCE.json').read_text())
    for row in provenance['source_records']:
        verify(row)
    return provenance, manifest_sha


def sealed_packet(manifest_record, seal_record):
    path = verify(manifest_record)
    require(json.loads(verify(seal_record).read_text())['manifest_sha256'] == manifest_record['sha256'], 'Original packet seal differs')
    for row in json.loads(path.read_text())['payload']:
        verify(dict(row, path=str(path.parent/row['path'])))


def admission(freeze_path, release_path, output):
    provenance, manifest_sha = source_guard()
    freeze_record, release_record = descriptor(freeze_path), descriptor(release_path)
    frozen, release = json.loads(Path(freeze_path).read_text()), json.loads(Path(release_path).read_text())
    require(frozen['schema'] == 'native15_closed_replay_audit_freeze_v1' and frozen['dataset'] == 'HGB-DBLP'
            and frozen['seeds'] == SEEDS and frozen['arms'] == ARMS and frozen['heldout_labels_closed'] is True,
            'Exact native15 closed development audit freeze required')
    require(release['execution_authorized'] is True and release['prepared_auditor_manifest_sha256'] == manifest_sha
            and release['audit_freeze_sha256'] == freeze_record['sha256'] and release['study_sha256'] == frozen['study']['sha256']
            and release['supervisor_receipt_sha256'] == frozen['supervisor_receipt']['sha256']
            and release['root_observed_closed_selected15'] is True and release['training_child_reaped'] is True
            and release['independent_source_review_observed'] is True and release['heldout_labels_closed'] is True
            and release['device'] == 'cpu' and release['threads'] == 1,
            'Separate explicit root audit release, critic and closed/reaped selected15 required')
    review = release['independent_source_review']; verify(review)
    require(Path(output).is_absolute() and str(Path(output)) == frozen['expected_output_path'] and not Path(output).exists(),
            'Exact fresh absolute root-bound audit output required')
    for key in ('source_manifest', 'source_seal', 'training_freeze', 'training_release', 'supervisor_manifest', 'supervisor_seal', 'supervisor_binding'):
        require(frozen[key]['sha256'] == provenance['authorities'][key]['sha256']
                and frozen[key]['bytes'] == provenance['authorities'][key]['bytes'], 'Original authority differs: '+key)
        verify(frozen[key])
    sealed_packet(frozen['source_manifest'], frozen['source_seal'])
    sealed_packet(frozen['supervisor_manifest'], frozen['supervisor_seal'])
    binding = json.loads(verify(frozen['supervisor_binding']).read_text())
    nf = json.loads(verify(frozen['training_freeze']).read_text())
    nr = json.loads(verify(frozen['training_release']).read_text())
    require(frozen['training_freeze'] == binding['freeze'] and nf['paired_HGT_freeze'] == binding['paired_HGT_freeze']
            and nr['prepared_manifest_sha256'] == binding['native_manifest_sha256'] == frozen['source_manifest']['sha256']
            and nr['study_freeze_sha256'] == frozen['training_freeze']['sha256']
            and nr['supervisor_manifest_sha256'] == frozen['supervisor_manifest']['sha256']
            and nr['device'] == 'cpu' and nr['threads'] == 1 and nr['execution_authorized'] is True
            and nr['mode'] == 'native_v2_serial_CPU_supervised' and nr['run_name'] == provenance['run_name'],
            'Actual immutable native-v2 serial CPU release/binding required')
    for key, value in binding['limits'].items():
        require(nr[key] == value, 'Original supervisor resource bound differs: '+key)
    native_provenance = json.loads((verify(frozen['source_manifest']).parent/'PROVENANCE.json').read_text())
    for record in native_provenance['inputs']:
        verify(record)
    require(native_provenance['loader_source'] == provenance['modules']['loader'], 'Original qualified reader differs')
    driver = load('native15_audit_original_driver', PHASE/provenance['modules']['driver'])
    driver.guard(nf, nr, frozen['training_freeze']['sha256'], frozen['source_manifest']['sha256'],
                 json.loads((verify(frozen['source_manifest']).parent/'FREEZE_TEMPLATE.json').read_text()))
    supervisor = load('native15_audit_original_supervisor_metadata', PHASE/provenance['modules']['supervisor'])
    original = [freeze_record, release_record, review] + [frozen[key] for key in
        ('source_manifest', 'source_seal', 'training_freeze', 'training_release', 'supervisor_manifest', 'supervisor_seal', 'supervisor_binding')]
    original.extend(native_provenance['inputs'])
    return provenance, frozen, release, nf, nr, binding, driver, supervisor, original


def audit_trace(driver, trace, terminal):
    maximum = 200 if terminal['arm'] == 'native_SeHGNN' else 300
    epochs = terminal['epochs_paid']
    require(1 <= epochs <= maximum and [r['epoch'] for r in trace] == list(range(1, epochs+1)), 'Complete consecutive native budget trace required')
    stopper = driver.NativeStopper(terminal['arm']); selected = None
    for index, row in enumerate(trace):
        require(row['heldout'] is False and row['calibrated'] is False and all(math.isfinite(row[key]) for key in
                ('TRAIN_CE', 'validation_NLL', 'validation_micro_F1', 'validation_macro_F1')), 'Finite original development trace required')
        save, stop = stopper.observe(row['epoch'], row['validation_NLL'])
        require(row['checkpoint_replaced'] == save and row['native_patience_counter'] == stopper.counter, 'Native selector/patience differs')
        require(not stop or index == len(trace)-1, 'Native trace continued after stopping')
        if save:
            selected = row
    require(epochs == maximum or stop, 'Short native trace without original stopping')
    require(selected is not None and all(selected[key] == value for key, value in terminal['selection'].items()),
            'Native selected score/epoch differs from original per-recipe selector')


def closed_family(provenance, frozen, nr, binding, driver, supervisor):
    """Complete selected15 closure/reaping before any label/tensor deserialization."""
    native_run = PHASE/'graph_heterogeneous_dblp_native_challengers_preparation_20261003_v2'/'runs'/nr['run_name']
    supervisor_run = PHASE/'graph_heterogeneous_dblp_native_cpu_supervisor_preparation_20261003_v1'/'runs'/nr['run_name']
    require(verify(frozen['study']) == native_run/'STUDY.json'
            and verify(frozen['graph_schema']) == native_run/'GRAPH_SCHEMA.json'
            and verify(frozen['supervisor_receipt']) == supervisor_run/'SUPERVISOR_RECEIPT.json', 'Exact original native/supervisor output slots required')
    receipt = json.loads(verify(frozen['supervisor_receipt']).read_text())
    expected_argv = [provenance['canonical_python'], '-B', '-c', supervisor.CHILD_BOOTSTRAP, binding['native_driver']['path'],
        '--freeze', binding['freeze']['path'], '--admission', frozen['training_release']['path'], '--run-name', nr['run_name'], '--device', 'cpu']
    require(receipt['status'] == 'child_completed' and receipt['exit_code'] == 0 and receipt['owned_child_reaped'] is True
            and receipt['originals_preserved'] is True and receipt['root_release'] == frozen['training_release']
            and receipt['canonical_study'] == frozen['study'] and receipt['native_output'] == str(native_run)
            and receipt['supervisor_manifest_sha256'] == frozen['supervisor_manifest']['sha256']
            and receipt['native_manifest_sha256'] == frozen['source_manifest']['sha256'] and receipt['argv'] == expected_argv
            and receipt['scientific_functions_or_arm_order_changed'] is False and receipt['successful_subset_scored'] is False
            and receipt['GPU_computation'] is False and receipt['canonical_study_created_or_modified_by_supervisor'] is False,
            'Actual original child completion/reaping and supervisor custody required')
    study = json.loads(verify(frozen['study']).read_text())
    expected = [(seed, arm) for seed in SEEDS for arm in ARMS]
    require([(r['seed'], r['arm']) for r in study['rows']] == expected
            and study['summary']['status'] == 'complete_development_challengers'
            and study['summary']['all_frozen_terminals'] is True and study['summary']['successful_subset_scored'] is False
            and study['summary']['TEST_label_reads'] == 0 and study['summary']['TEST_diagnostics'] == 0
            and study['original_inputs_unchanged'] is True and study['preservation_error'] is None
            and study['HGEN_included'] is False and study['study_superiority_claim'] is False,
            'Exact preserved complete15 family required; no successful-subset audit')
    require(all(r['status'] == 'selected' and r['selected_state_replay'] is True and r['TEST_label_reads'] == 0
                and r['TEST_diagnostics'] == 0 and r['classifier_classes_from_TRAIN'] == 4 and r['native_AMP_used'] is False
                and r['GPU_peak_allocated'] is None and r['GPU_peak_reserved'] is None for r in study['rows']),
            'All15 native CPU selected replays required before any payload deserialization')
    expected_admission = dict(study_freeze_sha256=frozen['training_freeze']['sha256'],
        prepared_manifest_sha256=frozen['source_manifest']['sha256'], root_admission_sha256=frozen['training_release']['sha256'], TEST_labels_closed=True)
    require(study['admission'] == expected_admission, 'Original native study admission differs')
    records = [frozen['study'], frozen['graph_schema'], frozen['supervisor_receipt']]
    for path, expected_value in ((native_run/'STUDY_STARTED.json', expected_admission),
        (supervisor_run/'SUPERVISOR_STARTED.json', dict(root_release=frozen['training_release'], supervisor_manifest_sha256=frozen['supervisor_manifest']['sha256'])),
        (supervisor_run/'OWNED_CHILD.json', dict(pid=receipt['owned_child_pid'], argv=expected_argv,
             preimport_RLIMIT_AS_bytes=binding['limits']['address_space_limit_bytes']))):
        record = descriptor(path); records.append(record)
        require(json.loads(verify(record).read_text()) == expected_value, 'Original start/owned-child receipt differs')
    cases = []
    for terminal in study['rows']:
        case_dir = native_run/f"seed{terminal['seed']}"/terminal['arm']
        case = dict(seed=terminal['seed'], arm=terminal['arm'])
        for key, filename in dict(selection_receipt='SELECTION.json', selected_checkpoint='selected.pt',
                                  selected_validation_logits='selected_validation_logits.pt', training_trace='TRACE.jsonl').items():
            require((case_dir/filename).resolve() == case_dir/filename, 'Native artifact must remain in exact original case')
            case[key] = descriptor(case_dir/filename); records.append(case[key])
        require(json.loads(verify(case['selection_receipt']).read_text()) == terminal, 'Native study/immutable selection receipt differs')
        audit_trace(driver, [json.loads(line) for line in verify(case['training_trace']).read_text().splitlines()], terminal)
        cases.append(case)
    require(study['summary']['validation_NLL'] == {arm: [next(r['selection']['validation_NLL'] for r in study['rows']
        if r['arm'] == arm and r['seed'] == seed) for seed in SEEDS] for arm in ARMS}, 'Original complete15 summary/terminal custody differs')
    return study, cases, records


def runtime(binding, nr):
    require(platform.system() == 'Linux' and sys.version_info[:3] == (3, 11, 14), 'Original qualified Linux/Python runtime required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Original CPU execution requires CUDA hidden before import')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key) == '1', 'Original one-thread preimport environment required: '+key)
    qualification = json.loads(verify(binding['qualification_result']).read_text())
    require(binding['qualification_result']['sha256'] == nr['qualification_result_sha256']
            and qualification['status'] == 'qualified' and qualification['validation_or_test_scored'] is False,
            'Original score-free qualification runtime authority required')
    import torch
    import numpy as np
    import scipy
    require(torch.__version__ == '2.1.2+cu118' and torch.get_default_dtype() == torch.float32, 'Original FP32 Torch runtime required')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    actual = dict(torch=torch.__version__, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                  threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(), CUDA_VISIBLE_DEVICES='')
    require(actual == qualification['runtime'], 'Original qualified native preprocessing/runtime versions differ')
    return torch, actual


def equal_tree(torch, left, right):
    if isinstance(left, torch.Tensor):
        require(isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.cpu(), right.cpu()), 'Restored tensor state differs')
        if left.is_floating_point():
            require(bool(torch.isfinite(left).all()), 'Nonfinite complete selected state')
    elif isinstance(left, dict):
        require(isinstance(right, dict) and left.keys() == right.keys(), 'Restored mapping state differs')
        for key in left:
            equal_tree(torch, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        require(type(left) is type(right) and len(left) == len(right), 'Restored sequence state differs')
        for a, b in zip(left, right):
            equal_tree(torch, a, b)
    else:
        require(type(left) is type(right) and left == right, 'Restored scalar state differs')


def run(freeze_path, release_path, output):
    provenance, frozen, release, nf, nr, binding, driver, supervisor, original = admission(freeze_path, release_path, output)
    study, cases, custody = closed_family(provenance, frozen, nr, binding, driver, supervisor); original.extend(custody)
    # Only this point admits Torch, graph/feature materialization, development
    # labels and selected tensor payloads. All15 and their child are closed.
    torch, versions = runtime(binding, nr); original.append(binding['qualification_result'])
    loader = load('native15_audit_original_loader', PHASE/provenance['modules']['loader'])
    data = load('native15_audit_original_preprocessing', PHASE/provenance['modules']['inputs'])
    models = load('native15_audit_original_model_factories', PHASE/provenance['modules']['models'])
    device = torch.device('cpu')
    schema, attributes, edges = data.stream_all_attributes(loader, nf['archive'], nf['members'])
    require(schema['node_counts'] == nf['expected_node_counts'] and schema['provided_attribute_widths'] == nf['expected_feature_widths']
            and schema['member_sha256'] == nf['member_sha256'], 'Original full native graph bytes/counts/features differ')
    development = loader.read_development_labels(nf['development_labels'], nf['archive']['sha256'])
    require(development['source_label_member_sha256'] == nf['source_label_member_sha256']
            and development['train_class_schema'] == [0, 1, 2, 3] and len(development['node_ids']) == 1217
            and max(development['node_ids']) < schema['node_counts']['0'], 'Exact source TRAIN/VAL compact label schema required')
    records = data.homogeneous_records(schema, edges)
    graph = data.materialize_homogeneous(records, models, device)
    adjs = data.normalized_native_adjacencies(schema, edges)
    features = data.feature_channels(schema, attributes, adjs); products = data.label_products(adjs)
    del attributes, edges, adjs
    schema.update(homogeneous_audit=records['audit'], SeHGNN_feature_channels={k: list(v.shape) for k, v in features.items()},
        SeHGNN_label_channels=sorted(products), SeHGNN_adjacency_convention='destination-first raw file row; native loader reverses HGT source convention', TEST_label_reads=0)
    require(schema == json.loads(verify(frozen['graph_schema']).read_text()), 'Reconstructed original full graph schema differs')
    original.extend([nf['archive'], nf['development_labels'], nf['paired_HGT_freeze']]+[r['descriptor'] for r in nf['splits']])
    for split_record in nf['splits']:
        seed = split_record['seed']
        split, train_labels, val_labels = loader.verify_split(split_record['descriptor'], development, seed)
        require(len(train_labels) == 974 and len(val_labels) == 243, 'Complete native80/20 source split required')
        label_features = data.train_only_label_channels(products, schema['node_counts']['0'], split['train_ids'], train_labels)
        eval_ids = data.native_evaluation_ids(schema['node_counts']['0'], split)
        used = set(split['train_ids']+split['validation_ids'])
        require(len(eval_ids) == 4057 and eval_ids == split['train_ids']+split['validation_ids']+
            [i for i in range(4057) if i not in used], 'Original complete4057 evaluation composition required')
        for arm in ARMS:
            terminal = next(r for r in study['rows'] if (r['seed'], r['arm']) == (seed, arm))
            case = next(r for r in cases if (r['seed'], r['arm']) == (seed, arm))
            driver.seed_all(torch, seed, device)
            if arm == 'native_SeHGNN':
                model = models.SeHGNN({k: v.shape[1] for k, v in features.items()}, label_features.keys()).to(device)
            else:
                model = models.HGBGAT([schema['node_counts'][str(i)] for i in range(4)], simple=arm == 'native_Simple_HGN').to(device)
            require(sum(p.numel() for p in model.parameters()) == terminal['parameters'], 'Original native parameter count differs')
            saved = torch.load(verify(case['selected_checkpoint']), map_location='cpu', weights_only=True)
            require(set(saved) == {'schema', 'arm', 'seed', 'model', 'optimizer', 'AMP', 'global_RNG', 'selection',
                    'optimizer_parameter_names', 'early_stopping', 'fixed_evaluation_ids', 'bindings'}
                    and saved['schema'] == 'DBLP_native_challenger_complete_state_v1'
                    and saved['arm'] == arm and saved['seed'] == seed and saved['selection'] == terminal['selection']
                    and saved['optimizer_parameter_names'] == [name for name, _ in model.named_parameters()]
                    and saved['AMP'] is None and saved['global_RNG']['torch_CUDA'] is None
                    and saved['early_stopping'] == dict(best=terminal['selection']['validation_NLL'], best_epoch=terminal['selection']['epoch'], counter=0)
                    and saved['fixed_evaluation_ids'] == (eval_ids if arm == 'native_SeHGNN' else None)
                    and saved['bindings'] == dict(study['admission'], archive=nf['archive'], development_labels=nf['development_labels'], graph_schema=schema, split=split_record),
                    'Exact original selected complete checkpoint/source/graph/split/selector binding differs')
            sehgnn = arm == 'native_SeHGNN'
            optimizer = torch.optim.Adam(model.parameters(), lr=.001 if sehgnn else .0005, weight_decay=0. if sehgnn else .0001)
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            driver.restore_global(torch, saved['global_RNG'], device)
            equal_tree(torch, saved['model'], model.state_dict())
            equal_tree(torch, saved['optimizer'], optimizer.state_dict())
            equal_tree(torch, saved['global_RNG'], driver.global_state(torch, device))
            raw = torch.load(verify(case['selected_validation_logits']), map_location='cpu', weights_only=True)
            require(isinstance(raw, torch.Tensor) and raw.dtype == torch.float32 and tuple(raw.shape) == (243, 4)
                    and bool(torch.isfinite(raw).all()), 'Original compact FP32 selected VAL logits required')
            replay = driver.predict_validation(torch, model, arm, graph, features, label_features, split, eval_ids)
            require(replay.dtype == torch.float32 and tuple(replay.shape) == (243, 4) and bool(torch.isfinite(replay).all()), 'Original FP32 replay shape/finite state differs')
            torch.testing.assert_close(replay.cpu(), raw, atol=1e-7, rtol=1e-7)
            score = driver.validation_metrics(torch, replay, torch.tensor(val_labels, dtype=torch.long))
            for key, value in score.items():
                require(value == terminal['selection'][key] if isinstance(value, bool) else abs(value-terminal['selection'][key]) <= 1e-7,
                        'Replayed original selected VAL score differs: '+key)
            case.update(status='audited', complete_selected_state_restored=True, independent_selected_inference_replay=True,
                original_selected_scores_verified=True, original_selector_and_budget_verified=True,
                full_native_evaluation_composition_preserved=True, TRAIN_only_propagated_labels=True)
            del model, optimizer, saved, raw, replay
        del label_features
    require(len(cases) == 15 and all(r['status'] == 'audited' for r in cases), 'Every native15 complete replay required')
    for record in original:
        verify(record)
    sealed_packet(frozen['source_manifest'], frozen['source_seal'])
    sealed_packet(frozen['supervisor_manifest'], frozen['supervisor_seal'])
    source_guard()
    result = dict(schema='complete15_native_development_audit_v1', status='complete', root_observed=True,
        all15_closed=True, training_child_reaped=True, all15_checkpoint_replays_audited=True,
        originals_preserved=True, dataset='HGB-DBLP', heldout_labels_closed=True,
        source_manifest=frozen['source_manifest'], freeze=frozen['training_freeze'], release=frozen['training_release'],
        study=frozen['study'], graph_schema=frozen['graph_schema'], supervisor_receipt=frozen['supervisor_receipt'],
        audit_manifest_sha256=digest(HERE/'MANIFEST.json'), audit_freeze=descriptor(freeze_path), audit_release=descriptor(release_path),
        cases=cases, source_inputs=original, runtime=versions, independent_native_inference_reperformed=True,
        source_validation_row_order_preserved=True, SeHGNN_full4057_evaluation_batch=True,
        propagated_labels_scope='TRAIN one-hot only; diagonal removed after complete normalized label products',
        TEST_label_reads=0, TEST_diagnostics=0, successful_subset_scored=False, original_scores_changed=False,
        new_metrics_or_calibration_or_tuning=False, scientific_gate_evaluated=False, superiority_claim=False)
    write(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', required=True); parser.add_argument('--admission', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = run(args.freeze, args.admission, args.output)
    print(json.dumps(dict(status=result['status'], all15_checkpoint_replays_audited=True, heldout_labels_closed=True, output=args.output)))
