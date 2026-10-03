"""Closed full40 development audit/calibration; explicit root release required."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SEEDS = [131, 137, 139, 149, 151]
DATASETS = ['HGB-DBLP', 'HGB-ACM']
POLICIES = ['own/own', 'pool/pool', 'pool/own', 'own/pool']
CLASSES = {'HGB-DBLP': [0, 1, 2, 3], 'HGB-ACM': [0, 1, 2]}
NATIVE_ARMS = ['native_GAT', 'native_Simple_HGN', 'native_SeHGNN']


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
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_guard():
    manifest_sha = digest(HERE/'MANIFEST.json')
    require(json.loads((HERE/'SEAL.json').read_text())['manifest_sha256'] == manifest_sha, 'Evaluation seal differs')
    for row in json.loads((HERE/'MANIFEST.json').read_text())['payload']:
        verify(dict(row, path=str(HERE/row['path'])))
    provenance = json.loads((HERE/'PROVENANCE.json').read_text())
    for row in provenance['source_records']:
        verify(row)
    return provenance, manifest_sha


def admission(freeze_path, release_path):
    provenance, manifest_sha = source_guard()
    freeze_record, release_record = descriptor(freeze_path), descriptor(release_path)
    frozen, release = json.loads(Path(freeze_path).read_text()), json.loads(Path(release_path).read_text())
    require(frozen['schema'] == 'mixed_full40_development_evaluation_freeze_v1'
            and frozen['evaluation_scope'] == 'source_validation' and frozen['heldout_labels_closed'] is True
            and frozen['datasets'] == DATASETS and frozen['seeds'] == SEEDS and frozen['policies'] == POLICIES
            and frozen['class_schemas'] == CLASSES, 'Exact full40 development/class freeze required')
    for key in ('training_manifest', 'training_freeze', 'training_release', 'scientific_design', 'calibration_core'):
        require(frozen[key]['sha256'] == provenance['authorities'][key]['sha256'], 'Frozen authority differs: '+key)
        verify(frozen[key])
    require(release['execution_authorized'] is True and release['evaluation_manifest_sha256'] == manifest_sha
            and release['evaluation_freeze_sha256'] == freeze_record['sha256']
            and release['study_sha256'] == frozen['study']['sha256']
            and release['root_observed_closed_full40'] is True and release['training_child_reaped'] is True
            and release['heldout_labels_closed'] is True and release['device'] == 'cpu' and release['threads'] == 1
            and release['independent_source_review_observed'] is True,
            'Explicit exact root release and closed/reaped full40 required')
    review_record = release['independent_source_review']; verify(review_record)
    require(Path(frozen['expected_output_path']).is_absolute(), 'Exact new absolute evaluation output required')
    training = json.loads(verify(frozen['training_freeze']).read_text())
    training_release = json.loads(verify(frozen['training_release']).read_text())
    require(training['schema'] == 'HGB_mixed_block_executable_freeze_v1' and training['datasets'] == DATASETS
            and training['seeds'] == SEEDS and training['policies'] == POLICIES and training['terminals'] == 40
            and training['baseline_reuse'] == {'mode': 'fresh_all40', 'audit': None}
            and training['model'] == 'global_BE' and training['members'] == 4 and training['CP_excluded'] is True,
            'This evaluator binds the actual fresh-all40 source run')
    require(training_release['execution_authorized'] is True
            and training_release['prepared_manifest_sha256'] == frozen['training_manifest']['sha256']
            and training_release['study_freeze_sha256'] == frozen['training_freeze']['sha256']
            and training_release['scientific_design_sha256'] == frozen['scientific_design']['sha256'],
            'Original training source/freeze/release differs')
    require(training['source_freezes'] == provenance['graph_authorities'], 'Immutable two-graph source authorities differ')
    return provenance, frozen, release, training, [freeze_record, release_record, review_record]


def latest_tie_trace(driver, trace, selection, updates):
    require(trace and [r['epoch'] for r in trace] == list(range(1, updates+1)) and updates <= 300,
            'All consecutive native post-update trace rows required')
    stopper = driver.NativeEarlyStop(); saved = []
    for index, row in enumerate(trace):
        save, stop = stopper.observe(row['validation_NLL'])
        require(row['checkpoint_replaced'] == save and row['patience_counter'] == stopper.counter,
                'Native latest-tie/patience trace differs')
        require(not stop or index == len(trace)-1, 'Trace continued after native early stop')
        if save:
            saved.append(row)
    require(updates == 300 or stop, 'Short trace without native stopping')
    require(saved and selection['epoch'] == saved[-1]['epoch'], 'Selected epoch differs from latest eligible minimum')
    for key, value in selection.items():
        require(saved[-1][key] == value, 'Selected trace metric differs: '+key)


def closed_full40(frozen, driver):
    """Before any Torch, labels or tensor payload: require the complete family."""
    study_path = verify(frozen['study'])
    require(study_path.name == 'STUDY.json' and str(study_path.parent) == frozen['expected_run_directory'],
            'Exact root full40 output slot required')
    study = json.loads(study_path.read_text())
    require(study['schema'] == 'HGB_mixed_block_full40_training_v1'
            and study['summary']['status'] == 'complete_development_summary'
            and study['summary']['all40_terminals'] is True and study['summary']['successful_subset_scored'] is False
            and study['originals_preserved'] is True and study['final_labels_closed'] is True
            and study['baseline_reuse_mode'] == 'fresh_all40' and study['CP_gate_dependency'] is False,
            'Complete immutable full40 closure required; no subset comparison')
    expected = [(d, s, p) for d in DATASETS for s in SEEDS for p in POLICIES]
    require([(r['dataset'], r['seed'], r['policy']) for r in study['rows']] == expected, 'Exactly40 ordered unique source terminals')
    require(all(row['status'] == 'selected' and row['selected_state_replay'] is True
                and row['checkpoint_bindings_verified'] is True and row['reused'] is False
                and row['arm'] == row['policy'] and row['final_labels_closed'] is True and row['label_payloads_saved'] is False
                for row in study['rows']), 'All40 selected binding-verified replays required')
    admission = study['admission']
    for key, authority in (('prepared_manifest_sha256', 'training_manifest'), ('study_freeze_sha256', 'training_freeze'),
                           ('root_release_sha256', 'training_release'), ('scientific_design_sha256', 'scientific_design')):
        require(admission[key] == frozen[authority]['sha256'], 'Source study admission differs: '+key)
    require(admission['device'] == 'cpu' and admission['threads'] == 1 and admission['test_labels_closed'] is True,
            'Original native CPU/development predictor differs')
    original = [frozen['study'], frozen['training_manifest'], frozen['training_freeze'], frozen['training_release'],
                frozen['scientific_design'], frozen['calibration_core']]
    started = descriptor(study_path.parent/'STUDY_STARTED.json'); original.append(started)
    require(json.loads(verify(started).read_text()) == admission, 'Study start/admission custody differs')
    artifact_names = dict(selected_checkpoint='selected.pt', selected_logits='selected_member_logits.pt',
        selection_receipt='SELECTION.json', training_trace='TRACE.jsonl', selected_descriptives='SELECTED_DESCRIPTIVES.json',
        fixed_final_and_optimizer_diagnostics='FIXED_FINAL_AND_OPTIMIZER_DIAGNOSTICS.json')
    for row in study['rows']:
        case = study_path.parent/row['dataset']/f"seed{row['seed']}"/row['policy'].replace('/', '__')
        for key, name in artifact_names.items():
            path = Path(row[key]['path'])
            require(path.is_absolute() and path == case/name and path.resolve() == path, 'Artifact outside exact source case: '+key)
            verify(row[key]); original.append(row[key])
        terminal = descriptor(case/'TERMINAL.json'); original.append(terminal)
        require(json.loads(verify(terminal).read_text()) == row, 'Closed study/terminal JSON differs')
        selection = json.loads(verify(row['selection_receipt']).read_text())
        # Wrapper wall_seconds includes audit/descriptive time and intentionally
        # differs from the original immutable fit's wall_seconds field.
        for key, value in selection.items():
            if key != 'wall_seconds':
                require(row.get(key) == value, 'Immutable selection/source row differs: '+key)
        trace = [json.loads(line) for line in verify(row['training_trace']).read_text().splitlines()]
        latest_tie_trace(driver, trace, row['selection'], row['updates'])
        auxiliary = json.loads(verify(row['fixed_final_and_optimizer_diagnostics']).read_text())
        require(auxiliary == dict(fixed_final_endpoint=row['fixed_final_endpoint'], optimizer_state_cost=row['optimizer_state_cost']),
                'Fixed-final/optimizer auxiliary source JSON differs')
    return study, original


def model_replay(torch, common, mods, trainer, blocks_module, row, study, graph_freeze, split_record, graph, features, schema):
    model = common.build(mods, row['dataset'], graph, schema, len(CLASSES[row['dataset']]), row['seed'])
    require(common.model_sha(model) == row['initial_model_sha256'], 'Paired initialization differs')
    require(mods['implementation'].parameter_count(model) == row['parameters'], 'Immutable selected model parameter accounting differs')
    blocks = blocks_module.partition(mods['implementation'], model)
    checkpoint = torch.load(verify(row['selected_checkpoint']), map_location='cpu', weights_only=True)
    require(checkpoint['schema'] == 'DBLP_selected_complete_state_v1' and checkpoint['arm'] == row['policy']
            and checkpoint['seed'] == row['seed'] and checkpoint['selection'] == row['selection']
            and checkpoint['optimizer_parameter_names'] == [n for n, _ in model.named_parameters()], 'Selected state/name/selector differs')
    bound = checkpoint['bindings']
    expected_bindings = dict(study['admission'], dataset=row['dataset'], model='global_BE', policy=row['policy'],
        archive=graph_freeze['archive'], development_labels=graph_freeze['development_labels'],
        implementation_manifest_sha256=graph_freeze['implementation_manifest_sha256'], graph_schema=schema,
        split=split_record, initial_model_sha256=row['initial_model_sha256'], module_semantic_partition=blocks['receipt'],
        fit_body_custody=json.loads((common.PACKET/'FIT_BODY_CUSTODY.json').read_text()))
    require(bound == expected_bindings, 'Exact selected source/graph/split/backward binding differs')
    require(checkpoint['early_stopping'] == dict(best=row['selection']['validation_NLL'], counter=0),
            'Selected native early-stopping state differs')
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=300, max_lr=1e-3, pct_start=.05)
    streams = mods['implementation'].MemberStreams([row['seed']+700001+1009*m for m in range(4)])
    model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
    mods['onecycle'].restore_state(scheduler, checkpoint['scheduler'])
    streams.load_state_dict(checkpoint['member_RNG']); torch.set_rng_state(checkpoint['torch_CPU_RNG'].cpu())
    require(checkpoint['torch_device_RNG'] is None, 'CPU selected predictor required')
    require(trainer.optimizer_tensor_cost(torch, checkpoint['optimizer']) == row['optimizer_state_cost'], 'Selected optimizer tensor accounting differs')
    model.eval()
    with torch.no_grad():
        replay = model(graph, features, '0')
    raw = torch.load(verify(row['selected_logits']), map_location='cpu', weights_only=True)
    require(isinstance(raw, torch.Tensor) and raw.dtype == torch.float32
            and tuple(raw.shape) == (4, schema['node_counts']['0'], len(CLASSES[row['dataset']])), 'Full FP32 saved member-logit shape differs')
    torch.testing.assert_close(replay, raw, atol=1e-7, rtol=1e-7)
    return raw, dict(full_state_restored=True, selected_inference_replay=True,
        checkpoint_source_binding_verified=True, optimizer_tensor_cost=row['optimizer_state_cost'])


def source_score_check(metrics, selection):
    for source, section, name in (('validation_NLL', 'raw', 'NLL_FP32'),
                                  ('validation_micro_F1', 'raw', 'micro_F1'), ('validation_macro_F1', 'raw', 'macro_F1')):
        require(abs(metrics[section][name]-selection[source]) <= 1e-7, 'Saved-logit/source FP32 score differs: '+source)


def native_secondary(torch, frozen, primary_rows, core, provenance):
    """Optional complete15 saved-VAL table, explicitly retaining native recipes."""
    if frozen.get('native_secondary') is None:
        return None, []
    lock_record = frozen['native_secondary']
    lock = json.loads(verify(lock_record).read_text()); original = [lock_record]
    require(lock['schema'] == 'complete15_native_development_audit_v1' and lock['root_observed'] is True
            and lock['all15_closed'] is True and lock['training_child_reaped'] is True and lock['all15_checkpoint_replays_audited'] is True
            and lock['originals_preserved'] is True and lock['dataset'] == 'HGB-DBLP'
            and lock['heldout_labels_closed'] is True, 'Separately bound complete15 native audit required')
    require(lock['source_manifest']['sha256'] == provenance['authorities']['native_source_manifest']['sha256']
            and lock['release']['sha256'] == provenance['authorities']['native_release']['sha256'],
            'Immutable native-v2 source/release authority differs')
    for key in ('source_manifest', 'freeze', 'release', 'study', 'graph_schema'):
        verify(lock[key]); original.append(lock[key])
    nf = json.loads(verify(lock['freeze']).read_text()); study = json.loads(verify(lock['study']).read_text())
    nr = json.loads(verify(lock['release']).read_text())
    native_schema = json.loads(verify(lock['graph_schema']).read_text())
    native_run = PHASE/'graph_heterogeneous_dblp_native_challengers_preparation_20261003_v2'/'runs'/nr['run_name']
    require(verify(lock['study']) == native_run/'STUDY.json' and verify(lock['graph_schema']) == native_run/'GRAPH_SCHEMA.json',
            'Native complete15 exact original output slots required')
    require(nr['execution_authorized'] is True and nr['study_freeze_sha256'] == lock['freeze']['sha256']
            and nr['prepared_manifest_sha256'] == lock['source_manifest']['sha256']
            and nr['device'] == 'cpu' and nr['threads'] == 1, 'Native original CPU release binding differs')
    require(nf['seeds'] == SEEDS and nf['arms'] == NATIVE_ARMS and nf['test_labels_closed'] is True
            and study['summary']['status'] == 'complete_development_challengers'
            and study['original_inputs_unchanged'] is True and study['summary']['successful_subset_scored'] is False
            and study['summary']['all_frozen_terminals'] is True and study['summary']['TEST_label_reads'] == 0
            and study['summary']['TEST_diagnostics'] == 0 and nf['study_adopted_by_root'] is True,
            'Native original complete15 source schema required')
    primary = json.loads(verify(next(x['record'] for x in frozen['source_freezes'] if x['dataset'] == 'HGB-DBLP')).read_text())
    for key in ('archive', 'development_labels', 'splits', 'source_label_member_sha256'):
        require(nf[key] == primary[key], 'Native/mixed paired development input differs: '+key)
    require(study['admission']['study_freeze_sha256'] == lock['freeze']['sha256']
            and study['admission']['prepared_manifest_sha256'] == lock['source_manifest']['sha256']
            and study['admission']['root_admission_sha256'] == lock['release']['sha256']
            and study['admission']['TEST_labels_closed'] is True, 'Native source admission differs')
    require(native_schema['node_counts'] == nf['expected_node_counts']
            and native_schema['provided_attribute_widths'] == nf['expected_feature_widths']
            and native_schema['member_sha256'] == nf['member_sha256']
            and native_schema['TEST_label_reads'] == 0
            and {str(r['raw_id']): [r['source'], r['target']] for r in native_schema['relations']} == nf['expected_relations'],
            'Native original full graph-schema binding differs')
    expected = [(s, a) for s in SEEDS for a in NATIVE_ARMS]
    require([(r['seed'], r['arm']) for r in study['rows']] == expected
            and [(r['seed'], r['arm']) for r in lock['cases']] == expected, 'Exactly all15 native records required')
    require(all(r['status'] == 'selected' and r['selected_state_replay'] is True
                and r['TEST_label_reads'] == 0 and r['classifier_classes_from_TRAIN'] == 4
                for r in study['rows']), 'All15 original native selected replays required')
    native_started = descriptor(native_run/'STUDY_STARTED.json'); original.append(native_started)
    require(json.loads(verify(native_started).read_text()) == study['admission'], 'Native original start/admission differs')
    labels = json.loads(verify(nf['development_labels']).read_text()); original.append(nf['development_labels'])
    require(labels['scope'] == 'TRAIN_VAL_ONLY' and labels['train_class_schema'] == CLASSES['HGB-DBLP'], 'Native VAL-only fixed-class input required')
    lookup = dict(zip(labels['node_ids'], labels['labels'])); rows = []
    for terminal, case in zip(study['rows'], lock['cases']):
        source_case = native_run/f"seed{terminal['seed']}"/terminal['arm']
        for key, filename in dict(selection_receipt='SELECTION.json', selected_checkpoint='selected.pt',
                                  selected_validation_logits='selected_validation_logits.pt', training_trace='TRACE.jsonl').items():
            require(Path(case[key]['path']) == source_case/filename and Path(case[key]['path']).resolve() == source_case/filename,
                    'Native artifact outside exact source case: '+key)
            verify(case[key]); original.append(case[key])
        require(json.loads(verify(case['selection_receipt']).read_text()) == terminal, 'Native immutable selection receipt differs')
        split_record = next(x for x in nf['splits'] if x['seed'] == terminal['seed'])['descriptor']
        split = json.loads(verify(split_record).read_text()); original.append(split_record)
        ids = torch.tensor(split['validation_ids'], dtype=torch.long)
        targets = torch.tensor([lookup[i] for i in split['validation_ids']], dtype=torch.long)
        trace = [json.loads(s) for s in verify(case['training_trace']).read_text().splitlines()]
        require([r['epoch'] for r in trace] == list(range(1, terminal['epochs_paid']+1)), 'Native complete epoch trace required')
        minimum = min(r['validation_NLL'] for r in trace)
        tied = [r for r in trace if r['validation_NLL'] == minimum]
        selected = tied[0] if terminal['arm'] == 'native_SeHGNN' else tied[-1]
        require(selected['epoch'] == terminal['selection']['epoch'] and minimum == terminal['selection']['validation_NLL'], 'Native per-recipe tie selector differs')
        checkpoint = torch.load(verify(case['selected_checkpoint']), map_location='cpu', weights_only=True)
        require(checkpoint['schema'] == 'DBLP_native_challenger_complete_state_v1'
                and checkpoint['seed'] == terminal['seed'] and checkpoint['arm'] == terminal['arm']
                and checkpoint['selection'] == terminal['selection']
                and checkpoint['bindings'] == dict(study['admission'], split=next(x for x in nf['splits'] if x['seed'] == terminal['seed']),
                    archive=nf['archive'], development_labels=nf['development_labels'], graph_schema=native_schema),
                'Native selected checkpoint custody differs')
        raw = torch.load(verify(case['selected_validation_logits']), map_location='cpu', weights_only=True)
        require(isinstance(raw, torch.Tensor) and raw.dtype == torch.float32 and tuple(raw.shape) == (len(ids), 4), 'Native source saved VAL row schema differs')
        calibration = core.fit_inverse_temperature(raw, ids, ids, targets, CLASSES['HGB-DBLP'], label_scope='SOURCE_VALIDATION_ONLY')
        metrics = core.evaluate_logits(raw, ids, ids, targets, CLASSES['HGB-DBLP'], calibration=calibration, evaluation_scope='source_validation')
        source_score_check(metrics, terminal['selection'])
        rows.append(dict(arm=terminal['arm'], seed=terminal['seed'], metrics=metrics, source_FP32_selection=terminal['selection'],
            native_original_checkpoint_replay_root_audited=True, evaluator_native_inference_reperformed=False,
            native_recipe_retained=True, fixed_final_trace={k: trace[-1][k] for k in ('epoch', 'validation_NLL', 'validation_macro_F1')}))
    binding = dict(dataset='HGB-DBLP', evaluation_scope='source_validation', class_schema=CLASSES['HGB-DBLP'],
        immutable=True, seeds=SEEDS, arms=POLICIES+NATIVE_ARMS, family_freeze_sha256=frozen['training_freeze']['sha256'],
        evaluation_manifest_sha256=frozen['_evaluation_manifest_sha256'], predictor_lock_sha256=lock_record['sha256'])
    combined = [dict(r, family_binding=binding, status='complete') for r in primary_rows if r['dataset'] == 'HGB-DBLP']
    combined.extend(dict(r, family_binding=binding, status='complete') for r in rows)
    summary = core.paired_summary(combined, binding, reference_arm='pool/own')
    require(summary['status'] == 'complete_paired_summary', 'Complete primary/native secondary table required')
    summary['uncertainty_scope'] = 'Conditional descriptive stability on fixed DBLP graph and overlapping development split blocks; not independent datasets or confirmatory testing'
    for record in original:
        verify(record)
    return dict(scope='secondary development table; original native recipes and selectors, unequal models/budgets',
                native_rows=rows, summary=summary, new_comparison_gate=False), original


def run(freeze_path, release_path, output):
    provenance, frozen, release, training, original = admission(freeze_path, release_path)
    require(Path(output).is_absolute() and str(Path(output)) == frozen['expected_output_path'] and not Path(output).exists(),
            'Exact fresh root-bound evaluation output required')
    core = load('mixed_eval_qualified_calibration_core', verify(frozen['calibration_core']))
    common = load('mixed_eval_immutable_common', PHASE/provenance['modules']['common'])
    driver = load('mixed_eval_native_fit_metadata', PHASE/provenance['modules']['driver'])
    study, custody = closed_full40(frozen, driver); original.extend(custody)
    # No label/tensor payload, inference or calibration enters before full40.
    torch, runtime = common.runtime()
    training_provenance, _ = common.packet_guard(); mods = common.modules(training_provenance)
    sys.modules['common'] = common
    blocks = load('block_policy', PHASE/provenance['modules']['block_policy'])
    trainer = load('mixed_eval_immutable_trainer', PHASE/provenance['modules']['trainer'])
    decomposition = load('mixed_eval_known_decomposition', PHASE/provenance['modules']['decomposition'])
    rows = []
    for dataset, source in zip(DATASETS, training['source_freezes']):
        gf = json.loads(verify(source['record']).read_text()); original.append(source['record'])
        inputs, graph, features, schema, development = common.graph_inputs(mods, dataset, gf, mods['implementation'], torch)
        require(development['train_class_schema'] == CLASSES[dataset], 'Graph development class schema differs')
        original.extend([gf['archive'], gf['development_labels']]+[r['descriptor'] for r in gf['splits']])
        graph_schema = descriptor(Path(frozen['study']['path']).parent/dataset/'GRAPH_SCHEMA.json'); original.append(graph_schema)
        require(json.loads(verify(graph_schema).read_text()) == schema, 'Reconstructed full graph schema differs')
        for split_record in gf['splits']:
            split, _, validation_labels = inputs.verify_split(split_record['descriptor'], development, split_record['seed'])
            ids = torch.tensor(split['validation_ids'], dtype=torch.long)
            targets = torch.tensor(validation_labels, dtype=torch.long)
            row_ids = torch.arange(schema['node_counts']['0'], dtype=torch.long)
            for policy in POLICIES:
                source_row = next(r for r in study['rows'] if (r['dataset'], r['seed'], r['policy']) == (dataset, split_record['seed'], policy))
                raw, replay = model_replay(torch, common, mods, trainer, blocks, source_row, study, gf, split_record, graph, features, schema)
                calibration = core.fit_inverse_temperature(raw, row_ids, ids, targets, CLASSES[dataset], label_scope='SOURCE_VALIDATION_ONLY')
                metrics = core.evaluate_logits(raw, row_ids, ids, targets, CLASSES[dataset], calibration=calibration, evaluation_scope='source_validation')
                source_score_check(metrics, source_row['selection'])
                descriptive = trainer.selected_descriptives(torch, driver, mods['implementation'], raw, split, validation_labels, len(CLASSES[dataset]))
                require(descriptive == json.loads(verify(source_row['selected_descriptives']).read_text()), 'Selected saved-logit descriptives differ')
                endpoint = trainer.fixed_final_endpoint(source_row['training_trace'], source_row['updates'])
                require(endpoint == source_row['fixed_final_endpoint'], 'Fixed final trace endpoint differs')
                rows.append(dict(dataset=dataset, seed=split_record['seed'], arm=policy, status='complete', metrics=metrics,
                    source_FP32_selection=source_row['selection'], checkpoint_replay=replay, selected_descriptives=descriptive,
                    paid_training_receipt={k: source_row[k] for k in ('updates', 'parameters', 'wall_seconds', 'memory', 'optimizer_state_cost')},
                    fixed_final_trace=endpoint, known_FP64_own_pool_decomposition=decomposition.decomposition(torch, raw[:, ids], targets),
                    decomposition_arithmetic='convert individual member logits to FP64 before mean; not served FP32 or FP64-after-served-mean scores'))
                del raw
        del graph, features, development
    summaries, accounting, final = {}, {}, {}
    for dataset in DATASETS:
        binding = dict(dataset=dataset, evaluation_scope='source_validation', class_schema=CLASSES[dataset], immutable=True,
            seeds=SEEDS, arms=POLICIES, family_freeze_sha256=frozen['training_freeze']['sha256'],
            evaluation_manifest_sha256=digest(HERE/'MANIFEST.json'), predictor_lock_sha256=frozen['study']['sha256'])
        graph_rows = [dict(r, family_binding=binding) for r in rows if r['dataset'] == dataset]
        summaries[dataset] = core.paired_summary(graph_rows, binding, reference_arm='pool/own')
        require(summaries[dataset]['status'] == 'complete_paired_summary', 'Allfive/four-policy complete calibration summary required')
        summaries[dataset]['uncertainty_scope'] = 'Conditional descriptive stability on one fixed graph and overlapping development split blocks; not independent datasets or confirmatory testing'
        accounting[dataset], final[dataset] = {}, {}
        for control in ('own/own', 'pool/pool', 'own/pool'):
            reference = [next(r for r in graph_rows if r['arm'] == 'pool/own' and r['seed'] == s) for s in SEEDS]
            others = [next(r for r in graph_rows if r['arm'] == control and r['seed'] == s) for s in SEEDS]
            accounting[dataset][control] = {key: core.descriptive_five([a['known_FP64_own_pool_decomposition'][key]-b['known_FP64_own_pool_decomposition'][key]
                for a, b in zip(reference, others)]) for key in ('mean_member_NLL', 'pool_NLL_FP64', 'mean_ambiguity')}
            final[dataset][control] = {key: core.descriptive_five([a['fixed_final_trace'][key]-b['fixed_final_trace'][key] for a, b in zip(reference, others)])
                for key in ('validation_NLL', 'validation_macro_F1')}
    frozen['_evaluation_manifest_sha256'] = digest(HERE/'MANIFEST.json')
    frozen['source_freezes'] = training['source_freezes']
    try:
        secondary, secondary_inputs = native_secondary(torch, frozen, rows, core, provenance)
        if secondary is not None:
            secondary['source_inputs'] = secondary_inputs
    except Exception as error:
        secondary = dict(status='not_admitted', reason=type(error).__name__+': '+str(error),
                         audit_record=frozen.get('native_secondary'), successful_subset_scored=False,
                         native_table_scored=False, primary_evaluation_blocked=False)
    for record in original:
        verify(record)
    source_guard()
    primary_contrasts = {dataset: {metric: {control: summaries[dataset]['reference_minus_control'][metric][control]
        for control in ('own/own', 'pool/pool')} for metric in ('raw.NLL_FP32', 'raw.macro_F1')} for dataset in DATASETS}
    role_contrasts = {dataset: {metric: summaries[dataset]['reference_minus_control'][metric]['own/pool']
        for metric in ('raw.NLL_FP32', 'raw.macro_F1')} for dataset in DATASETS}
    result = dict(schema='mixed_full40_closed_development_evaluation_v1', status='complete', rows=rows,
        frozen_primary_selected_raw_contrasts=primary_contrasts, frozen_role_assignment_selected_raw_contrasts=role_contrasts,
        contrast_direction='pool/own minus named control; negative NLL and positive macro-F1 favor pool/own',
        paired_selected_raw_and_calibrated=summaries, known_decomposition_contrasts=accounting,
        fixed_final_trace_paired=final, native_secondary=secondary, source_inputs=original, runtime=runtime,
        full40_comparison_denominator=True, successful_subset_scored=False, originals_preserved=True,
        heldout_labels_opened=False, evaluation_scope='source_validation',
        calibration_fit_and_evaluation_share_source_VAL=True, calibration_core_unchanged=True,
        known_decomposition_new_theorem_claim=False, scientific_design_sha256=frozen['scientific_design']['sha256'],
        uncertainty_scope='Conditional on two fixed graphs and their overlapping development split blocks; illustrative intervals and LOBO are descriptive only',
        calibration_or_new_contrast_may_rescue_primary_gate_failure=False,
        practical_scientific_gate_evaluated=False, verified_predictive_advantage_claim=False, paper_acceptance_claim=False)
    write(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', required=True); parser.add_argument('--admission', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = run(args.freeze, args.admission, args.output)
    print(json.dumps(dict(status=result['status'], heldout_labels_opened=False, output=args.output)))
