"""Inactive centered-prior extension gates; exact V2 role/native custody retained."""


import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent


def source_checks():
    seal=read(HERE/'SEAL.json')
    require(seal['source_only'] is True and seal['execution_enabled'] is False and sha(HERE/'MANIFEST.json')==seal['manifest_sha256'],'Exact inactive centered source seal')
    for root,rows in ((HERE,read(HERE/'MANIFEST.json')['files']),(PHASE,read(HERE/'SOURCE_BINDINGS.json')['files'])):
        for row in rows:
            path=(root/row['path']).resolve(strict=True)
            require(path.is_relative_to(root) and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Exact centered/pinned source bytes')
    return read(HERE/'SOURCE_BINDINGS.json')


def borrowed(pins):
    core=load(PHASE/pins['scientific_core_directory']/'support.py','_centered_original_V2_support')
    core.source_checks()
    return core.borrowed(core.source_checks())


def load_centered_fit(pins):
    absent=object(); previous={name:sys.modules.get(name,absent) for name in ('bank','support')}
    try:
        sys.modules['support']=sys.modules[__name__]
        sys.modules['bank']=load(PHASE/pins['scientific_core_directory']/'bank.py','_centered_original_V2_bank')
        return load(HERE/'centered_fit.py','_centered_shared_fit')
    finally:
        for name,value in previous.items():
            if value is absent: sys.modules.pop(name,None)
            else: sys.modules[name]=value


def hashed(row):
    path=Path(row['path']).resolve(strict=True)
    require(path.is_relative_to(PHASE) and sha(path)==row['sha256'],'Exact reused artifact/revised scope')
    if 'bytes' in row: require(path.stat().st_size==row['bytes'],'Exact reused artifact bytes')
    return path


def validate_revision(release,pins):
    scope=read(hashed(release['revised_scientific_scope']))
    require(scope['release_owner']=='root' and scope['enabled'] is True and scope['declared_before_any_scientific_fit'] is True
            and scope['scientific_fit_launches_at_declaration']==0 and scope['comparative_outcomes_accessed_at_declaration'] is False
            and scope['centered_candidate_co_primary'] is True and scope['vanilla_V2_required_control'] is True
            and scope['required_logical_records']==21 and scope['paired_base_seeds']==[7409,8501,9607]
            and scope['centered_source_seal_sha256']==sha(HERE/'SEAL.json')
            and scope['vanilla_source_seal_sha256']==pins['scientific_core_seal_sha256']
            and scope['centered_prior_coefficient']==0.0005 and scope['centered_fast_optimizer_decay']==0.0
            and scope['original18_required_before_reference_reuse'] is True and scope['all21_required_before_comparative_opening'] is True,
            'Explicit prospective two-recipe co-primary freeze before any SCI, no after-outcome tuning')


def key(row): return (row['kind'],row['base_seed'],row.get('member',-1))


def validate_origins(release,receipt,config,pins):
    origins=read(hashed(release['original18_origins']))
    complete=read(hashed(origins['original_complete']))
    bases=[7409,8501,9607]
    expected={('shared_fit',base,-1) for base in bases}|{('independent_pool',base,-1) for base in bases}|{('independent_member',base,m) for base in bases for m in range(4)}
    require(complete['complete'] is True and len(complete['records'])==18 and {key(row) for row in complete['records']}==expected
            and all(row['status']=='complete' for row in complete['records']) and origins['comparative_outcomes_unopened'] is True
            and origins['all_original_costs_charged'] is True and len(origins['records'])==18 and {key(row) for row in origins['records']}==expected,
            'Complete unchanged original18 required; all paired origins/costs retained before reference reuse')
    checkpoints={(row['base_seed'],row['member']):next(a['sha256'] for a in row['artifacts'] if a['label']=='checkpoint')
                 for row in origins['records'] if row['kind']=='independent_member'}
    for origin in origins['records']:
        record=read(hashed(origin['result']))
        require(key(record)==key(origin) and record['status']==origin['status']=='complete','Exact complete original component header')
        identity=(record['native_result']['identity'] if record['kind']=='independent_member' else record['identity']
                  if record['kind']=='shared_fit' else record['components'][0]['native_result']['identity'])
        require(identity['source_seal_sha256']==pins['scientific_core_seal_sha256'] and identity['configuration']==config
                and identity['role_archive_sha256']==receipt['roles_sha256'] and identity['role_metadata_sha256']==receipt['role_metadata_sha256'],
                'Same complete paired native architecture/source/role origins')
        if record['kind']=='independent_member':
            require(record['seed']==record['base_seed']+1000003*record['member'] and record['reused'] is False,'Originally fresh own independent body/seed')
        if record['kind']=='independent_pool':
            require(len(record['member_bindings'])==4 and {row['member'] for row in record['member_bindings']}==set(range(4))
                    and all(row['seed']==record['base_seed']+1000003*row['member']
                    and row['checkpoint_sha256']==checkpoints[(record['base_seed'],row['member'])] for row in record['member_bindings']),
                    'Pool uses these exact four own selected paired independent origins')
        required={'native_result','history','checkpoint'} if record['kind']=='independent_member' else {'history','checkpoint','serving_reference'} if record['kind']=='shared_fit' else {'serving_reference'}
        require(required.issubset({row['label'] for row in origin['artifacts']}),'Complete selected/history/fresh-serving origin bindings')
        for row in origin['artifacts']: hashed(row)
        artifacts={row['label']:row for row in origin['artifacts']}
        if record['kind'] in ('shared_fit','independent_member'):
            require(artifacts['checkpoint']['sha256']==record['checkpoint_sha256'],'Exact own selected checkpoint origin')
            require(Path(artifacts['history']['path']).resolve().parent==Path(artifacts['checkpoint']['path']).resolve().parent,'Own complete history beside selected state')
        if record['kind'] in ('shared_fit','independent_pool'):
            require(artifacts['serving_reference']['sha256']==record['serving_reference_sha256'],'Exact actual fresh selected-serving reference origin')
        require(origin['costs_charged_in_full'] is True and bool(origin['cost_fields']),'Every original native fit/serving/constructor/checkpoint/cleanup cost charged')
        minimum=({'complete_attempt_seconds','CPU_user_seconds','CPU_system_seconds','native_result.complete_attempt_seconds',
                  'native_result.complete_attempt_cuda_peak_allocated_bytes','native_result.process_peak_rss_bytes_cumulative',
                  'native_result.forward_counts.train_forwards','native_result.forward_counts.validation_forwards','native_result.forward_counts.restored_report_forwards'}
                 if record['kind']=='independent_member' else {'complete_attempt_seconds','CPU_user_seconds','CPU_system_seconds','peak_CUDA_allocated_bytes','peak_CUDA_reserved_bytes'}
                 if record['kind']=='shared_fit' else {'selected_serving_seconds','selected_serving_CPU_user_seconds','selected_serving_CPU_system_seconds',
                  'selected_serving_peak_CUDA_allocated_bytes','selected_serving_peak_CUDA_reserved_bytes','serving_constructors_completed','serving_forwards_completed'})
        require(minimum.issubset(origin['cost_fields']),'Full original fit and serving cost fields, no free reused controls')
        for name,value in origin['cost_fields'].items():
            require('scores' not in name and any(word in name for word in ('seconds','bytes','cost','forwards','constructors','backwards','Adam')),'Cost-only reference receipt fields')
            actual=record
            for part in name.split('.'): actual=actual[part]
            require(actual==value,'Exact original charged cost field')
    return origins


def require(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def topology_bytes(torch, models):
    names = ('edge_index', 'full_left_right_idx', 'left_right_idx', 'vertex_tril_idx', 'diag_indices',
             'tril_indices', 'fixed_diag_indices', 'fixed_tril_indices', 'deg')
    seen, total = set(), 0
    for model in models:
        for owner in (model, model.laplacian_builder):
            for name in names:
                value = getattr(owner, name, None)
                if isinstance(value, torch.Tensor):
                    storage = value.untyped_storage()
                    key = (str(value.device), storage.data_ptr())
                    if key not in seen:
                        seen.add(key); total += storage.nbytes()
    return total


def replay_materiality(torch, scores, role_logp, saved_scores, saved_logp):
    require(all(role_logp[role].shape == saved_logp[role].shape and role_logp[role].ndim == 2
                and role_logp[role].shape[1] == 2 and torch.isfinite(role_logp[role]).all().item()
                and torch.isfinite(saved_logp[role]).all().item() for role in ('train','valid')),
            'Finite matching binary reconstructed/saved serving log probabilities')
    diagnostics = {role: dict(max_abs_logp=float((role_logp[role]-saved_logp[role]).abs().max().item()),
        mean_abs_logp=float((role_logp[role]-saved_logp[role]).abs().mean().item()),
        max_abs_probability=float((role_logp[role].exp()-saved_logp[role].exp()).abs().max().item()),
        mean_abs_probability=float((role_logp[role].exp()-saved_logp[role].exp()).abs().mean().item()),
        prediction_changes=int((role_logp[role].argmax(-1) != saved_logp[role].argmax(-1)).sum().item()),
        metric_signed_differences={name: float(scores[role][name]-saved_scores[role][name]) for name in scores[role]},
        metric_absolute_differences={name: abs(float(scores[role][name]-saved_scores[role][name])) for name in scores[role]})
        for role in ('train', 'valid')}
    return diagnostics


def admission(args, pins, protocol):
    release, receipt = read(args.release), read(args.admission)
    require(release['enabled'] is True and release['release_owner'] == 'root'
            and release['action'] == 'identity_centered_fast_prior_extension'
            and release['source_seal_sha256'] == sha(HERE / 'SEAL.json')
            and release['admission_receipt_sha256'] == sha(args.admission), 'Exact root exploratory admission')
    require(release['runtime_qualification_passed'] is True and release['geometry_only_core_reviewed'] is True
            and release['prior_exposure_audit_complete'] is True and release['test_truth_excluded'] is True,
            'Root runtime/core/exposure review')
    require(receipt['admission_enabled'] is True and receipt['admission_mode'] == 'post_screen_exploratory_architecture_fixed' and receipt['release_owner'] == 'root'
            and receipt['original_native15_protocol_pass_claimed'] is False and receipt['original_incomplete_no_freeze_preserved'] is True
            and receipt['new_seeds_are_independent_confirmation'] is False and receipt['same_already_used_Tolokers_split0'] is True
            and receipt['replay_policy'] == protocol['replay_policy'] == 'exact_state_finite_serving_recorded_materiality',
            'Transparent exploratory scope, original failures preserved, prospective report policy')
    require(release['reuse_native_member0'] is False and receipt['reuse_native_member0'] is False, 'All three references fresh; historical reuse disabled')
    decision_path = PHASE / pins['root_decision']['path']
    require(receipt['root_decision_sha256'] == sha(decision_path) == pins['root_decision']['sha256'], 'Exact published root decision before new fits')
    decision = read(decision_path)
    require(decision['release_owner'] == 'root' and decision['all15_fit_attempts_terminal'] is True
            and decision['original_failures_preserved'] is True and decision['original_screen_status_preserved'] == 'incomplete_no_freeze'
            and decision['original_screen_protocol_pass_claimed'] is False and decision['original_scores_modified'] is False
            and decision['TEST_truth_accessed'] is False and decision['unused_confirmation_claimed'] is False,
            'Root post-screen decision retains scientific history')
    require(decision['configuration_id'] == receipt['configuration_id'] == protocol['configuration_id'] == 'd4_f16_L4'
            and decision['new_base_seeds'] == receipt['new_base_seeds'] == protocol['seeds'] == [7409,8501,9607]
            and decision['reuse_native_member0'] is False, 'Exact prospectively declared new trajectories')
    require(receipt['choice_recorded_before_new_fits'] is True and receipt['architecture_fixed'] is True,
            'Post-screen architecture and seeds fixed before this new execution')
    for row in decision['evidence']:
        path = (PHASE / row['path']).resolve(strict=True)
        require(path.is_relative_to(PHASE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Exact all-attempt/diagnostic evidence')
    overview = read(PHASE / pins['all_attempts_path'])
    rows = overview['records']
    v2 = read(PHASE / pins['NSD_V2_directory'] / 'RUNNER_PROTOCOL.json')
    expected = {('native_single', cfg['id'], seed) for cfg in v2['configs'] for seed in v2['seeds']}
    expected.update(('feature_mlp','feature_mlp_10_64_64_2',seed) for seed in v2['seeds'])
    require(overview['all15_terminal_before_read'] is True and len(rows) == 15
            and {(r['family'],r['config_id'],r['seed']) for r in rows} == expected
            and sum(r['status']=='complete' for r in rows) == 13 and sum(r['status']=='failed' for r in rows) == 2
            and {(r['run_id'],r['status'],r['epochs_completed'],r['selected_epoch']) for r in rows}
                == {(r['run_id'],r['status'],r['epochs_completed'],r['selected_epoch']) for r in decision['all_original_attempts']},
            'Every original attempt retained; no survivor reconstruction of the screen')
    summary = read(PHASE / pins['original_screen_summary_path'])
    require(summary['selection_status'] == 'incomplete_no_freeze' and summary['panel_complete'] is False
            and summary['architecture_frozen'] is False and summary['failures'] == 2, 'Original failed screen unchanged')
    materiality = read(PHASE / pins['materiality_summary_path'])
    failed_ids = {r['run_id'] for r in rows if r['status']=='failed'}
    require({r['run_id'] for r in materiality['records']} == failed_ids
            and materiality['original_failed_attempts_preserved'] is True and materiality['thresholds_or_acceptance_gate_added'] is False
            and materiality['original_family_status'] == 'incomplete_no_freeze' and materiality['architecture_selection_allowed'] is False
            and materiality['training_forwards'] == materiality['backwards'] == materiality['optimizer_steps'] == 0,
            'Once-only materiality evidence, no retraining/reselection/tolerance campaign')
    diagnostic_ids = set()
    for row in pins['diagnostic_reports']:
        path = PHASE / row['path']
        require(sha(path) == row['sha256'], 'Exact individual materiality report')
        diagnostic = read(path)
        diagnostic_ids.add(diagnostic['run_id'])
        require(diagnostic['run_id'] in failed_ids and diagnostic['original_attempt_status'] == 'failed'
                and diagnostic['parameter_buffer_restore_exact'] is True and diagnostic['retry_or_reselection'] is False
                and diagnostic['training_forwards'] == diagnostic['backwards'] == diagnostic['optimizer_steps'] == 0
                and diagnostic['fresh_model_forwards_completed'] == 1, 'Actual one-time selected-state diagnostics retained')
    require(len(pins['diagnostic_reports']) == 2 and diagnostic_ids == failed_ids, 'Both individual failed-state diagnostics bound exactly')
    config = next(value for value in v2['configs'] if value['id'] == protocol['configuration_id'])
    require(config['native_args'] == receipt['native_args'] == decision['native_args']
            and receipt['optimizer'] == v2['optimizer'] == decision['optimizer']
            and receipt['checkpoint_rule'] == v2['checkpoint_rule'] == decision['checkpoint_rule']
            and receipt['NSD_V2_source_seal_sha256'] == pins['NSD_V2_source_seal_sha256'], 'Unchanged native architecture/optimizer/selector')
    require(receipt['roles_sha256'] == release['roles_sha256'] == sha(args.roles) == summary['role_identity']['role_archive_sha256']
            and receipt['role_metadata_sha256'] == release['role_metadata_sha256'] == sha(args.roles.parent/'ROLE.json')
                == summary['role_identity']['role_metadata_sha256']
            and summary['role_identity']['source_seal_sha256'] == pins['NSD_V2_source_seal_sha256'], 'Same original official role exposure/source')
    require(receipt['runtime_versions'] == release['expected_runtime_versions'] and receipt['device'] == release['device']
            and receipt['deterministic_algorithms'] == release['deterministic_algorithms']
            and type(release['deterministic_algorithms']) is bool and release['allow_tf32'] is False, 'Exact runtime/device policy')
    require(len(release['execution_source_commit']) == 40 and all(c in '0123456789abcdef' for c in release['execution_source_commit']), 'Committed execution source')
    if release['action'] == 'identity_centered_fast_prior_extension':
        row = release['engineering_qualification']
        require(sha(row['path']) == row['sha256'], 'Exact full-input engineering qualification receipt')
        qualification = read(row['path'])
        require(qualification['qualification_passed'] is True and qualification['status'] == 'complete'
                and qualification['identity']['source_seal_sha256'] == sha(HERE/'SEAL.json')
                and qualification['identity']['roles_sha256'] == receipt['roles_sha256']
                and qualification['identity']['role_metadata_sha256'] == receipt['role_metadata_sha256']
                and qualification['identity']['configuration_id'] == protocol['configuration_id']
                and qualification['identity']['configuration'] == config and qualification['identity']['optimizer'] == receipt['optimizer']
                and qualification['identity']['admission_receipt_sha256'] == sha(args.admission)
                and qualification['identity']['execution_source_commit'] == release['execution_source_commit']
                and qualification['identity']['runtime_versions'] == release['expected_runtime_versions']
                and qualification['identity']['device'] == release['device']
                and qualification['identity']['deterministic_algorithms'] == release['deterministic_algorithms']
                and qualification['identity']['allow_tf32'] is False
                and qualification['centered_prior_probe_passed'] is True and qualification['full_graph'] is True
                and qualification['all_TRAIN_rows'] is True and qualification['centered_update_qualified'] is True
                and qualification['validation_metric_access'] is False and qualification['TEST_truth_present'] is False,
                'Actual sharing/gradient qualification before new fits, exact same source/input/runtime/admission')
    output = args.output.resolve()
    forbidden = [HERE]+[PHASE/row['directory'] for row in pins['immutable_packets']]+[PHASE/value for value in pins['protected_directories']]
    require(str(output) == release['output_directory'] and not output.exists() and output.is_relative_to(PHASE)
            and not any(output.is_relative_to(path) for path in forbidden), 'Fresh exact output outside original artifacts')
    validate_revision(release, pins)
    origins = validate_origins(release, receipt, config, pins)
    require(not output.is_relative_to(Path(origins['original_complete']['path']).resolve().parent), 'Centered output outside original18 artifacts')
    return release, receipt, config, output, origins
