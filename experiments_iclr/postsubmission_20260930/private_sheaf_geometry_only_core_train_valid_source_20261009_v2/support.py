"""Inactive source/admission gates and exact existing NSD/V2 helper imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


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
            and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Exact inactive core source seal')
    for row in read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed core source')
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    for row in pins['files']:
        path = (PHASE / row['path']).resolve(strict=True)
        require(path.is_relative_to(PHASE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed pinned native/helper source')
    return pins


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def borrowed(pins):
    existing = load(PHASE / pins['BSNN_wrapper_directory'] / 'support.py', '_geometry_existing_stdlib_support')
    existing.source_checks()
    common, helpers = existing.borrowed(read(PHASE / pins['BSNN_wrapper_directory'] / 'SOURCE_BINDINGS.json'))
    placement = sys.modules['_bsnn_original_placement_reference']
    return existing, common, helpers, placement


def admission(args, pins, protocol):
    release, receipt = read(args.release), read(args.admission)
    require(release['enabled'] is True and release['release_owner'] == 'root'
            and release['action'] in ('post_screen_geometry_core', 'geometry_engineering_qualify')
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
    if release['action'] == 'post_screen_geometry_core':
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
                and qualification['validation_metric_access'] is False and qualification['TEST_truth_present'] is False,
                'Actual sharing/gradient qualification before new fits, exact same source/input/runtime/admission')
    output = args.output.resolve()
    forbidden = [HERE]+[PHASE/row['directory'] for row in pins['immutable_packets']]+[PHASE/value for value in pins['protected_directories']]
    require(str(output) == release['output_directory'] and not output.exists() and output.is_relative_to(PHASE)
            and not any(output.is_relative_to(path) for path in forbidden), 'Fresh exact output outside original artifacts')
    return release, receipt, config, output

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
