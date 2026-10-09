"""Inactive source/freeze gates and exact existing NSD/V2 helper imports."""
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


def freeze(args, pins, protocol):
    release, receipt = read(args.release), read(args.freeze)
    require(release['enabled'] is True and release['release_owner'] == 'root'
            and release['action'] == 'frozen_geometry_only_core'
            and release['source_seal_sha256'] == sha(HERE / 'SEAL.json')
            and release['freeze_receipt_sha256'] == sha(args.freeze), 'Explicit exact root activation and freeze receipt')
    require(release['runtime_qualification_passed'] is True and release['geometry_only_core_reviewed'] is True
            and release['prior_exposure_audit_complete'] is True and release['test_truth_excluded'] is True,
            'Root runtime/core/exposure review required')
    require(receipt['frozen'] is True and receipt['release_owner'] == 'root'
            and receipt['native15_complete'] is True and receipt['native_competence_review_passed'] is True
            and receipt['native_curve_stability_review_passed'] is True
            and receipt['NSD_V2_source_seal_sha256'] == pins['NSD_V2_source_seal_sha256'], 'Actual competent native15 root freeze required')
    v2 = read(PHASE / pins['NSD_V2_directory'] / 'RUNNER_PROTOCOL.json')
    configs = [value for value in v2['configs'] if value['id'] == receipt['configuration_id']]
    require(len(configs) == 1 and configs[0]['native_args'] == receipt['native_args']
            and v2['optimizer'] == receipt['optimizer'] and v2['checkpoint_rule'] == receipt['checkpoint_rule']
            and v2['seeds'] == protocol['seeds'], 'One exact previously screened configuration/budget/selector')
    summary_artifact = receipt['native_screen_summary']
    summary_path = Path(summary_artifact['path']).resolve(strict=True)
    require(summary_path.is_relative_to(PHASE) and sha(summary_path) == summary_artifact['sha256'], 'Exact complete native15 screen summary receipt')
    summary = read(summary_path)
    require(summary['panel_complete'] is True and summary['native_successes'] == 12 and summary['mlp_successes'] == 3
            and summary['failures'] == 0 and summary['selected_configuration'] == receipt['configuration_id']
            and summary['provisional_fit_gate'] is True and summary['provisional_reference_gate'] is True,
            'All native15 fits and provisional native competence before root freeze')
    require(receipt['roles_sha256'] == release['roles_sha256'] == sha(args.roles)
            and receipt['role_metadata_sha256'] == release['role_metadata_sha256'] == sha(args.roles.parent / 'ROLE.json'), 'Same official role archive/metadata')
    require(summary['role_identity']['source_seal_sha256'] == pins['NSD_V2_source_seal_sha256']
            and summary['role_identity']['role_archive_sha256'] == receipt['roles_sha256']
            and summary['role_identity']['role_metadata_sha256'] == receipt['role_metadata_sha256'], 'Actual screen source/role identity before reuse')
    require(receipt['runtime_versions'] == release['expected_runtime_versions']
            and receipt['deterministic_algorithms'] == release['deterministic_algorithms']
            and receipt['device'] == release['device']
            and type(release['deterministic_algorithms']) is bool and release['allow_tf32'] is False, 'Frozen runtime and deterministic policy')
    require(type(release['reuse_native_member0']) is bool and release['reuse_native_member0'] == receipt['reuse_native_member0'], 'Explicit root member0 reuse decision')
    require(len(release['execution_source_commit']) == 40 and all(c in '0123456789abcdef' for c in release['execution_source_commit']), 'Committed execution identity')
    output = args.output.resolve()
    forbidden = [HERE] + [PHASE / row['directory'] for row in pins['immutable_packets']]
    require(str(output) == release['output_directory'] and not output.exists() and output.is_relative_to(PHASE)
            and not any(output.is_relative_to(path) for path in forbidden), 'Fresh bound normal phase output')
    if release['reuse_native_member0']:
        require(set(receipt['member0_artifacts']) == {str(seed) for seed in protocol['seeds']}, 'Exactly three frozen member0 artifact receipts')
        for seed in protocol['seeds']:
            for name in ('result', 'checkpoint', 'history', 'runtime_metadata'):
                artifact = receipt['member0_artifacts'][str(seed)][name]
                path = Path(artifact['path']).resolve(strict=True)
                require(path.is_relative_to(PHASE) and sha(path) == artifact['sha256'], 'Exact frozen member0 artifact: ' + name)
    return release, receipt, configs[0], output


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


def gross_diagnostics(torch, scores, role_logp, saved_scores, saved_logp):
    diagnostics = {role: dict(max_abs_logp=float((role_logp[role]-saved_logp[role]).abs().max().item()),
        max_abs_probability=float((role_logp[role].exp()-saved_logp[role].exp()).abs().max().item()),
        prediction_changes=int((role_logp[role].argmax(-1) != saved_logp[role].argmax(-1)).sum().item()),
        metric_signed_differences={name: float(scores[role][name]-saved_scores[role][name]) for name in scores[role]},
        metric_absolute_differences={name: abs(float(scores[role][name]-saved_scores[role][name])) for name in scores[role]})
        for role in ('train', 'valid')}
    require(all(value['max_abs_logp'] <= 0.001 and value['metric_absolute_differences']['auroc'] <= 0.001
                for value in diagnostics.values()), 'V2 practical gross restored-output diagnostics')
    return diagnostics
