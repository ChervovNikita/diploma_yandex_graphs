"""Inactive three-seed original deterministic bundle d2/f32/L2 fairness control."""
import argparse
import importlib
import importlib.metadata
import importlib.util
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
VERSIONS = ('torch', 'numpy', 'scipy', 'torch-geometric', 'torch-sparse', 'torch-scatter', 'torch-householder', 'scikit-learn')


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
            and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Exact inactive source seal')
    for row in read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed fairness-control source')
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    for row in pins['files']:
        path = (PHASE / row['path']).resolve(strict=True)
        require(path.is_relative_to(PHASE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed original author/helper source')
    return pins


def borrowed(pins):
    # Import only the sealed stdlib support; its importer loads actual V2 helpers.
    path = PHASE / pins['BSNN_wrapper_directory'] / 'support.py'
    spec = importlib.util.spec_from_file_location('_deterministic_bundle_source_support', path)
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    original_pins = support.source_checks()
    common, helpers = support.borrowed(original_pins)
    return common, helpers


def native_class(pins):
    root = (PHASE / pins['author_directory']).resolve(strict=True)
    require(not any(name == 'models' or name.startswith('models.') or name == 'lib' or name.startswith('lib.')
                    for name in sys.modules), 'Fresh original models/lib namespace required')
    sys.path.insert(0, str(root))
    value = importlib.import_module('models.disc_models')
    for name, item in list(sys.modules.items()):
        if name == 'models' or name.startswith('models.') or name == 'lib' or name.startswith('lib.'):
            require(Path(item.__file__).resolve().is_relative_to(root), 'Exact original author namespace origin')
    return value.DiscreteBundleSheafDiffusion


def adapter(torch, cls, device, native_args):
    """Original constructor, plus placement of per-layer plain edge indices."""
    class OriginalBundleAdapter:
        @staticmethod
        def make_native_factory(cpu_edges, args):
            def construct():
                model = cls(cpu_edges, dict(args))
                require(len(model.sheaf_learners) == len(model.weight_learners) == native_args['layers'], 'Original separate per-layer learners')
                # V2's original CPU/float32 parameter checks and remaining placement follow.
                # The original tensor values, source forward and per-layer objects are retained.
                for learner in model.weight_learners:
                    learner.full_left_right_idx = learner.full_left_right_idx.to(device)
                return model
            return construct
    return OriginalBundleAdapter()


def extended_topology_counter(torch, original):
    def count(model):
        total = original(model)
        seen = {(str(value.device), value.untyped_storage().data_ptr())
                for owner in (model, model.laplacian_builder) for value in vars(owner).values()
                if isinstance(value, torch.Tensor)}
        for learner in model.weight_learners:
            value = learner.full_left_right_idx
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            if key not in seen:
                total += storage.nbytes()
                seen.add(key)
        return total
    return count


def run(args):
    pins, protocol = source_checks(), read(HERE / 'PROTOCOL.json')
    release = read(args.release)
    require(release['enabled'] is True and release['source_seal_sha256'] == sha(HERE / 'SEAL.json')
            and release['runtime_qualification_passed'] is True
            and release['native_bundle_source_and_placement_reviewed'] is True, 'Explicit root runtime/source/placement release')
    require(release['release_owner'] == 'root' and release['action'] == 'deterministic_bundle_fairness_control'
            and release['prior_exposure_audit_complete'] is True and release['test_truth_excluded'] is True, 'Root fairness-control and role authorization')
    require(type(release['deterministic_algorithms']) is bool and release['allow_tf32'] is False
            and set(release['expected_runtime_versions']) == set(VERSIONS), 'Exact runtime and deterministic policy')
    require(len(release['execution_source_commit']) == 40 and all(c in '0123456789abcdef' for c in release['execution_source_commit']), 'Pinned committed source identity')
    require(sha(args.roles) == release['roles_sha256'] and sha(args.roles.parent / 'ROLE.json') == release['role_metadata_sha256'], 'Exact same official role archive/metadata')
    output = args.output.resolve()
    immutable_roots = [PHASE / row['directory'] for row in pins['immutable_packets']]
    require(str(output) == release['output_directory'] and not output.exists() and output.is_relative_to(PHASE)
            and not output.is_relative_to(HERE) and not any(output.is_relative_to(root) for root in immutable_roots), 'Fresh bound normal-phase output outside sealed packets')
    started, usage0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    common, helpers = borrowed(pins)
    output.mkdir(parents=True, exist_ok=False)
    records, failure, original_counter = [], None, helpers.static_topology_bytes
    family = 'native_single'  # Exact V2 fit_one API; identities name the deterministic bundle control.
    config = protocol['configs'][0]
    def folder(seed): return output / (family + '__' + config['id'] + '__seed' + str(seed))
    try:
        common.write_json(output / 'SCHEDULE.json', dict(protocol=protocol, release=release, required_seeds=protocol['seeds'], automatic_retry=False))
        import numpy as np
        import torch
        from sklearn.metrics import roc_auc_score
        versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        require(versions == release['expected_runtime_versions'] and str(torch.__version__) == '2.1.2+cu118'
                and np.__version__ == '1.26.4', 'Same recorded existing runtime')
        device = torch.device(release['device'])
        require(device == torch.device('cuda:0') and torch.cuda.device_count() == 1, 'One root-owned visible cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(release['deterministic_algorithms'])
        arrays, role_meta, role_sha = common.read_roles(np, args.roles)
        require(role_meta.get('exposure_classification') == 'original_paper_benchmark_exploratory', 'Same corrected original-benchmark exposure class')
        data = {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}
        data['cpu_edge_index'] = torch.from_numpy(arrays['edge_index'])
        cls = native_class(pins)
        native_adapter = adapter(torch, cls, device, config['native_args'])
        helpers.static_topology_bytes = extended_topology_counter(torch, original_counter)
        householder = sys.modules['torch_householder']
        identity = dict(source_manifest_sha256=read(HERE / 'SEAL.json')['manifest_sha256'], source_commit=release['execution_source_commit'],
            implementation_class='models.disc_models.DiscreteBundleSheafDiffusion', author_commit=pins['author_commit'],
            author_model_program_sha256=pins['author_model_program_sha256'], actual_NSD_V2_seal_sha256=pins['actual_NSD_V2_seal_sha256'],
            roles_sha256=release['roles_sha256'], role_metadata_sha256=role_sha, runtime_versions=versions, protocol=protocol,
            actual_householder_provider_path=str(Path(householder.__file__).resolve()), actual_householder_provider_sha256=sha(householder.__file__),
            experiment_family='authentic_deterministic_bundle_fairness_control', pure_sampling_or_KL_ablation=False, TEST_truth_present=False)
        common.write_json(output / 'RUN_IDENTITY.json', identity)
        for seed in protocol['seeds']:
            fit_usage = resource.getrusage(resource.RUSAGE_SELF)
            record = helpers.fit_one(np, torch, roc_auc_score, native_adapter, protocol, config, seed, family,
                                     data, role_meta, identity, output, device)
            fit_final_usage = resource.getrusage(resource.RUSAGE_SELF)
            record.update(experiment_family='authentic_deterministic_bundle_fairness_control', full_evaluation_calls=1,
                          full_serving_calls=1, pure_sampling_or_KL_ablation=False, configuration_search=False,
                          CPU_user_seconds=fit_final_usage.ru_utime-fit_usage.ru_utime,
                          CPU_system_seconds=fit_final_usage.ru_stime-fit_usage.ru_stime)
            # Preserve V2's complete returned record; missing cost/cleanup evidence blocks this panel.
            if any(record.get(key) for key in ('cost_query_failure', 'cleanup_failure', 'cost_finalization_failure')):
                record['status'] = 'failed'
            common.write_json(folder(seed) / 'RESULT.json', record)
            records.append(record)
            common.append_jsonl(output / 'ALL_FITS.jsonl', record)
    except BaseException as error:
        failure = common.failure_record(error, 'deterministic_bundle_panel')
        raise
    finally:
        helpers.static_topology_bytes = original_counter
        missing_returns = [seed for seed in protocol['seeds'] if seed not in {row['seed'] for row in records}]
        for seed in missing_returns:
            partial = folder(seed) / 'RESULT.json'
            record = read(partial) if partial.exists() else dict(seed=seed, status='failed_before_fit', failure=failure,
                actual_fit_started=False, recorded_fit_cost_seconds=0, TEST_truth_present=False, automatic_retry=False)
            if record['status'] == 'started': record.update(status='interrupted', failure=failure)
            records.append(record)
        common.write_json(output / 'ALL_FITS.json', records)
        complete = failure is None and len(records) == 3 and {row['seed'] for row in records} == set(protocol['seeds']) and all(row['status'] == 'complete' for row in records)
        if complete:
            common.write_json(output / 'PANEL_SUMMARY.json', dict(configuration=config['id'], complete=True,
                mean_selected_reconstructed_scores={role: {metric: sum(row['scores'][role][metric] for row in records)/3
                    for metric in ('auroc', 'nll', 'accuracy', 'brier')} for role in ('train', 'valid')},
                competence='requires_root_learning_curve_and_reference_review', configuration_search=False,
                comparison_opening_authorized=False, pure_sampling_or_KL_ablation=False, TEST_truth_present=False))
        usage = resource.getrusage(resource.RUSAGE_SELF)
        common.write_json(output / 'COMPLETE.json', dict(complete=complete, required_seeds=protocol['seeds'], records=records,
            seeds_without_returned_records=missing_returns, failure=failure, complete_panel_seconds=time.perf_counter()-started,
            CPU_user_seconds=usage.ru_utime-usage0.ru_utime, CPU_system_seconds=usage.ru_stime-usage0.ru_stime,
            process_RSS_high_water_cumulative_bytes=common.process_peak_rss_bytes(),
            includes_setup_role_imports_construction_all_fit_and_restore_costs=True, all3_required_before_comparison=True,
            automatic_retry=False, configuration_search=False, comparative_opening_authorized=False, TEST_truth_present=False))
    require(complete, 'All three paired attempts required; failures retained, no survivor mean')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--roles', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(inactive=True, protocol=read(HERE / 'PROTOCOL.json'), numeric_or_model_import_or_role_load=False)))
        return
    require(args.release and args.roles and args.output, 'Explicit root release/roles/fresh output required')
    run(args)


if __name__ == '__main__': main()
