"""Inactive fixed richer BSNN tuple; dispatch the byte-exact original engine."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SEEDS = (7409, 8501, 9607)
ACTION = 'fixed_richer_bsnn_cayley_d2_f32_L4_baseline'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def source_gate():
    seal = read(HERE / 'SEAL.json')
    require(seal['source_only'] is True and seal['execution_enabled'] is False
            and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Exact inactive richer source seal')
    for row in read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes']
                and sha(path) == row['sha256'], 'Changed richer source/evidence payload')
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    require(sha(HERE / 'original_runner.py') == pins['copied_original_runner_sha256']
            and sha(HERE / 'support.py') == pins['copied_original_support_sha256'],
            'Original BSNN engine/support must remain byte exact')
    return pins


def release_gate(args, pins):
    protocol = read(HERE / 'PROTOCOL.json')
    require(tuple(protocol['seeds']) == SEEDS and protocol['configuration_count'] == 1
            and protocol['configuration_search'] is False and protocol['max_epochs'] == 500
            and protocol['patience'] == 200, 'Only the fixed fresh three-seed tuple is admitted')
    native = protocol['native_args']
    require(native['d'] == 2 and native['hidden_channels'] == 32 and native['layers'] == 4
            and native['second_linear'] is True and native['input_dropout'] == .4
            and native['dropout'] == .2 and native['orth'] == 'householder'
            and native['add_lp'] is False and native['add_hp'] is False
            and protocol['native_class'] == 'models.bayes_disc_models.BayesBundleSheafDiffusion',
            'Exact original richer Cayley/Householder BSNN configuration required')
    release = read(args.release)
    require(release.get('enabled') is True and release.get('release_owner') == 'root'
            and release.get('action') == ACTION and release.get('source_seal_sha256') == sha(HERE / 'SEAL.json'),
            'Explicit root release for this exact tuple required')
    require(release.get('prior_exposure_audit_complete') is True
            and release.get('test_truth_excluded') is True
            and release.get('chosen_after_weak_baseline_results_before_new_GNNM_science') is True
            and release.get('only_one_configuration') is True
            and release.get('original_probabilistic_math_unchanged') is True
            and release.get('richer_configuration_full_input_work_qualified') is True
            and release.get('predecessor_groups_and_cuda_absent') is True,
            'Post-baseline classification, exact scope and fresh work custody required')
    receipt = release['owner_release_receipt']
    require(sha(receipt['path']) == receipt['sha256'], 'Normal root owner receipt hash mismatch')
    require(release['roles_sha256'] == pins['same_original_role_archive_sha256']
            and release['role_metadata_sha256'] == pins['same_original_role_metadata_sha256']
            and release['expected_runtime_versions'] == pins['same_original_runtime_versions']
            and release.get('allow_tf32') is False, 'Same recorded roles/runtime and TF32 policy required')
    output = args.output.resolve()
    immutable = [PHASE / row['directory'] for row in pins['immutable_prior_packets']]
    immutable += [PHASE / 'bsnn_cayley_d2_train_valid_runner_source_20261009_v1',
                  PHASE / 'bsnn_deterministic_bundle_d2_train_valid_runner_source_20261009_v2',
                  PHASE / 'bsnn_full_three_seed_baseline_execution_root_20261009_v1',
                  PHASE / 'private_sheaf_train_valid_runner_20261009_v2']
    require(str(output) == release['output_directory'] and not output.exists()
            and output.is_relative_to(PHASE) and not output.is_relative_to(HERE)
            and not any(output.is_relative_to(root) for root in immutable),
            'Fresh separate normal-phase output outside original sources/results required')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def dispatch(args):
    pins = source_gate()
    release_gate(args, pins)
    support = module(HERE / 'support.py', '_richer_bsnn_original_support')
    absent = object()
    previous = sys.modules.get('support', absent)
    try:
        sys.modules['support'] = support
        engine = module(HERE / 'original_runner.py', '_richer_bsnn_original_engine')
    finally:
        if previous is absent:
            sys.modules.pop('support', None)
        else:
            sys.modules['support'] = previous
    # Original fit/evaluation/optimizer/KL schedule/RNG/restore/cost code is unchanged.
    engine.run(args)
    complete = read(args.output / 'COMPLETE.json')
    require(complete['complete'] is True and complete.get('failure') is None
            and len(complete['records']) == 3
            and {row['seed'] for row in complete['records']} == set(SEEDS)
            and all(row['status'] == 'complete' for row in complete['records']),
            'All three fresh records and clean wrapper completion required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--roles', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'inactive': True, 'protocol': read(HERE / 'PROTOCOL.json'),
                          'numeric_model_role_or_checkpoint_access': False}))
        return
    require(args.release and args.roles and args.output, 'Root release/roles/fresh output required')
    dispatch(args)


if __name__ == '__main__':
    main()
