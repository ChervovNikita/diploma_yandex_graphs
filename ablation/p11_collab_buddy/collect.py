"""Derive paired BASE, ENS4 and GNNM4 results from completed native fits."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from ablation.common.resumable import JobStore, atomic_torch_save, atomic_write_json
from .protocol import inside_repository
from .run import evaluate_scores


IDENTITY_FIELDS = {'cohort', 'variant', 'seed', 'member', 'members', 'reports_base'}


def common_binding(config):
    """Retain every protocol, data, source and runtime field across paired fits."""
    return {key: value for key, value in config.items() if key not in IDENTITY_FIELDS}


def load_fit(store, cohort, variant, member=None):
    config = json.loads(store.config_path.read_text())
    expected = {'study': 'p11_collab_buddy', 'dataset': 'ogbl-collab',
                'variant': variant, 'cohort': cohort, 'seed': cohort * 1000}
    if variant == 'native':
        expected.update(member=member, seed=cohort * 1000 + member, reports_base=member == 0)
    else:
        expected['members'] = 4
    for field, value in expected.items():
        if config.get(field) != value:
            raise RuntimeError(f'Invalid {field} for expected {store.dir.name}: {store.dir}')
    result = store.read_done()
    artifact = torch.load(store.dir / 'predictions.pt', map_location='cpu', weights_only=False)
    if artifact['config'] != config:
        raise RuntimeError(f'Prediction/config.json mismatch: {store.dir}')
    if any(result.get(key) != value for key, value in config.items()):
        raise RuntimeError(f'Completed result/config.json mismatch: {store.dir}')
    return config, result, artifact


def require_matching_predictions(first, current):
    if (current['edge_order_sha256'] != first['edge_order_sha256']
            or current['num_positive'] != first['num_positive']
            or current['scores'].shape != first['scores'].shape):
        raise RuntimeError('Paired prediction edge ordering or shape differs between fits')


def collect_cohort(root, cohort, include_smoke=False, output_root=None, extensions=False, valid_only=False):
    members = [JobStore(root / f'cohort{cohort}/native{member}') for member in range(4)]
    if not all(member.is_done() for member in members):
        return None
    loaded = [load_fit(store, cohort, 'native', member)
              for member, store in enumerate(members)]
    first_config = loaded[0][0]
    if first_config['smoke'] and not include_smoke:
        return None
    identities = {(config['cohort'], config['member'], config['seed'])
                  for config, _, _ in loaded}
    if len(identities) != 4 or len({store.dir.resolve() for store in members}) != 4:
        raise RuntimeError('The ensemble requires four distinct native fit identities')
    binding = common_binding(first_config)
    for config, _, _ in loaded:
        if common_binding(config) != binding:
            raise RuntimeError('Incompatible ensemble source/protocol/data/runtime binding')
    results = [result for _, result, _ in loaded]
    artifacts = [artifact for _, _, artifact in loaded]
    gnnm = JobStore(root / f'cohort{cohort}/gnnm')
    gnnm_result = None
    gnnm_artifact = None
    if gnnm.is_done():
        gnnm_config, gnnm_result, gnnm_artifact = load_fit(gnnm, cohort, 'gnnm')
        if common_binding(gnnm_config) != binding:
            raise RuntimeError('Incompatible paired GNNM source/protocol/data/runtime binding')
    extension_results = {}
    extension_artifacts = []
    if extensions:
        for variant in ('gnnm_contrast', 'gnnm_private_bn', 'gnnm_private_bn_contrast'):
            extension_store = JobStore(root / f'cohort{cohort}' / variant)
            if not extension_store.is_done():
                return None
            extension_config, extension_result, extension_artifact = load_fit(extension_store, cohort, variant)
            if common_binding(extension_config) != binding:
                raise RuntimeError('Incompatible extension source/protocol/data/runtime binding')
            extension_results[variant] = extension_result
            extension_artifacts.append(extension_artifact)
    config = {key: value for key, value in first_config.items()
              if key not in ('seed', 'member', 'reports_base')}
    config.update(variant='ensemble', members=4, selected_member_checkpoints='own_validation_best')
    output_root = root if output_root is None else output_root
    store = JobStore(output_root / f'cohort{cohort}/ensemble')
    store.ensure_config(config)
    result = dict(config, metric='Hits@50', test_evaluated=not bool(config['smoke']) and not valid_only,
                  member_best_epochs=[item['best_epoch'] for item in results],
                  train_seconds=sum(item['train_seconds'] for item in results),
                  num_params=sum(item['num_params'] for item in results),
                  predictions_file='predictions.pt')
    predictions = {'config': config}
    splits = ['valid'] if config['smoke'] or valid_only else ['valid', 'test']
    for split in splits:
        first = artifacts[0][split]
        for artifact in artifacts[1:]:
            require_matching_predictions(first, artifact[split])
        if gnnm_artifact is not None:
            require_matching_predictions(first, gnnm_artifact[split])
        for extension_artifact in extension_artifacts:
            require_matching_predictions(first, extension_artifact[split])
        scores = torch.stack([artifact[split]['scores'] for artifact in artifacts]).mean(0)
        predictions[split] = dict(first, scores=scores,
                                  member_scores=torch.stack([artifact[split]['scores'] for artifact in artifacts]))
        prefix = 'val' if split == 'valid' else 'test'
        result[f'{prefix}_metric'] = evaluate_scores(scores, first['num_positive'])
    atomic_torch_save(predictions, store.dir / 'predictions.pt')
    store.write_done(result)
    return {'cohort': cohort, 'binding': binding,
            'BASE': results[0], 'ENS4': result, 'GNNM4': gnnm_result, **extension_results}


def collect_groups(root, cohorts, include_smoke=False, output_root=None, extensions=False, valid_only=False):
    complete, incomplete = [], []
    binding = None
    for cohort in cohorts:
        missing = [name for name in ('native0', 'native1', 'native2', 'native3', 'gnnm')
                   if not JobStore(root / f'cohort{cohort}' / name).is_done()]
        if extensions:
            missing.extend(variant for variant in ('gnnm_contrast', 'gnnm_private_bn', 'gnnm_private_bn_contrast')
                           if not JobStore(root / f'cohort{cohort}' / variant).is_done())
        row = collect_cohort(root, cohort, include_smoke, output_root, extensions, valid_only)
        if row is None or row['GNNM4'] is None:
            incomplete.append({'cohort': cohort, 'missing': missing,
                               'reason': 'missing_fits' if missing else 'smoke_excluded'})
            continue
        if binding is not None and row['binding'] != binding:
            raise RuntimeError(f'Incompatible cross-cohort source/protocol/data/runtime binding: cohort{cohort}')
        binding = row['binding']
        complete.append(row)
    return complete, incomplete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path,
                        help='Write collected ensemble artifacts and results.json under this directory.')
    parser.add_argument('--cohorts', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--extensions', action='store_true')
    parser.add_argument('--valid-only', action='store_true')
    parser.add_argument('--include-smoke', action='store_true',
                        help='Include explicitly labelled TRAIN/VALID engineering runs.')
    args = parser.parse_args(argv)
    if any(cohort < 0 for cohort in args.cohorts) or len(set(args.cohorts)) != len(args.cohorts):
        parser.error('Cohorts must be distinct nonnegative integers')
    root = inside_repository(args.root)
    output_root = inside_repository(args.out) if args.out is not None else None
    complete, incomplete = collect_groups(root, args.cohorts, args.include_smoke, output_root, args.extensions, args.valid_only)
    output = {'complete_cohorts': len(complete), 'requested_cohorts': len(args.cohorts),
              'incomplete_cohorts': incomplete, 'smoke_included': args.include_smoke, 'settings': {}}
    settings = ['BASE', 'ENS4', 'GNNM4']
    if args.extensions:
        settings.extend(['gnnm_contrast', 'gnnm_private_bn', 'gnnm_private_bn_contrast'])
    for setting in settings:
        rows = [row[setting] for row in complete]
        metrics = {}
        for field in ('val_metric', 'test_metric', 'train_seconds', 'num_params'):
            values = [row[field] for row in rows if field in row]
            if values:
                metrics[field] = {'mean': statistics.mean(values),
                                  'sample_std': statistics.stdev(values) if len(values) > 1 else None,
                                  'values': values}
        output['settings'][setting] = metrics
    output['paired_test_differences'] = {
        setting: [row[setting]['test_metric'] - row['BASE']['test_metric']
                  for row in complete if 'test_metric' in row[setting]]
        for setting in settings[1:]
    }
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output, output_root / 'results.json')
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if complete and not incomplete else 1


if __name__ == '__main__':
    raise SystemExit(main())
