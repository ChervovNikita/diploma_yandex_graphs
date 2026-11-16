"""Report BASE, independent ensemble and GNNM from completed paired cohorts."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ogb.graphproppred import Evaluator

from ablation.common.resumable import atomic_write_json
from .protocol import EXTENDED_METHODS, METHODS, RECIPES, confined_path
from .training import score


def requested_groups(datasets=None, models=None, cohorts=None):
    datasets = list(RECIPES) if datasets is None else datasets
    cohorts = list(range(10)) if cohorts is None else cohorts
    groups = {(dataset, model, cohort)
              for dataset in datasets
              for model in RECIPES[dataset]
              if models is None or model in models
              for cohort in cohorts}
    if not groups:
        raise ValueError('Dataset/model/cohort selection produced no groups')
    return groups


def collect(root, datasets=None, models=None, cohorts=None, include_smoke=False, extensions=False, valid_only=False):
    methods = EXTENDED_METHODS if extensions else METHODS
    expected = requested_groups(datasets, models, cohorts)
    marker = root / 'protocol_role.json'
    root_role = json.loads(marker.read_text()) if marker.exists() else None
    groups = defaultdict(dict)
    for path in sorted(root.rglob('done.json')):
        result = json.loads(path.read_text())
        if result.get('protocol') != 'ogb-molecular-native-gnnm-v1':
            continue
        key = (result['dataset'], result['model'], result['cohort'])
        if key not in expected or (result.get('smoke') and not include_smoke):
            continue
        if root_role is not None:
            for field in ('smoke', 'smoke_epochs', 'smoke_updates'):
                if result.get(field) != root_role.get(field):
                    raise ValueError(f'Result differs from root role marker: {path}')
        method = result['method']
        if method not in methods:
            raise ValueError(f'Unexpected physical fit method: {path}')
        if method in groups[key]:
            previous = groups[key][method][0]
            raise ValueError(f'Duplicate cohort member {key}/{method}: {previous} and {path.parent}')
        groups[key][method] = (path.parent, result)
    rows, incomplete = [], []
    recipe_bindings = {}
    splits = ('valid',) if include_smoke or valid_only else ('valid', 'test')
    role = 'engineering_validation' if include_smoke else 'scientific_valid_only' if valid_only else 'scientific'
    for dataset, model, cohort in sorted(expected):
        runs = groups[(dataset, model, cohort)]
        missing = sorted(set(methods) - set(runs))
        if missing:
            incomplete.append({'dataset': dataset, 'model': model, 'cohort': cohort,
                               'missing': missing})
            continue
        predictions = {}
        common = None
        for method in methods:
            folder, result = runs[method]
            payload = torch.load(folder / 'predictions.pt', map_location='cpu', weights_only=False)
            config = json.loads((folder / 'config.json').read_text())
            if payload['config'] != config:
                raise ValueError(f'Prediction/config mismatch: {folder}')
            if any(result.get(key) != value for key, value in config.items()):
                raise ValueError(f'Result/config mismatch: {folder}')
            if payload['data_binding'] != result['data_binding']:
                raise ValueError(f'Prediction/data mismatch: {folder}')
            expected_members = 4 if method.startswith('gnnm') else 1
            expected_seed = cohort * 1000 + (int(method[-1]) if method.startswith('ens_m') else 0)
            if config['members'] != expected_members or config['seed'] != expected_seed:
                raise ValueError(f'Invalid member cardinality or seed: {folder}')
            for split in splits:
                if split not in payload['splits']:
                    raise ValueError(f'Missing {split} predictions: {folder}')
                prediction = payload['splits'][split]
                expected_shape = (expected_members, len(prediction['indices']), 1)
                if tuple(prediction['member_logits'].shape) != expected_shape:
                    raise ValueError(f'Invalid {split} member-logit shape: {folder}')
            binding = {key: value for key, value in config.items()
                       if key not in ('method', 'members', 'seed', 'normalization')}
            binding['data_binding'] = payload['data_binding']
            if common is not None and common != binding:
                raise ValueError(f'Cohort has incompatible fitted protocols: {folder}')
            common = binding
            predictions[method] = payload['splits']
        recipe_binding = {key: value for key, value in common.items() if key != 'cohort'}
        previous_binding = recipe_bindings.setdefault((dataset, model), recipe_binding)
        if previous_binding != recipe_binding:
            raise ValueError(f'Recipe cohorts have incompatible fitted protocols: {dataset}/{model}')
        evaluator = Evaluator(name=dataset)
        comparisons = [('base', ['ens_m0']), ('ensemble', [f'ens_m{i}' for i in range(4)])]
        comparisons.extend((method, [method]) for method in methods if method.startswith('gnnm'))
        for method, members in comparisons:
            row = {'dataset': dataset, 'model': model, 'cohort': cohort,
                   'method': method, 'role': role, 'smoke': common['smoke'],
                   'source_sha256': common['source_sha256']}
            for split in splits:
                reference = predictions[members[0]][split]
                logits = []
                for member in members:
                    prediction = predictions[member][split]
                    if not torch.equal(reference['indices'], prediction['indices']):
                        raise ValueError('Ensemble split indices are misaligned')
                    if not torch.allclose(reference['y_true'], prediction['y_true'], equal_nan=True):
                        raise ValueError('Ensemble labels are misaligned')
                    logits.append(prediction['member_logits'])
                pooled = torch.cat(logits, dim=0).mean(dim=0)
                row[f'{split}_rocauc'] = score(evaluator, reference['y_true'], pooled)
            rows.append(row)
    if len({row['source_sha256'] for row in rows}) > 1:
        raise ValueError('Refusing to aggregate different source fingerprints')
    return rows, incomplete


def summarize(rows):
    groups = defaultdict(list)
    paired = defaultdict(dict)
    for row in rows:
        groups[(row['dataset'], row['model'], row['method'])].append(row)
        paired[(row['dataset'], row['model'], row['cohort'])][row['method']] = row
    summary = []
    for (dataset, model, method), records in sorted(groups.items()):
        item = {'dataset': dataset, 'model': model, 'method': method, 'cohorts': len(records),
                'role': records[0]['role'], 'smoke': records[0]['smoke']}
        splits = ('valid',) if item['role'] in ('engineering_validation', 'scientific_valid_only') else ('valid', 'test')
        for split in splits:
            values = np.array([row[f'{split}_rocauc'] for row in records])
            deltas = np.array([row[f'{split}_rocauc'] -
                               paired[(dataset, model, row['cohort'])]['base'][f'{split}_rocauc']
                               for row in records])
            item[f'{split}_mean'] = float(values.mean())
            item[f'{split}_std'] = float(values.std(ddof=1)) if len(values) > 1 else None
            item[f'{split}_paired_delta_vs_base_mean'] = float(deltas.mean())
            item[f'{split}_paired_delta_vs_base_std'] = float(deltas.std(ddof=1)) if len(values) > 1 else None
        summary.append(item)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=list(RECIPES), default=list(RECIPES))
    parser.add_argument('--models', nargs='+', choices=['gcn', 'gin'])
    parser.add_argument('--cohorts', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--extensions', action='store_true')
    parser.add_argument('--valid-only', action='store_true')
    parser.add_argument('--include-smoke', action='store_true',
                        help='Include smoke fits and emit engineering-only VALID aggregates.')
    args = parser.parse_args(argv)
    if any(cohort < 0 for cohort in args.cohorts) or len(set(args.cohorts)) != len(args.cohorts):
        parser.error('Cohorts must be distinct nonnegative integers')
    root, out = confined_path(args.root), confined_path(args.out)
    rows, incomplete = collect(root, args.datasets, args.models, args.cohorts, args.include_smoke, args.extensions, args.valid_only)
    role = 'engineering_validation' if args.include_smoke else 'scientific_valid_only' if args.valid_only else 'scientific'
    marker = root / 'protocol_role.json'
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json({'rows': rows, 'summary': summarize(rows), 'role': role,
                       'include_smoke': args.include_smoke,
                       'root_role': json.loads(marker.read_text()) if marker.exists() else None,
                       'selection': {'datasets': args.datasets, 'models': args.models,
                                     'cohorts': args.cohorts},
                       'incomplete_cohorts': incomplete,
                       'units': 'ROC-AUC fraction, sample standard deviation'}, out / 'results.json')
    fields = ['dataset', 'model', 'cohort', 'method', 'role', 'smoke', 'source_sha256', 'valid_rocauc']
    if not args.include_smoke and not args.valid_only:
        fields.append('test_rocauc')
    with (out / 'cohorts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'role={role} completed paired cohorts={len(rows) // (4 if args.extensions else 3)}, '
          f'incomplete cohorts={len(incomplete)}')
    return 0 if rows and not incomplete else 1


if __name__ == '__main__':
    raise SystemExit(main())
