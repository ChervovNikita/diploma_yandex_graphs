"""Hits@20, Hits@50 and Hits@100 on Collab from saved prediction files; no GPU.

    python -m ablation.p14_collab_ogb.rescore --root CAMPAIGN_ROOT --pair collab_buddy \
        --out ablation/p14_collab_ogb/fixed_protocol_buddy_hits

Reads every scored ``predictions.pt`` of one transfer pair (BUDDY runs of the
earlier campaign, or the new GCN/GraphSAGE runs) and recomputes the official
OGB Hits@K for BASE (native member 0), ENS (mean raw score of the four native
members, each at its own selected checkpoint) and every shared-model variant.
Nothing is retrained and checkpoints are not loaded.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from ablation.fixed_protocol.protocol import fresh_output_dir
from .protocol import HITS, REPOSITORY
from .run import all_hits

ROLES = {'valid': 'val', 'test': 'test'}


def method_name(config):
    if config['variant'] == 'native':
        return f"native{config['member']}"
    return config['variant']


def find_predictions(root, pair):
    found = defaultdict(dict)
    base = Path(root) / 'tasks' / 'transfer' / pair
    for path in sorted(base.glob('**/scoring/**/predictions.pt')):
        artifact = torch.load(path, map_location='cpu', weights_only=False)
        if 'test' not in artifact:
            continue
        config = artifact['config']
        method = method_name(config)
        if method in found[config['cohort']]:
            raise ValueError(f'Two scored predictions for cohort {config["cohort"]} {method}')
        found[config['cohort']][method] = artifact
    if not found:
        raise FileNotFoundError(f'No scored predictions under {base}')
    return found


def cohort_rows(cohort, artifacts):
    natives = [artifacts.get(f'native{member}') for member in range(4)]
    settings = {name: artifact for name, artifact in artifacts.items() if not name.startswith('native')}
    if natives[0] is not None:
        settings['BASE'] = natives[0]
    rows = []
    reference = next(iter(artifacts.values()))
    for role, prefix in ROLES.items():
        expected = reference[role]
        for name, artifact in artifacts.items():
            current = artifact[role]
            if (current['edge_order_sha256'] != expected['edge_order_sha256']
                    or current['num_positive'] != expected['num_positive']):
                raise ValueError(f'Edge order differs for cohort {cohort} {name}')
        scored = {name: artifact[role]['scores'] for name, artifact in settings.items()}
        if all(item is not None for item in natives):
            scored['ENS'] = torch.stack([item[role]['scores'] for item in natives]).mean(0)
        for name, scores in scored.items():
            metrics = all_hits(scores, expected['num_positive'])
            rows.append(dict(cohort=cohort, setting=name, role=prefix,
                             **{key: 100 * value for key, value in metrics.items()}))
    return rows


def summarise(rows):
    frame = pd.DataFrame(rows)
    metrics = [f'hits@{k}' for k in HITS]
    summary = frame.groupby(['role', 'setting'])[metrics].agg(['mean', 'std', 'count'])
    wide = frame.pivot_table(index=['role', 'cohort'], columns='setting', values=metrics)
    paired = []
    for metric in metrics:
        for setting in wide[metric].columns:
            for reference in ('ENS', 'BASE'):
                if setting in (reference, 'BASE') or reference not in wide[metric].columns:
                    continue
                difference = wide[metric][setting] - wide[metric][reference]
                for role, values in difference.groupby(level='role'):
                    paired.append(dict(role=role, metric=metric, setting=setting, reference=reference,
                                       mean_difference=values.mean(), cohorts=int(values.count()),
                                       positive=int((values > 0).sum())))
    return frame, summary, pd.DataFrame(paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='campaign root with transfer tasks')
    parser.add_argument('--pair', required=True, help='collab_buddy, collab_gcn or collab_sage')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = fresh_output_dir(args.out, REPOSITORY)
    found = find_predictions(args.root, args.pair)
    rows = [row for cohort, artifacts in sorted(found.items()) for row in cohort_rows(cohort, artifacts)]
    frame, summary, paired = summarise(rows)
    frame.to_csv(out / 'per_cohort.csv', index=False)
    summary.to_csv(out / 'summary.csv')
    paired.to_csv(out / 'paired_differences.csv', index=False)
    (out / 'source.json').write_text(json.dumps(
        {'root': str(Path(args.root).resolve()), 'pair': args.pair,
         'cohorts': sorted(found), 'methods': {str(c): sorted(a) for c, a in found.items()}}, indent=2))
    print(summary.round(2).to_string())


if __name__ == '__main__':
    main()
