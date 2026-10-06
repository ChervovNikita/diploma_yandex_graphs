#!/usr/bin/env python3
"""Audit previously recorded TRAIN loss scalars. No numeric/model libraries."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def reject_nonfinite(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def summarize(values):
    return {'n': len(values), 'mean': math.fsum(values)/len(values),
            'median': statistics.median(values), 'minimum': min(values),
            'maximum': max(values), 'first': values[0], 'last': values[-1]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--local-bindings', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.spec.read_text(), parse_constant=reject_nonfinite)
    bindings = json.loads(args.local_bindings.read_text(), parse_constant=reject_nonfinite)
    require(bindings['spec_sha256'] == sha(args.spec), 'Sealed audit spec changed')
    require(not args.output.exists(), 'Fresh result required')
    observed = {row['source_path']: row for row in bindings['traces']}
    require(len(observed) == len(spec['traces']) == 10
            and set(observed) == {r['path'] for r in spec['traces']}, 'All ten fixed traces required')
    # Verify source inventory and local bytes for every trace before parsing one.
    for row in observed.values():
        path = Path(row['local_path'])
        require(path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Source-bound scalar trace bytes changed: ' + row['source_path'])
    results = []
    for wanted in spec['traces']:
        binding = observed[wanted['path']]
        rows = [json.loads(line, parse_constant=reject_nonfinite)
                for line in Path(binding['local_path']).read_text().splitlines()]
        require(len(rows) == wanted['expected_rows'], 'Wrong trace row count')
        for i, row in enumerate(rows, 1):
            require(set(row) == set(wanted['columns']) and type(row['update']) is int
                    and row['update'] == i, 'Trace fields or contiguous update clock differ')
            if wanted['family'] == 'native':
                expected = wanted['role'] == 'SR' or i > 200
                require(type(row['global_stage']) is bool and row['global_stage'] == expected,
                        'Recorded native stage differs')
            else:
                require(row['objective'] == wanted['objective'], 'Ordinary objective differs')
            for key in wanted['columns']:
                if key in ('update', 'global_stage', 'objective'):
                    continue
                require(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0,
                        'Finite nonnegative loss scalar required')
            if wanted['family'] == 'ordinary':
                expected = row['own_CE'] if wanted['objective'] == 'own' else .5*(row['own_CE']+row['probability_pool_NLL'])
                require(abs(row['optimized_loss']-expected) <= 5e-7*max(1., abs(expected)),
                        'Recorded FP32 loss combination differs')
        scalar_keys = [k for k in wanted['columns'] if k not in ('update', 'global_stage', 'objective')]
        windows = [(1, 16), (185, 200), (201, 216), (385, 400)] if wanted['expected_rows'] == 400 else [(1, 16), (85, 100), (385, 400), (1185, 1200), (2285, 2300)]
        losses = {}
        for key in scalar_keys:
            values = [float(r[key]) for r in rows]
            losses[key] = {'whole_trace': summarize(values),
                'fixed_windows': {f'{a}_to_{b}': summarize(values[a-1:b]) for a,b in windows},
                'last16_minus_first16_mean': math.fsum(values[-16:])/16 - math.fsum(values[:16])/16}
        results.append({'source_path': wanted['path'], 'source_sha256': binding['sha256'],
                        'family': wanted['family'], 'member': wanted.get('member'),
                        'role': wanted.get('role'), 'objective': wanted.get('objective'),
                        'verified_rows': len(rows), 'losses': losses})
    result = {'schema': 'existing_scalar_TRAIN_curve_audit_v1', 'spec_sha256': sha(args.spec),
              'all_ten_trace_hashes_schema_stage_clocks_passed': True, 'traces': results,
              'model_forwards': 0, 'fits': 0, 'held_label_reads': 0, 'payload_reads': 0,
              'new_predictive_scores': False, 'new_checkpoint_or_gate_selection': False,
              'information_limit': 'Recorded train mode CE with dropout; TRAIN accuracy/confidence and held trajectories are not recorded.'}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
