"""Planning scenarios from bounded node calibration, with explicit extrapolation."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from .plan import build_plan


def capacity(plan, calibration, additional=()):
    measured = {}
    for row in [*calibration['records'], *additional]:
        spec = row['spec']
        method = spec.get('variant') or spec['method']
        measured[(spec['dataset'], spec['model'], spec['num_layers'], method)] = row['steady_step_seconds']

    def interpolate(dataset, model, depth, method):
        known = sorted((key[2], value) for key, value in measured.items()
                       if key[0] == dataset and key[1] == model and key[3] == method)
        if not known:
            return None
        for layers, value in known:
            if layers == depth:
                return value
        if depth < known[0][0]:
            return known[0][1] * depth / known[0][0]
        if depth > known[-1][0]:
            return known[-1][1] * depth / known[-1][0]
        low = max((layers, value) for layers, value in known if layers < depth)
        high = min((layers, value) for layers, value in known if layers > depth)
        return low[1] + (high[1] - low[1]) * (depth - low[0]) / (high[0] - low[0])

    def rate(spec):
        dataset, model, depth = spec['dataset'], spec['model'], spec['num_layers']
        method = spec.get('variant') or spec['method']
        direct = interpolate(dataset, model, depth, method)
        if direct is not None and (method in ['base', 'gnnm'] or depth == 3):
            return direct, 'measured_or_depth_interpolated'
        if method not in ['base', 'gnnm']:
            ratios = {
                backbone: measured[('roman-empire', backbone, 3, method)] /
                measured[('roman-empire', backbone, 3, 'gnnm')]
                for backbone in ['GCN', 'GAT']
            }
            factor = ratios.get(model, sum(ratios.values()) / 2)
            return interpolate('roman-empire', model, depth, 'gnnm') * factor, 'variant_extrapolation'
        factor = interpolate(dataset, 'GAT', depth, method) / interpolate('roman-empire', 'GAT', depth, method)
        return interpolate('roman-empire', model, depth, method) * factor, 'dataset_extrapolation'

    parts = defaultdict(lambda: dict(fits=0, seconds_per_common_update=0.0,
                                    measured_or_depth_interpolated=0,
                                    variant_extrapolation=0, dataset_extrapolation=0,
                                    unknown_selected_depth_assumed_three=0))
    for task in plan['tasks']:
        if task['metadata']['action'] != 'node_fit':
            continue
        spec = dict(task['metadata']['spec'])
        resolver = task['metadata'].get('resolve')
        selected = resolver and 'selection' in resolver
        if selected:
            spec['num_layers'] = 3
        seconds, source = rate(spec)
        part = parts['conditional_expansion' if task['metadata'].get('stage', '').startswith('expanded') else 'core']
        task["estimated_seconds"] = round(seconds * 7500, 2)
        part['fits'] += 1
        part['seconds_per_common_update'] += seconds
        part[source] += 1
        part['unknown_selected_depth_assumed_three'] += int(bool(selected))
    rows = []
    for updates in [5000, 7500, 10000, 20000, 40000]:
        row = dict(updates_per_fit=updates)
        hours = {}
        for name, part in parts.items():
            hours[name] = part['seconds_per_common_update'] * updates / 3600
        hours['all_node'] = sum(hours.values())
        for name, value in hours.items():
            row[name] = dict(isolated_gpu_hours=value,
                            gpus_four_days_80pct=math.ceil(value / (96 * .8)),
                            gpus_four_days_80pct_concurrency1p2=math.ceil(value / (96 * .8 * 1.2)))
        rows.append(row)
    return dict(
        role='capacity_scenarios', calibration_gpu=calibration['gpu'], parts=dict(parts),
        scenarios=rows,
        caveats=[
            'Node fits only. Transfer epochs and dependent profiling/scoring are additional.',
            'The conditional expansion may remain closed.',
            'These scenarios are not predictions of stopping time or convergence.',
            'Selected confirmation depth is unknown before validation; depth three is assumed.',
            'Some backbones and variants use disclosed topology/rate extrapolations.',
            'LR, dropout and attention-head timing effects are not exhaustively measured.',
            'Concurrency factor 1.2 is provisional from one GAT workload on A100 80GB.',
            'A twenty-percent scheduling allowance is applied.',
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--additional', type=Path, nargs='*', default=[])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--annotated-plan', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text()) if args.plan else build_plan()
    calibration = json.loads(args.calibration.read_text())
    extra = [json.loads(path.read_text()) for path in args.additional]
    report = capacity(plan, calibration, extra)
    if args.annotated_plan:
        for task in plan["tasks"]:
            if "estimated_seconds" not in task:
                action = task["metadata"]["action"]
                task["estimated_seconds"] = 14400 if action == "transfer_fit" else 300 if task["kind"] == "gpu" else 20
        args.annotated_plan.write_text(json.dumps(plan, indent=2) + '\n')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
