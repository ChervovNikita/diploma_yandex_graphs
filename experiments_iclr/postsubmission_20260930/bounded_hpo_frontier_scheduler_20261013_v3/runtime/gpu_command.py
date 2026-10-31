"""GPU-specific preflight/start companion for the unchanged host supervisor."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import launcher
from spec import canonical


def selected_worker(plan, placement, observed, host, gpu_index):
    if plan.get('slots_per_gpu') != 2:
        raise ValueError('The frozen plan must declare two slots per GPU')
    if type(gpu_index) is not int or gpu_index not in {r['index'] for r in observed}:
        raise ValueError('Requested GPU index is absent from verified hardware')
    worker = host + '/gpu' + str(gpu_index)
    if worker not in {'allocation/gpu0', 'server77/gpu0', 'server77/gpu1'}:
        raise ValueError('GPU is outside the three-GPU assignment contract')
    if not any(p['worker'] == worker for p in placement.values()):
        raise ValueError('Selected GPU has no frozen assigned records')
    return worker


def preflight(root, plan_path, host, python, gpu_index):
    # Verify the allocation route before reading plan or input metadata.
    if host == 'allocation':
        launcher.host_check(host, {})
    plan, digest, studies, runs, placement, dependencies = launcher.load_plan(root, plan_path)
    observed = launcher.host_check(host, plan)
    worker = selected_worker(plan, placement, observed, host, gpu_index)
    runtime = launcher.runtime_preflight(plan, python)
    selected = [key for key in runs if placement[key]['worker'] == worker]
    stages = Counter(runs[key]['run']['config'].get('stage', 'unspecified') for key in selected)
    return {'status': 'gpu_preflight_passed', 'host': host, 'gpu_index': gpu_index,
            'worker': worker, 'slots_per_gpu': 2, 'plan_hash': digest,
            'selected_run_records': len(selected), 'selected_stage_records': dict(sorted(stages.items())),
            'verified_hardware': observed, 'runtime_compatibility': runtime,
            'global_gates': sorted(plan.get('global_gates', {})),
            'assignment_dependencies_unchanged': True, 'models_constructed': 0,
            'scientific_workers_started': 0, 'approval_written': False, 'GO': False}


def start(root, plan_path, approval_path, host, python, gpu_index):
    if not approval_path:
        raise ValueError('Explicit GO receipt is required')
    if host == 'allocation':
        launcher.host_check(host, {})
    plan, _, _, _, placement, _ = launcher.load_plan(root, plan_path, approval_path)
    selected_worker(plan, placement, launcher.host_check(host, plan), host, gpu_index)
    return launcher.detach_host(root, plan_path, approval_path, host, python, gpu_index=gpu_index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('preflight', 'start'))
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval')
    parser.add_argument('--host', choices=('allocation', 'server77'), required=True)
    parser.add_argument('--gpu-index', type=int, required=True)
    parser.add_argument('--python', required=True, help='Exact worker interpreter')
    args = parser.parse_args()
    if args.host == 'allocation':
        launcher.host_check(args.host, {})
    root = Path(args.project_root).resolve()
    for value in (args.plan, args.approval):
        if value and not Path(value).resolve().is_relative_to(root):
            raise ValueError('Plan and approval must stay in the project')
    if args.operation == 'preflight':
        if args.approval:
            raise ValueError('GPU preflight does not consume or write an approval')
        result = preflight(root, Path(args.plan), args.host, Path(args.python).resolve(), args.gpu_index)
    else:
        result = start(root, Path(args.plan), Path(args.approval) if args.approval else None,
                       args.host, Path(args.python).resolve(), args.gpu_index)
    print(canonical(result))


if __name__ == '__main__':
    main()
