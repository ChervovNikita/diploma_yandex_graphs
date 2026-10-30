"""Check exact runtime import closures and masked CUDA on the newly bound host."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
E = HERE.parent.parent
from host_extension import bind, pin, require, install_native_authority, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--task-check')
    args = parser.parse_args()
    config = bind(args.worker)
    manifest = json.loads((HERE / 'MANIFEST.json').read_text())
    require(pin(HERE / 'MANIFEST.json')['sha256'] == args.source_sha256, 'Preflight source changed')
    for row in manifest['files']:
        require(all(pin(HERE / row['path'])[key] == row[key] for key in ('bytes', 'sha256')), 'Source member changed')
    plan = json.loads((E / 'campaign_execution_20261016_v7/PLAN-allocation_gpu0.json').read_text())
    if args.task_check:
        task = next(row for row in plan['tasks'] if row['id'] == args.task_check)
        require(pin(task['runtime_entrypoint']['path']) == task['runtime_entrypoint'], 'Native runtime changed')
        if task['kind'] == 'fit':
            require(pin(task['descriptor']['path']) == task['descriptor'], 'Scientific descriptor changed')
        entry = Path(task['runtime_entrypoint']['path'])
        authority = install_native_authority('/composed/' in str(entry))
        sys.path.insert(0, str(entry.parent))
        spec = importlib.util.spec_from_file_location('_technical_import_check', entry)
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        if task['kind'] == 'fit':
            job = json.loads(Path(task['descriptor']['path']).read_text())
            runtime.require_allocation()
            runtime.validate_host_job(job, ROOT)
        print(json.dumps({'task_id': task['id'], 'runtime_source': pin(entry),
                          'authority': authority, 'status': 'PASS'}))
        return
    unique = {}
    for row in plan['tasks']:
        unique.setdefault(row['runtime_entrypoint']['path'], row['id'])
    imports = []
    for task in unique.values():
        command = [sys.executable, '-B', str(__file__), '--worker', args.worker,
                   '--source-sha256', args.source_sha256, '--task-check', task]
        result = subprocess.run(command, text=True, capture_output=True, timeout=120)
        require(result.returncode == 0, 'Native import/config check failed: ' + result.stdout + result.stderr)
        imports.append(json.loads(result.stdout.splitlines()[-1]))
    # Exercise inherited invocation assembly without taking custody, creating an
    # attempt or running a scientific fit. This catches MRO/launcher regressions.
    from types import SimpleNamespace
    from shard_worker import ShardWorker, four
    task = next(row for row in plan['tasks'] if row['kind'] == 'fit')
    draft = ShardWorker.__new__(ShardWorker)
    draft.args = SimpleNamespace(worker=args.worker, plan=str(E / 'campaign_execution_20261016_v7/PLAN-allocation_gpu0.json'))
    draft.plan = plan; draft.states = {task['id']: {'status': 'pending', 'attempt': 0}}
    draft.index = {task['id']: task}; draft.retry = set(); draft.recovery_config = {'repairs': {}}
    draft.canonical_root = Path(plan['state_root']); draft.partition_pin = {'technical_preflight_only': True}
    draft._scheduler_source = {'technical_preflight_only': True}
    draft.named_source = four.recovery.source(four.REPAIR_SHA)
    draft.process_title_dependency_dir = str(E / 'campaign_process_title_dependency_20261021_v1')
    request = draft.invocation(task, 'full', HERE / 'engineering_only' / args.worker.replace('/', '_'))
    require(request['argv'][3] == str(HERE / 'dispatch_native.py'), 'Additive launcher not selected')
    require(request['native_argv'][3] == plan['reporting_entrypoint']['path'], 'Original scientific/reporting argv lost')
    require(request['environment']['CUDA_VISIBLE_DEVICES'] == config['physical_gpu_uuid'], 'Physical mask lost')
    import torch
    require(torch.cuda.device_count() == 1, 'Only one masked GPU may be visible')
    model = torch.nn.Linear(16, 4).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(torch.ones(8, 16, device='cuda')).square().mean()
        require(bool(torch.isfinite(loss).item()), 'Nonfinite CUDA engineering loss')
        loss.backward(); optimizer.step()
    torch.cuda.synchronize()
    result = {'status': 'PASS', 'schema': 'allocation-extension-technical-preflight-v1',
              'config': config, 'source': pin(HERE / 'MANIFEST.json'), 'native_imports': imports,
              'torch_version': torch.__version__, 'masked_cuda_device_count': 1,
              'cuda_name': torch.cuda.get_device_name(0), 'technical_updates': 2,
              'inherited_invocation_assembly': 'PASS',
              'scientific_training_performed': False, 'TEST_accessed': False}
    (HERE / ('PREFLIGHT-' + args.worker.replace('/', '_') + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
