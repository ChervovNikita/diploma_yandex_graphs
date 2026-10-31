"""Real128-training-graph CUDA function/gradient/AdamW engineering preflight."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
PACKET = PHASE / 'sharing_policy/gine_transfer_grouping_fix_v6'
sys.path.insert(0, str(PACKET))
import torch
from torch_geometric.data import Batch
from probe import new_model, prepare, gpu_guard
from gine_collector import equal_graph_bce


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-bundle', type=Path, required=True)
    p.add_argument('--expected-bundle-sha256', required=True)
    p.add_argument('--export-supervisor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    bundle, parent, output = args.train_bundle.resolve(), args.export_supervisor.resolve(), args.output.resolve()
    if not all(path.is_relative_to(PHASE) for path in (bundle, parent, output)) or output.exists():
        raise ValueError('Require confined inputs and exclusive new output')
    auth = json.loads((parent / 'authorization.json').read_text())
    done = json.loads((parent / 'completion.json').read_text())
    if auth['ssh_destination'] != 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru' or auth['visible_gpu_count'] != 1:
        raise ValueError('Only newly authorized exporter evidence qualifies')
    if done['exit_code'] != 0 or auth['inner_completion_sha256'] != sha(parent / 'completion.json'):
        raise ValueError('Exporter did not complete successfully')
    if sha(bundle) != args.expected_bundle_sha256:
        raise ValueError('Training bundle changed')
    resource = gpu_guard(SimpleNamespace(device='cuda:0', physical_gpu=0))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(31209)
    units, _, mask, _ = prepare(bundle, .1)
    chosen = sorted(units, key=lambda unit: unit.unit_id)[:128]
    if len(chosen) != 128:
        raise ValueError('Need complete real128-graph batch')
    batch = Batch.from_data_list([unit.graph for unit in chosen])
    reference = new_model().eval()
    model = copy.deepcopy(reference).to('cuda:0').eval()
    gpu_batch = batch.clone().to('cuda:0')
    if any(not torch.equal(cpu, gpu.detach().cpu()) for cpu, gpu in zip(reference.parameters(), model.parameters())):
        raise AssertionError('Initial parameter copy changed')
    cpu_logits = reference(batch, batch.x)
    gpu_logits = model(gpu_batch, gpu_batch.x)
    if not torch.allclose(cpu_logits, gpu_logits.detach().cpu(), atol=5e-4, rtol=5e-4):
        raise AssertionError('CPU/CUDA real-graph logits differ beyond predeclared engineering tolerance')
    cpu_loss = equal_graph_bce(cpu_logits, batch.y.float())
    gpu_loss = equal_graph_bce(gpu_logits, gpu_batch.y.float())
    cpu_loss.backward()
    gpu_loss.backward()
    gradient_error = 0.
    for a, b in zip(reference.parameters(), model.parameters()):
        if a.grad is None or b.grad is None or not bool(torch.isfinite(b.grad).all()):
            raise AssertionError('Missing/nonfinite real-batch gradient')
        error = float((a.grad - b.grad.detach().cpu()).abs().max())
        gradient_error = max(gradient_error, error)
        if not torch.allclose(a.grad, b.grad.detach().cpu(), atol=5e-4, rtol=5e-4):
            raise AssertionError('CPU/CUDA real-batch gradients differ beyond predeclared engineering tolerance')
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0., eps=1e-8)
    if optimizer.state:
        raise AssertionError('Need a fresh optimizer')
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    step_loss = equal_graph_bce(model(gpu_batch, gpu_batch.x), gpu_batch.y.float())
    step_loss.backward()
    if not bool(torch.isfinite(step_loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise AssertionError('Real CUDA training step has missing/nonfinite gradients')
    optimizer.step()
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - start
    changed = sum(not torch.equal(old, new) for old, new in zip(initial, model.parameters()))
    if not changed:
        raise AssertionError('CUDA AdamW step did not change the model')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'gnnm-authorized-molhiv-cuda-preflight-v1', 'status': 'passed',
              'scope': 'One real training batch; implementation engineering, not predictive evaluation.',
              'batch_selection': 'First128 sorted observed training units under frozen tenpercent mask',
              'graphs': 128, 'nodes': int(batch.num_nodes), 'directed_edges': int(batch.num_edges),
              'unit_ids': [unit.unit_id for unit in chosen], 'label_mask': mask,
              'resource_guard': resource, 'bundle_sha256': sha(bundle),
              'authorized_exporter_record_sha256': sha(parent / 'authorization.json'),
              'maximum_CPU_CUDA_logit_error': float((cpu_logits.detach() - gpu_logits.detach().cpu()).abs().max()),
              'maximum_CPU_CUDA_gradient_error': gradient_error, 'engineering_atol': 5e-4, 'engineering_rtol': 5e-4,
              'CPU_BCE': float(cpu_loss.detach()), 'CUDA_BCE': float(gpu_loss.detach()),
              'optimizer_steps': 1, 'changed_parameter_tensors': changed, 'step_seconds': elapsed,
              'CUDA_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
              'CUDA_peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
              'validation_or_test_scored': False, 'predictive_improvement_claimed': False,
              'source_sha256': sha(Path(__file__))}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'graphs': 128, 'optimizer_steps': 1,
                      'step_seconds': elapsed, 'maximum_gradient_error': gradient_error,
                      'CUDA_peak_allocated_bytes': report['CUDA_peak_allocated_bytes']}))


if __name__ == '__main__':
    main()
