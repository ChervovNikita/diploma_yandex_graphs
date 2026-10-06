#!/usr/bin/env python3
"""Fixed ordinary WikiCS pilot; root adoption/review required before training."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
REPO = ROOT.parents[2]
ARMS = ('S', 'E1', 'EG', 'EI', 'I1', 'IG')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): value.update(block)
    return value.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def bound(record):
    path = (PHASE / record['path']).resolve(strict=True)
    if not path.is_relative_to(PHASE.resolve()) or not path.is_file() or sha(path) != record['sha256']:
        raise ValueError('Exact phase file binding changed')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); job = json.loads(args.job.read_text()); output = args.output.resolve()
    if job.get('root_execution_authorized') is not True or job.get('source_review_approved') is not True or job.get('TEST_access') is not False:
        raise ValueError('Root-adopted reviewed source/job required; TEST closed')
    if output.exists() or not output.is_relative_to(PHASE.resolve()) or str(output) != job['output_directory']:
        raise ValueError('Fresh exact authorized output required')
    plan_path = bound(job['plan']); plan = json.loads(plan_path.read_text())
    if not plan['root_adopted'] or plan['seeds'] != [17, 29, 43] or plan['arms'] != list(ARMS) or job['arm'] not in ARMS or job['seed'] not in plan['seeds']:
        raise ValueError('Exact fixed six-control/three-seed plan required')
    if sha(__file__) != job['program_sha256'] or sha(ROOT/'MANIFEST.json') != job['source_manifest_sha256']:
        raise ValueError('Reviewed source bytes changed')
    for row in json.loads((ROOT/'MANIFEST.json').read_text())['files']:
        if sha(ROOT/row['path']) != row['sha256']: raise ValueError('Sealed source dependency changed')
    if not job['source_review_evidence']: raise ValueError('Exact source-review evidence required')
    for evidence in job['source_review_evidence']: bound(evidence)
    if socket.gethostname() != job['expected_hostname'] or str(REPO) != job['repository'] or Path.cwd().resolve() != REPO.resolve():
        raise ValueError('Exact allowed repository/host required')
    if job['repository'] not in ('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs', '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'):
        raise ValueError('Only existing authorized repos may execute')
    targets = {
        '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs': ('anogena-2-0', ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']),
        '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git': ('peptide', ['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998', 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'])}
    hostname, inventory = targets[job['repository']]
    observed = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
    if socket.gethostname() != hostname or observed != inventory or job['physical_gpu_uuid'] not in inventory:
        raise ValueError('Exact authorized physical host/GPU inventory required')
    if str(Path(sys.executable).absolute()) != job['python_executable'] or os.environ.get('CUDA_VISIBLE_DEVICES') != job['physical_gpu_uuid']:
        raise ValueError('Exact existing interpreter and one visible physical GPU required')
    if job.get('retry') is not False or job.get('external_hard_bound_confirmed') is not True or not isinstance(job['soft_seconds'], int) or job['soft_seconds'] <= 0:
        raise ValueError('Fixed owned bounds and no retry required')
    import numpy as np
    import torch
    import torch_geometric
    import torch_scatter
    import torch_sparse
    from vendor.native_polynormer import Polynormer
    from vendor.backbone_boundary_adapter import PolynormerBoundaryFamily
    from weights import make_weights
    if str(torch.__version__) != '2.1.2+cu118' or torch_geometric.__version__ != '2.7.0' or np.__version__ != '1.26.4' or torch_scatter.__version__ != '2.1.2+pt21cu118' or torch_sparse.__version__ != '0.6.18+pt21cu118' or torch.cuda.device_count() != 1:
        raise ValueError('Pinned runtime and one visible GPU required')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    manifest_path = bound(job['data_manifest']); manifest = json.loads(manifest_path.read_text())
    if (manifest['nodes'], manifest['features'], manifest['classes'], manifest['official_split']) != (11701, 300, 10, 0) or manifest['TEST_labels_available_to_trainer'] is not False:
        raise ValueError('Official WikiCS split0 safe roles required')
    data_path = bound(manifest['available']); data = torch.load(data_path, map_location='cpu', weights_only=True)
    if set(data) != {'x', 'edge_index', 'train_ids', 'train_y', 'valid_ids', 'valid_y'}: raise ValueError('No TEST role may enter trainer')
    if tuple(data['x'].shape) != (11701, 300) or data['x'].dtype != torch.float32: raise ValueError('Full FP32 raw features required')
    output.mkdir(); started = time.monotonic(); device = torch.device('cuda:0')
    try:
        before_cpu = torch.get_rng_state().clone(); before_cuda = torch.cuda.get_rng_state().clone()
        graph_weights, iid_weights, field_audit = make_weights(torch, data['edge_index'], data['train_ids'], 11701, job['seed'])
        if not torch.equal(before_cpu, torch.get_rng_state()) or not torch.equal(before_cuda, torch.cuda.get_rng_state()): raise ValueError('Field generation advanced model/global RNG')
        torch.save({'graph': graph_weights, 'iid': iid_weights}, output/'FIXED_WEIGHTS.pt'); write(output/'FIELD_AUDIT.json', field_audit)
        x = data['x'].to(device); edge = data['edge_index'].to(device); train_ids = data['train_ids'].to(device)
        train_y = data['train_y'].to(device); valid_ids = data['valid_ids']; valid_y = data['valid_y']
        graph_weights, iid_weights = graph_weights.to(device), iid_weights.to(device)
        independent = job['arm'] in ('I1', 'IG'); shared = job['arm'] in ('E1', 'EG', 'EI')

        def cpu_tree(value):
            if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
            if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)): return type(value)(cpu_tree(v) for v in value)
            return value

        def core(model): return model.core if shared else model

        def construct(seed):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = Polynormer(300, 512, 10, local_layers=7, global_layers=2, in_dropout=.5, dropout=.5,
                global_dropout=.5, heads=1, beta=-1, pre_ln=False).to(device)
            model.reset_parameters(); model._global = False
            if shared: model = PolynormerBoundaryFamily(model, members=4)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0., betas=(.9, .999), eps=1e-8)
            names = [name for name, _ in model.named_parameters()]
            if len({id(p) for p in model.parameters()}) != len(names) or {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in model.parameters()}:
                raise ValueError('Optimizer must own exactly this complete model')
            streams = [{'cpu': torch.get_rng_state().clone(), 'cuda': torch.cuda.get_rng_state().clone()}]
            for member in range(1, 4 if shared else 1):
                streams.append({'cpu': torch.Generator(device='cpu').manual_seed(seed+1009*member).get_state(),
                    'cuda': torch.Generator(device=device).manual_seed(seed+1009*member).get_state()})
            return model, optimizer, streams, names

        @contextmanager
        def route_rng(stream):
            old_cpu, old_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
            torch.set_rng_state(stream['cpu']); torch.cuda.set_rng_state(stream['cuda'])
            try: yield
            finally:
                stream.update(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state().clone())
                torch.set_rng_state(old_cpu); torch.cuda.set_rng_state(old_cuda)

        def score(probabilities):
            pooled = probabilities.mean(0)[valid_ids]
            return {'correct': int((pooled.argmax(1) == valid_y).sum()), 'nodes': len(valid_ids),
                'accuracy': float((pooled.argmax(1) == valid_y).float().mean()),
                'NLL': float(-pooled[torch.arange(len(valid_ids)), valid_y].double().clamp_min(1e-300).log().mean()),
                'members_accuracy': [float((p[valid_ids].argmax(1) == valid_y).float().mean()) for p in probabilities]}

        def evaluate(model):
            model.eval()
            with torch.no_grad():
                logits = [model.forward_member(x, edge, member) for member in range(4)] if shared else [model(x, edge)]
                return torch.stack([torch.softmax(z, 1).cpu() for z in logits])

        selected_members = []; selector_records = []; model_counters = []
        for member in range(4 if independent else 1):
            seed = job['seed']+1009*member if independent else job['seed']
            model, optimizer, streams, names = construct(seed)
            destination = output/('member_'+str(member)); destination.mkdir()
            torch.save({'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
                'streams': cpu_tree(streams), 'stage_global': False, 'optimizer_owned_names': names, 'seed': seed}, destination/'INITIAL_STATE.pt')
            best = -1; selected_epoch = selected_global = None; updates = route_forwards = 0
            torch.cuda.reset_peak_memory_stats()
            with (destination/'HISTORY.jsonl').open('x') as history:
                for epoch in range(1100):
                    epoch_started = time.monotonic()
                    if epoch == 100:
                        checkpoint = torch.load(destination/'selected_checkpoint.pt', map_location='cpu', weights_only=True)
                        if checkpoint['stage_global']: raise ValueError('Local transition requires selected local state')
                        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
                        core(model)._global = True  # Selected local model/Adam, live end-local route RNG retained.
                        write(destination/'TRANSITION.json', {'selected_local_epoch': checkpoint['epoch'], 'model_Adam_restored': True,
                            'live_end_local_RNG_retained': True, 'selector_reset': False, 'global_stage': True})
                    model.train(); optimizer.zero_grad(set_to_none=True); losses = []
                    for route in range(4 if shared else 1):
                        with route_rng(streams[route]):
                            z = model.forward_member(x, edge, route) if shared else model(x, edge)
                            log_prob = torch.nn.functional.log_softmax(z, 1)[train_ids]
                            per_node = None
                            if job['arm'] in ('EG', 'IG', 'EI'):
                                per_node = torch.nn.functional.nll_loss(log_prob, train_y, reduction='none')
                                weight_route = route if shared else member
                                weight = graph_weights[weight_route, train_ids] if job['arm'] in ('EG', 'IG') else iid_weights[route, train_ids]
                                loss = (weight*per_node).sum()/len(train_ids)
                            else:
                                loss = torch.nn.functional.nll_loss(log_prob, train_y)  # Released native unweighted objective.
                            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite TRAIN loss')
                            (loss/(4 if shared else 1)).backward(); losses.append(float(loss.detach()))
                            route_forwards += 1
                        del z, log_prob, per_node, loss
                    for name, parameter in model.named_parameters():
                        inactive = name.startswith('core.global_attn.' if shared else 'global_attn.') or name.startswith('core.ln.' if shared else 'ln.') or name.startswith('global_head.' if shared else 'pred_global.')
                        if core(model)._global: inactive = name.startswith('local_head.' if shared else 'pred_local.')
                        if not inactive and (parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())): raise ValueError('Disconnected/nonfinite active gradient: '+name)
                    optimizer.step(); updates += 1
                    probabilities = evaluate(model); result = score(probabilities)
                    if result['correct'] > best:
                        best = result['correct']; selected_epoch = epoch+1; selected_global = core(model)._global
                        torch.save({'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()), 'streams': cpu_tree(streams),
                            'global_cpu_RNG': torch.get_rng_state().clone(), 'global_cuda_RNG': torch.cuda.get_rng_state().cpu().clone(),
                            'stage_global': selected_global, 'training': model.training, 'epoch': selected_epoch,
                            'metrics': result, 'optimizer_owned_names': names, 'seed': seed, 'job': job}, destination/'selected_checkpoint.pt')
                    torch.cuda.synchronize()
                    history.write(json.dumps({'epoch': epoch+1, 'stage_global': core(model)._global, 'TRAIN_member_losses': losses, 'complete_VALID': result,
                        'seconds': time.monotonic()-epoch_started, 'ordinary_Adam_updates': updates, 'TRAIN_route_forwards': route_forwards,
                        'CUDA_peak_allocated': torch.cuda.max_memory_allocated(), 'CUDA_peak_reserved': torch.cuda.max_memory_reserved()})+'\n'); history.flush()
                    write(output/'PROGRESS.json', {'member': member, 'epoch': epoch+1, 'stage_global': core(model)._global, 'ordinary_Adam_updates': updates, 'scores_read_by_owner': False})
                    if time.monotonic()-started > job['soft_seconds']: raise TimeoutError('Fixed root cap; no horizon shortening or retry')
                torch.save({'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()), 'streams': cpu_tree(streams),
                    'stage_global': core(model)._global, 'optimizer_owned_names': names}, destination/'FINAL_STATE.pt')
            checkpoint = torch.load(destination/'selected_checkpoint.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer']); core(model)._global = checkpoint['stage_global']
            model.train(checkpoint['training']); probabilities = evaluate(model)
            replay_metrics = score(probabilities)
            if replay_metrics['correct'] != checkpoint['metrics']['correct']: raise ValueError('Selected forward-mode replay failed')
            selected_members.extend(probabilities.unbind(0)); selector_records.append({'member': member, 'seed': seed, 'epoch': selected_epoch,
                'stage_global': selected_global, 'checkpoint_sha256': sha(destination/'selected_checkpoint.pt'), 'selected_metrics': replay_metrics})
            model_counters.append({'member': member, 'ordinary_Adam_updates': updates, 'TRAIN_route_forwards': route_forwards})
            del model, optimizer, streams, checkpoint, probabilities
        bank = torch.stack(selected_members)
        torch.save({'member_VALID_probabilities': bank[:, valid_ids], 'mean_VALID_probabilities': bank.mean(0)[valid_ids],
            'selector_records': selector_records, 'input_manifest_sha256': job['data_manifest']['sha256']}, output/'SELECTED_VALID_PROBABILITIES.pt')
        write(output/'FREEZE.json', {'arm': job['arm'], 'seed': job['seed'], 'complete': True, 'schedule': [100, 1000],
            'selectors': selector_records, 'selected_bank_VALID': score(bank), 'model_counters': model_counters,
            'independent_optimizers': independent, 'pooled_training_loss': False, 'selector_budget': 4*1100 if independent else 1100,
            'program_sha256': sha(__file__), 'job_sha256': sha(args.job), 'plan_sha256': job['plan']['sha256'],
            'fixed_weights_sha256': sha(output/'FIXED_WEIGHTS.pt'), 'selected_probabilities_sha256': sha(output/'SELECTED_VALID_PROBABILITIES.pt'),
            'inclusive_seconds': time.monotonic()-started, 'TEST_access': False, 'retry': False})
    except BaseException as error:
        write(output/'FAILURE.json', {'error': type(error).__name__+': '+str(error), 'partial_outputs_preserved': True, 'TEST_access': False, 'retry': False})
        raise


if __name__ == '__main__': main()
