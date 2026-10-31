"""Post-training resource profile for every validation-selected tuning arm.

Run only after the complete global validation lock and after training has ended
on this GPU. No test labels or scores are used. This file does not alter the
frozen training, model, or validation-selection source.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import torch_geometric
import tuning as study

ROOT = Path(__file__).resolve().parent
ROUNDS = 3
WARMUPS = 20
REPETITIONS = 100
SEED = 0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_context():
    try:
        return subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,name,uuid,driver_version,temperature.gpu,utilization.gpu,memory.used',
            '--format=csv,noheader'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', default='INFERENCE_PROFILE.json')
    a = p.parse_args()
    output = ROOT / a.output
    if output.exists():
        raise RuntimeError('Refusing to overwrite an existing profile')
    device = torch.device(a.device)
    if device.type != 'cuda':
        raise ValueError('This protocol uses synchronized CUDA profiling')
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    freeze_sha = study.check_freeze()
    lock_file = ROOT / 'VALIDATION_SELECTION_LOCK.json'
    lock = json.loads(lock_file.read_text())
    if lock['freeze_sha256'] != freeze_sha or len(lock['cells']) != 432:
        raise RuntimeError('Complete global validation lock required')
    report = {
        'protocol': 'selected_inference_profile_v1',
        'source_sha256': sha(Path(__file__)),
        'freeze_sha256': freeze_sha,
        'selection_lock_sha256': sha(lock_file),
        'python': platform.python_version(), 'torch': torch.__version__,
        'pyg': torch_geometric.__version__, 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(device),
        'device_properties': str(torch.cuda.get_device_properties(device)),
        'cpu_threads': torch.get_num_threads(), 'optimization_seed': SEED,
        'rounds': ROUNDS, 'warmups_per_block': WARMUPS,
        'repetitions_per_block': REPETITIONS,
        'scope': 'full resident graph, eval mode, float32, sequential member forwards, stack and mean raw logits; excludes loading and host transfers',
        'timing': 'host wall clock bounded by CUDA synchronization for each forward; median and quartiles across 300 repeats',
        'memory': 'peak torch CUDA allocated bytes including resident graph/model, plus increment above idle graph/model; excludes reserved allocator memory and external process memory',
        'gpu_before': gpu_context(), 'graphs': {},
    }
    for dataset in study.DATASETS:
        bundle, desc = study.load_graph(dataset, device, include_test=False)
        records = {arm: {'blocks': []} for arm in study.ARMS}
        for rnd in range(ROUNDS):
            order = list(study.ARMS)
            if rnd == 1:
                order.reverse()
            elif rnd == 2:
                order = order[3:] + order[:3]
            for arm in order:
                lr, wd = lock['selections'][dataset][arm]['selected_candidate']
                key = f'{dataset}/{arm}/{study.candidate_name(lr, wd)}/seed{SEED}'
                cell = study.cell_path(dataset, arm, lr, wd, SEED)
                cp = cell / 'checkpoint.pt'
                if sha(cp) != lock['cells'][key]['checkpoint_sha256']:
                    raise RuntimeError(f'Checkpoint mismatch: {key}')
                study.seed_all(SEED)
                model, _ = study.make_model(arm, bundle, device)
                payload = torch.load(cp, map_location='cpu', weights_only=True)
                model.load_state_dict(payload['state_dict'], strict=True)
                model.eval()
                del payload
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                def predict():
                    return torch.stack(study.member_logits(model, arm, bundle)).mean(0)
                with torch.inference_mode():
                    for _ in range(WARMUPS):
                        prediction = predict()
                        if not torch.isfinite(prediction).all():
                            raise RuntimeError('Nonfinite inference')
                        del prediction
                    torch.cuda.synchronize(device)
                    gc.collect()
                    idle = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    prediction = predict()
                    torch.cuda.synchronize(device)
                    peak = torch.cuda.max_memory_allocated(device)
                    del prediction
                    durations = []
                    for _ in range(REPETITIONS):
                        torch.cuda.synchronize(device)
                        start = time.perf_counter_ns()
                        prediction = predict()
                        torch.cuda.synchronize(device)
                        durations.append((time.perf_counter_ns() - start) / 1e6)
                        del prediction
                rec = records[arm]
                rec.update({'selected_candidate': [lr, wd], 'checkpoint_sha256': sha(cp),
                            'parameter_count': sum(v.numel() for v in model.parameters()),
                            'parameter_bytes': sum(v.numel()*v.element_size() for v in model.parameters())})
                rec['blocks'].append({'round': rnd, 'position': order.index(arm),
                                      'latency_ms': durations, 'idle_allocated_bytes': idle,
                                      'peak_allocated_bytes': peak, 'peak_increment_bytes': peak-idle})
                del model
                gc.collect()
                torch.cuda.empty_cache()
        for rec in records.values():
            vals = [v for block in rec['blocks'] for v in block['latency_ms']]
            rec['latency_ms_median'] = statistics.median(vals)
            rec['latency_ms_q25'], rec['latency_ms_q75'] = np.quantile(vals,[.25,.75]).tolist()
            rec['block_latency_medians_ms'] = [statistics.median(b['latency_ms']) for b in rec['blocks']]
            rec['peak_allocated_bytes_max'] = max(b['peak_allocated_bytes'] for b in rec['blocks'])
            rec['peak_increment_bytes_max'] = max(b['peak_increment_bytes'] for b in rec['blocks'])
        report['graphs'][dataset] = {'descriptor': desc, 'arms': records}
        del bundle
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({'profiled_graph': dataset, 'arms': len(records)}), flush=True)
    report['gpu_after'] = gpu_context()
    study.write_json(output, report)
    print(json.dumps({'profile_file': output.name, 'sha256': sha(output)}), flush=True)


if __name__ == '__main__':
    main()
