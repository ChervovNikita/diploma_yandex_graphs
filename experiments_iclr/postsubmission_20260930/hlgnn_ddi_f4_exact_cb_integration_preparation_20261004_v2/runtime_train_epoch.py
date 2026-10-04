"""One actual complete TRAIN epoch for runtime feasibility; no VALID scoring.

Uses fresh seed-0 state and the exact native recipe/selected arm. This entry
point does not release the disabled 500-epoch scientific family. Runtime state
is discarded and cannot seed a scientific fit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import resource
import statistics
import sys
import time

sys.dont_write_bytecode = True
from runtime_paths import require_project_output
from train_f4_cb_ddi import verify_dependencies
from paired_comparison import ARMS, validate_receipt


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def support_aggregate(receipt):
    """Population denominators retain all selected rows, including extremes."""
    if 'support_counts' not in receipt:
        return None
    nl, nr = receipt['support_counts']
    kl, kr = receipt['teacher_counts']
    strata = {}
    positive = len(receipt['positive_positions'])
    for name, indices in (('positive', range(positive)), ('native_negative', range(positive, len(nl)))):
        indices = list(indices)
        vl = [0 < kl[i] < nl[i] for i in indices]
        vr = [0 < kr[i] < nr[i] for i in indices]
        strata[name] = {
            'queries': len(indices), 'candidate_slots': sum(nl[i] + nr[i] for i in indices),
            'teacher_ones': sum(kl[i] + kr[i] for i in indices),
            'empty_sides': sum(nl[i] == 0 for i in indices) + sum(nr[i] == 0 for i in indices),
            'zero_count_sides': sum(kl[i] == 0 for i in indices) + sum(kr[i] == 0 for i in indices),
            'full_count_sides': sum(kl[i] == nl[i] for i in indices) + sum(kr[i] == nr[i] for i in indices),
            'variable_sides': sum(vl) + sum(vr),
            'one_variable_side_queries': sum(a != b for a, b in zip(vl, vr)),
            'two_variable_sides_queries': sum(a and b for a, b in zip(vl, vr)),
        }
    return strata


class RuntimeObserver:
    def __init__(self, torch, device, output, arm):
        self.torch, self.device, self.output, self.arm = torch, device, output, arm
        self.rows = []
        self.maximum_allocated = self.maximum_reserved = 0

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def capture_peak(self):
        allocated = self.torch.cuda.max_memory_allocated(self.device)
        reserved = self.torch.cuda.max_memory_reserved(self.device)
        self.maximum_allocated = max(self.maximum_allocated, allocated)
        self.maximum_reserved = max(self.maximum_reserved, reserved)
        return allocated, reserved

    def start_update(self):
        self.synchronize()
        self.capture_peak()  # Also retains setup/epoch negative-draw allocations.
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()

    def finish_update(self, receipt):
        self.synchronize()
        elapsed = time.perf_counter() - self.started
        allocated, reserved = self.capture_peak()
        errors = validate_receipt(receipt, self.arm, 0, 1, len(self.rows), receipt['native_records'])
        if errors:
            raise RuntimeError(f'Runtime receipt schema failed: {errors}')
        row = {**receipt, 'runtime_only': True, 'update_wall_seconds': elapsed,
               'peak_cuda_allocated_bytes': allocated, 'peak_cuda_reserved_bytes': reserved,
               'support_teacher_aggregates': support_aggregate(receipt)}
        self.rows.append(row)
        with (self.output / 'runtime_updates.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps({'phase': 'runtime_update', 'arm': self.arm, 'batch': row['batch'],
                          'native_records': row['native_records'], 'wall_seconds': elapsed,
                          'peak_cuda_allocated_bytes': allocated, 'peak_cuda_reserved_bytes': reserved}), flush=True)

    def summary(self):
        aggregate = None
        if self.arm != 'target_only':
            aggregate = {}
            for stratum in ('positive', 'native_negative'):
                aggregate[stratum] = {
                    field: sum(row['support_teacher_aggregates'][stratum][field] for row in self.rows)
                    for field in self.rows[0]['support_teacher_aggregates'][stratum]
                } if self.rows else {}
        return {
            'completed_native_updates': len(self.rows),
            'completed_native_records': sum(row['native_records'] for row in self.rows),
            'last_batch_records': self.rows[-1]['native_records'] if self.rows else None,
            'update_wall_seconds': [row['update_wall_seconds'] for row in self.rows],
            'mean_update_wall_seconds': statistics.mean(row['update_wall_seconds'] for row in self.rows)
            if self.rows else None,
            'peak_cuda_allocated_bytes': self.maximum_allocated,
            'peak_cuda_reserved_bytes': self.maximum_reserved,
            'support_teacher_aggregates': aggregate,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0', help='Use a root-admitted CUDA device.')
    parser.add_argument('--cuda-memory-fraction', type=float, default=0.30,
                        help='Per-process PyTorch allocator limit; root defaults to 30%% of total device memory.')
    args = parser.parse_args()
    packet = Path(__file__).resolve().parent
    output = require_project_output(args.output_dir, packet)
    config = json.loads((packet / 'config.json').read_text())
    binding = json.loads((packet / 'INPUT_BINDINGS.json').read_text())
    verify_dependencies(packet, binding)
    sys.path.insert(1, str((packet.parent / binding['f4_packet']).resolve()))
    import baseline_train_ddi as baseline
    if (config['author_commit'] != baseline.PIN or config['recipe'] != baseline.AUTHOR_RECIPE
            or config['arms'] != list(ARMS) or config['seeds'] != [0, 1, 2]
            or config['artifact_contract']['train_weight_present'] is not False
            or config['qualification']['train_valid_contract'] is not True):
        raise ValueError('The qualified artifact and frozen scientific recipe are required.')
    contract = config['artifact_contract']
    if (contract['schema'] != 'hlgnn-ddi-train-valid-v1'
            or contract['sha256'] != binding['artifact']['sha256']):
        raise ValueError('TRAIN/VALID artifact contract changed.')
    artifact_path = Path(contract['path'])
    if not artifact_path.is_absolute():
        raise ValueError('Qualified artifact path must be absolute.')
    with artifact_path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != contract['sha256']:
            raise ValueError('TRAIN/VALID artifact SHA256 changed.')
    report_path = Path(config['qualification']['report'])
    report_bytes = report_path.read_bytes()
    report_pin = next(pin for pin in binding['evidence_pins']
                      if pin['path'].endswith('/owned_monitor01/run01/QUALIFICATION.json'))
    if hashlib.sha256(report_bytes).hexdigest() != report_pin['sha256']:
        raise ValueError('Qualified TRAIN/VALID report digest changed.')
    report = json.loads(report_bytes)
    if (report['artifact_sha256'] != contract['sha256']
            or report['native_graph_and_fixed_VALID_contract_qualified'] is not True
            or report['single_graph']['num_nodes'] != 4267
            or report['graph_qualification']['TRAIN_rows'] != 1067911):
        raise ValueError('Artifact qualification report does not bind the actual DDI graph.')
    # Runtime feasibility has its own entry point: no false runtime/full-budget gate is set true.
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from torch_geometric.transforms import ToSparseTensor
    from f4_cb_model import F4ConditionalPatternModel, AUXILIARY
    if config['auxiliary'] != AUXILIARY:
        raise ValueError('The fixed exact-CB masking/query/scaling recipe changed.')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This real runtime command requires an admitted CUDA device.')
    if not 0 < args.cuda_memory_fraction <= 1:
        raise ValueError('CUDA allocator fraction must be in (0, 1].')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'schema': 'hlgnn-ddi-f4-real-train-epoch-runtime-v2', 'status': 'in_progress',
                'runtime_only': True, 'arm': args.arm, 'seed': 0, 'epochs_requested': 1,
                'scientific_epochs': 500, 'scientific_seeds': [0, 1, 2],
                'artifact_sha256': contract['sha256'], 'recipe': config['recipe'],
                'source_manifest_sha256': hashlib.sha256((packet / 'MANIFEST.json').read_bytes()).hexdigest(),
                'auxiliary': config['auxiliary'], 'device': str(device),
                'cuda_allocator_fraction': args.cuda_memory_fraction,
                'VALID_scored': False, 'TEST_read': False, 'checkpoints_saved': False,
                'runtime_state_reusable_for_fits': False,
                'limits': 'Runtime feasibility only; no predictive evidence, full-budget admission or guaranteed extrapolation.'}
    write_json(output / 'runtime_summary.json', metadata)
    print(json.dumps({'phase': 'real_train_epoch_runtime_start', 'arm': args.arm,
                      'seed': 0, 'native_records': 1067911, 'native_updates': 17,
                      'device': str(device), 'output': str(output)}), flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    observer = RuntimeObserver(torch, device, output, args.arm)
    started = time.perf_counter()
    try:
        data, split_edge, num_nodes = baseline.load_train_valid(artifact_path, contract, torch, Data, ToSparseTensor)
        if num_nodes != 4267 or len(split_edge['train']['edge']) != 1067911:
            raise ValueError('Actual TRAIN dimensions differ from qualified DDI metadata.')
        data = data.to(device)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        recipe = config['recipe']
        model = F4ConditionalPatternModel(
            lr=recipe['lr'], dropout=recipe['dropout'], grad_clip_norm=recipe['grad_clip_norm'],
            gnn_num_layers=recipe['gnn_num_layers'], mlp_num_layers=recipe['mlp_num_layers'],
            emb_hidden_channels=recipe['emb_hidden_channels'], gnn_hidden_channels=recipe['gnn_hidden_channels'],
            mlp_hidden_channels=recipe['mlp_hidden_channels'], num_nodes=num_nodes, num_node_feats=0,
            gnn_encoder_name=recipe['encoder'], predictor_name=recipe['predictor'],
            loss_func=recipe['loss_func'], optimizer_name=recipe['optimizer'], device=device,
            use_node_feats=recipe['use_node_feats'], train_node_emb=recipe['train_node_emb'],
            pretrain_emb=recipe['pretrain_emb'], alpha=recipe['alpha'], init=recipe['init'],
            arm=args.arm, training_seed=0, stream_path=output / 'paired_stream.jsonl', runtime_observer=observer)
        model.param_init()
        observer.synchronize()
        epoch_started = time.perf_counter()
        model.train(data, split_edge, batch_size=recipe['batch_size'],
                    neg_sampler_name=recipe['neg_sampler'], num_neg=recipe['num_neg'])
        observer.synchronize()
        metadata.update(observer.summary(), epoch_wall_seconds=time.perf_counter() - epoch_started,
                        total_wall_seconds=time.perf_counter() - started)
        if (metadata['completed_native_updates'] != 17 or metadata['completed_native_records'] != 1067911
                or metadata['last_batch_records'] != 19335):
            raise RuntimeError('Runtime epoch omitted native records or final partial batch.')
        metadata['status'] = 'COMPLETE_REAL_TRAIN_EPOCH_RUNTIME_ONLY'
    except BaseException as error:
        observer.capture_peak()
        metadata.update(observer.summary(), status='FAILED_RUNTIME_ONLY',
                        error_type=type(error).__name__, error=str(error),
                        total_wall_seconds=time.perf_counter() - started,
                        host_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                        (1 if sys.platform == 'darwin' else 1024))
        write_json(output / 'runtime_summary.json', metadata)
        raise
    # Linux ru_maxrss is KiB; macOS is bytes. This includes Python/framework/setup.
    metadata['host_peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    metadata['framework_versions'] = {'torch': torch.__version__, 'numpy': np.__version__}
    write_json(output / 'runtime_summary.json', metadata)
    print(json.dumps(metadata, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
