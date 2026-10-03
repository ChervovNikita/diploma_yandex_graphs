"""Bounded, synthetic CUDA dropout semantics probe; no study or model access."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
HERE = Path(__file__).resolve().parent
GPU = 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'
SEED = 20261003
CASES = [
    ('native_input_control', (235868, 128), .25, False, False),
    ('encoder_normalized_output', (235868, 64), .1, True, False),
    ('decoder_full_node', (235868, 64), .3, True, False),
    ('decoder_native_batch', (65536, 64), .3, True, False),
]


def deadline(signum, frame):
    raise TimeoutError('Data-free probe exceeded its 90-second wall bound')


def main():
    assert Path.cwd().resolve() == REPO
    assert HERE == REPO / 'experiments_iclr/postsubmission_20260930/graph_ncNC_cuda_dropout_probe_root_20261003_v1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
    uuid = subprocess.run(['nvidia-smi', '--id=' + GPU, '--query-gpu=uuid', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.strip()
    assert uuid == GPU
    assert str(Path(sys.executable).resolve()) == '/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12'
    assert hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest() == '14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
    assert not (HERE / 'PROBE.json').exists()
    resource.setrlimit(resource.RLIMIT_CPU, (60, 65))
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(90)
    started = time.monotonic()
    receipt = dict(schema='ncnc-data-free-native-CUDA-dropout-semantics-v1', UTC=datetime.now(timezone.utc).isoformat(), status='STARTED', GPU_UUID=GPU, seed=SEED, synthetic_input='float32 ones, nonleaf activation=leaf*1.0', benchmark_data_checkpoint_quality_access=False, study_models_imported=False, broad_grid=False, cases=[])
    code = 1
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        assert torch.__version__ == '2.7.1' and torch.version.cuda == '12.6'
        assert torch.cuda.device_count() == 1
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        torch.manual_seed(SEED)
        receipt['runtime'] = dict(torch_version=torch.__version__, cuda_version=torch.version.cuda, GPU_name=torch.cuda.get_device_name(0), visible_device_count=torch.cuda.device_count())

        def rng():
            state = torch.cuda.get_rng_state(0)
            raw = bytes(state.tolist())
            detail = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            if len(raw) == 16:
                seed, offset = struct.unpack('<QQ', raw)
                detail.update(seed_uint64_le=seed, offset_uint64_le=offset)
            return state, detail

        def branch(shape, probability, kind, inplace, initial_state):
            leaf = torch.ones(shape, device='cuda', dtype=torch.float32, requires_grad=True)
            activation = leaf * 1.0
            layout = dict(shape=list(activation.shape), stride=list(activation.stride()), contiguous=activation.is_contiguous(), storage_offset=activation.storage_offset(), dtype=str(activation.dtype), device=str(activation.device))
            torch.cuda.set_rng_state(initial_state, 0)
            _, before = rng()
            if kind == 'author_nn_Dropout':
                output = nn.Dropout(p=probability, inplace=inplace).train()(activation)
            else:
                output = F.dropout(activation, p=probability, training=True, inplace=inplace)
            torch.cuda.synchronize()
            _, after_forward = rng()
            output.sum().backward()
            torch.cuda.synchronize()
            _, after_backward = rng()
            value, gradient = output.detach().clone(), leaf.grad.detach().clone()
            detail = dict(kind=kind, p=probability, training=True, inplace=inplace, layout=layout, output_aliases_activation=output.data_ptr() == activation.data_ptr(), before=before, after_forward=after_forward, after_backward=after_backward, backward_consumed_no_RNG=after_forward == after_backward, output_nonzero_count=int(torch.count_nonzero(value)), input_gradient_equals_output_for_ones=torch.equal(value, gradient))
            if 'offset_uint64_le' in before and 'offset_uint64_le' in after_forward:
                detail['forward_offset_delta_uint64_le'] = (after_forward['offset_uint64_le'] - before['offset_uint64_le']) % (2**64)
            return value, gradient, detail

        def compare(a, b):
            av, ag, ad = a
            bv, bg, bd = b
            return dict(masks_equal=torch.equal(av != 0, bv != 0), mask_mismatch_count=int(torch.count_nonzero((av != 0) != (bv != 0))), outputs_exactly_equal=torch.equal(av, bv), maximum_absolute_output_difference=float((av - bv).abs().max()), input_gradients_exactly_equal=torch.equal(ag, bg), maximum_absolute_input_gradient_difference=float((ag - bg).abs().max()), final_RNG_state_equal=ad['after_backward'] == bd['after_backward'])

        for label, shape, probability, native_inplace, original_inplace in CASES:
            torch.manual_seed(SEED)
            initial_state, initial_detail = rng()
            native = branch(shape, probability, 'author_nn_Dropout', native_inplace, initial_state)
            original = branch(shape, probability, 'current_F_dropout', original_inplace, initial_state)
            successor = branch(shape, probability, 'matched_F_dropout', native_inplace, initial_state)
            row = dict(case=label, shape=list(shape), p=probability, initial_RNG=initial_detail, author=native[2], current=original[2], matched=successor[2], current_vs_author=compare(original, native), matched_vs_author=compare(successor, native))
            receipt['cases'].append(row)
            assert row['matched_vs_author']['outputs_exactly_equal'] and row['matched_vs_author']['input_gradients_exactly_equal'] and row['matched_vs_author']['final_RNG_state_equal']
            del native, original, successor
        differences = [row for row in receipt['cases'] if not row['current_vs_author']['masks_equal']]
        receipt['inplace_mask_mismatch_confirmed'] = bool(differences)
        receipt['mismatched_cases'] = [row['case'] for row in differences]
        receipt['status'] = 'DATA_FREE_PROBE_COMPLETED'
        code = 0
    except BaseException as error:
        receipt.update(status='FAILED', error_type=type(error).__name__, error_condition=str(error))
    finally:
        signal.alarm(0)
        receipt.update(terminal_UTC=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-started, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), process_exit_code=code)
        with (HERE / 'PROBE.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
        print(json.dumps(receipt))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
