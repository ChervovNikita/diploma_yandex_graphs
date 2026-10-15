"""CPU-only optimistic readout bound on three completed residual VALID banks.

No model, checkpoint, training, gate fit, TEST read or change to served scores.
This bounds the already-proposed convex gate with a .05 weight floor per route.
The label-informed endpoint choices are a diagnostic oracle, not a predictor.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
E = ROOT / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'


def verified(pin):
    p = Path(pin['path']).resolve()
    assert p.is_relative_to(ROOT)
    raw = p.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pin['sha256']
    assert len(raw) == pin['bytes']
    return p


def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                   text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    started = time.monotonic()
    report_path = E / 'analysis_outputs/residual_only_complete_v1/RESULT.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'complete' and report['TEST_accessed'] is False
    assert len(report['rows']) == 33 and len(report['training_fit_outcomes']) == 39
    records = report['route_collection_outcomes'][-3:]
    import torch
    torch.set_num_threads(2)
    rows = []
    for seed, record in zip((101, 202, 303), records):
        collection = json.loads(verified(record['receipt']).read_text())
        assert collection['status'] == 'complete' and collection['TEST_accessed'] is False
        tensor = torch.load(verified(collection['raw_tensor_receipt']), map_location='cpu', weights_only=False)
        assert tensor['role'] == 'VALID' and tensor['TEST_accessed'] is False
        z, y = tensor['member_raw_logits'].double().squeeze(-1), tensor['labels'].reshape(-1)
        assert z.shape == (4, 151) and int((y == 1).sum()) == 130 and int((y == 0).sum()) == 21
        assert torch.isfinite(z).all()
        low, high = .05 * z.sum(0) + .8 * z.min(0).values, .05 * z.sum(0) + .8 * z.max(0).values
        margin = high[y == 1, None] - low[None, y == 0]
        assert margin.numel() == 2730
        impossible, tied = int((margin < 0).sum()), int((margin == 0).sum())
        actual = next(r for r in report['rows'] if r['seed'] == seed and r['bank'] == 'residual_only_private_BN')
        rows.append({'seed': seed, 'saved_pool_AUC': actual['pool_AUC'],
                     'unrepairable_pairs_for_any_declared_convex_weights': impossible,
                     'oracle_endpoint_ties': tied,
                     'optimistic_AUC_upper_bound': 1 - (impossible + .5 * tied) / 2730,
                     'bound_minus_saved_pool_AUC_pp': 100 * (1 - (impossible + .5 * tied) / 2730 - actual['pool_AUC']),
                     'collection_receipt': record['receipt']})
    out = Path(__file__).with_name('RESULT.json')
    assert not out.exists()
    result = {'status': 'complete', 'rows': rows, 'weight_law': 'alpha=.05+.8*softmax(score), four routes',
              'interpretation': 'Optimistic label-informed endpoint ceiling in float64 on fixed collected logits. Softmax endpoints are limiting values; no TRAIN-learnable gate is established. Scores and checkpoint selection are unchanged.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'analysis_input_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
              'new_models': 0, 'native_forwards': 0, 'optimizer_steps': 0, 'TEST_accessed': False,
              'wall_seconds': time.monotonic() - started}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
