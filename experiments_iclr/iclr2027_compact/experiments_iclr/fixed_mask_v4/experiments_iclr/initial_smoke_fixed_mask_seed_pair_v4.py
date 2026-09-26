"""No-score, all-seed full-graph initial-equivalence gate for v4."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from fixed_mask_seed_pair_v4 import (
    ARGS, LOGIT_TOLERANCE, MEMBERS, REPO, RESULT, SEEDS, VARIANTS,
    canonical_state, check_graph_compatibility, hash_bytes, hash_state,
    verify_sources,
)
from projector_controls import all_logits, make_model
from datasets import load_dataset
from run_common import set_seed


def digest_object(value) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def make_initial(variant, seed, data, output_dim, device):
    set_seed(seed)
    model = make_model(ARGS, variant, data.x.shape[1], output_dim, device)
    state = hash_state(canonical_state(model, variant))
    rng = {
        'python': digest_object(random.getstate()),
        'numpy': digest_object(np.random.get_state()),
        'cpu': hash_bytes(torch.get_rng_state().numpy().tobytes()),
        'cuda': hash_bytes(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()),
    }
    model.eval()
    with torch.no_grad():
        logits = all_logits(model, variant, data, MEMBERS).detach().cpu().numpy()
    if not np.isfinite(logits).all() or logits.shape != (MEMBERS, 22662, output_dim):
        raise RuntimeError(f'Invalid initial logits: {seed} {variant}')
    del model
    torch.cuda.empty_cache()
    return state, rng, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if list(RESULT.glob('seed*')):
        raise RuntimeError('Initial smoke must precede every v4 training run')
    source_sha = verify_sources()
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Full-graph smoke requires CUDA')
    data, train, val, test, _, output_dim, is_binary = load_dataset(
        'roman-empire', add_self_loops=True, device=device, data_dir=str(REPO / 'data'))
    check_graph_compatibility(data, train, val, test)
    if is_binary or output_dim != 18:
        raise RuntimeError('Unexpected Roman label configuration')
    records = []
    for seed in SEEDS:
        tied = make_initial('gnnm', seed, data, output_dim, device)
        untied = make_initial('untied_backbone', seed, data, output_dim, device)
        if tied[0] != untied[0] or tied[1] != untied[1]:
            raise RuntimeError(f'Canonical state or post-construction RNG differs: seed {seed}')
        max_diff = float(np.max(np.abs(tied[2] - untied[2])))
        if max_diff > LOGIT_TOLERANCE:
            raise RuntimeError(f'Initial CUDA logits differ: seed {seed}, {max_diff}')
        changed_decisions = int(np.count_nonzero(tied[2].argmax(-1) != untied[2].argmax(-1)))
        records.append({
            'seed': seed, 'state_sha256': tied[0], 'rng_sha256': tied[1],
            'initial_logits_max_abs_difference': max_diff,
            'initial_member_decision_differences': changed_decisions,
        })
        print('INITIAL_PAIR_PASS', seed, max_diff, changed_decisions, flush=True)
    output = {
        'status': 'FULL_GRAPH_ALL_SEED_INITIAL_SMOKE_PASS',
        'source_manifest_sha256': source_sha,
        'device': str(device), 'records': records,
        'no_validation_or_test_metric_scored': True,
    }
    path = RESULT / 'gpu_initial_smoke.json'
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    print('FULL_GRAPH_ALL_SEED_INITIAL_SMOKE_PASS', len(records), flush=True)


if __name__ == '__main__':
    main()
