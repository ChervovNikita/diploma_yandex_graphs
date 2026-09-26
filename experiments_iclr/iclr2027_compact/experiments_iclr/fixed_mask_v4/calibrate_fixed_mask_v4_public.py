"""No-score CUDA numerical calibration for a separate Roman fixed-mask v4 study."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE / 'experiments_iclr'))
sys.path.insert(0, str(SOURCE))
from fixed_mask_seed_pair_v4 import ARGS, MEMBERS, canonical_state, check_graph_compatibility, hash_state
from projector_controls import all_logits, make_model
from datasets import load_dataset
from run_common import set_seed

DEVICE = torch.device('cuda:0')
REPEATS = 6


def model_for(seed: int, variant: str, data, classes: int):
    set_seed(seed)
    model = make_model(ARGS, variant, data.x.shape[1], classes, DEVICE)
    return model, hash_state(canonical_state(model, variant)), {
        'cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        'cuda': hashlib.sha256(torch.cuda.get_rng_state(DEVICE).cpu().numpy().tobytes()).hexdigest(),
    }


def repeats(model, variant, data):
    model.eval()
    outputs = []
    with torch.no_grad():
        for _ in range(REPEATS):
            outputs.append(all_logits(model, variant, data, MEMBERS).detach().cpu().numpy())
    return outputs


def max_abs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def main():
    data, train, val, test, _, classes, is_binary = load_dataset(
        'roman-empire', add_self_loops=True, device=DEVICE, data_dir=str(SOURCE / 'data'))
    check_graph_compatibility(data, train, val, test)
    assert classes == 18 and not is_binary
    rows = []
    for seed in range(3):
        entries = {}
        for variant in ('gnnm', 'untied_backbone'):
            model, state_sha, rng_sha = model_for(seed, variant, data, classes)
            logits = repeats(model, variant, data)
            within = max(max_abs(logits[i], logits[j])
                         for i in range(REPEATS) for j in range(i + 1, REPEATS))
            entries[variant] = (logits, state_sha, rng_sha, within)
            del model
            torch.cuda.empty_cache()
        tied, untied = entries['gnnm'], entries['untied_backbone']
        assert tied[1] == untied[1] and tied[2] == untied[2]
        cross = max(max_abs(a, b) for a in tied[0] for b in untied[0])
        changes = max(int(np.count_nonzero(a.argmax(-1) != b.argmax(-1)))
                      for a in tied[0] for b in untied[0])
        rows.append({'seed': seed, 'state_sha256': tied[1], 'rng_sha256': tied[2],
                     'tied_repeated_forward_max_abs': tied[3],
                     'untied_repeated_forward_max_abs': untied[3],
                     'cross_variant_repeated_forward_max_abs': cross,
                     'cross_variant_max_member_decision_differences': changes})
    output = {'purpose': 'no-label, no-score numerical calibration before any v4 training',
              'source_protocol': 'v4-public-recheck', 'repeats_per_seed_variant': REPEATS,
              'rows': rows, 'torch': torch.__version__, 'cuda': torch.version.cuda}
    path = SOURCE / 'reference/v4_numeric_calibration_recheck.json'
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    print(json.dumps(output, sort_keys=True))


if __name__ == '__main__':
    main()
