"""Check whether matching split seeds align initial SAGE backbone tensors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from projector_controls import make_model
from datasets import load_dataset
from run_common import set_seed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments_iclr' / 'recomputed_main' / 'init_audit.json'
VARIANTS = ('gnnm', 'independent_projectors', 'heads_only', 'input_only',
            'output_only', 'base', 'gnnm_m1')


def backbone_state(model):
    state = {}
    for prefix in ('residual_modules', 'output_normalization'):
        for key, tensor in getattr(model, prefix).state_dict().items():
            state[f'{prefix}.{key}'] = tensor.detach().cpu().clone()
    return state


def digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    device = torch.device('cpu')
    data, _, _, _, _, out_dim, _ = load_dataset(
        'roman-empire', add_self_loops=True, device=device, data_dir='data')
    args = SimpleNamespace(model='SAGE', num_layers=5, hidden_dim=512, m=4)
    report = {'dataset': 'roman-empire', 'model': 'SAGE', 'split_seed': 0,
              'comparison': 'Initial residual modules and output normalization after set_seed(0) before each model constructor',
              'variants': {}}
    reference = None
    for variant in VARIANTS:
        set_seed(0)
        model = make_model(args, variant, data.x.size(1), out_dim, device)
        state = backbone_state(model)
        if reference is None:
            reference = state
        if set(state) != set(reference):
            raise ValueError(f'Backbone state keys differ for {variant}')
        report['variants'][variant] = {
            'sha256': digest(state),
            'tensor_count': len(state),
            'exactly_equal_tensors_to_gnnm': sum(torch.equal(state[k], reference[k])
                                                for k in state),
            'max_abs_difference_to_gnnm': max((state[k] - reference[k]).abs().max().item()
                                              for k in state if state[k].is_floating_point()),
        }
        del model
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
