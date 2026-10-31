"""Single-pass conditional projector pilot from a frozen GNNM checkpoint.

This deliberately tests an architectural extension separately from GNNM.
The member projector *functions* are mixed per node, then the shared SAGE
backbone runs once. A static mixture is the matched single-pass control.
No test label is used until both gates have been selected on validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'experiments_iclr' / 'conditional_projector_results'
SOURCE_CSV = ROOT / 'experiments_iclr' / 'results' / 'projector_controls.csv'
SOURCE_CACHE = (ROOT / 'experiments_iclr' / 'moe_gate_results'
                / 'roman_empire_sage_gnnm_split0_all_logits.npz')
SOURCE_META = (ROOT / 'experiments_iclr' / 'moe_gate_results'
               / 'roman_empire_sage_gnnm_split0_export.json')


def source_row(variant: str) -> dict[str, str] | None:
    with SOURCE_CSV.open(newline='') as handle:
        rows = [row for row in csv.DictReader(handle)
                if (row['dataset'], row['model'], row['variant'], row['split'])
                == ('roman-empire', 'SAGE', variant, '0')]
    if len(rows) > 1:
        raise RuntimeError(f'Duplicate {variant} split-0 rows')
    return rows[0] if rows else None


def load_frozen_expert(device: torch.device, data):
    from ablation.models_ablation import TABMAblationModel

    row = source_row('gnnm')
    if row is None:
        raise FileNotFoundError('Completed SAGE GNNM split-0 control row is required')
    path = ROOT / row['checkpoint']
    with path.open('rb') as handle:
        sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
    base = TABMAblationModel(
        model_name='SAGE', num_layers=int(row['num_layers']),
        input_dim=data.x.shape[1], hidden_dim=int(row['hidden_dim']),
        output_dim=int(data.y.max().item()) + 1,
        hidden_dim_multiplier=1, num_heads=8, normalization='LayerNorm',
        dropout=0.2, tabm_inits=int(row['m']), init_scheme='default',
        device=device,
    ).to(device)
    base.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    base.requires_grad_(False)
    base.eval()
    return base, row, sha256


def mix_projector(block: nn.Module, x: torch.Tensor,
                  weights: torch.Tensor) -> torch.Tensor:
    """Exact convex mixture of member BE maps, before/after one backbone pass."""
    result = None
    for member in range(weights.shape[1]):
        member_out = block(x, tabm_seed=member)
        term = weights[:, member:member + 1] * member_out
        result = term if result is None else result + term
    assert result is not None
    return result


class ConditionalProjector(nn.Module):
    def __init__(self, base: nn.Module, train_features: torch.Tensor,
                 mode: str, selected_member: int):
        super().__init__()
        if mode not in ('static', 'node'):
            raise ValueError(mode)
        self.base = base
        self.mode = mode
        m = base.tabm_inits
        d = train_features.shape[1]
        mean = train_features.mean(dim=0)
        std = train_features.std(dim=0).clamp_min(1e-6)
        self.register_buffer('feature_mean', mean)
        self.register_buffer('feature_std', std)
        self.alpha = nn.Parameter(torch.zeros(m, device=mean.device))
        with torch.no_grad():
            # Both modes start from the same validation-selected near-pure path.
            self.alpha[selected_member] = 4.0
        self.gate = nn.Linear(d, m, bias=False, device=mean.device) if mode == 'node' else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()  # Frozen experts keep dropout disabled.
        return self

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        score = self.alpha.expand(x.shape[0], -1)
        if self.gate is not None:
            normalized = ((x - self.feature_mean) / self.feature_std).clamp(-5, 5)
            score = score + self.gate(normalized)
        return score.softmax(dim=1)

    def forward_with_weights(self, graph, x: torch.Tensor,
                             weights: torch.Tensor) -> torch.Tensor:
        h = mix_projector(self.base.input_be_block, x, weights)
        h = self.base.act(self.base.dropout(h))
        for module in self.base.residual_modules:
            h = module(graph, h)
        h = self.base.output_normalization(h)
        return mix_projector(self.base.output_be_block, h, weights)

    def forward(self, graph, x: torch.Tensor):
        weights = self.weights(x)
        return self.forward_with_weights(graph, x, weights), weights


def metrics(logits: torch.Tensor, labels: torch.Tensor,
            mask: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        z, y = logits[mask], labels[mask]
        return {
            'accuracy': (z.argmax(dim=1) == y).float().mean().item(),
            'cross_entropy': F.cross_entropy(z, y).item(),
        }


def gate_split(original_val_mask: torch.Tensor):
    """Predeclared 50/50 partition of original validation nodes, seed 2027."""
    indices = original_val_mask.nonzero(as_tuple=False).flatten().cpu().numpy()
    indices = np.random.default_rng(2027).permutation(indices)
    half = len(indices) // 2
    fit = torch.zeros_like(original_val_mask)
    selection = torch.zeros_like(original_val_mask)
    fit[torch.as_tensor(indices[:half], device=fit.device)] = True
    selection[torch.as_tensor(indices[half:], device=selection.device)] = True
    if not torch.equal(fit | selection, original_val_mask) or (fit & selection).any():
        raise RuntimeError('Gate split does not partition original validation nodes')
    return fit, selection


def choose_initial_member(labels: torch.Tensor, gate_fit_mask: torch.Tensor):
    if not SOURCE_CACHE.exists():
        raise FileNotFoundError('All-node member-logit cache from moe_gate.py export is required')
    with np.load(SOURCE_CACHE) as cache:
        cached_labels = torch.from_numpy(cache['labels']).to(labels.device)
        cached_val_mask = torch.from_numpy(cache['val_mask']).to(gate_fit_mask.device)
        member_logits = torch.from_numpy(cache['member_logits']).to(labels.device)
    if (not torch.equal(cached_labels[gate_fit_mask], labels[gate_fit_mask])
            or (gate_fit_mask & ~cached_val_mask).any()):
        raise RuntimeError('Cached gate-fit labels/validation mask differ from loaded graph')
    scores = [metrics(member_logits[:, m], labels, gate_fit_mask)
              for m in range(member_logits.shape[1])]
    selected = max(range(len(scores)),
                   key=lambda m: (scores[m]['accuracy'], -scores[m]['cross_entropy']))
    return selected, scores


def fit_gate(model: ConditionalProjector, data, labels: torch.Tensor,
             gate_fit_mask: torch.Tensor, gate_selection_mask: torch.Tensor,
             max_epochs: int, validation_every: int,
             patience_checks: int):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.005, weight_decay=0)
    best_key = (-float('inf'), -float('inf'))
    best_state = None
    best_epoch = 0
    stagnant_checks = 0
    start = time.monotonic()
    for epoch in range(max_epochs + 1):
        if epoch % validation_every == 0 or epoch == max_epochs:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(data, data.x)
            val = metrics(val_logits, labels, gate_selection_mask)
            key = (val['accuracy'], -val['cross_entropy'])
            if key > best_key:
                best_key = key
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                    if key == 'alpha' or key.startswith('gate.')
                }
                best_epoch = epoch
                stagnant_checks = 0
            else:
                stagnant_checks += 1
            if epoch % 25 == 0:
                print(f'{model.mode} epoch={epoch} val_acc={val["accuracy"]:.6f} '
                      f'best_epoch={best_epoch}', flush=True)
            if stagnant_checks >= patience_checks:
                break
        if epoch == max_epochs:
            break
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data, data.x)
        loss = F.cross_entropy(logits[gate_fit_mask], labels[gate_fit_mask])
        loss.backward()
        optimizer.step()
    assert best_state is not None
    current = model.state_dict()
    current.update({key: value.to(data.x.device) for key, value in best_state.items()})
    model.load_state_dict(current)
    model.eval()
    with torch.no_grad():
        all_logits, all_weights = model(data, data.x)
    return {
        'mode': model.mode,
        'trainable_parameters': sum(p.numel() for p in params),
        'frozen_parameters': sum(p.numel() for p in model.base.parameters()),
        'optimizer': 'Adam', 'learning_rate': 0.005, 'weight_decay': 0,
        'max_epochs': max_epochs, 'validation_every': validation_every,
        'patience_checks': patience_checks,
        'best_epoch': best_epoch, 'last_epoch': epoch,
        'fit_seconds': time.monotonic() - start,
        'gate_fit': metrics(all_logits, labels, gate_fit_mask),
        'gate_validation': metrics(all_logits, labels, gate_selection_mask),
        'mean_member_weight_gate_validation': all_weights[gate_selection_mask].mean(dim=0).tolist(),
        'std_member_weight_gate_validation': all_weights[gate_selection_mask].std(dim=0).tolist(),
    }, best_state, all_logits, all_weights


def benchmark_forward(fn, device: torch.device, warmup=5, reps=20) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.monotonic()
        for _ in range(reps):
            fn()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    return 1000 * (time.monotonic() - start) / reps


def run_train(device: torch.device, max_epochs: int):
    from datasets import load_dataset

    torch.set_num_threads(2)
    data, train_masks, val_masks, test_masks, _, _, is_binary = load_dataset(
        'roman-empire', add_self_loops=True, device=device,
        data_dir=str(ROOT / 'data'),
    )
    if is_binary:
        raise RuntimeError('Roman Empire must be multiclass')
    labels = data.y.long()
    train_mask = train_masks[:, 0].to(device)
    val_mask = val_masks[:, 0].to(device)
    test_mask = test_masks[:, 0].to(device)
    if (train_mask & val_mask).any() or (train_mask & test_mask).any() or (val_mask & test_mask).any():
        raise RuntimeError('Official masks overlap')
    base, row, sha256 = load_frozen_expert(device, data)
    source_meta = json.loads(SOURCE_META.read_text())
    if source_meta['checkpoint_sha256'] != sha256:
        raise RuntimeError('Cached logits are from a different expert checkpoint')
    gate_fit_mask, gate_selection_mask = gate_split(val_mask)
    selected_member, member_fit = choose_initial_member(labels, gate_fit_mask)
    print(f'CONFIG frozen={row["checkpoint"]} selected_member={selected_member} '
          f'device={device} max_epochs={max_epochs}', flush=True)
    models = {
        mode: ConditionalProjector(base, data.x[gate_fit_mask], mode,
                                   selected_member).to(device)
        for mode in ('static', 'node')
    }
    results = {}
    selected_states = {}
    final_logits = {}
    final_weights = {}
    for mode in ('static', 'node'):
        result, state, logits, weights = fit_gate(
            models[mode], data, labels, gate_fit_mask, gate_selection_mask,
            max_epochs=max_epochs, validation_every=5, patience_checks=20,
        )
        results[mode] = result
        selected_states[mode] = state
        final_logits[mode] = logits.detach()
        final_weights[mode] = weights.detach()
    # The two gate models have now been selected. Only now inspect test labels.
    for mode in ('static', 'node'):
        results[mode]['test'] = metrics(final_logits[mode], labels, test_mask)
        results[mode]['mean_member_weight_test'] = final_weights[mode][test_mask].mean(dim=0).tolist()
        results[mode]['std_member_weight_test'] = final_weights[mode][test_mask].std(dim=0).tolist()
    cached = np.load(SOURCE_CACHE)
    all_member_logits = torch.tensor(cached['member_logits'], device=device)
    ensemble_logits = all_member_logits.mean(dim=1)
    selected_member_logits = all_member_logits[:, selected_member]
    uniform = {
        'source': 'four frozen GNNM forward paths, uniform raw logit mean',
        'gate_validation': metrics(ensemble_logits, labels, gate_selection_mask),
        'test': metrics(ensemble_logits, labels, test_mask),
    }
    pure_member = {
        'source': 'one frozen member path, chosen by gate-fit accuracy',
        'member_index': selected_member,
        'gate_validation': metrics(selected_member_logits, labels, gate_selection_mask),
        'test': metrics(selected_member_logits, labels, test_mask),
    }
    cached.close()
    base_row = source_row('base')
    base_control = None if base_row is None else {
        'validation_accuracy': float(base_row['val_metric']),
        'test_accuracy': float(base_row['test_metric']),
        'note': 'matched 5-layer split-0 BASE control from projector_controls.csv',
    }
    timing = None
    if device.type == 'cuda':
        timing = {
            'static_single_pass_ms': benchmark_forward(
                lambda: models['static'](data, data.x), device),
            'node_single_pass_ms': benchmark_forward(
                lambda: models['node'](data, data.x), device),
            'four_path_gnnm_ms': benchmark_forward(
                lambda: [base(data, data.x, tabm_seed=m)
                         for m in range(base.tabm_inits)], device),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, state in selected_states.items():
        torch.save({'gate_state': state, 'source_checkpoint_sha256': sha256},
                   OUT / f'roman_empire_sage_split0_{mode}_gate.pt')
    report = {
        'protocol': {
            'dataset': 'roman-empire', 'model': 'SAGE', 'official_split': 0,
            'source_checkpoint': row['checkpoint'],
            'source_checkpoint_sha256': sha256,
            'source_best_step': int(row['best_step']),
            'frozen_member_count': int(row['m']),
            'selected_initial_member_by_gate_fit_labels': selected_member,
            'member_gate_fit_metrics': member_fit,
            'original_train_nodes': int(train_mask.sum()),
            'original_validation_nodes': int(val_mask.sum()),
            'gate_partition_seed': 2027,
            'gate_fit_nodes': int(gate_fit_mask.sum()),
            'gate_validation_nodes': int(gate_selection_mask.sum()),
            'test_nodes': int(test_mask.sum()),
            'gate_input': 'node features standardized using gate-fit-node mean/std and clipped to [-5,5]',
            'projector_mix': 'convex mix of four complete BE input maps per node, one frozen backbone pass, convex mix of four complete BE output maps using the same node weights',
            'backbone_and_be_parameters': 'frozen',
            'initial_gate': 'selected member alpha=4, other alphas=0, node linear coefficients=0',
            'gate_training': 'cross-entropy on gate-fit labels, a held-out half of original validation nodes',
            'validation_selection': 'highest accuracy then lower cross-entropy on the other half of original validation nodes',
            'test_rule': 'test labels evaluated only after both gate checkpoints fixed',
            'validation_reuse_caveat': 'The frozen source checkpoint was already selected using the entire original validation split.',
            'supervision_comparison_caveat': 'Conditional gates use half of original validation labels for training; BASE and original GNNM do not. Static-vs-node is supervision-matched, but comparisons to BASE/uniform are descriptive.',
            'claim_scope': 'single-split exploratory comparison; no novelty or generalization claim',
        },
        'four_path_uniform': uniform,
        'selected_pure_member': pure_member,
        'base_control': base_control,
        'static_single_pass': results['static'],
        'node_single_pass': results['node'],
        'timing': timing,
    }
    report_path = OUT / 'roman_empire_sage_split0.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def selftest():
    from ablation.models_ablation import TABMAblationModel

    torch.manual_seed(17)
    torch.set_num_threads(2)
    device = torch.device('cpu')
    base = TABMAblationModel(
        model_name='SAGE', num_layers=1, input_dim=3,
        hidden_dim=8, output_dim=3, hidden_dim_multiplier=1,
        num_heads=8, normalization='LayerNorm', dropout=0,
        tabm_inits=4, init_scheme='default', device=device,
    ).eval()
    base.requires_grad_(False)
    graph = type('Graph', (), {})()
    graph.edge_index = torch.tensor([[0, 1, 2, 0, 2], [1, 0, 1, 2, 0]])
    x = torch.randn(3, 3)
    model = ConditionalProjector(base, x, 'node', 0).eval()
    for member in range(4):
        weights = F.one_hot(torch.full((3,), member), 4).float()
        expected = base(graph, x, tabm_seed=member)
        actual = model.forward_with_weights(graph, x, weights)
        if not torch.allclose(actual, expected, rtol=1e-6, atol=1e-6):
            raise AssertionError(f'One-hot member {member} mismatch: '
                                 f'{(actual - expected).abs().max().item()}')
    trainable = [p for p in model.parameters() if p.requires_grad]
    output, _ = model(graph, x)
    F.cross_entropy(output, torch.tensor([0, 1, 2])).backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable):
        raise AssertionError('Gate gradient missing/nonfinite')
    if any(p.grad is not None for p in base.parameters()):
        raise AssertionError('Frozen expert received a gradient')
    # A neighbor's gate can alter a target node through the one shared graph pass.
    weights = torch.randn(3, 4).softmax(dim=1).detach().requires_grad_(True)
    conditional_logits = model.forward_with_weights(graph, x, weights)
    conditional_gradient = torch.autograd.grad(
        conditional_logits[1, 0], weights,
    )[0]
    if conditional_gradient[0].abs().max().item() <= 1e-8:
        raise AssertionError('Neighbor gate did not affect target node')
    with torch.no_grad():
        fixed_member_logits = torch.stack(
            [base(graph, x, tabm_seed=m) for m in range(4)], dim=1,
        )
    posthoc_logits = (fixed_member_logits * weights.unsqueeze(-1)).sum(dim=1)
    posthoc_gradient = torch.autograd.grad(posthoc_logits[1, 0], weights)[0]
    if posthoc_gradient[0].abs().max().item() != 0:
        raise AssertionError('Posthoc target logit depends on neighbor gate')
    print('SELFTEST PASS: one-hot equivalence, finite gate gradients, '
          'and neighbor-gate effect unique to pre-propagation mixing')


def dryrun():
    from datasets import load_dataset

    torch.set_num_threads(2)
    device = torch.device('cpu')
    data, train_masks, val_masks, _, _, _, is_binary = load_dataset(
        'roman-empire', add_self_loops=True, device=device,
        data_dir=str(ROOT / 'data'),
    )
    if is_binary:
        raise RuntimeError('Roman Empire must be multiclass')
    train_mask, val_mask = train_masks[:, 0], val_masks[:, 0]
    gate_fit_mask, gate_selection_mask = gate_split(val_mask)
    base, _, sha256 = load_frozen_expert(device, data)
    source_meta = json.loads(SOURCE_META.read_text())
    if source_meta['checkpoint_sha256'] != sha256:
        raise RuntimeError('Cached logits are from a different expert checkpoint')
    selected, member_fit = choose_initial_member(data.y.long(), gate_fit_mask)
    model = ConditionalProjector(base, data.x[gate_fit_mask], 'node', selected).eval()
    with torch.no_grad():
        logits, weights = model(data, data.x)
    print(json.dumps({
        'selected_initial_member': selected,
        'member_gate_fit_accuracy': [x['accuracy'] for x in member_fit],
        'single_pass_shape': list(logits.shape),
        'initial_gate_validation': metrics(logits, data.y.long(), gate_selection_mask),
        'initial_member_weights': weights[0].tolist(),
        'frozen_checkpoint_sha256': sha256,
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['selftest', 'dryrun', 'train'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-epochs', type=int, default=300)
    args = parser.parse_args()
    if args.stage == 'selftest':
        selftest()
    elif args.stage == 'dryrun':
        dryrun()
    else:
        run_train(torch.device(args.device), args.max_epochs)


if __name__ == '__main__':
    main()
