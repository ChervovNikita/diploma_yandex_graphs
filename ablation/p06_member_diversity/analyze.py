"""Step 2 of the diversity study: compare the members of GNNM and ENS on the test nodes.

    python -m ablation.p06_member_diversity.analyze --root RUN_DIR --out RESULT_DIR

Scores are fractions, not percentage points. Output files:

    runs.csv                       prediction and representation metrics of each model
    representation_similarity.csv  linear CKA between members after each stage of the network
    parameters.csv                 similarity of the member factors of GNNM
    comparison.csv                 GNNM next to ENS for each dataset and backbone
    summary.json                   means over cells and the link between diversity and accuracy
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from datasets import compute_metrics, load_dataset

from ablation.common.study import HIDDEN_DIM
from ablation.p05_hparam_grid import grid
from ablation.p05_hparam_grid.collect import load_rows
from ablation.p05_hparam_grid.models import PlainModel

from .models import build_model
from .select_depths import SELECTED_PATH


COMPARED = ['ensemble_metric', 'mean_member_metric', 'ensemble_gain', 'ensemble_acc',
            'mean_member_acc', 'pair_disagreement', 'error_jaccard', 'double_fault',
            'error_correlation', 'at_least_one_correct', 'probability_correlation',
            'js_divergence', 'cka_input', 'cka_final']
EXPLAINERS = ['pair_disagreement', 'error_correlation', 'probability_correlation',
              'js_divergence', 'cka_final']
RECHECK_TOLERANCE = 1e-4


@torch.no_grad()
def capture_member(model, data, member, mask):
    """Run one member and return its test logits and the test activations of each stage."""
    inner = model.model if isinstance(model, PlainModel) else model
    stages = {}

    def recorder(name):
        def hook(_module, _inputs, output):
            stages[name] = output[mask].detach()
        return hook

    named = [('input', inner.act)]
    named += [(f'block_{i + 1}', block) for i, block in enumerate(inner.residual_modules)]
    named.append(('final', inner.output_normalization))
    handles = [module.register_forward_hook(recorder(name)) for name, module in named]
    try:
        model.eval()
        logits = model(data, data.x, tabm_seed=member)
    finally:
        for handle in handles:
            handle.remove()
    return logits[mask].detach(), stages


def linear_cka(first, second):
    """Linear CKA in feature space; invariant to rotation and isotropic scaling."""
    first = (first - first.mean(dim=0)).double()
    second = (second - second.mean(dim=0)).double()
    numerator = torch.linalg.norm(second.T @ first) ** 2
    denominator = torch.linalg.norm(first.T @ first) * torch.linalg.norm(second.T @ second)
    return (numerator / denominator).item() if denominator > 0 else float('nan')


def class_probabilities(logits, is_binary):
    """Probabilities with shape (members, nodes, classes)."""
    if is_binary:
        positive = torch.sigmoid(logits.squeeze(-1) if logits.dim() == 3 else logits)
        return torch.stack([1 - positive, positive], dim=-1)
    return F.softmax(logits, dim=-1)


def pearson(first, second):
    first, second = first.double() - first.double().mean(), second.double() - second.double().mean()
    denominator = first.norm() * second.norm()
    return (first @ second / denominator).item() if denominator > 0 else float('nan')


def jensen_shannon(first, second):
    middle = (first + second) / 2
    def divergence(p):
        return (p * (p.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(dim=-1)
    return ((divergence(first) + divergence(second)) / 2).mean().item()


def mean_defined(values):
    defined = [v for v in values if v == v]
    return sum(defined) / len(defined) if defined else float('nan')


def prediction_metrics(logits, y, is_binary, dataset):
    """Ensemble quality and member diversity from logits of shape (members, nodes, ...)."""
    everyone = torch.ones(y.shape[0], dtype=torch.bool, device=y.device)
    score = lambda output: compute_metrics(output, y, everyone, is_binary, dataset_name=dataset)
    members = logits.shape[0]
    ensemble = score(logits.mean(dim=0))
    member_scores = [score(logits[i]) for i in range(members)]
    probabilities = class_probabilities(logits, is_binary)
    predictions = probabilities.argmax(dim=-1)
    target = y.long()
    errors = predictions.ne(target)
    # With two classes the second probability column carries all the information.
    compared = probabilities[..., 1] if is_binary else probabilities.flatten(start_dim=1)

    pairs = list(combinations(range(members), 2))
    jaccard, double_fault, error_correlation, correlation, divergence, disagreement = (
        [], [], [], [], [], [])
    for i, j in pairs:
        union = (errors[i] | errors[j]).sum().item()
        both = (errors[i] & errors[j]).sum().item()
        jaccard.append(both / union if union else 1.0)
        double_fault.append(both / target.shape[0])
        error_correlation.append(pearson(errors[i].float(), errors[j].float()))
        correlation.append(pearson(compared[i], compared[j]))
        divergence.append(jensen_shannon(probabilities[i], probabilities[j]))
        disagreement.append(predictions[i].ne(predictions[j]).float().mean().item())

    mean_member_metric = sum(s['metric'] for s in member_scores) / members
    return {
        'ensemble_metric': ensemble['metric'],
        'mean_member_metric': mean_member_metric,
        'best_member_metric': max(s['metric'] for s in member_scores),
        'ensemble_gain': ensemble['metric'] - mean_member_metric,
        'ensemble_acc': ensemble['acc'],
        'mean_member_acc': sum(s['acc'] for s in member_scores) / members,
        'pair_disagreement': mean_defined(disagreement),
        'error_jaccard': mean_defined(jaccard),
        'double_fault': mean_defined(double_fault),
        'error_correlation': mean_defined(error_correlation),
        'at_least_one_correct': (~errors).any(dim=0).float().mean().item(),
        'probability_correlation': mean_defined(correlation),
        'js_divergence': mean_defined(divergence),
        'member_metrics': [s['metric'] for s in member_scores],
    }


def stage_similarity(captured):
    """Mean pairwise linear CKA for each stage, in network order."""
    rows = []
    for index, name in enumerate(captured[0]):
        values = [linear_cka(captured[i][name], captured[j][name])
                  for i, j in combinations(range(len(captured)), 2)]
        rows.append({'stage_index': index, 'stage': name, 'cka': mean_defined(values)})
    return rows


def factor_similarity(matrix):
    """Mean pairwise cosine and relative spread across the first dimension of ``matrix``."""
    flat = matrix.flatten(start_dim=1).double()
    cosines = [F.cosine_similarity(flat[i], flat[j], dim=0).item()
               for i, j in combinations(range(flat.shape[0]), 2)]
    rms = flat.pow(2).mean().sqrt().item()
    spread = flat.std(dim=0).mean().item() / rms if rms > 0 else float('nan')
    return {'mean_pair_cosine': mean_defined(cosines), 'relative_spread': spread}


def parameter_rows(state):
    """Similarity of R, S, B and of the full member matrices of both GNNM projectors."""
    rows = []
    for block, prefix in (('input', 'input_be_block'), ('output', 'output_be_block')):
        factors = {name: state[f'{prefix}.{name}'] for name in 'RSB'}
        weight = state[f'{prefix}.W.weight']
        members = factors['S'].unsqueeze(-1) * weight.unsqueeze(0) * factors['R'].unsqueeze(1)
        for name, matrix in {**factors, 'effective': members}.items():
            rows.append({'block': block, 'factor': name, **factor_similarity(matrix)})
    return rows


def split_mask(test_masks, split):
    return test_masks[:, split]


def load_group(group, method, bundle, device):
    """Models and member indices of one GNNM model or one ENS of independent models."""
    data, _, _, _, _, targets, _ = bundle
    first = group.iloc[0]
    if method == 'gnnm':
        job_dirs = [first.job_dir]
    else:
        job_dirs = group.sort_values('method').job_dir.tolist()
    models = []
    for job_dir, config in zip(job_dirs, group.sort_values('method').itertuples()):
        name = 'gnnm' if method == 'gnnm' else config.method
        model = build_model(name, first.model, int(first.num_layers), data.x.size(1),
                            HIDDEN_DIM, targets, int(config.members), device)
        state = torch.load(Path(job_dir) / 'model.pt', map_location=device, weights_only=True)
        model.load_state_dict(state)
        models.append(model)
    if method == 'gnnm':
        return [(models[0], i) for i in range(int(first.members))]
    return [(model, 0) for model in models]


def analyse_group(group, method, bundle, device):
    data, _, _, test_masks, _, _, is_binary = bundle
    first = group.iloc[0]
    mask = split_mask(test_masks, int(first.split))
    members = load_group(group, method, bundle, device)
    outputs = [capture_member(model, data, index, mask) for model, index in members]
    logits = torch.stack([output[0] for output in outputs])
    y = data.y[mask]
    metrics = prediction_metrics(logits, y, is_binary, first.dataset)
    similarity = stage_similarity([output[1] for output in outputs])

    identity = {'dataset': first.dataset, 'model': first.model, 'method': method,
                'num_layers': int(first.num_layers), 'split': int(first.split)}
    member_metrics = metrics.pop('member_metrics')
    if method == 'gnnm':
        recheck = abs(metrics['ensemble_metric'] - float(first.test_metric))
        state = members[0][0].state_dict()
        parameters = [{**identity, **row} for row in parameter_rows(
            {key: value.detach().cpu() for key, value in state.items()})]
    else:
        reported = group.sort_values('method').test_metric.astype(float).tolist()
        recheck = max(abs(a - b) for a, b in zip(member_metrics, reported))
        parameters = []

    run = {**identity, **metrics,
           'cka_input': similarity[0]['cka'], 'cka_final': similarity[-1]['cka'],
           'best_step': int(group.best_step.max()), 'recheck_abs_delta': recheck}
    return run, [{**identity, **row} for row in similarity], parameters


def group_runs(runs):
    """Yield complete (method, group) pairs and report the incomplete ones."""
    runs = runs.assign(group_method=runs.method.map(
        lambda method: 'ens' if method.startswith('ens_m') else method))
    keys = ['dataset', 'model', 'group_method', 'num_layers', 'split']
    complete, incomplete = [], []
    for key, group in runs.groupby(keys):
        expected = grid.ENSEMBLE_MEMBERS if key[2] == 'ens' else 1
        if group.method.nunique() == expected:
            complete.append((key[2], group))
        else:
            incomplete.append(key)
    return complete, incomplete


def compare(frame):
    """One row per dataset and backbone, keeping each method at its own winning depth."""
    keep = ['dataset', 'model', 'num_layers', *COMPARED]
    gnnm = frame[frame.method == 'gnnm'][keep]
    ens = frame[frame.method == 'ens'][keep]
    merged = gnnm.merge(ens, on=['dataset', 'model'], suffixes=('_gnnm', '_ens'))
    for metric in COMPARED:
        merged[f'{metric}_gap'] = merged[f'{metric}_gnnm'] - merged[f'{metric}_ens']
    return merged


def summarise(merged):
    means = {
        metric: {
            'gnnm': float(merged[f'{metric}_gnnm'].mean()),
            'ens': float(merged[f'{metric}_ens'].mean()),
            'mean_gap': float(merged[f'{metric}_gap'].mean()),
            'cells_gnnm_higher': int((merged[f'{metric}_gap'] > 0).sum()),
        }
        for metric in COMPARED
    }
    return {
        'cells': int(len(merged)),
        'metrics': means,
        'spearman_with_accuracy_gap': {
            metric: float(merged['ensemble_metric_gap'].corr(merged[f'{metric}_gap'], method='spearman'))
            for metric in EXPLAINERS
        },
    }


def analyse_runs(runs, load, device):
    """Return the run, representation and parameter tables; ``load`` gives a dataset bundle."""
    complete, incomplete = group_runs(runs)
    for key in incomplete:
        print(f'[skip] incomplete: {key}', flush=True)
    run_rows, stage_rows, parameter_table = [], [], []
    for method, group in complete:
        run, stages, parameters = analyse_group(group, method, load(group.iloc[0].dataset), device)
        run_rows.append(run)
        stage_rows += stages
        parameter_table += parameters
        if run['recheck_abs_delta'] > RECHECK_TOLERANCE:
            print(f"[warning] saved model of {run['dataset']}/{run['model']}/{method} differs from "
                  f"the trained result by {run['recheck_abs_delta']:.6f}", flush=True)
    return pd.DataFrame(run_rows), pd.DataFrame(stage_rows), pd.DataFrame(parameter_table)


def add_archive_reference(frame, selected):
    reference = selected[['dataset', 'model', 'method', 'num_layers', 'archive_test_metric']]
    merged = frame.merge(reference, on=['dataset', 'model', 'method', 'num_layers'], how='left')
    merged['archive_delta'] = merged.ensemble_metric - merged.archive_test_metric
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    cache = {}

    def load(dataset):
        if dataset not in cache:
            cache.clear()
            cache[dataset] = load_dataset(dataset, add_self_loops=True, device=device,
                                          data_dir=args.data_dir)
        return cache[dataset]

    frame, stages, parameters = analyse_runs(load_rows(args.root), load, device)
    frame = add_archive_reference(frame, pd.read_csv(SELECTED_PATH))
    merged = compare(frame)
    summary = summarise(merged)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / 'runs.csv', index=False)
    stages.to_csv(args.out / 'representation_similarity.csv', index=False)
    parameters.to_csv(args.out / 'parameters.csv', index=False)
    merged.to_csv(args.out / 'comparison.csv', index=False)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
