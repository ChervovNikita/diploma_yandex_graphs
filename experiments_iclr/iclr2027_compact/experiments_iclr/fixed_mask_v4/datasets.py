import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops, coalesce, to_undirected
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


DATASET_CHOICES = ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']
DATASET_VAL_METRIC = {
    'roman-empire': 'acc',
    'amazon-ratings': 'acc',
    'minesweeper': 'roc_auc',
    'tolokers': 'roc_auc',
    'questions': 'roc_auc',
}


def load_dataset(name, add_self_loops=False, device='cpu', data_dir='data'):
    npz_path = os.path.join(data_dir, f'{name.replace("-", "_")}.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'Dataset not found: {npz_path}')

    data = np.load(npz_path)
    node_features = torch.tensor(data['node_features'], dtype=torch.float32)
    labels = torch.tensor(data['node_labels'], dtype=torch.int64)
    edges = torch.tensor(data['edges'])

    edge_index = edges.t().contiguous()
    if 'directed' not in name:
        edge_index = to_undirected(
            coalesce(edge_index, num_nodes=len(node_features)),
            num_nodes=len(node_features),
        )
    if add_self_loops:
        edge_index, _ = add_remaining_self_loops(
            edge_index, num_nodes=len(node_features),
        )

    num_classes = int(labels.max().item()) + 1
    num_targets = 1 if num_classes == 2 else num_classes
    is_binary = num_targets == 1
    if is_binary:
        labels = labels.float()

    train_masks = torch.tensor(data['train_masks'])
    val_masks = torch.tensor(data['val_masks'])
    test_masks = torch.tensor(data['test_masks'])
    if train_masks.dim() == 2 and train_masks.shape[0] < train_masks.shape[1]:
        train_masks = train_masks.t()
        val_masks = val_masks.t()
        test_masks = test_masks.t()

    pyg_data = Data(edge_index=edge_index, num_nodes=len(node_features))
    pyg_data.x = node_features
    pyg_data.y = labels
    pyg_data = pyg_data.to(device)

    return pyg_data, train_masks, val_masks, test_masks, num_classes, num_targets, is_binary


def get_validation_metric_name(dataset_name, is_binary):
    metric_name = DATASET_VAL_METRIC.get(dataset_name, 'roc_auc' if is_binary else 'acc')
    if metric_name == 'roc_auc' and not is_binary:
        return 'acc'
    return metric_name


def compute_metrics(logits, y, mask, is_binary, dataset_name=None):
    preds = logits[mask].argmax(dim=1) if not is_binary else (logits[mask].squeeze(-1) > 0).long()
    y_masked = y[mask]

    if is_binary:
        acc = (preds == y_masked.long()).float().mean().item()
        try:
            roc_auc = roc_auc_score(y_masked.cpu().numpy(), logits[mask].squeeze(-1).cpu().numpy())
        except ValueError:
            roc_auc = 0.0
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[mask].squeeze(-1), y_masked
        ).item()
        metrics = {
            'acc': acc,
            'roc_auc': roc_auc,
            'loss': loss,
        }
        metric_name = get_validation_metric_name(dataset_name, is_binary)
        metrics['metric_name'] = metric_name
        metrics['metric'] = metrics[metric_name]
        return metrics
    else:
        y_true = y_masked.cpu().numpy()
        y_pred = preds.cpu().numpy()
        acc = (preds == y_masked).float().mean().item()
        loss = torch.nn.functional.cross_entropy(logits[mask], y_masked.long()).item()
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics = {
            'acc': acc,
            'loss': loss,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'prec_macro': prec_macro,
            'rec_macro': rec_macro,
        }
        metric_name = get_validation_metric_name(dataset_name, is_binary)
        metrics['metric_name'] = metric_name
        metrics['metric'] = metrics[metric_name]
        return metrics
