import random
import numpy as np
import torch


ALL_MODELS = ['ResNet', 'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep', 'TAG']
NUM_LAYERS_RANGE = range(1, 6)
NUM_SPLITS = 10
NUM_STEPS = 5000
PATIENCE = 3

COMMON_RESULT_FIELDNAMES = [
    'train_metric_name', 'train_metric', 'train_acc', 'train_loss', 'train_roc_auc',
    'train_f1_macro', 'train_f1_weighted', 'train_prec_macro', 'train_rec_macro',
    'val_metric_name', 'val_metric', 'val_acc', 'val_loss', 'val_roc_auc',
    'val_f1_macro', 'val_f1_weighted', 'val_prec_macro', 'val_rec_macro',
    'test_metric_name', 'test_metric', 'test_acc', 'test_loss', 'test_roc_auc',
    'test_f1_macro', 'test_f1_weighted', 'test_prec_macro', 'test_rec_macro',
]


def combo_slug(hidden_dim, lr):
    """Stable, filename-safe token for (hidden_dim, lr) in checkpoint paths."""
    hd = int(hidden_dim)
    x = float(lr)
    lr_s = f'{x:.10g}'.replace('.', 'p').replace('-', 'm').replace('+', '')
    return f'h{hd}_lr{lr_s}'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics_to_str(prefix, metrics):
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f'{prefix}_{key}={value:.4f}')
        else:
            parts.append(f'{prefix}_{key}={value}')
    return ' '.join(parts)


def metrics_to_prefixed_dict(prefix, metrics):
    out = {}
    for key, value in metrics.items():
        out[f'{prefix}_{key}'] = value
    return out


def prefixed_metrics_to_str(prefix, result):
    parts = []
    for key in sorted(result.keys()):
        if not key.startswith(f'{prefix}_'):
            continue
        value = result[key]
        if isinstance(value, float):
            parts.append(f'{key}={value:.4f}')
        else:
            parts.append(f'{key}={value}')
    return ' '.join(parts)
