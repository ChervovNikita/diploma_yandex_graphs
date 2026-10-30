"""CPU-only summaries of one exact already-scored VALID export; no model import."""
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
root = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
path = root / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/campaign_execution_20261016_v7/runs/server51_gpu0/full/42727be85df43bc7e43cd1abfc3c5faf849829d58f0933b0fc03576d4e562408/attempts/0001/output/selected_VALID.pt'
expected = 'f78419bf7a78fb2174b698cd8a4ce3294d60525119f234434a441ae305167d0c'
assert path.resolve().is_relative_to(root) and path.stat().st_size == 194284
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
value = torch.load(path, map_location='cpu', weights_only=True)
assert value['schema'] == 'own-selected-VALID-logits-v1'
assert value['role'] == 'VALID' and value['TEST_accessed'] is False
assert value['selected_epoch'] == 152 and value['procedure'] == 'aggregation_global_globalJ_v1'
labels = value['labels'].flatten().double()
logits = value['raw_logits'].flatten().double()
assert len(labels) == len(logits) == 4113 and torch.isfinite(logits).all()
assert set(labels.tolist()) == {0., 1.}
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
probability = logits.sigmoid()
q = torch.tensor([0., .01, .1, .5, .9, .99, 1.], dtype=torch.float64)
def summarize(z, y):
    l = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction='none')
    return {'records': len(z), 'logit_quantiles': torch.quantile(z, q).tolist(),
            'mean_BCE': float(l.mean()), 'mean_logit': float(z.mean()),
            'positive_prediction_fraction': float((z > 0).double().mean()),
            'BCE_sum': float(l.sum())}
largest = torch.topk(loss, 10)
out = {'selected_export': {'path': str(path), 'bytes': path.stat().st_size, 'sha256': expected},
       'schema': 'one-saved-VALID-export-CPU-diagnostic-v1', 'selected_epoch': 152,
       'quantile_levels': q.tolist(), 'overall': summarize(logits, labels),
       'classwise': {str(c): summarize(logits[labels == c], labels[labels == c]) for c in (0, 1)},
       'Brier': float(((probability-labels)**2).mean()),
       'top10_BCE_fraction': float(largest.values.sum()/loss.sum()),
       'top10': [{'record_index': int(i), 'graph_id': int(value['ids'].flatten()[i]),
                  'label': int(labels[i]), 'logit': float(logits[i]), 'BCE': float(loss[i])}
                 for i in largest.indices],
       'same_bank_mean_available': value.get('same_bank_mean_VALID') is not None,
       'export_keys': sorted(value.keys()), 'model_forwards': 0, 'optimizer_steps': 0,
       'TEST_accessed': False, 'artifact_mutated': False}
mean = value.get('same_bank_mean_VALID')
if mean is not None:
    out['same_bank_mean'] = summarize(mean['raw_logits'].flatten().double(), labels)
print(json.dumps(out, sort_keys=True, allow_nan=False))
