"""Frozen Collab recipe for the GCN and GraphSAGE backbones.

Source: the official OGB example ``examples/linkproppred/collab/gnn.py``
(snap-stanford/ogb). Its defaults are kept: three message-passing layers of
width 256, a three-layer Hadamard MLP predictor, no dropout, Adam with learning
rate 0.001, batches of 65536 positive edges with as many uniformly random
negative edges, gradient-norm clipping at 1.0, validation after every epoch and
400 epochs. The checkpoint with the best validation Hits@50 is kept.

Documented differences from the example, identical for every method:
  * validation edges enter the message-passing graph at TEST time only, as in
    the BUDDY runs (``use_valedges_as_input``);
  * the graph is stored as an edge list; GCN uses the symmetric normalisation of
    the collaboration-count weights with self-loops, and is not cached, so the
    TEST graph really includes validation edges;
  * the step limit can extend once: if the best validation epoch is within the
    last 50 epochs of 400, training continues to 800 epochs.
"""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
BACKBONES = ('gcn', 'sage')
METHODS = ('native', 'gnnm', 'gnnm_contrast')
ROUTES = 4

RECIPE = {
    'dataset_name': 'ogbl-collab', 'reference': 'ogb/examples/linkproppred/collab/gnn.py',
    'num_layers': 3, 'hidden_channels': 256, 'predictor_layers': 3, 'dropout': 0.0,
    'lr': 0.001, 'weight_decay': 0.0, 'batch_size': 64 * 1024, 'grad_clip': 1.0,
    'eval_steps': 1, 'use_valedges_as_input': True, 'selection_metric': 'hits@50',
    'cap_schedule': [400, 800], 'extend_if_best_within': 50,
    'prediction_average': 'raw_logits', 'route_count': ROUTES,
    'contrast': {'coefficient': 0.05, 'temperature': 0.2, 'anchors': 1024,
                 'alternatives': 16, 'private': 'all route factors'},
}
HITS = (20, 50, 100)


def dependency_versions():
    return {name: version(name) for name in ('torch', 'torch-geometric', 'ogb', 'numpy')}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    here = Path(__file__).parent
    sources = [here / name for name in ('protocol.py', 'data.py', 'models.py', 'run.py')]
    sources += [REPOSITORY / name for name in
                ('models.py', 'ablation/common/worker.py', 'ablation/common/resumable.py',
                 'ablation/common/contrastive.py')]
    hashes = {str(path.relative_to(REPOSITORY)): file_hash(path) for path in sources}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def inside_repository(path):
    path = Path(path).resolve()
    if not path.is_relative_to(REPOSITORY):
        raise ValueError(f'Paths must remain inside {REPOSITORY}: {path}')
    return path
