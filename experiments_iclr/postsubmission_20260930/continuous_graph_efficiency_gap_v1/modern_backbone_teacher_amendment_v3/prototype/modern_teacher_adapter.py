"""Prospective modern teachers. SOURCE ONLY: not imported or run by the author.

Exact attributed native bodies live beside this adapter. Four native trajectories
remain separate through every block, graph attention score and global reduction.
Primary pooling remains softmax(mean raw logits), including all selection.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import random
from types import SimpleNamespace
import torch
from torch import nn

FAMILIES = ('single_author', 'gnnm_boundary_4', 'independent_author_4_same_width')
SEEDS = (17, 29, 43)
CONFIGS = (
    dict(id='source', lr_multiplier=1.0, dropout_delta=0.0),
    dict(id='lr_half', lr_multiplier=0.5, dropout_delta=0.0),
    dict(id='lr_double', lr_multiplier=2.0, dropout_delta=0.0),
    dict(id='lower_dropout', lr_multiplier=1.0, dropout_delta=-0.2),
)
NATIVE = {
    'polyformer_mono': dict(variant='PolyFormer-Mono', nodes=2223, features=2089, classes=5,
        hidden=256, K=12, nlayer=2, n_head=4, d_ffn=128, q=1.4, multi=1.0,
        dropout=0.3, dprate=0.8, lr=0.0001, weight_decay=0.0,
        attention_lr=0.001, attention_weight_decay=1e-7,
        max_epochs=2000, patience=250, base='mono', dataset='squirrel_filtered',
        author_commit='d390f39e88d0eaac80318fdc7704bd3bf3cf8b13'),
    'polynormer_r': dict(variant='Polynormer-r', nodes=7650, features=745, classes=8,
        hidden_per_head=64, heads=8, local_layers=7, global_layers=2,
        in_dropout=0.2, dropout=0.7, global_dropout=0.7, beta=-1, pre_ln=False,
        qk_shared=True, lr=0.001, weight_decay=5e-5,
        local_epochs=200, global_epochs=1000,
        author_commit='fc8c276c9c5dfbd616d83f65338a3392188a5e08'),
}
PROTOCOL = dict(version='modern_native_teacher_v1', families=list(FAMILIES), seeds=list(SEEDS),
    configurations=list(CONFIGS), primary_pooling='softmax(mean raw member logits)',
    checkpoint_selection='mean raw-logit predictor-validation NLL, earliest exact tie',
    configuration_selection='arithmetic mean of selected NLL across all three paired seeds; listed config tie',
    training_loss='arithmetic mean member cross entropy', native=NATIVE,
    label_scope=['train', 'validation'], family_cells=72,
    photo_stage_selection='best local model and Adam state; global selector starts fresh; global-only final',
    photo_updates=1200, squirrel_patience=250,
    secondary_pooling='mean member softmax probabilities from the same selected saved logits; never selected',
    graph_roles='derived_roles_v2; new20percent_train protocol, not native paper split reproduction')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def specification(backbone, family, config, seed):
    require(backbone in NATIVE and family in FAMILIES and seed in SEEDS, 'Unregistered cell')
    require(config in range(len(CONFIGS)), 'Unregistered configuration')
    native = copy.deepcopy(NATIVE[backbone])
    delta = CONFIGS[config]['dropout_delta']
    native['dropout'] = max(0.0, native['dropout'] + delta)
    if backbone == 'polynormer_r':
        native['global_dropout'] = max(0.0, native['global_dropout'] + delta)
    return dict(backbone=backbone, family=family, config=config, seed=seed,
        members=1 if family == 'single_author' else 4, native=native,
        lr_multiplier=CONFIGS[config]['lr_multiplier'],
        initialization='paired native core seed; independent member m seed+100003*m; GNNM stem R Rademacher',
        primary_pooling=PROTOCOL['primary_pooling'], protocol_version=PROTOCOL['version'])


def fix_seed(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def native_model(spec, seed):
    # Scientific imports happen only in future authorized runtime construction.
    fix_seed(seed)
    n = spec['native']
    if spec['backbone'] == 'polyformer_mono':
        from native_polyformer_outer import PolyFormer
        args = SimpleNamespace(**n, num_features=n['features'], num_classes=n['classes'])
        return PolyFormer(None, args)
    from native_polynormer import Polynormer
    model = Polynormer(n['features'], n['hidden_per_head'], n['classes'],
        local_layers=n['local_layers'], global_layers=n['global_layers'],
        in_dropout=n['in_dropout'], dropout=n['dropout'], global_dropout=n['global_dropout'],
        heads=n['heads'], beta=n['beta'], pre_ln=n['pre_ln'])
    # Native main.py calls this before each fit; wrap boundaries only afterward.
    model.reset_parameters()
    return model


class TeacherFamily(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.specification = copy.deepcopy(spec)
        self.members = spec['members']
        self.backbone = spec['backbone']
        self.global_stage = False
        if spec['family'] == 'gnnm_boundary_4':
            from backbone_boundary_adapter import PolyFormerBoundaryFamily, PolynormerBoundaryFamily
            native = native_model(spec, spec['seed'])
            wrapper = PolyFormerBoundaryFamily if self.backbone == 'polyformer_mono' else PolynormerBoundaryFamily
            self.boundary = wrapper(native, members=4)
        else:
            self.models = nn.ModuleList([native_model(spec, spec['seed'] + 100003*m)
                for m in range(self.members)])
        # Independent initialization consumption cannot alter training RNG pairing.
        fix_seed(spec['seed'] + 70000)

    def set_global_stage(self, enabled):
        require(self.backbone == 'polynormer_r' or not enabled, 'PolyFormer has no global stage')
        self.global_stage = bool(enabled)
        if self.backbone == 'polynormer_r':
            if hasattr(self, 'boundary'):
                self.boundary.set_global_stage(enabled)
            else:
                for model in self.models:
                    model._global = bool(enabled)

    def forward(self, graph, x=None):
        require(graph.teacher_backbone == self.backbone, 'Wrong teacher preprocessing')
        data = graph.teacher_input
        if hasattr(self, 'boundary'):
            if self.backbone == 'polyformer_mono':
                return self.boundary(data)
            return self.boundary(data, graph.teacher_edge_index)
        if self.backbone == 'polyformer_mono':
            native_data = SimpleNamespace(list_mat=data.unbind(1))
            return torch.stack([model(native_data) for model in self.models], 0)
        return torch.stack([model(data, graph.teacher_edge_index) for model in self.models], 0)


def prepare_graph(x, canonical_edges, backbone, environment, input_binding):
    """No labels. No author filename-only cache is read or written.

    Native polynomial tokens are recomputed once per admitted cell/request. The
    content identity binds raw graph/features, preprocessing, source and runtime;
    cold serving charges all preprocessing and full token materialization.
    """
    require(backbone in NATIVE, 'Unknown modern preprocessing')
    n = NATIVE[backbone]
    require(tuple(x.shape) == (n['nodes'], n['features']) and x.dtype == torch.float32,
        'Only the full verified graph/feature dimensions are admitted')
    require(canonical_edges.dtype == torch.int64 and canonical_edges.ndim == 2 and canonical_edges.shape[0] == 2,
        'Expected canonical int64 edges')
    require(bool(((canonical_edges >= 0) & (canonical_edges < len(x))).all()) and
        bool((canonical_edges[0] < canonical_edges[1]).all()), 'Invalid canonical edge identities')
    require(canonical_edges.T.unique(dim=0).shape[0] == canonical_edges.shape[1], 'Duplicate canonical edges')
    directed = torch.cat((canonical_edges, canonical_edges.flip(0)), 1)
    from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
    if backbone == 'polyformer_mono':
        from native_polyformer_preprocess import mono_base
        # Keep native orientation/SciPy COO conversion and FP32 sparse behavior.
        teacher_edges = to_undirected(directed.cpu(), num_nodes=len(x))
        teacher_input = torch.stack(mono_base(12, x, teacher_edges, None), dim=1)
        recipe = 'raw features; to_undirected; native gcn_norm+SciPy COO FP32; X..Ahat^12X'
    else:
        from torch_geometric.data import Data
        from torch_geometric.transforms import NormalizeFeatures
        # Actual versioned PyG transform exactly once, after the verified provider.
        teacher_input = NormalizeFeatures()(Data(x=x.clone())).x
        teacher_edges = to_undirected(directed, num_nodes=len(x))
        teacher_edges, _ = remove_self_loops(teacher_edges)
        teacher_edges, _ = add_self_loops(teacher_edges, num_nodes=len(x))
        recipe = 'verified provider raw features; PyG NormalizeFeatures once; undirected; remove/add loops once'
    binding = dict(backbone=backbone, input_binding=input_binding, recipe=recipe,
        environment=environment, order=12 if backbone == 'polyformer_mono' else None,
        dtype='FP32', cache_policy='no persistent reuse; full recomputation',
        source_commit=n['author_commit'])
    identity = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    graph = SimpleNamespace(edge_index=directed, classes=n['classes'], teacher_backbone=backbone,
        teacher_input=teacher_input, teacher_edge_index=teacher_edges.to(x.device),
        preprocessing=dict(identity=identity, **binding))
    return graph


def optimizer_for(model):
    spec = model.specification
    n, mult = spec['native'], spec['lr_multiplier']
    if spec['backbone'] == 'polyformer_mono':
        groups = [dict(params=[parameter], lr=mult*(n['attention_lr'] if 'attnmodule' in name else n['lr']),
            weight_decay=n['attention_weight_decay'] if 'attnmodule' in name else n['weight_decay'])
            for name, parameter in model.named_parameters()]
        return torch.optim.Adam(groups)
    return torch.optim.Adam(model.parameters(), lr=n['lr']*mult, weight_decay=n['weight_decay'])


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def fit_teacher(spec, graph, train, validation, qualify, out):
    """Exactly train/validation labels; no correction or final-pool label argument."""
    model = TeacherFamily(spec).to(graph.teacher_input.device)
    optimizer = optimizer_for(model)
    n = spec['native']
    stages = [('local', 2 if qualify else n['local_epochs']),
              ('global', 2 if qualify else n['global_epochs'])] if spec['backbone'] == 'polynormer_r' else [
              ('native', 3 if qualify else n['max_epochs'])]
    history, best, local_best, updates = [], None, None, 0
    with open(out/'teacher_trace.jsonl', 'x') as trace:
        for stage, cap in stages:
            if stage == 'global':
                require(local_best is not None, 'No finite local checkpoint')
                model.load_state_dict(local_best['state'])
                optimizer.load_state_dict(local_best['optimizer'])
                model.set_global_stage(True)
            best_nll, best_epoch, best = float('inf'), None, None
            for epoch in range(1, cap+1):
                updates += 1
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(graph)
                require(tuple(logits.shape) == (spec['members'], n['nodes'], n['classes']), 'Full teacher trajectory contract')
                loss = torch.nn.functional.cross_entropy(logits[:, train.nodes].reshape(-1, n['classes']),
                    train.labels.repeat(spec['members']))
                require(bool(torch.isfinite(loss)), 'Nonfinite train loss')
                loss.backward()
                grad_count = sum(int(p.grad is not None and bool((p.grad != 0).any())) for p in model.parameters())
                require(grad_count > 0, 'No nonzero teacher gradients')
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    z = model(graph)
                    value = float(torch.nn.functional.cross_entropy(z.mean(0)[validation.nodes], validation.labels))
                require(math.isfinite(value), 'Nonfinite primary validation NLL')
                trace.write(json.dumps(dict(stage=stage, stage_epoch=epoch, actual_update=updates,
                    loss=float(loss.detach()), validation_nll=value, nonzero_gradient_tensors=grad_count))+'\n')
                trace.flush()
                if value < best_nll:
                    best_nll, best_epoch = value, epoch
                    best = dict(state=_cpu_copy(model.state_dict()), optimizer=_cpu_copy(optimizer.state_dict()),
                        stage=stage, global_stage=model.global_stage, stage_epoch=epoch, actual_update=updates,
                        primary_validation_nll=value)
                if stage == 'native' and not qualify and epoch-best_epoch >= n['patience']:
                    break
            require(best is not None, 'No selected finite checkpoint')
            history.append(dict(stage=stage, updates_completed=epoch, selected_epoch=best_epoch,
                primary_validation_nll=best_nll))
            if stage == 'local':
                local_best = best
                torch.save(dict(schema='modern-teacher-local-transition-v1', specification=spec,
                    preprocessing=graph.preprocessing, **best), out/'teacher_local_transition.pt')
    require(spec['backbone'] != 'polynormer_r' or best['global_stage'], 'Final Photo teacher must be global')
    model.load_state_dict(best['state'])
    model.set_global_stage(best['global_stage'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint = dict(schema='modern-teacher-checkpoint-v1', specification=spec,
        preprocessing=graph.preprocessing, qualification_only=qualify, **best)
    torch.save(checkpoint, out/'teacher_checkpoint.pt')
    selection = dict(schema='modern-teacher-selection-v1', specification=spec, stages=history,
        updates_completed=updates, selected_stage=best['stage'], selected_epoch=best['stage_epoch'],
        primary_validation_nll=best['primary_validation_nll'], global_stage=best['global_stage'],
        model_tensor_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values()),
        parameter_count=sum(t.numel() for t in model.parameters()), preprocessing=graph.preprocessing,
        label_scope=['train', 'validation'], qualification_only=qualify, report_eligible=not qualify,
        primary_pooling=PROTOCOL['primary_pooling'], secondary_selected=False)
    (out/'teacher_selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False)+'\n')
    del optimizer
    return model, selection


def restore_teacher(checkpoint_path, graph, expected_spec):
    checkpoint = torch.load(checkpoint_path, map_location=graph.teacher_input.device, weights_only=True)
    require(checkpoint['schema'] == 'modern-teacher-checkpoint-v1' and
        checkpoint['specification'] == expected_spec and not checkpoint['qualification_only'],
        'Wrong/unqualified teacher checkpoint')
    require(checkpoint['preprocessing'] == graph.preprocessing, 'Replay preprocessing differs')
    model = TeacherFamily(expected_spec).to(graph.teacher_input.device)
    model.load_state_dict(checkpoint['state'])
    model.set_global_stage(checkpoint['global_stage'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def saved_probability_views(logits):
    """Fixed primary/secondary, never a selectable pooling grid."""
    return dict(primary=logits.mean(0).softmax(-1), secondary_mean_probabilities=logits.softmax(-1).mean(0))
