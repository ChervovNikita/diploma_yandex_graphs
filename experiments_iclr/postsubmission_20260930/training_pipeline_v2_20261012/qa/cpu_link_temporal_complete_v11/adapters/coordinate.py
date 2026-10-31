"""Exact coordinate-family source bridge; no generic all-map substitution.

Historical SAGE contexts, live boundary BE, private biases, fixed index buffers,
shared norms/skips, capable heads and the separate GT-sep single are imported
from the immutable source factory. No numerical work occurs at module import.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

try:
    from .classifier import ClassifierAdapter
    from ..spec import ORIGINAL_PAPER_DATASETS, normalized_id
    from ..engine import AdapterStepError
except ImportError:
    from adapters.classifier import ClassifierAdapter
    from spec import ORIGINAL_PAPER_DATASETS, normalized_id
    from engine import AdapterStepError


COORDINATE_SHA256 = 'a74a87dc26b7675a2d3fa0aaf0e7734b4786fa6c49411ef52f9bb21c0516dbc1'
NATIVE_SHA256 = '07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303'
ARMS = ('original', 'factor', 'coordinate', 'permutation', 'bias_only',
        'single', 'untied', 'heads', 'gt_sep_single')
SOURCE_CONTRACTS = {
    'source': 'coordinate_ensemble_source_v1/models.py::build_model',
    'source_sha256': COORDINATE_SHA256, 'native_sha256': NATIVE_SHA256,
    'hidden_map_sites': ('SAGE mean-neighbor lin_l', 'SAGE bias-free root lin_r', 'FF1', 'FF2'),
    'coordinate_order': 'D_s P_out W P_in D_r h + private_bias',
    'norm_skip': 'shared original affine LayerNorm and unpermuted identity skips',
    'boundary': 'live source R/S/B at input and output; no input/class-axis permutation',
    'factor': 'unit-initialized interior R/S and private original-bias-presence rows',
    'permutation': 'independent fixed input/output gathers and private biases; interior R/S absent',
    'bias_only': 'interior R/S/indices absent; private biases only; live boundary BE remains',
    'heads': 'one complete ordinary SAGE trunk computed once; private h->h->C GELU heads',
    'single_initialization': 'same-width TABM reference common maps/norms copied; private boundary factors omitted',
    'untied': 'four physical bodies and disjoint optimizer objects; sum own CE backward; one cohort selector',
    'fusion': 'mean_probabilities for every historical arm including original alias',
    'selector': 'strict earliest SELECT pooled NLL, every10, checkpoint0 false',
    'source_budget': 'min1000/max2000 cycles; patience200',
    'v2_budget': 'full2000 cycles; historical min/patience recorded as diagnostics, never terminate',
    'unsupported': ('non-SAGE context placement', 'coherent full-network gauge/conjugation',
                    'binary BCE', 'sampler/heterogeneous/task-specific providers'),
    'exact_historical_data_recipe_reproduction': False,
}


def confined(root, relative):
    root = Path(root).resolve()
    raw = Path(relative)
    if raw.is_absolute() or '..' in raw.parts:
        raise ValueError('Coordinate source/input paths must be project-relative')
    path = (root / raw).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing or unconfined coordinate source/input')
    return path


def load_source(root, coordinate_source, native_source):
    path, dependency = confined(root, coordinate_source), confined(root, native_source)
    if hashlib.sha256(path.read_bytes()).hexdigest() != COORDINATE_SHA256:
        raise ValueError('Coordinate source differs from the exact audited recipe')
    if hashlib.sha256(dependency.read_bytes()).hexdigest() != NATIVE_SHA256:
        raise ValueError('Coordinate native dependency differs from the exact audited source')
    name = 'v2_coordinate_' + hashlib.sha256((str(path) + str(dependency)).encode()).hexdigest()[:20]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            # Relocation of one explicitly hashed dependency, without changing
            # source bytes, constructors, parameter ownership or forward order.
            module.FROZEN_SOURCE = dependency
            module._load_frozen.cache_clear()
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


class OptimizerBundle:
    """Checkpoint facade retaining distinct source optimizer objects/states."""
    def __init__(self, optimizers):
        self.optimizers = tuple(optimizers)
        if not self.optimizers:
            raise ValueError('At least one actual optimizer object required')
        identities = [id(p) for group in self.param_groups for p in group['params']]
        if len(identities) != len(set(identities)):
            raise ValueError('Source optimizer objects overlap parameter ownership')

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, **kwargs):
        for optimizer in self.optimizers:
            optimizer.zero_grad(**kwargs)

    def step(self, operation_counts=None):
        for optimizer in self.optimizers:
            if operation_counts is not None:
                operation_counts['optimizer_step_calls'] += 1
            try:
                optimizer.step()
            except BaseException:
                if operation_counts is not None:
                    operation_counts['failed_optimizer_step_calls'] = 1
                    operation_counts['partial_optimizer_cycle_requires_manual_recovery'] = 1
                raise
            if operation_counts is not None:
                operation_counts['optimizer_updates'] += 1

    def state_dict(self):
        return {'schema': 'coordinate-disjoint-optimizers-v1', 'optimizer_count': len(self.optimizers),
                'optimizer_states': [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, value):
        if (value.get('schema') != 'coordinate-disjoint-optimizers-v1'
                or value.get('optimizer_count') != len(self.optimizers)
                or len(value.get('optimizer_states', [])) != len(self.optimizers)):
            raise ValueError('Checkpoint optimizer-object roster differs')
        for optimizer, state in zip(self.optimizers, value['optimizer_states']):
            optimizer.load_state_dict(state)


class CoordinateAdapter(ClassifierAdapter):
    def train_step(self, step):
        counts = {'optimizer_updates': 0, 'optimizer_step_calls': 0, 'source_cycles': 0,
                  'backward_calls': 0, 'model_forwards': 0, 'trunk_forwards': 0,
                  'route_predictions': 0, 'supervised_examples': 0}
        try:
            return self._train_step(step, counts)
        except BaseException as error:
            raise AdapterStepError(str(error), operation_counts=counts) from error

    def _train_step(self, step, counts):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.logits_fn(self.model, True, step)
        selected = logits.index_select(1, self.train_ids)
        members, rows, classes = selected.shape
        counts.update(model_forwards=1 if self.source_arm == 'heads' else members,
            trunk_forwards=1 if self.source_arm == 'heads' else members,
            route_predictions=members, supervised_examples=rows * members)
        losses = self.F.cross_entropy(selected.reshape(members * rows, classes),
            self.train_y.repeat(members), reduction='none').view(members, rows).mean(1)
        objective = losses.sum() if self.source_arm == 'untied' else losses.mean()
        if not bool(self.t.isfinite(objective)):
            raise FloatingPointError('Nonfinite coordinate objective')
        value = float(objective.detach())
        mean_ce = float(losses.detach().mean())
        counts['backward_calls'] += 1
        objective.backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None and not bool(self.t.isfinite(parameter.grad).all()):
                raise FloatingPointError('Nonfinite coordinate gradient')
        self.optimizer.step(counts)
        for parameter in self.model.parameters():
            if not bool(self.t.isfinite(parameter).all()):
                raise FloatingPointError('Nonfinite coordinate parameter after source update')
        counts['source_cycles'] = 1
        return {'objective': value, 'ce': mean_ce, 'regularization': 0., 'operation_counts': counts}

    def validate(self, step):
        self.model.eval()
        with self.t.no_grad():
            logits = self.logits_fn(self.model, False, step).index_select(1, self.select_ids)
            if not bool(self.t.isfinite(logits).all()):
                raise FloatingPointError('Nonfinite coordinate SELECT logits')
            log_pool = self.t.logsumexp(self.F.log_softmax(logits, -1), 0) - math.log(len(logits))
            nll = self.F.nll_loss(log_pool, self.select_y)
            own = self.t.stack([self.F.cross_entropy(z, self.select_y) for z in logits]).mean()
            sensitivity = self.F.log_softmax(logits.mean(0), -1)
            correct = int(log_pool.argmax(-1).eq(self.select_y).sum())
            return {'objective': float(own), 'ce': float(own), 'regularization': 0.,
                'accuracy': correct / len(self.select_y), 'correct': correct, 'examples': len(self.select_y),
                'pooled_nll': float(nll), 'selection_metric': float(nll), 'fusion': 'mean_probabilities',
                'same_checkpoint_mean_logit_nll': float(self.F.nll_loss(sensitivity, self.select_y)),
                'same_checkpoint_mean_logit_accuracy': float(sensitivity.argmax(-1).eq(self.select_y).float().mean())}

    def observe_validation(self, step, metrics):
        decision = super().observe_validation(step, metrics)
        decision['stop_requested'] &= step >= self.source_minimum_cycles
        decision['historical_patience_is_diagnostic_only'] = True
        return decision


def make_adapter(*, torch, project_root, study, run, device):
    from torch.nn import functional as F
    config = run['config']
    arm = config['source_arm']
    if arm not in ARMS or config['target_law'] != 'multiclass_ce' or config['classes'] < 3:
        raise ValueError('Declared coordinate multiclass source arm required')
    if config['backbone'] != ('GT-sep' if arm == 'gt_sep_single' else 'SAGE'):
        raise ValueError('Exact coordinate contexts support SAGE only; GT-sep is a separate ordinary single')
    if normalized_id(study['dataset']) in ORIGINAL_PAPER_DATASETS and arm == 'original':
        raise ValueError('Original-paper datasets may not enter the original GNNM alias')
    if config['fusion'] != 'mean_probabilities' or config['normalization'] != 'LayerNorm':
        raise ValueError('Keep the source probability pool and affine LayerNorm')
    if run['selection']['metric'] != 'pooled_nll' or run['selection']['direction'] != 'min' or run['selection']['ties'] != 'first':
        raise ValueError('Coordinate source requires strict earliest SELECT pooled NLL')
    source = load_source(project_root, config['coordinate_source'], config['native_model_source'])
    def load(role):
        return torch.load(confined(project_root, config['node_inputs'][role]), map_location='cpu', weights_only=True)
    inputs, train, select = load('graph'), load('train'), load('select')
    if set(inputs) != {'x', 'edge_index'} or set(train) != {'ids', 'y'} or set(select) != {'ids', 'y'}:
        raise ValueError('Prepared graph and separate TRAIN/SELECT input schema required')
    if bool(torch.isin(train['ids'], select['ids']).any()):
        raise ValueError('TRAIN and SELECT overlap')
    seed = config.get('initialization_seed', run['seed'] + 1000003 * run['body'])
    permutation_seed = seed + config['permutation_seed_offset']
    torch.manual_seed(seed)
    random.seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    model = source.build_model(arm, inputs['x'].shape[1], config['width'], config['classes'],
        seed, permutation_seed, members=config['source_requested_members'], num_layers=config['depth'],
        dropout=config['dropout'], hidden_dim_multiplier=config['ffn_multiplier'],
        normalization='layer', num_heads=config['heads']).to(device)
    if model.members != config['members']:
        raise ValueError('Source actual route count differs from the approved packet')
    if arm == 'untied':
        parameters = [list(body.parameters()) for body in model.models]
    else:
        parameters = [list(model.parameters())]
    optimizers = [torch.optim.AdamW(values, lr=config['learning_rate'], weight_decay=config['weight_decay'],
        betas=(.9, .999), eps=1e-8) for values in parameters]
    bundle = OptimizerBundle(optimizers)
    if len(optimizers) != config['source_optimizer_count']:
        raise ValueError('Actual source optimizer-object count differs from approved packet')
    if set(id(p) for group in bundle.param_groups for p in group['params']) != set(id(p) for p in model.parameters()):
        raise ValueError('Source optimizer objects do not cover every model parameter')
    graph, x = SimpleNamespace(edge_index=inputs['edge_index'].to(device)), inputs['x'].to(device)
    selector = {'metric': 'pooled_nll', 'direction': 'min', 'ties': 'first',
                'patience_steps': config['source_patience_cycles'], 'save_every': 1}
    adapter = CoordinateAdapter(torch, F, model, bundle, logits_fn=lambda model, training, step: model(graph, x),
        train_ids=train['ids'].to(device), train_y=train['y'].to(device),
        select_ids=select['ids'].to(device), select_y=select['y'].to(device),
        regularizer=lambda model: x.new_zeros(()), fusion='mean_probabilities', selector=selector,
        provenance={'run_id': run['run_id'], 'data': study['data'], 'source': study['source'], 'config': config,
            'actual_model_seed': seed, 'actual_permutation_seed': permutation_seed,
            'source_contract': SOURCE_CONTRACTS, 'actual_source_specification': model.specification,
            'model_storage': source.storage_report(model), 'optimizer_object_count': len(optimizers),
            'separately_selected_I4': False})
    adapter.source_arm, adapter.source_minimum_cycles = arm, config['source_minimum_cycles']
    return adapter
