"""Host-independent library adapter for the frozen internal-BE core.

Importing this file loads only stdlib. Numerical dependencies load when a
Session is constructed. There is no author-runtime admission or download path.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
_CORE = None


def recipe(task):
    if task not in ('wikics', 'collab', 'molhiv'):
        raise ValueError('Unknown task: ' + task)
    return json.loads((ROOT / 'recipes' / (task + '.json')).read_text())


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'core' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _core():
    global _CORE
    if _CORE is None:
        factors = _module('_portable_internal_be_factors', 'factors.py')
        old = sys.modules.get('factors')
        try:
            sys.modules['factors'] = factors
            models = _module('_portable_internal_be_models', 'models.py')
        finally:
            if old is None:
                sys.modules.pop('factors', None)
            else:
                sys.modules['factors'] = old
        _CORE = dict(factors=factors, models=models,
                     objectives=_module('_portable_internal_be_objectives', 'objectives.py'),
                     selection=_module('_portable_internal_be_selection', 'selection.py'))
    return _CORE


def native_sources(task, polynormer=None, ncn_model=None, ncn_utils=None):
    """Load only the task's external source, supplied by the researcher."""
    models = _core()['models']
    pins = json.loads((ROOT / 'NATIVE_SOURCES.json').read_text())
    provenance = {}
    if task == 'molhiv':
        return (None, None), provenance
    if task == 'wikics':
        if polynormer is None:
            raise ValueError('Supply Polynormer model.py at the documented commit')
        digest = pins['polynormer']['model_sha256']
        module = models.load_source(polynormer, digest, '_portable_native_polynormer')
        provenance['polynormer_model_sha256'] = digest
        return (module, None), provenance
    if task == 'collab':
        if ncn_model is None or ncn_utils is None:
            raise ValueError('Supply NCN model.py and utils.py at the documented commit')
        pin = pins['ncn']
        utils = models.load_source(ncn_utils, pin['utils_sha256'], '_portable_native_ncn_utils')
        old = sys.modules.get('utils')
        try:
            sys.modules['utils'] = utils
            module = models.load_source(ncn_model, pin['model_sha256'], '_portable_native_ncn')
        finally:
            if old is None:
                sys.modules.pop('utils', None)
            else:
                sys.modules['utils'] = old
        provenance.update(ncn_model_sha256=pin['model_sha256'], ncn_utils_sha256=pin['utils_sha256'])
        return (None, module), provenance
    raise ValueError(task)


class Session:
    """Fresh model, Adam bank and persistent member dropout streams.

    This library implements update/serving/snapshot primitives. It does not
    implement the benchmark epoch loop or select a checkpoint using validation.
    """
    def __init__(self, task, arm='be_init_contrastive', seed=6101, device='cpu',
                 polynormer=None, ncn_model=None, ncn_utils=None):
        import numpy as np
        import torch
        self.torch, self.np = torch, np
        self.task, self.arm, self.seed = task, arm, seed
        self.config = recipe(task)
        if arm not in self.config['arms'] or type(seed) is not int:
            raise ValueError('Registered arm and integer seed required')
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('This adapter supports cpu or cuda[:index]')
        self.cuda_index = None
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('Requested CUDA is unavailable')
            self.cuda_index = 0 if self.device.index is None else self.device.index
            torch.cuda.set_device(self.cuda_index)
            self.device = torch.device('cuda', self.cuda_index)
        self.core = _core()
        self.core_provenance = {name: hashlib.sha256((ROOT / 'core' / name).read_bytes()).hexdigest()
                                for name in ('factors.py', 'models.py', 'objectives.py', 'selection.py')}
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if self.cuda_index is not None:
            torch.cuda.manual_seed(seed)
        sources, self.native_provenance = native_sources(task, polynormer, ncn_model, ncn_utils)
        self.model = self.core['models'].Ensemble(task, arm, seed, self.config['model'], sources).to(self.device)
        if task == 'wikics':
            self.model.set_global(False)
        def optimizer(body):
            if task == 'collab':
                return torch.optim.Adam([{'params': body.encoder.parameters(), 'lr': .0082},
                                         {'params': body.decoder.parameters(), 'lr': .0037}], weight_decay=0.)
            return torch.optim.Adam(body.parameters(), lr=self.config['training']['lr'], weight_decay=0., eps=1e-8)
        self.optimizers = [optimizer(body) for body in self.model.models]
        self.streams = []
        for member in range(self.model.members):
            with torch.random.fork_rng(devices=self.cuda_devices):
                torch.manual_seed(seed + 1009 * member + 300001)
                stream = {'cpu': torch.get_rng_state()}
                if self.cuda_index is not None:
                    torch.cuda.manual_seed(seed + 1009 * member + 300001)
                    stream['cuda'] = torch.cuda.get_rng_state(self.cuda_index)
                self.streams.append(stream)
        self.steps = 0

    @property
    def cuda_devices(self):
        return [] if self.cuda_index is None else [self.cuda_index]

    def forward(self, batch):
        torch = self.torch
        rows = []
        for member, stream in enumerate(self.streams):
            with torch.random.fork_rng(devices=self.cuda_devices):
                torch.set_rng_state(stream['cpu'])
                if self.cuda_index is not None:
                    torch.cuda.set_rng_state(stream['cuda'], self.cuda_index)
                rows.append(self.model.member_forward(batch, member))
                stream['cpu'] = torch.get_rng_state()
                if self.cuda_index is not None:
                    stream['cuda'] = torch.cuda.get_rng_state(self.cuda_index)
        return torch.stack([row[0] for row in rows]), torch.stack([row[1] for row in rows])

    def serving(self, logits):
        return logits.softmax(-1).mean(0) if self.task == 'wikics' else logits.mean(0)

    def train_step(self, batch, labels):
        """Exact two-own-view update and bounded auxiliary target selection."""
        torch = self.torch
        if labels.device != self.device:
            raise ValueError('Move batch and labels to Session.device before updating')
        self.model.train()
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        la, ha = self.forward(batch); lb, hb = self.forward(batch)
        selection = self.core['selection']; losses = self.core['objectives']
        selection.finite_predictions(la, self.serving(la)); selection.finite_predictions(lb, self.serving(lb))
        if not torch.isfinite(ha).all() or not torch.isfinite(hb).all():
            raise FloatingPointError('TRAIN representation')
        size = min(len(labels), self.config['contrastive']['max_objects'])
        index = torch.linspace(0, len(labels)-1, steps=size, device=labels.device).long()
        own = .5 * (losses.own_supervision(la, labels, self.task) + losses.own_supervision(lb, labels, self.task))
        auxiliary = own.sum() * 0
        if self.model.contrastive:
            identity = batch['query'][index] if self.task == 'collab' else None
            auxiliary = .05 * losses.alignment_loss(ha[:, index], hb[:, index], labels[index], self.task, identities=identity)
            auxiliary = auxiliary + .05 * losses.residual_member_contrast(ha[:, index], hb[:, index], labels[index])
        total = own.sum() + self.model.members * auxiliary if self.model.independent else own.mean() + auxiliary
        if not torch.isfinite(total):
            raise FloatingPointError('TRAIN loss')
        total.backward()
        active = [parameter for parameter in self.model.parameters() if parameter.grad is not None]
        if not active or any(not torch.isfinite(parameter.grad).all() for parameter in active):
            raise FloatingPointError('TRAIN gradient')
        for optimizer in self.optimizers:
            optimizer.step()
        selection.finite_state(self.model, self.optimizers)
        self.steps += 1
        return {'loss': total.detach(), 'own_mean': own.mean().detach(), 'auxiliary': auxiliary.detach()}

    def _cpu_tree(self, value):
        if isinstance(value, self.torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: self._cpu_tree(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._cpu_tree(item) for item in value)
        return value

    def save_training_state(self, path, epoch=0):
        """Fresh continuation snapshot; no validation-selection claim."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        torch = self.torch
        state = {'schema': 'portable-internal-be-continuation-v1', 'task': self.task, 'arm': self.arm,
                 'seed': self.seed, 'config': self.config, 'native_provenance': self.native_provenance,
                 'core_provenance': self.core_provenance,
                 'backend': self.device.type, 'model': self.model.state_dict(),
                 'optimizers': [optimizer.state_dict() for optimizer in self.optimizers],
                 'streams': self.streams, 'steps': self.steps, 'epoch': epoch,
                 'body_global': [getattr(getattr(body, 'body', None), '_global', False) for body in self.model.models],
                 'python_rng': random.getstate(), 'numpy_rng': self.np.random.get_state(),
                 'cpu_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state(self.cuda_index) if self.cuda_index is not None else None,
                 'checkpoint_selection': 'none; model/optimizer/RNG snapshot primitive only',
                 'data_iterator_state_included': False, 'selector_history_included': False,
                 'author_execution_equivalence': False}
        torch.save(self._cpu_tree(state), path)

    def restore_training_state(self, path):
        """Restore a trusted snapshot from this interface; never mix own epochs."""
        torch = self.torch
        state = torch.load(path, map_location='cpu', weights_only=False)
        expected = {'schema': 'portable-internal-be-continuation-v1', 'task': self.task,
                    'arm': self.arm, 'seed': self.seed, 'config': self.config,
                    'native_provenance': self.native_provenance, 'core_provenance': self.core_provenance,
                    'backend': self.device.type}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError('Snapshot task/arm/recipe/source/backend differs')
        if len(state['optimizers']) != len(self.optimizers) or len(state['streams']) != self.model.members:
            raise ValueError('Snapshot optimizer or member bank differs')
        self.model.load_state_dict(state['model'])
        for optimizer, saved in zip(self.optimizers, state['optimizers']):
            optimizer.load_state_dict(saved)
        if self.task == 'wikics':
            for body, global_mode in zip(self.model.models, state['body_global']):
                body.set_global(global_mode)
        self.streams, self.steps = state['streams'], state['steps']
        random.setstate(state['python_rng']); self.np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['cpu_rng'])
        if self.cuda_index is not None:
            torch.cuda.set_rng_state(state['cuda_rng'], self.cuda_index)
        self.core['selection'].finite_state(self.model, self.optimizers)
        return state['epoch']
