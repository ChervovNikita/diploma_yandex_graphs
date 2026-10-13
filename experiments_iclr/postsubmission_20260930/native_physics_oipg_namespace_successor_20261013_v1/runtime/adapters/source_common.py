"""Inert bound-source and complete RNG helpers for prospective native ports."""
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import random
import sys
from spec import confined

def bound(root, row):
    path = confined(root, row['path'])
    if path.stat().st_size != row['bytes']:
        raise ValueError('Bound file size changed: ' + row['path'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    if digest.hexdigest() != row['sha256']:
        raise ValueError('Bound file changed: ' + row['path'])
    return path

def module(root, row, name):
    path = bound(root, row)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

class Streams:
    def __init__(self, torch, np, device):
        self.t, self.np, self.device, self.states = torch, np, torch.device(device), []
    def capture(self):
        state = self.np.random.get_state()
        return {'python': random.getstate(), 'numpy': (state[0],state[1].tolist(),state[2],state[3],state[4]),
                'cpu': self.t.get_rng_state().clone(),
                'cuda': self.t.cuda.get_rng_state(self.device).clone() if self.device.type=='cuda' else None}
    def restore(self, value):
        random.setstate(value['python']);state=value['numpy']
        self.np.random.set_state((state[0],self.np.asarray(state[1],dtype=self.np.uint32),state[2],state[3],state[4]))
        self.t.set_rng_state(value['cpu'].cpu())
        if self.device.type=='cuda':self.t.cuda.set_rng_state(value['cuda'].cpu(),self.device)
    def seed(self, seed):
        random.seed(seed);self.np.random.seed(seed);self.t.manual_seed(seed)
        if self.device.type=='cuda':self.t.cuda.manual_seed(seed);self.t.cuda.manual_seed_all(seed)
    @contextmanager
    def scope(self, body):
        outer=self.capture();self.restore(self.states[body])
        try:yield
        finally:self.states[body]=self.capture();self.restore(outer)

class OptimizerBank:
    """Serializer for actual own optimizers; creates no optimizer or transition."""
    def __init__(self, optimizers):
        self.optimizers=optimizers
        self.param_groups=[group for opt in optimizers for group in opt.param_groups]
    def state_dict(self):return {'own_optimizers':[opt.state_dict() for opt in self.optimizers]}
    def load_state_dict(self, state):
        if len(state['own_optimizers'])!=len(self.optimizers):raise ValueError('Optimizer bank size changed')
        for opt,value in zip(self.optimizers,state['own_optimizers']):opt.load_state_dict(value)

def roles(torch, root, config):
    packs={role:torch.load(bound(root,row),map_location='cpu',weights_only=True) for role,row in config['node_inputs'].items()}
    graph=packs['graph']
    if set(graph)!={'x','edge_index'} or any(set(packs[r])!={'ids','y'} for r in ('train','select')):
        raise ValueError('Exact separated graph and TRAIN/SELECT schemas required')
    if graph['x'].dtype!=torch.float32 or graph['x'].ndim!=2 or not torch.isfinite(graph['x']).all():raise ValueError('Full finite float32 features')
    if graph['edge_index'].dtype!=torch.long or graph['edge_index'].shape[0]!=2:raise ValueError('Full int64 graph support')
    for role in ('train','select'):
        pack=packs[role]
        if pack['ids'].dtype!=torch.long or pack['y'].dtype!=torch.long or pack['ids'].shape!=pack['y'].shape or not len(pack['ids']) or len(torch.unique(pack['ids']))!=len(pack['ids']):raise ValueError('Unique complete source role')
    if torch.isin(packs['train']['ids'],packs['select']['ids']).any():raise ValueError('TRAIN/SELECT overlap')
    return packs
