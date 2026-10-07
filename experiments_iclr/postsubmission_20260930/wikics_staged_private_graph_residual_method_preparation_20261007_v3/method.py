"""Actual full-WikiCS residual model/step operators. Imported only after root guards."""
from contextlib import contextmanager
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv
from vendor.native_polynormer import Polynormer

ARMS = ('E_stage', 'E_joint', 'E_own', 'S_continue', 'S_paths', 'I_native', 'U_stage')
NATIVE_ARGUMENTS = dict(in_channels=300, hidden_channels=512, out_channels=10,
    local_layers=7, global_layers=2, in_dropout=.5, dropout=.5,
    global_dropout=.5, heads=1, beta=-1, pre_ln=False)


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return type(value)(cpu_tree(v) for v in value)
    return value


def rng_snapshot():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state().clone())


def rng_restore(state):
    random.setstate(state[0]); np.random.set_state(state[1])
    torch.set_rng_state(state[2]); torch.cuda.set_rng_state(state[3])


@contextmanager
def route_rng(stream):
    old_cpu, old_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    torch.set_rng_state(stream['cpu']); torch.cuda.set_rng_state(stream['cuda'])
    try: yield
    finally:
        stream.update(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state().clone())
        torch.set_rng_state(old_cpu); torch.cuda.set_rng_state(old_cuda)


def adam(parameters):
    return torch.optim.Adam(parameters, lr=.001, weight_decay=0., betas=(.9, .999), eps=1e-8)


def construct_native(seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = Polynormer(**NATIVE_ARGUMENTS).to(device)
    model.reset_parameters(); model._global = False
    stream = {'cpu': torch.get_rng_state().clone(), 'cuda': torch.cuda.get_rng_state().clone()}
    return model, adam(model.parameters()), stream


def native_hidden_logits(model, x, edge):
    """Capture the real classifier input; never reconstruct the native arithmetic."""
    received = []
    head = model.pred_global if model._global else model.pred_local
    handle = head.register_forward_pre_hook(lambda module, arguments: received.append(arguments[0]))
    try: logits = model(x, edge)
    finally: handle.remove()
    if len(received) != 1 or received[0].shape != (11701, 512):
        raise ValueError('Actual full native classifier interface changed')
    return received[0], logits


def active_native(name, global_stage):
    if global_stage: return not name.startswith('pred_local.')
    return not name.startswith(('global_attn.', 'ln.', 'pred_global.'))


def gradients_finite(model, active=None):
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or (active is not None and not active(name)): continue
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise ValueError('Disconnected/nonfinite active gradient: '+name)


def optimizer_ownership(model, optimizers):
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    owned = [id(p) for opt in optimizers for g in opt.param_groups for p in g['params']]
    if len(owned) != len(set(owned)) or set(owned) != expected:
        raise ValueError('Optimizers must own every active parameter exactly once')


def native_step(model, optimizer, stream, x, edge, ids, labels):
    model.train(); optimizer.zero_grad(set_to_none=True)
    with route_rng(stream):
        z = model(x, edge)
        loss = F.nll_loss(F.log_softmax(z, 1)[ids], labels)
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Native TRAIN loss')
        loss.backward()
    gradients_finite(model, lambda name: active_native(name, model._global))
    optimizer.step()
    return {'own_CE': float(loss.detach()), 'native_training_forwards': 1, 'updates': 1}


class GraphPath(nn.Module):
    """Full raw-feature affine plus one private native local Polynormer block."""
    def __init__(self):
        super().__init__()
        self.raw = nn.Linear(300, 512)
        self.h_lin = nn.Linear(512, 512)
        self.conv = GATConv(512, 512, heads=1, concat=True, add_self_loops=False, bias=False)
        self.root_lin = nn.Linear(512, 512)
        self.ln = nn.LayerNorm(512)
        self.beta = nn.Parameter(torch.zeros(1, 512))
        self.reset_parameters()

    def reset_parameters(self):
        for block in (self.raw, self.h_lin, self.conv, self.root_lin, self.ln): block.reset_parameters()
        nn.init.xavier_normal_(self.beta)

    def forward(self, x, hidden, edge):
        u = hidden+self.raw(x)
        h = F.relu(self.h_lin(u))
        value = F.relu(self.conv(u, edge)+self.root_lin(u))
        value = F.dropout(value, p=.5, training=self.training)
        beta = F.sigmoid(self.beta)
        return (1-beta)*self.ln(h*value)+beta*value


class ResidualPath(nn.Module):
    def __init__(self):
        super().__init__(); self.graph = GraphPath(); self.output = nn.Linear(512, 10)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, x, hidden, edge): return self.output(self.graph(x, hidden, edge))


def isolated_path(seed, device, graph_only=False):
    old = rng_snapshot()
    try:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        path = (GraphPath() if graph_only else ResidualPath()).to(device)
        stream = {'cpu': torch.get_rng_state().clone(), 'cuda': torch.cuda.get_rng_state().clone()}
    finally: rng_restore(old)
    return path, stream


class ResidualBank(nn.Module):
    def __init__(self, donors, base_seed, device):
        super().__init__()
        if len(donors) not in (1, 4): raise ValueError('One common or four genuinely acquired donors')
        self.donors = nn.ModuleList(donors); self.paths = nn.ModuleList(); self.streams = []
        for donor in self.donors:
            donor.eval()
            for parameter in donor.parameters(): parameter.requires_grad_(False)
        for member in range(4):
            path, stream = isolated_path(base_seed+1009*member, device)
            self.paths.append(path); self.streams.append(stream)
        self.cache = None
        self.optimizers = [adam(path.parameters()) for path in self.paths]
        optimizer_ownership(self, self.optimizers)
        self.assert_storage()

    def assert_storage(self):
        storages = [p.untyped_storage().data_ptr() for p in self.parameters()]
        if len(storages) != len(set(storages)): raise ValueError('Private paths/donors alias storage')

    def train(self, mode=True):
        super().train(mode)
        for donor in self.donors: donor.eval()
        return self

    def make_cache(self, x, edge):
        if any(p.requires_grad for d in self.donors for p in d.parameters()):
            raise ValueError('Only a frozen donor may cache hidden states/logits')
        self.eval()
        with torch.no_grad():
            self.cache = [tuple(t.detach().clone() for t in native_hidden_logits(d, x, edge)) for d in self.donors]
        return self.cache

    def donor_cache(self, member):
        if self.cache is None: raise ValueError('Frozen native donor cache is absent')
        return self.cache[0 if len(self.cache) == 1 else member]

    def member_logits(self, member, x, edge):
        h, z = self.donor_cache(member)
        return z+self.paths[member](x, h, edge)

    def serving(self, x, edge):
        self.eval()
        with torch.no_grad(): return torch.stack([F.softmax(self.member_logits(m, x, edge), 1) for m in range(4)])


def correction_loss(logits, peer_probabilities, ids, labels, include_residual=True):
    p = F.softmax(logits, 1)[ids]
    peers = peer_probabilities[:, ids].detach()
    pool = (peers.sum(0)+p)/4
    target = F.one_hot(labels, num_classes=10).to(p.dtype)
    own = F.nll_loss(F.log_softmax(logits, 1)[ids], labels)
    brier = (pool-target).square().sum(1).mean()
    total = own+2*brier if include_residual else own
    if not bool(torch.isfinite(total)): raise FloatingPointError('Correction TRAIN loss')
    return total, own, brier


def bank_epoch(bank, x, edge, ids, labels, members, include_residual=True, diagnostics=False, gradient_observer=None):
    """The very same route loss is used for staged and interleaved fitting."""
    peers = bank.serving(x, edge)  # Common pre-update bank; deterministic peers.
    records = []
    for m in members:
        path = bank.paths[m]; path.train(); opt = bank.optimizers[m]; opt.zero_grad(set_to_none=True)
        with route_rng(bank.streams[m]):
            logits = bank.member_logits(m, x, edge)
            total, own, brier = correction_loss(logits, peers[[j for j in range(4) if j != m]], ids, labels, include_residual)
            record = {'member': m, 'own_CE': float(own.detach()), 'pool_Brier': float(brier.detach()), 'J': float(total.detach())}
            if diagnostics and include_residual:
                parameters = tuple(path.parameters())
                g_own = torch.autograd.grad(own, parameters, retain_graph=True)
                g_pool = torch.autograd.grad(2*brier, parameters, retain_graph=True)
                aa = sum((g*g).sum() for g in g_own); bb = sum((g*g).sum() for g in g_pool)
                ab = sum((a*b).sum() for a,b in zip(g_own,g_pool))
                record['TRAIN_gradient_terms'] = {'own_norm': float(aa.sqrt()), 'pool_norm': float(bb.sqrt()),
                    'dot': float(ab), 'cosine': float(ab/(aa*bb).sqrt()) if float(aa*bb)>0 else None}
            if gradient_observer is not None:
                gradient_observer(m,path,opt,logits,peers[[j for j in range(4) if j != m]],ids,labels,include_residual)
            total.backward()
        gradients_finite(path); records.append(record)
        del logits, total, own, brier
    # No path is changed until every requested gradient uses the same peer state.
    for m in members: bank.optimizers[m].step()
    return {'routes': records, 'private_training_forwards': len(members),
            'private_serving_forwards': 4, 'updates': len(members),
            'extra_diagnostic_VJPs':2*len(members) if diagnostics and include_residual else 0}


class PathSingle(nn.Module):
    """One full native learner, four full graph paths, a nonlinear joint decoder."""
    def __init__(self, donor, native_optimizer_state, base_stream, base_seed, device):
        super().__init__(); self.donor = donor; self.base_stream = copy.deepcopy(base_stream)
        for parameter in donor.parameters(): parameter.requires_grad_(True)
        self.paths = nn.ModuleList(); self.streams = []
        for m in range(4):
            path, stream = isolated_path(base_seed+1009*m, device, graph_only=True)
            self.paths.append(path); self.streams.append(stream)
        old = rng_snapshot()
        try:
            torch.manual_seed(base_seed+7919); torch.cuda.manual_seed_all(base_seed+7919)
            self.readout = nn.Sequential(nn.Linear(2048,512), nn.ReLU(), nn.Linear(512,10)).to(device)
            nn.init.zeros_(self.readout[-1].weight); nn.init.zeros_(self.readout[-1].bias)
        finally: rng_restore(old)
        self.native_optimizer = adam(donor.parameters()); self.native_optimizer.load_state_dict(native_optimizer_state)
        self.new_optimizer = adam(list(self.paths.parameters())+list(self.readout.parameters()))
        optimizer_ownership(self, [self.native_optimizer, self.new_optimizer])
        self.assert_storage()

    def assert_storage(self):
        pointers = [p.untyped_storage().data_ptr() for p in self.parameters()]
        if len(pointers)!=len(set(pointers)): raise ValueError('Single predictor storage alias')

    def forward(self, x, edge, checkpointed=True):
        if self.training:
            with route_rng(self.base_stream):
                h,z = checkpoint(lambda a: native_hidden_logits(self.donor,a,edge), x,
                    use_reentrant=False, preserve_rng_state=True) if checkpointed else native_hidden_logits(self.donor,x,edge)
        else: h,z = native_hidden_logits(self.donor,x,edge)
        representations = []
        for m,path in enumerate(self.paths):
            if self.training:
                with route_rng(self.streams[m]):
                    b = checkpoint(path,x,h,edge,use_reentrant=False,preserve_rng_state=True) if checkpointed else path(x,h,edge)
            else: b = path(x,h,edge)
            representations.append(b)
        return z+self.readout(torch.cat(representations,1))

    def serving(self, x, edge):
        self.eval()
        with torch.no_grad(): return F.softmax(self(x,edge),1).unsqueeze(0)


def single_epoch(model, x, edge, ids, labels, checkpointed=True, forward_observer=None):
    model.train()
    for opt in (model.native_optimizer,model.new_optimizer): opt.zero_grad(set_to_none=True)
    logits = model(x,edge,checkpointed=checkpointed)
    p = F.softmax(logits,1)[ids]; target = F.one_hot(labels,10).to(p.dtype)
    own = F.nll_loss(F.log_softmax(logits,1)[ids],labels)
    brier = (p-target).square().sum(1).mean(); loss = own+2*brier
    if not bool(torch.isfinite(loss)): raise FloatingPointError('Single TRAIN loss')
    if forward_observer is not None:forward_observer(logits,loss)
    loss.backward()
    gradients_finite(model, lambda name: not name.startswith('donor.') or active_native(name[6:],model.donor._global))
    model.native_optimizer.step(); model.new_optimizer.step()
    return {'own_CE':float(own.detach()),'served_Brier':float(brier.detach()),
        'native_training_forwards':1,'private_training_forwards':4,'updates':2,
        'checkpoint_rematerialization':checkpointed,'extra_backward_message_passes_charged':True}
