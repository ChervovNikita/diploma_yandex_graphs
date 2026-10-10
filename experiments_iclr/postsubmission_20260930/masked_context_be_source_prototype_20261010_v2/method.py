"""Ordinary scalar-gradient masked-context prototype. Import only after admission."""
from contextlib import contextmanager
import hashlib
import random
import time
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from native import construct, factors_module, native_preprocess, seeded
from plan import NATIVE_MONO, RULE, condition


def balanced_hash_partition(eligible_ids, seed=190101):
    """Public node ID ordering only; labels never enter masks."""
    ids = [int(x) for x in eligible_ids]
    ordered = sorted(ids, key=lambda node: (hashlib.sha256(f'{seed}:{node}'.encode('ascii')).digest(), node))
    return tuple(tuple(ordered[m::4]) for m in range(4))


def negative_indices(size, generator):
    """Exactly32 other reconstructions, uniform without replacement per anchor."""
    if size < 33:
        raise ValueError('Every masked route needs at least33 eligible anchors')
    # Floyd sampling on [0,size-2] avoids a full quadratic random-key matrix.
    rows = []
    for anchor in range(size):
        selected = set()
        for high in range(size-1-32, size-1):
            draw = generator.randrange(high+1)
            selected.add(high if draw in selected else draw)
        rows.append(sorted(value+(value >= anchor) for value in selected))
    return torch.tensor(rows, dtype=torch.long)


def core_loss(reconstruction, targets, indices, chunk=512):
    """CORE Eq12 finite-T instance contrast; negative z_k, never raw x_k."""
    if reconstruction.shape != targets.shape or reconstruction.ndim != 2 or len(targets) < 33:
        raise ValueError('Complete matched eligible masked vectors required')
    if targets.requires_grad or indices.shape != (len(targets), 32):
        raise ValueError('Frozen feature target and exact32-negative identities required')
    z = F.normalize(reconstruction, dim=-1, eps=1e-8)
    x = F.normalize(targets, dim=-1, eps=1e-8)
    total = z.sum()*0
    # Gradients pass through anchor z_i AND all gathered negative z_k.
    for start in range(0, len(z), chunk):
        end = min(start+chunk, len(z)); anchor = z[start:end]
        positive = (anchor*x[start:end]).sum(-1, keepdim=True)
        negative = (anchor[:, None, :]*z[indices[start:end]]).sum(-1)
        scores = torch.cat((positive, negative), dim=1)/.2
        target = torch.zeros(end-start, dtype=torch.long, device=z.device)
        total = total+F.cross_entropy(scores, target, reduction='sum')
    return total/len(z)


def verify_auxiliary_edges(original, rewired, n):
    """Rewire input is prepared/frozen separately; graph cannot be outcome-selected."""
    def edge_set(value):
        if value.dtype != torch.long or value.ndim != 2 or value.shape[0] != 2:
            raise ValueError('Exact long edge_index required')
        pairs = [tuple(row) for row in value.detach().cpu().t().tolist()]
        if len(set(pairs)) != len(pairs) or any(min(a,b) < 0 or max(a,b) >= n for a,b in pairs):
            raise ValueError('Simple ordered undirected input without duplicates required')
        items = set(pairs)
        if any((b,a) not in items for a,b in items):
            raise ValueError('Undirected source and auxiliary graphs required')
        return items
    old, new = edge_set(original), edge_set(rewired)
    if old == new or {p for p in old if p[0] == p[1]} != {p for p in new if p[0] == p[1]}:
        raise ValueError('Nontrivial rewire with all original self edges preserved required')
    def degrees(items):
        result = [0]*n
        for src,dst in items:
            if src != dst: result[src] += 1
        return result
    if degrees(old) != degrees(new):
        raise ValueError('Every public node degree must be preserved')


class DropoutStreams:
    """Persistent independent member/view RNG; outer caller state is restored."""
    def __init__(self, seed, device):
        self.device = device
        self.cuda_devices = [] if device.type != 'cuda' else [device.index]
        self.states = {}
        # Factual slots0..3; auxiliary slots0..3. Single uses all4 aux slots.
        for member in range(4):
            for view in ('factual', 'masked'):
                value = seed+300001+1009*member+(0 if view == 'factual' else 500009)
                with seeded(value, device):
                    self.states[(member, view)] = dict(cpu=torch.get_rng_state(),
                        cuda=None if not self.cuda_devices else torch.cuda.get_rng_state(device.index))

    @contextmanager
    def use(self, member, view):
        state = self.states[(member, view)]
        with torch.random.fork_rng(devices=self.cuda_devices):
            torch.set_rng_state(state['cpu'])
            if state['cuda'] is not None: torch.cuda.set_rng_state(state['cuda'], self.device.index)
            yield
            state['cpu'] = torch.get_rng_state()
            if state['cuda'] is not None: state['cuda'] = torch.cuda.get_rng_state(self.device.index)


class Session:
    """One declared condition, ordinary old-state accumulation then optimizer step.

    This surface does not own a validation selector, fit horizon, checkpoint
    campaign, data exporter, science release, worker or serving score call.
    """
    def __init__(self, name, seed, x, edges, train_ids, train_y, device='cpu', auxiliary_edges=None):
        self.spec = condition(name); self.name, self.seed = name, seed
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            self.device = torch.device('cuda', 0 if self.device.index is None else self.device.index)
        if self.device.type not in ('cpu', 'cuda'): raise ValueError('Ordinary CPU/CUDA runtime only')
        if x.dtype != torch.float32 or x.shape != (19717,500) or x.requires_grad or not torch.isfinite(x).all():
            raise ValueError('Complete finite public PubMed float32 normalized features required')
        if train_ids.dtype != torch.long or train_y.dtype != torch.long or train_ids.ndim != 1 or train_y.shape != train_ids.shape:
            raise ValueError('Only complete projected TRAIN identities/labels required')
        if len(train_ids) == 0 or train_ids.unique().numel() != len(train_ids) or train_ids.min() < 0 or train_ids.max() >= len(x):
            raise ValueError('Nonempty unique TRAIN identities in full graph required')
        if train_y.min() < 0 or train_y.max() >= 3: raise ValueError('Native3 class TRAIN targets required')
        if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] == 0:
            raise ValueError('Complete nonempty long PubMed edge_index required')
        if edges.min() < 0 or edges.max() >= len(x): raise ValueError('Every edge endpoint must be in the complete graph')
        self.x, self.edges = x.to(self.device), edges.to(self.device)
        self.train_ids, self.train_y = train_ids.to(self.device), train_y.to(self.device)
        self.factors = factors_module(); self.bodies = nn.ModuleList()
        for member in range(self.spec['bodies']):
            # Member0 exactly matches single/native seed; no best-member reuse.
            body = construct(NATIVE_MONO, seed+1009*member, self.device)
            if self.spec['shared_BE']:
                self.factors.install_factors(body, 4)
                body.to(self.device)  # New factor vectors were constructed on CPU.
            if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in body.modules()):
                raise ValueError('This native provider uses LayerNorm only')
            self.bodies.append(body)
        self.decoder = None
        if self.spec['core']:
            with seeded(seed+900001, self.device): self.decoder = nn.Linear(256,500).to(self.device)
        self.optimizers = [self._optimizer(body) for body in self.bodies]
        if self.decoder is not None:
            self.optimizers.append(torch.optim.Adam(self.decoder.parameters(), lr=.005, weight_decay=.001, eps=1e-8))
        self.streams = DropoutStreams(seed, self.device)
        self.negative_generators = [random.Random(seed+700001+1009*m) for m in range(4)]
        self.ownership_generator = random.Random(seed+800001)
        self.counters = dict(updates=0, factual_forwards=0, masked_forwards=0, backwards=0,
                             optimizer_steps=0, serving_forwards=0, preprocessing_banks=0)
        self.preparation_seconds = 0.; started = time.monotonic()
        eligible = torch.nonzero(self.x.norm(dim=-1) > 0, as_tuple=False).flatten().cpu().tolist()
        self.partitions = balanced_hash_partition(eligible)
        if any(len(ids) < 33 for ids in self.partitions): raise ValueError('Four complete32-negative anchor populations required')
        self.anchor_ids = [torch.tensor(ids, dtype=torch.long, device=self.device) for ids in self.partitions]
        self.factual = self._view(self.x, self.edges)
        self.masked = []
        if self.spec['masked']:
            aux = self.edges
            if self.spec['auxiliary_rewire']:
                if auxiliary_edges is None: raise ValueError('Separately frozen degree-preserving auxiliary graph required')
                verify_auxiliary_edges(self.edges, auxiliary_edges, len(x)); aux = auxiliary_edges.to(self.device)
            elif auxiliary_edges is not None:
                raise ValueError('Auxiliary graph changes forbidden in this condition')
            for ids in self.anchor_ids:
                masked_x = self.x.clone(); masked_x[ids] = 0
                # All native polynomial tokens are recomputed from masked X.
                self.masked.append(self._view(masked_x, aux))
        elif auxiliary_edges is not None:
            raise ValueError('No auxiliary graph in own-only controls')
        self.preparation_seconds = time.monotonic()-started

    def _optimizer(self, body):
        groups = []
        for name, parameter in body.named_parameters():
            attention = 'attnmodule' in name
            groups.append(dict(params=[parameter], lr=.0005 if attention else .005,
                               weight_decay=1e-8 if attention else .001))
        return torch.optim.Adam(groups, eps=1e-8)

    def _view(self, x, edges):
        banks = native_preprocess(x, edges, K=2)
        self.counters['preprocessing_banks'] += 1
        return SimpleNamespace(list_mat=banks)

    def _forward(self, member, view, quarter=None):
        body = self.bodies[member if self.spec['untied'] else 0]
        captured = []
        handle = body.lin3.register_forward_pre_hook(lambda module, arguments: captured.append(arguments[0]))
        try:
            with self.factors.member_context(body, member if self.spec['shared_BE'] else 0):
                stream_member = member if view == 'factual' or self.spec['members'] == 4 else quarter
                with self.streams.use(stream_member, view):
                    logits = body(self.factual if view == 'factual' else self.masked[quarter])
            if len(captured) != 1 or logits.shape != (19717,3) or captured[0].shape != (19717,256):
                raise ValueError('Complete native fullgraph logits/prehead representation required')
            if not torch.isfinite(logits).all() or not torch.isfinite(captured[0]).all():
                raise FloatingPointError('Nonfinite complete forward')
            self.counters[view+'_forwards'] += 1
            return logits, captured[0]
        finally:
            handle.remove()

    def _masked_terms(self, member, quarter, audit=False):
        logits, hidden = self._forward(member, 'masked', quarter)
        ce = F.cross_entropy(logits[self.train_ids], self.train_y)
        core = ce*0
        if self.decoder is not None:
            ids = self.anchor_ids[quarter]
            reconstruction = self.decoder(hidden[ids])
            negatives = negative_indices(len(ids), self.negative_generators[member if self.spec['members'] == 4 else quarter]).to(self.device)
            core = core_loss(reconstruction, self.x[ids], negatives)
            if audit: self.audit = self._auxiliary_audit(core)
        return ce, core

    def _auxiliary_audit(self, core):
        # Inspection only; optimizer gradients still come solely from backwards.
        body = self.bodies[0]; params = list(body.named_parameters())
        groups = dict(upstream=[], factors=[], head=[], decoder=list(self.decoder.parameters()))
        for name, p in params:
            if name.startswith('lin3.'): groups['head'].append(p)
            elif name.endswith(('.r', '.s')): groups['factors'].append(p)
            else: groups['upstream'].append(p)
        flat = [p for ps in groups.values() for p in ps]
        gradients = torch.autograd.grad(core, flat, retain_graph=True, allow_unused=True)
        result = {}; offset = 0
        for name, ps in groups.items():
            subset = gradients[offset:offset+len(ps)]; offset += len(ps)
            if any(g is not None and not torch.isfinite(g).all() for g in subset): raise FloatingPointError('Auxiliary audit gradient')
            result[name] = dict(parameters=len(ps), nonzero_gradients=sum(g is not None and bool(g.abs().sum() > 0) for g in subset))
        if result['upstream']['nonzero_gradients'] == 0 or result['decoder']['nonzero_gradients'] == 0:
            raise ValueError('CORE must reach shared/upstream weights and common decoder')
        if self.spec['shared_BE'] and result['factors']['nonzero_gradients'] == 0:
            raise ValueError('CORE must reach live private factors')
        if result['head']['nonzero_gradients'] != 0: raise ValueError('Reconstruction cannot reach classifier head')
        result['upstream_is_shared_BE_body'] = self.spec['shared_BE']
        return result

    def train_step(self, audit=False):
        for body in self.bodies: body.train()
        if self.decoder is not None: self.decoder.train()
        for optimizer in self.optimizers: optimizer.zero_grad(set_to_none=True)
        ownership = list(range(4))
        if self.spec['ownership_shuffle']: self.ownership_generator.shuffle(ownership)
        metrics = dict(factual_ce=0., masked_ce=0., core=0., scalar_objective=0.)
        # This decomposition differentiates the ordinary scalar sum at one
        # parameter state. No block gradient filtering or early optimizer step.
        scale = 1./self.spec['members'] if not self.spec['untied'] else 1.
        for member in range(self.spec['members']):
            logits, _ = self._forward(member, 'factual')
            factual = F.cross_entropy(logits[self.train_ids], self.train_y)
            total = factual*scale
            metrics['factual_ce'] += float(factual.detach())/self.spec['members']
            if self.spec['masked'] and not self.spec['all_view_single']:
                ce, core = self._masked_terms(member, ownership[member], audit=audit and member == 0)
                total = total+scale*(.5*ce+.1*core)
                metrics['masked_ce'] += float(ce.detach())/4; metrics['core'] += float(core.detach())/4
            if not torch.isfinite(total): raise FloatingPointError('Scalar objective')
            total.backward(); self.counters['backwards'] += 1
            metrics['scalar_objective'] += float(total.detach())
        if self.spec['all_view_single']:
            for quarter in range(4):
                ce, core = self._masked_terms(0, quarter, audit=audit and quarter == 0)
                total = (.5*ce+.1*core)/4
                if not torch.isfinite(total): raise FloatingPointError('Single all-view objective')
                total.backward(); self.counters['backwards'] += 1
                metrics['masked_ce'] += float(ce.detach())/4; metrics['core'] += float(core.detach())/4
                metrics['scalar_objective'] += float(total.detach())
        # Untied bodies retain native unscaled own gradients. The common
        # decoder, like its counterpart in BE4/all-view single, learns the
        # mean reconstruction objective across the four participating routes.
        decoder_gradient_divisor = self.spec['members'] if self.spec['untied'] and self.decoder is not None else 1
        if decoder_gradient_divisor != 1:
            for parameter in self.decoder.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(decoder_gradient_divisor)
        active = [p for body in self.bodies for p in body.parameters() if p.grad is not None]
        if self.decoder is not None: active += [p for p in self.decoder.parameters() if p.grad is not None]
        if not active or any(not torch.isfinite(p.grad).all() for p in active): raise FloatingPointError('Accumulated gradient')
        for optimizer in self.optimizers: optimizer.step(); self.counters['optimizer_steps'] += 1
        self.counters['updates'] += 1
        parameters = [p for body in self.bodies for p in body.parameters()]
        if self.decoder is not None: parameters += list(self.decoder.parameters())
        if any(not torch.isfinite(p).all() for p in parameters): raise FloatingPointError('Parameter state')
        for optimizer in self.optimizers:
            if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
                   for state in optimizer.state.values() for value in state.values()):
                raise FloatingPointError('Optimizer state')
        return dict(**metrics, ownership=ownership, TRAIN_label_count=len(self.train_ids),
                    complete_graph_nodes=len(self.x), decoder_data_gradient_divisor=decoder_gradient_divisor,
                    gradient_assignment='ordinary old-state scalar backwards; common untied decoder uses route-mean data gradient')

    def factual_probabilities(self):
        """Factual serving with exact module-mode restoration on every exit."""
        modes = [(module, module.training) for body in self.bodies for module in body.modules()]
        try:
            for body in self.bodies: body.eval()
            logits = []
            with torch.no_grad():
                for member in range(self.spec['members']):
                    value, _ = self._forward(member, 'factual'); logits.append(value)
                    self.counters['serving_forwards'] += 1
            bank = torch.stack(logits)
            return bank.softmax(-1).mean(0), bank
        finally:
            # Direct flags preserve callers' mixed submodule modes too.
            for module, previous in modes: module.training = previous
