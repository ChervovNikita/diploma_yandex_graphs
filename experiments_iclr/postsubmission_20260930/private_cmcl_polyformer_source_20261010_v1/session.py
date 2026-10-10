"""Closed full-graph FP32 native session: current scores, exact owned replay, VJPs."""
from contextlib import contextmanager
import math
from types import ModuleType

from .caps import CLOSED
from .native import require, source_path, build, seeded
from .objective import node_terms, assignment, member_terms
from .plan import MEMBERS, CLASSES, SEEDS, NATIVE, condition
from .gate import source_gate


def state_helpers(caps=CLOSED):
    caps.require('source_bound', 'runtime')
    path = source_path('native_state', caps)
    module = ModuleType('_private_cmcl_exact_state_helpers')
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    return module


def gradient_stamp(parameters):
    return tuple((id(p.grad), None if p.grad is None else int(p.grad._version)) for p in parameters)


class Session:
    def __init__(self, name, seed, inputs, device, costs, identity, caps=CLOSED):
        caps.require('source_bound', 'model', 'data', 'runtime')
        require(name in ('own_floor', 'private_cmcl', 'all_block_cmcl', 'vanilla_cmcl', 'private_uniform', 'private_constant_credit')
                and seed in SEEDS and isinstance(identity, dict) and identity, 'One declared condition/seed and root-bound identity')
        source_gate(identity, caps)
        import torch
        import numpy as np
        self.torch, self.np, self.caps = torch, np, caps
        self.device, self.costs, self.inputs = torch.device(device), costs, inputs
        self.spec, self.seed, self.identity = condition(name), seed, identity
        if self.device.type == 'cuda':
            self.device = torch.device('cuda', 0 if self.device.index is None else self.device.index)
        require(self.device.type in ('cuda', 'cpu') and inputs.split_seed == 190111
                and inputs.counts['nodes'] == 19717 and inputs.counts['classes'] == 3,
                'Full declared public PubMed role/graph interface')
        require(identity['input_custody_sha256'] == inputs.input_custody_sha256
                and identity['TRAIN_custody_sha256'] == inputs.train.custody_sha256
                and identity['VALID_custody_sha256'] == inputs.validation.custody_sha256,
                'Exact root-bound actual feature/edge and isolated role custody')
        require(all(x.device == self.device and x.shape == (19717, 500) and x.dtype == torch.float32
                for x in inputs.factual.list_mat), 'Full fixed native feature banks on the model device')
        self.state = state_helpers(caps)
        with costs.measure('native_constructor_full23_site_BE_and_Adam'):
            self.model, self.factors, self.shared, self.private = build(seed, self.device, caps)
            self.parameters = tuple(self.model.parameters())
            self.private_ids = {id(p) for p in self.private}
            groups = [dict(params=[p], lr=.0005 if 'attnmodule' in n else .005,
                           weight_decay=1e-8 if 'attnmodule' in n else .001)
                      for n, p in self.model.named_parameters()]
            self.optimizer = torch.optim.Adam(groups, eps=1e-8)
        self.streams = []
        for member in range(MEMBERS):
            with seeded(seed+300001+1009*member, self.device, caps):
                self.streams.append(self.capture_stream())
        self.counters = dict(updates=0, score_fullgraph_forwards=0, gradient_fullgraph_forwards=0,
                             own_gradient_VJPs=0, auxiliary_gradient_VJPs=0,
                             combined_gradient_VJPs=0, optimizer_steps=0,
                             serving_fullgraph_forwards=0)
        self.cache_guard = self.cache_stamp()
        self.verify_optimizer()

    def capture_stream(self):
        torch = self.torch
        return dict(cpu=torch.get_rng_state().clone(), cuda=None if self.device.type != 'cuda'
                    else torch.cuda.get_rng_state(self.device.index).clone())

    @contextmanager
    def use_stream(self, stream):
        torch = self.torch
        devices = [] if self.device.type != 'cuda' else [self.device.index]
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(stream['cpu'])
            if stream['cuda'] is not None:
                torch.cuda.set_rng_state(stream['cuda'], self.device.index)
            yield

    def cache_stamp(self):
        return tuple((id(v), int(v._version), tuple(v.shape), str(v.device), str(v.dtype)) for v in
            (*self.inputs.factual.list_mat, self.inputs.train.ids, self.inputs.train.targets,
             self.inputs.validation.ids, self.inputs.validation.targets))

    def verify_optimizer(self):
        require(len({id(p) for p in self.parameters}) == len(self.parameters)
                and {id(p) for p in self.shared}.isdisjoint(self.private_ids)
                and {id(p) for p in self.parameters} == {id(p) for p in (*self.shared, *self.private)}, 'Complete disjoint gradient blocks')
        actual = [p for g in self.optimizer.param_groups for p in g['params']]
        require(len(actual) == len(self.parameters) and {id(p) for p in actual} == {id(p) for p in self.parameters}, 'One deduplicated Adam owner for every parameter')
        for (name, p), group in zip(self.model.named_parameters(), self.optimizer.param_groups):
            require(group['params'][0] is p and group['lr'] == (.0005 if 'attnmodule' in name else .005)
                    and group['weight_decay'] == (1e-8 if 'attnmodule' in name else .001)
                    and group['betas'] == (.9, .999) and group['eps'] == 1e-8
                    and group['amsgrad'] is False and group['maximize'] is False, 'Unchanged published optimizer grouping and Adam defaults')

    def forward(self, member):
        with self.factors.member_context(self.model, member):
            result = self.model(self.inputs.factual)
        require(result.shape == (19717, CLASSES) and result.dtype == self.torch.float32
                and self.torch.isfinite(result).all().item(), 'Complete finite full-native FP32 categorical logits')
        return result

    def accumulate(self, gradients, parameters):
        torch = self.torch
        for gradient, p in zip(gradients, parameters):
            require(gradient is not None and gradient.shape == p.shape and torch.isfinite(gradient).all().item(), 'Finite complete parameter-only VJP')
            value = gradient.detach()
            if p.grad is None:
                p.grad = value.clone()
            else:
                p.grad.add_(value)

    def step(self, *, numerical_qualification=False):
        self.caps.require('source_bound', 'model', 'data', 'runtime')
        if not numerical_qualification or self.counters['updates']:
            self.caps.require('scientific')
        torch = self.torch
        require(self.cache_stamp() == self.cache_guard, 'Immutable native feature/role/target input caches')
        self.model.train()
        before_global = self.state.capture_rng(self.np, torch)
        self.optimizer.zero_grad(set_to_none=True)
        owners, replay, scored, after = None, [], [], []
        with self.costs.measure('full_graph_current_CMCL_scores_and_streamed_gradient_update') as cost:
            if self.spec['assignment']:
                ces, kls = [], []
                # First pass holds no tapes; replay uses the exact SAME incoming
                # parameters, native TRAIN dropout draws and complete graph.
                for member in range(MEMBERS):
                    stream = self.state.cpu_tree(torch, self.streams[member])
                    replay.append(stream)
                    with self.use_stream(stream), torch.no_grad():
                        logits = self.forward(member)[self.inputs.train.ids]
                        ce, kl = node_terms(torch, logits, self.inputs.train.targets, self.caps)
                        scored.append(logits.detach().clone())
                        ces.append(ce.detach()); kls.append(kl.detach())
                        after.append(self.capture_stream())
                    self.counters['score_fullgraph_forwards'] += 1
                owners = assignment(torch, torch.stack(ces), torch.stack(kls), self.caps)
                require(not owners.requires_grad, 'Discrete ownership stopped; no fitted initial router')
                del ces, kls
            losses = []
            for member in range(MEMBERS):
                incoming = replay[member] if self.spec['assignment'] else self.streams[member]
                with self.use_stream(incoming):
                    output = self.forward(member)[self.inputs.train.ids]
                    if self.spec['assignment']:
                        require(torch.allclose(output.detach(), scored[member], atol=1e-5, rtol=1e-5),
                                'FP32 same-incoming dropout replay within declared tolerance')
                    ce, kl = node_terms(torch, output, self.inputs.train.targets, self.caps)
                    floor, aux = member_terms(torch, ce, kl, self.spec,
                                              None if owners is None else owners[member], self.caps)
                    self.counters['gradient_fullgraph_forwards'] += 1
                    if self.spec['private_auxiliary_only']:
                        # Common own CE reaches ALL original weights and factors.
                        own_grad = torch.autograd.grad(floor, self.parameters, retain_graph=True, allow_unused=False)
                        self.accumulate(own_grad, self.parameters)
                        self.counters['own_gradient_VJPs'] += 1
                        before = gradient_stamp(self.parameters)
                        extra = torch.autograd.grad(aux, self.private, retain_graph=False, allow_unused=False)
                        require(gradient_stamp(self.parameters) == before, 'Auxiliary VJP does not write any current grad field')
                        require(all(g[torch.arange(MEMBERS, device=g.device) != member].count_nonzero().item() == 0
                                    for g in extra), 'Private auxiliary credit reaches only the current member factor rows')
                        self.accumulate(extra, self.private)
                        self.counters['auxiliary_gradient_VJPs'] += 1
                        del own_grad, extra
                    else:
                        total = aux if floor is None else floor if aux is None else floor+aux
                        combined = torch.autograd.grad(total, self.parameters, allow_unused=False)
                        self.accumulate(combined, self.parameters)
                        self.counters['combined_gradient_VJPs'] += 1
                        del combined, total
                    outgoing = self.capture_stream()
                    if self.spec['assignment']:
                        require(self.state.exact(torch, outgoing, after[member]), 'Exact score/gradient replay RNG post-state')
                        self.streams[member] = after[member]
                    else:
                        self.streams[member] = outgoing
                    losses.append(dict(member=member, own_CE=float(ce.detach().mean().item()),
                                       uniform_KL=float(kl.detach().mean().item()),
                                       owners=None if owners is None else int(owners[member].sum().item()),
                                       floor=None if floor is None else float(floor.detach().item()),
                                       auxiliary=None if aux is None else float(aux.detach().item())))
                    del output, ce, kl, floor, aux
            require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in self.parameters), 'One finite complete simultaneous Adam data gradient')
            self.optimizer.step()
            self.counters['optimizer_steps'] += 1
            self.counters['updates'] += 1
            cost.update(fullgraph_nodes=19717, complete_TRAIN_targets_per_member=len(self.inputs.train.ids),
                score_forwards_this_update=4 if self.spec['assignment'] else 0, gradient_forwards_this_update=4,
                private_only_extra_credit=self.spec['private_auxiliary_only'],
                shared_gradient_is_current_own_data_gradient_only=self.spec['private_auxiliary_only'] or self.spec['name'] == 'own_floor',
                Adam_decay_and_moments_remain_live=True, FP32_no_AMP=True)
        require(self.state.exact(torch, self.state.capture_rng(self.np, torch), before_global)
                and self.cache_stamp() == self.cache_guard, 'Forward/update preserves caller RNG and fixed factual inputs')
        self.verify_optimizer()
        return dict(member_terms=losses, counters=dict(self.counters),
                    owners_tie_rule='stable member-index0,1,2,3',
                    state_specific_gradient_guarantee_not_trajectory_or_descent_guarantee=True)

    def evaluate(self):
        torch = self.torch
        caller = self.state.capture_rng(self.np, torch)
        streams = self.state.cpu_tree(torch, self.streams)
        modes = tuple((m, m.training) for m in self.model.modules())
        try:
            self.model.eval()
            with self.costs.measure('complete_factual_M4_TRAIN_VALID_probability_serving'), torch.no_grad():
                full = torch.stack([self.forward(m).cpu() for m in range(MEMBERS)])
                self.counters['serving_fullgraph_forwards'] += MEMBERS
                result, outputs = {}, {}
                for name, role in (('TRAIN', self.inputs.train), ('VALID', self.inputs.validation)):
                    ids, truth = role.ids.cpu(), role.targets.cpu()
                    logits = full[:, ids]
                    logp = torch.log_softmax(logits, dim=-1)
                    pool_logp = torch.logsumexp(logp, dim=0)-math.log(MEMBERS)
                    prediction, member_prediction = pool_logp.argmax(dim=-1), logits.argmax(dim=-1)
                    correct = member_prediction == truth[None]
                    nll = -pool_logp.gather(1, truth[:, None]).mean().item()
                    acc = float((prediction == truth).float().mean().item())
                    class_accuracy, class_nll, f1 = [], [], []
                    for c in range(CLASSES):
                        selected = truth == c
                        tp = ((prediction == c) & selected).sum().item()
                        fp = ((prediction == c) & ~selected).sum().item()
                        fn = ((prediction != c) & selected).sum().item()
                        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
                        class_accuracy.append(float((prediction[selected] == truth[selected]).float().mean().item()))
                        class_nll.append(float(-pool_logp[selected, c].mean().item()))
                    wrong = prediction != truth
                    rival = prediction
                    z_y = logits.gather(2, truth[None, :, None].expand(MEMBERS, -1, 1)).squeeze(2)
                    z_r = logits.gather(2, rival[None, :, None].expand(MEMBERS, -1, 1)).squeeze(2)
                    strict_common_rival = wrong & (z_r > z_y).all(dim=0)
                    result[name] = dict(accuracy=acc, macro_F1=sum(f1)/CLASSES, NLL=float(nll),
                        class_accuracy=class_accuracy, class_NLL=class_nll,
                        member_accuracy=[float(x.float().mean().item()) for x in correct],
                        member_NLL=[float(-v.gather(1, truth[:, None]).mean().item()) for v in logp],
                        available_correct_nodes=int(correct.any(dim=0).sum().item()),
                        pooled_errors=int(wrong.sum().item()),
                        no_correct_member_errors=int((wrong & ~correct.any(dim=0)).sum().item()),
                        pool_lost_correct_alternatives=int((wrong & correct.any(dim=0)).sum().item()),
                        strict_common_rival_errors=int(strict_common_rival.sum().item()))
                    require(all(math.isfinite(x) for x in (acc, nll, *class_accuracy, *class_nll, *f1)), 'Finite complete competence/served-risk metrics')
                    outputs[name] = dict(ids=ids.tolist(), member_logits=logits.clone(),
                        pool_log_probabilities=pool_logp.clone(), prediction=prediction.clone(),
                        truth=truth.clone(), strict_common_rival=strict_common_rival.clone(),
                        fixed_rival=rival.clone())
                return result, outputs
        finally:
            for module, training in modes:
                module.training = training
            self.state.restore_rng(self.np, torch, caller)
            require(self.state.exact(torch, self.state.capture_rng(self.np, torch), caller)
                    and self.state.exact(torch, self.streams, streams)
                    and self.cache_stamp() == self.cache_guard, 'Evaluation preserves all owned/global streams and inputs')

    def snapshot(self, epoch, scores, outputs):
        torch = self.torch
        with self.costs.measure('selected_full_model_Adam_RNG_CPU_snapshot'):
            return dict(schema='private-CMCL-PolyFormer-selected-v1', identity=self.identity,
                condition=self.spec, seed=self.seed, epoch=epoch,
                input_custody_sha256=self.inputs.input_custody_sha256,
                TRAIN_custody_sha256=self.inputs.train.custody_sha256,
                VALID_custody_sha256=self.inputs.validation.custody_sha256,
                model=self.state.cpu_tree(torch, self.model.state_dict()),
                optimizer=self.state.cpu_tree(torch, self.optimizer.state_dict()),
                streams=self.state.cpu_tree(torch, self.streams), global_rng=self.state.capture_rng(self.np, torch),
                scores=scores, outputs=outputs, pooling='arithmetic mean member softmax probabilities')

    def restore(self, saved):
        torch = self.torch
        require(saved['schema'] == 'private-CMCL-PolyFormer-selected-v1' and saved['identity'] == self.identity
                and saved['condition'] == self.spec and saved['seed'] == self.seed
                and saved['input_custody_sha256'] == self.inputs.input_custody_sha256
                and saved['TRAIN_custody_sha256'] == self.inputs.train.custody_sha256
                and saved['VALID_custody_sha256'] == self.inputs.validation.custody_sha256, 'Exact selected source/method/role/input identity')
        self.model.load_state_dict(saved['model'], strict=True)
        self.optimizer.load_state_dict(saved['optimizer'])
        self.streams = self.state.cpu_tree(torch, saved['streams'])
        self.state.restore_rng(self.np, torch, saved['global_rng'])
        require(self.state.exact(torch, self.state.cpu_tree(torch, self.model.state_dict()), saved['model'])
                and self.state.exact(torch, self.state.cpu_tree(torch, self.optimizer.state_dict()), saved['optimizer'])
                and self.state.exact(torch, self.streams, saved['streams'])
                and self.state.exact(torch, self.state.capture_rng(self.np, torch), saved['global_rng']), 'Exact selected model/buffers/Adam/owned and caller RNG')
        self.verify_optimizer()
