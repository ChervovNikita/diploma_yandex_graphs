"""Exact V3 native Session plus one ordinary mean-four-dropout scalar update.

Import only after source, role, provider and external-owner admission.
"""
from source import load_v3


def fresh_single(condition, seed, tensor, device='cuda:0'):
    import torch
    from torch.nn import functional as F
    method = load_v3(numerical=True)['method']
    if condition not in ('single_native', 'single_mean4_dropout', 'independent4_own'):
        raise ValueError('Frozen reference condition required')

    class MeanFour(method.Session):
        def train_step(self, audit=False):
            if audit: raise ValueError('No auxiliary audit in a factual-only reference')
            self.bodies[0].train()
            optimizer = self.optimizers[0]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            # Four separate graphs, all at the same parameter state; one Adam.
            for view in range(4):
                logits, _ = self._forward(view, 'factual')
                ce = F.cross_entropy(logits[self.train_ids], self.train_y)
                if not torch.isfinite(ce): raise FloatingPointError('Factual CE')
                (ce/4).backward()
                self.counters['backwards'] += 1
                losses.append(float(ce.detach()))
            parameters = list(self.bodies[0].parameters())
            active = [p for p in parameters if p.grad is not None]
            if not active or any(not torch.isfinite(p.grad).all() for p in active):
                raise FloatingPointError('Complete native mean-four gradient')
            optimizer.step()
            self.counters['optimizer_steps'] += 1
            self.counters['updates'] += 1
            if any(not torch.isfinite(p).all() for p in parameters):
                raise FloatingPointError('Native parameter state')
            if any(isinstance(v,torch.Tensor) and not torch.isfinite(v).all()
                   for state in optimizer.state.values() for v in state.values()):
                raise FloatingPointError('Native Adam state')
            return dict(factual_ce=sum(losses)/4, factual_view_CE=losses,
                        scalar_objective=sum(losses)/4, TRAIN_label_count=len(self.train_ids),
                        complete_graph_nodes=len(self.x), mean_four_old_state_CE=True,
                        probability_pool_CE=False, Adam_steps=1)

    cls = MeanFour if condition == 'single_mean4_dropout' else method.Session
    session = cls('single_native', seed, tensor['x'], tensor['edge_index'],
                  tensor['train_ids'], tensor['train_y'], device=device)
    assert_native(session, 1)
    return session


def assembled_I4(seed, tensor, selected_states, device='cuda:0'):
    method = load_v3(numerical=True)['method']
    if len(selected_states) != 4: raise ValueError('Four individually selected states required')
    session = method.Session('independent4_native', seed, tensor['x'], tensor['edge_index'],
                             tensor['train_ids'], tensor['train_y'], device=device)
    assert_native(session, 4)
    for body,state in zip(session.bodies, selected_states): body.load_state_dict(state, strict=True)
    # No optimizer step, pooled VALID selection, router, factor, fit or calibration.
    return session


def assert_native(session, expected):
    if len(session.bodies) != expected or len(session.optimizers) != expected or session.decoder is not None:
        raise ValueError('Complete private native bodies/Adams and no decoder required')
    ids = [set(id(p) for p in body.parameters()) for body in session.bodies]
    if any(ids[a] & ids[b] for a in range(expected) for b in range(a)):
        raise ValueError('Native body parameters cannot be shared')
    for body,optimized in zip(session.bodies,session.optimizers):
        if sum(p.numel() for p in body.parameters()) != 2069875:
            raise ValueError('Exact native predictor parameter count required')
        if any(isinstance(m,session.factors.FactorLinear) for m in body.modules()):
            raise ValueError('Reference native bodies cannot contain factor wrappers')
        if {id(p) for g in optimized.param_groups for p in g['params']} != {id(p) for p in body.parameters()}:
            raise ValueError('A private Adam must optimize exactly its own native body')
