"""Source-only binding hooks; call only after an exact finite numerical admission.

No data loading, worker, fit loop, numerical imports or source edits here.
The completed adapter remains disabled on disk; admission is scoped in memory.
"""
from contextlib import contextmanager


@contextmanager
def _admitted(hop, numerical_admitted):
    if numerical_admitted is not True:
        raise RuntimeError('Explicit reviewed numerical admission required')
    previous = hop.ENABLED
    hop.ENABLED = True
    try:
        yield
    finally:
        hop.ENABLED = previous


def fresh(method, hop, condition, seed, tensor, device='cuda:0', *, numerical_admitted=False):
    """Exact V3 Session plus the existing complete HopCredit update."""
    if condition not in hop.M4 + hop.M1:
        raise ValueError('One declared private-hop/M1 condition required')
    base = 'shared4_own' if condition in hop.M4 else 'single_native'
    with _admitted(hop, numerical_admitted):
        session = method.Session(base, seed, tensor['x'], tensor['edge_index'],
                                 tensor['train_ids'], tensor['train_y'], device=device)
        if condition in hop.M1:
            hop.prepare_factor1(session)
        credit = hop.HopCredit(session, condition)
    session.name = condition
    session._private_hop_credit = credit

    def train_step(audit=False):
        if audit:
            raise ValueError('No legacy CORE audit in the private-hop entry')
        with _admitted(hop, numerical_admitted):
            return credit.step()

    session.train_step = train_step
    return session


def expected_work(session, updates, evaluations=0):
    """Native counters: each factual_probabilities call pays members paths."""
    credit = session._private_hop_credit
    members = credit.members
    auxiliary = 3 if members == 4 or credit.condition == 'factor1_allview' else 0
    expected_credit = dict(updates=updates, factual_forwards=members*updates,
                           auxiliary_forwards=auxiliary*updates,
                           gradient_calls=(members+auxiliary)*updates,
                           optimizer_steps=updates)
    if credit.counters != expected_credit:
        raise ValueError('Exact complete HopCredit counters required')
    return dict(updates=updates, factual_forwards=members*(updates+evaluations),
                masked_forwards=auxiliary*updates, backwards=(members+auxiliary)*updates,
                optimizer_steps=updates, serving_forwards=members*evaluations,
                preprocessing_banks=1)


def independent_probabilities(sessions):
    """Pool four already own-selected complete M1 bodies, with no fitting."""
    if len(sessions) != 4 or any(s._private_hop_credit.members != 1 for s in sessions):
        raise ValueError('Four complete separately fit factorized M1 Sessions required')
    parameter_ids = [{id(p) for p in s.bodies[0].parameters()} for s in sessions]
    if (len({id(s.bodies[0]) for s in sessions}) != 4 or
            len({id(s.optimizers[0]) for s in sessions}) != 4 or
            any(parameter_ids[a] & parameter_ids[b] for a in range(4) for b in range(a))):
        raise ValueError('Genuine disjoint complete bodies/native Adams required')
    torch = sessions[0]._private_hop_credit.torch
    bank = torch.cat([s.factual_probabilities()[1] for s in sessions], dim=0)
    return bank.softmax(-1).mean(0), bank
