"""Conditional PNA residual channel under one later-frozen allocation policy."""
import math
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_degree_channel", HERE / "channel.py")
channel_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(channel_module)

POLICIES = ('O', 'I', 'P', 'G')
REVERSE_CALLS = {'O': 2, 'I': 2, 'P': 3, 'G': 3}  # G also collects the new shared scalar's own risk.
SUPERVISION = {
    'O': {'shared_own': 'own', 'boundary_private_own': 'own', 'internal_private': 'own'},
    'I': {'shared_own': 'own', 'boundary_private_own': 'own', 'internal_private': 'J'},
    'P': {'shared_own': 'own', 'boundary_private_own': 'J', 'internal_private': 'J'},
    'G': {'shared_own': 'J', 'boundary_private_own': 'J', 'internal_private': 'J'},
}

def validate(policy, lambda_value):
    if policy not in POLICIES:
        raise ValueError('Explicit O/I/P/G allocation policy required')
    if isinstance(lambda_value, bool) or not isinstance(lambda_value, (int, float)) or not math.isfinite(lambda_value) or not 0 <= lambda_value <= 1:
        raise ValueError('Explicit fixed finite Python lambda in [0,1] required; no source strength adoption')
    return float(lambda_value)

class CardinalityAdapter:
    """No model/optimizer creation; attach only to a fresh sealed shared Session."""
    def __init__(self, session, audited_adapter, train, *, kind, policy, lambda_value):
        self.lambda_value = validate(policy, lambda_value)
        # The unchanged original auditor enforces exact native/core bytes,
        # backbone/factor names/shapes, exhaustive roles, and native Adam groups.
        auditor = audited_adapter.PrivateSteeringAdapter(session, mode='gncl_private', lambda_value=self.lambda_value)
        self.session, self.policy = session, policy
        original = auditor.partition
        self.channel = channel_module.CardinalityChannel(session, train, kind=kind, base_partition=original)
        self.partition = {'all': self.channel.all, 'roles': self.channel.roles,
            'shared_own': original['shared_own'] + self.channel.new_shared,
            'boundary_own': original['boundary_own'],
            'internal_private': original['internal_private'] + self.channel.new_internal}
        self.degree_shared = self.channel.new_shared
        self.pool_supervision = audited_adapter.served_pool_supervision
        self.counters = {'two_view_updates': 0, 'Session_forward_calls': 0, 'member_forwards': 0,
            'autograd_grad_calls': 0, 'Adam_calls': 0, 'Adam_steps': 0}

    def metadata(self):
        return {'scope': 'conditional first-local cardinality channel under fixed allocation', 'mode': 'cardinality_residual',
            'policy': self.policy, 'lambda': self.lambda_value, 'lambda_used_by_supervision': self.policy != 'O',
            'lambda_learned': False, 'strength_adopted_by_source': False,
            'constructor_arm': self.session.arm, 'supervision_by_role': {**SUPERVISION[self.policy], 'degree_shared_own': 'own'},
            'auxiliary': 'unchanged .05 alignment A; gradients ONLY on audited internal phi in every policy',
            'permission_roles': dict(self.partition['roles']), 'degree_shared_risk': 'own ONLY',
            'channel': self.channel.metadata(), 'counters': dict(self.counters),
            'charged_reverse_calls_per_update': REVERSE_CALLS[self.policy], 'new_coefficient_parameters': True,
            'teacher_router_or_new_network': False, 'new_noncoefficient_parameters': False,
            'global_scalar_objective_claimed': False, 'novelty_or_quality_guarantee': False}

    def annotate_run(self, run):
        if 'cardinality_control' in run:
            raise ValueError('Do not overwrite an existing cardinality identity')
        return {**run, 'cardinality_control': self.metadata()}

    def train_step(self, batch, labels):
        session, policy = self.session, self.policy; torch = session.torch
        self.channel.check_inventory()
        if labels.device != session.device or not len(labels):
            raise ValueError('Original complete nonempty TRAIN batch on Session.device required')
        current = list(session.model.named_parameters(remove_duplicate=False))
        if len(current) != len(self.partition['all']) or any(name != old_name or parameter is not old_parameter
                for (name, parameter), (old_name, old_parameter) in zip(current, self.partition['all'])):
            raise ValueError('Parameter objects/permissions changed')
        selection, losses = session.core['selection'], session.core['objectives']
        selection.finite_state(session.model, session.optimizers); session.model.train()
        for optimizer in session.optimizers: optimizer.zero_grad(set_to_none=True)
        self.counters['Session_forward_calls'] += 1
        la, ha = session.forward(batch); self.counters['member_forwards'] += session.model.members
        self.counters['Session_forward_calls'] += 1
        lb, hb = session.forward(batch); self.counters['member_forwards'] += session.model.members
        for logits, representation in ((la, ha), (lb, hb)):
            selection.finite_predictions(logits, session.serving(logits))
            if logits.dtype != torch.float32 or not torch.isfinite(representation).all():
                raise FloatingPointError('Original finite float32 TRAIN forward required')
        own_a = losses.own_supervision(la, labels, session.task).mean()
        own_b = losses.own_supervision(lb, labels, session.task).mean()
        own = .5 * (own_a + own_b)
        size = min(len(labels), session.config['contrastive']['max_objects'])
        index = torch.linspace(0, len(labels) - 1, steps=size, device=labels.device).long()
        alignment = losses.alignment_loss(ha[:, index], hb[:, index], labels[index], session.task, identities=None)
        auxiliary = .05 * alignment
        pool = mixture = None
        if policy != 'O':
            pool_a = self.pool_supervision(la, labels, session.task, torch)
            pool_b = self.pool_supervision(lb, labels, session.task, torch)
            # Exact task member/pool mixture PER VIEW, then two-view average.
            mixture_a = (1 - self.lambda_value) * own_a + self.lambda_value * pool_a
            mixture_b = (1 - self.lambda_value) * own_b + self.lambda_value * pool_b
            mixture = .5 * (mixture_a + mixture_b)
            pool = .5 * (pool_a + pool_b)
        internal_loss = (own if policy == 'O' else mixture) + auxiliary
        if any(value is not None and not torch.isfinite(value) for value in (own, pool, mixture, alignment, internal_loss)):
            raise FloatingPointError('Nonfinite own/pool/mixture/alignment objective')
        all_rows, phi = self.partition['all'], self.partition['internal_private']
        theta, psi = self.partition['shared_own'], self.partition['boundary_own']

        reverse_before = self.counters['autograd_grad_calls']
        def collect(loss, rows, retain):
            self.counters['autograd_grad_calls'] += 1
            gradients = torch.autograd.grad(loss, tuple(parameter for _, parameter in rows),
                retain_graph=retain, create_graph=False, allow_unused=True)
            if any(value is not None and not torch.isfinite(value).all() for value in gradients):
                raise FloatingPointError('Nonfinite old-state gradient collection')
            return {id(parameter): value for (_, parameter), value in zip(rows, gradients)}

        def add(first, second):
            if first is None: return second
            if second is None: return first
            return first + second

        # Every reverse collection precedes the sole original Adam transition.
        # There is no upstream detach, theta-first update, or extra cost padding.
        if policy == 'O':
            supplied = collect(own, all_rows, True)
            aligned = collect(auxiliary, phi, False)
            for _, parameter in phi: supplied[id(parameter)] = add(supplied[id(parameter)], aligned[id(parameter)])
        elif policy == 'I':
            own_rows = [(name, parameter) for name, parameter in all_rows if self.partition['roles'][name] != 'internal_private']
            supplied = collect(own, own_rows, True)
            supplied.update(collect(internal_loss, phi, False))
        elif policy == 'P':
            supplied = collect(own, theta, True)
            supplied.update(collect(mixture, psi, True))
            supplied.update(collect(internal_loss, phi, False))
        else:  # G on original basis + member slots; NEW shared coefficient remains own.
            mixture_rows = [(name, parameter) for name, parameter in all_rows if self.partition['roles'][name] != 'degree_shared_own']
            supplied = collect(mixture, mixture_rows, True)
            aligned = collect(auxiliary, phi, True)
            for _, parameter in phi: supplied[id(parameter)] = add(supplied[id(parameter)], aligned[id(parameter)])
            supplied.update(collect(own, self.degree_shared, False))
        if set(supplied) != {id(parameter) for _, parameter in all_rows}:
            raise ValueError('Incomplete exhaustive block gradient assignment')
        if not any(value is not None for value in supplied.values()) or any(value is not None and not torch.isfinite(value).all() for value in supplied.values()):
            raise FloatingPointError('No active or nonfinite supplied block gradient')
        for _, parameter in all_rows:
            gradient = supplied[id(parameter)]
            parameter.grad = None if gradient is None else gradient.detach()
        for optimizer in session.optimizers:
            self.counters['Adam_calls'] += 1
            optimizer.step(); self.counters['Adam_steps'] += 1
        selection.finite_state(session.model, session.optimizers)
        session.steps += 1; self.counters['two_view_updates'] += 1
        return {'own_mean': own.detach(), 'pool_mean': None if pool is None else pool.detach(),
            'mixture_mean': None if mixture is None else mixture.detach(), 'alignment': alignment.detach(),
            'auxiliary': auxiliary.detach(), 'internal_loss': internal_loss.detach(),
            'charged_reverse_passes': self.counters['autograd_grad_calls'] - reverse_before,
            'extra_reverse_passes_over_source_shared_update': self.counters['autograd_grad_calls'] - reverse_before - 1,
            'member_forwards': 2 * session.model.members, 'Adam_steps': 1}
