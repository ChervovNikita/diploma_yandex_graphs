"""GNCL/NCL and mixed-block prior adaptation; no adoption or new parameters."""
import math
from .permissions import partition_parameters


def served_pool_supervision(logits, labels, task, torch):
    """Exact declared task pool, one stochastic view, all TRAIN labels."""
    if logits.ndim != 3 or logits.shape[1] != len(labels) or not len(labels):
        raise ValueError('Member/object/output axes and complete nonempty label rows required')
    functional = torch.nn.functional
    if task == 'wikics':
        if labels.dtype != torch.long or labels.ndim != 1:
            raise ValueError('Original integer WikiCS targets required')
        # Stable -log(mean_m softmax(z_m)_y), with no epsilon/clamp or
        # substitution of softmax(mean logits). Views are never mixed here.
        member_log_probability = functional.log_softmax(logits, dim=-1)
        true_log_probability = member_log_probability.gather(-1, labels[None, :, None].expand(logits.shape[0], -1, 1)).squeeze(-1)
        return -(torch.logsumexp(true_log_probability, dim=0) - math.log(logits.shape[0])).mean()
    if task not in ('collab', 'molhiv') or logits.shape[-1] != 1:
        raise ValueError('Audited single-output binary backbone required')
    target = labels.reshape(-1, 1).to(logits)
    if not torch.isfinite(target).all() or not ((target == 0) | (target == 1)).all():
        raise ValueError('All original finite binary TRAIN labels required')
    pooled_logits = logits.mean(dim=0)
    losses = functional.binary_cross_entropy_with_logits(pooled_logits, target, reduction='none')
    if task == 'collab':
        positive, negative = target[:, 0] == 1, target[:, 0] == 0
        if not positive.any() or not negative.any():
            raise ValueError('Both Collab target groups required')
        return losses[positive].mean() + losses[negative].mean()
    return losses.mean()


class PrivateSteeringAdapter:
    """Wrapper of a fresh publicV2 shared Session; no model/optimizer creation.

    theta/shared and psi/input-score boundary factors receive exact mean-own.
    Only audited existing phi/internal factors receive the selected private
    objective. Both gradient collections precede one original Adam transition.
    """
    MODES = ('alignment_private', 'hidden_private', 'gncl_private')

    def __init__(self, session, *, mode, lambda_value=None):
        if mode not in self.MODES:
            raise ValueError('Explicit matched private-steering mode required')
        if mode == 'gncl_private':
            if isinstance(lambda_value, bool) or not isinstance(lambda_value, (int, float)) or not math.isfinite(lambda_value) or not 0 <= lambda_value <= 1:
                raise ValueError('GNCL lambda must be an explicit fixed finite Python scalar in [0,1]')
            lambda_value = float(lambda_value)
        elif lambda_value is not None:
            raise ValueError('Reference modes take no GNCL lambda')
        if session.steps != 0:
            raise ValueError('Attach to a fresh prospective Session; no warm acquisition or running-family edit')
        self.session, self.mode, self.lambda_value = session, mode, lambda_value
        self.partition = partition_parameters(session)
        config = session.config['contrastive']
        if config != {'alignment_weight': .05, 'max_objects': 512, 'residual_weight': .05, 'temperature': .2}:
            raise ValueError('Original fixed alignment/residual source recipe required')
        self.counters = {'two_view_updates': 0, 'member_forwards': 0, 'autograd_grad_calls': 0, 'Adam_steps': 0}

    def metadata(self):
        return {'scope': 'prospective attributed private-steering control', 'mode': self.mode,
                'lambda': self.lambda_value, 'lambda_learned': False, 'strength_adopted_by_source': False,
                'constructor_arm': self.session.arm, 'shared_gradient': 'mean-own',
                'boundary_private_gradient': 'mean-own', 'internal_private_gradient': self.mode,
                'permission_roles': dict(self.partition['roles']), 'counters': dict(self.counters),
                'teacher_router_new_parameters': False, 'novelty_or_quality_guarantee': False}

    def annotate_run(self, run):
        """Pass this run identity to unchanged publicV2 joint_snapshot()."""
        if 'private_steering_control' in run:
            raise ValueError('Do not overwrite an existing method identity')
        return {**run, 'private_steering_control': self.metadata()}

    def train_step(self, batch, labels):
        session = self.session; torch = session.torch
        if labels.device != session.device or not len(labels):
            raise ValueError('Original complete nonempty TRAIN batch on Session.device required')
        current = [(name, parameter) for name, parameter in session.model.named_parameters(remove_duplicate=False)]
        if len(current) != len(self.partition['all']) or any(name != old_name or parameter is not old_parameter for (name, parameter), (old_name, old_parameter) in zip(current, self.partition['all'])):
            raise ValueError('Model parameter objects/permissions changed')
        session.core['selection'].finite_state(session.model, session.optimizers)
        session.model.train()
        for optimizer in session.optimizers:
            optimizer.zero_grad(set_to_none=True)
        la, ha = session.forward(batch); lb, hb = session.forward(batch)
        self.counters['member_forwards'] += 2 * session.model.members
        selection = session.core['selection']; losses = session.core['objectives']
        for logits, representation in ((la, ha), (lb, hb)):
            selection.finite_predictions(logits, session.serving(logits))
            if logits.dtype != torch.float32 or not torch.isfinite(representation).all():
                raise FloatingPointError('Original finite float32 TRAIN forward required')
        mean_a = losses.own_supervision(la, labels, session.task).mean()
        mean_b = losses.own_supervision(lb, labels, session.task).mean()
        own_mean = .5 * (mean_a + mean_b)
        size = min(len(labels), session.config['contrastive']['max_objects'])
        index = torch.linspace(0, len(labels)-1, steps=size, device=labels.device).long()
        identity = batch['query'][index] if session.task == 'collab' else None
        alignment = losses.alignment_loss(ha[:, index], hb[:, index], labels[index], session.task, identities=identity)
        residual = None; pool_mean = None
        if self.mode == 'gncl_private':
            pool_a = served_pool_supervision(la, labels, session.task, torch)
            pool_b = served_pool_supervision(lb, labels, session.task, torch)
            # Mixture per stochastic view BEFORE the two-view average.
            mix_a = (1-self.lambda_value)*mean_a + self.lambda_value*pool_a
            mix_b = (1-self.lambda_value)*mean_b + self.lambda_value*pool_b
            private_loss = .5*(mix_a+mix_b) + .05*alignment
            pool_mean = .5*(pool_a+pool_b)
        else:
            private_loss = own_mean + .05*alignment
            if self.mode == 'hidden_private':
                residual = losses.residual_member_contrast(ha[:, index], hb[:, index], labels[index])
                private_loss = private_loss + .05*residual
        if not torch.isfinite(own_mean) or not torch.isfinite(private_loss):
            raise FloatingPointError('Nonfinite own/private loss')
        all_parameters = tuple(parameter for _, parameter in self.partition['all'])
        phi_parameters = tuple(parameter for _, parameter in self.partition['internal_private'])
        # autograd.grad returns gradients without accumulating .grad. Shared
        # predictor operations remain connected; there is no upstream detach.
        own_gradients = torch.autograd.grad(own_mean, all_parameters, retain_graph=True, create_graph=False, allow_unused=True)
        self.counters['autograd_grad_calls'] += 1
        private_gradients = torch.autograd.grad(private_loss, phi_parameters, retain_graph=False, create_graph=False, allow_unused=True)
        self.counters['autograd_grad_calls'] += 1
        for collection in (own_gradients, private_gradients):
            if any(gradient is not None and not torch.isfinite(gradient).all() for gradient in collection):
                raise FloatingPointError('Nonfinite old-state gradient collection')
        private_by_id = {id(parameter): gradient for parameter, gradient in zip(phi_parameters, private_gradients)}
        supplied = []
        for parameter, own_gradient in zip(all_parameters, own_gradients):
            gradient = private_by_id[id(parameter)] if id(parameter) in private_by_id else own_gradient
            if gradient is not None and not torch.isfinite(gradient).all():
                raise FloatingPointError('Nonfinite supplied old-state block gradient')
            supplied.append(gradient)
        if not any(gradient is not None for gradient in supplied):
            raise FloatingPointError('No active own/private gradient')
        for parameter, gradient in zip(all_parameters, supplied):
            parameter.grad = None if gradient is None else gradient.detach()
        # One unchanged optimizer; no theta-first/phi-later parameter update.
        for optimizer in session.optimizers:
            optimizer.step()
            self.counters['Adam_steps'] += 1
        selection.finite_state(session.model, session.optimizers)
        session.steps += 1; self.counters['two_view_updates'] += 1
        return {'own_mean': own_mean.detach(), 'private_loss': private_loss.detach(),
                'alignment': alignment.detach(), 'residual': None if residual is None else residual.detach(),
                'pool_mean': None if pool_mean is None else pool_mean.detach(),
                'charged_reverse_passes': 2, 'extra_reverse_passes_over_source_shared_update': 1,
                'member_forwards': 2*session.model.members, 'Adam_steps': 1}
