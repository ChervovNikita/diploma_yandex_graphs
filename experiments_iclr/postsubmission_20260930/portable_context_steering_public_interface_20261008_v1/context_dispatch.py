"""Explicit context-target construction and checkpoint policies; no arm alias.

The legacy contrastive bank is coupled. Only this registered residual-free,
route-separable objective permits own-selected context checkpoints.
"""
from dataclasses import dataclass
import copy


METHODS = {
    'shared_common': ('be_unit_contrastive', 'common', False, False),
    'shared_route': ('be_unit_contrastive', 'route', False, False),
    'shared_route_permuted': ('be_unit_contrastive', 'route', True, False),
    'single_native': ('single', None, False, False),
    'independent4_native': ('independent4', None, False, True),
    'single_common_context': ('single_contrastive', 'common', False, False),
    'independent4_route_context': ('independent4_contrastive', 'route', False, True),
}


@dataclass(frozen=True)
class SelectionPolicy:
    method: str
    underlying_arm: str
    own_selected_four: bool
    objective_separable: bool


def configured_recipe(public, method):
    arm, mode, permuted, own_selected = METHODS[method]
    value = copy.deepcopy(public.recipe('wikics'))
    value['arms'] = [method]
    value['metric'] = 'complete_development_union_split0_validation_and_stopping_accuracy'
    value['contrastive'].update(alignment_weight=.05 if mode else 0.,
                                residual_weight=0., temperature=.2)
    value['context_target_method'] = {
        'method': method, 'underlying_session_arm': arm, 'mode': mode,
        'permuted': permuted, 'own_selected_four': own_selected,
        'fixed_target_objective_separable': mode is not None,
        'no_inter_member_repulsion': True,
        'original_stochastic_views': 2,
        'TEST_access': False, 'exploratory': True,
    }
    return value


def make_session(public, method, seed, device, polynormer, train, frozen_targets):
    """Use caller TRAIN-derived immutable targets; validate before construction."""
    import torch
    from train_targets import PreparedTargets
    from session_objectives_adapter import install as install_objectives
    if not isinstance(frozen_targets, PreparedTargets):
        raise ValueError('The validated caller TRAIN-derived target bundle is required')
    frozen_targets.validate_current(train, torch, device)
    arm, mode, permuted, own_selected = METHODS[method]
    session = public.Session('wikics', arm, seed, device, polynormer)
    session.config = configured_recipe(public, method)
    facade = None
    preparation = dict(frozen_targets.metadata)
    if mode is not None:
        key = 'permuted_masks' if permuted else 'masks'
        facade = install_objectives(session, train['y'].to(session.device),
            frozen_targets.arrays[key], mode)
        if not torch.equal(facade.index.cpu(), torch.from_numpy(
                frozen_targets.arrays['panel_rows'].copy())):
            raise ValueError('Actual cached Session panel differs from frozen panel')
    policy = selection_policy(method, session, facade)
    return session, facade, policy, preparation


def selection_policy(method, session, facade):
    arm, mode, permuted, own_selected = METHODS[method]
    model = session.model
    if session.arm != arm or model.arm != arm:
        raise ValueError('Underlying original arm identity must not be normalized or aliased')
    if model.members != (1 if arm.startswith('single') else 4):
        raise ValueError('Registered predictor count differs')
    if model.independent != arm.startswith(('single', 'independent4')):
        raise ValueError('Registered sharing semantics differs')
    separable = mode is not None
    if separable:
        if facade is None or session.core['objectives'] is not facade:
            raise ValueError('Explicit fixed-target facade required')
        if session.config['contrastive'] != {
            'alignment_weight': .05, 'max_objects': 512,
            'residual_weight': 0., 'temperature': .2}:
            raise ValueError('Exact residual-free context objective required')
        if not model.contrastive or facade.mode != mode:
            raise ValueError('Registered context mode differs')
    elif facade is not None or model.contrastive:
        raise ValueError('Native references must retain only ordinary own supervision')
    if own_selected and (not model.independent or model.members != 4 or len(session.optimizers) != 4):
        raise ValueError('Four independent original Adam histories required')
    return SelectionPolicy(method, arm, own_selected, separable)


def local_transition(session, load, policy, facade):
    """Restore each own local model/Adam state for an explicitly separable bank.

    Keep live end-local dropout/RNG streams, as in the original native schedule.
    No global flags, member streams or optimizer histories are spliced silently.
    """
    checked = selection_policy(policy.method, session, facade)
    if checked != policy:
        raise ValueError('Live objective/checkpoint contract changed')
    if policy.own_selected_four:
        for member, body in enumerate(session.model.models):
            state = load('own_local_' + str(member) + '.pt')
            body.load_state_dict(state['model'])
            session.optimizers[member].load_state_dict(state['optimizer'])
        session.model.set_global(True)
    else:
        session.core['selection'].local_transition(
            session.model, session.optimizers, load, ordinary_independent=False)


def restore_own_best_bank(session, load, policy, facade):
    """Evaluation-only own-selected bank; no mixed-epoch training resume."""
    if selection_policy(policy.method, session, facade) != policy or not policy.own_selected_four:
        raise ValueError('Only explicit ordinary/separable four-model own-selected banks')
    epochs = []
    for member, body in enumerate(session.model.models):
        state = load('own_best_' + str(member) + '.pt')
        body.load_state_dict(state['model'])
        body.set_global(state['global'])
        epochs.append(state['epoch'])
    return epochs
