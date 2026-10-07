"""Patch one live original ROUTE facade; preserve target tensors and own loss."""
from types import MethodType
import time
from denominator_objective import masked_context_alignment, retained_entries


def alignment_loss(self, a, b, labels, task, temperature=.2, identities=None):
    if task != 'wikics' or temperature != .2 or identities is not None or self.mode != 'route':
        raise ValueError('Only original WikiCS ROUTE target/view contract')
    if not self.session.torch.equal(labels, self.complete_labels[self.index]):
        raise ValueError('Original deterministic512 TRAIN order required')
    self.alignment_source_calls += 1
    return masked_context_alignment(a, b, self.weight_cache, labels, temperature, self.denominator_retained)


def install(session, facade, policy):
    started = time.monotonic()
    if (session.task != 'wikics' or session.arm != 'be_unit_contrastive'
        or session.model.arm != 'be_unit_contrastive' or session.model.independent
        or session.model.members != 4 or policy.method != 'shared_route' or policy.own_selected_four
        or facade.mode != 'route' or session.core['objectives'] is not facade
        or session.config['contrastive'] != {'alignment_weight': .05, 'max_objects': 512,
                                            'residual_weight': 0., 'temperature': .2}):
        raise ValueError('Inactive denominator contrast is only original shared ROUTE')
    torch = session.torch
    labels = facade.complete_labels[facade.index]
    weights = facade.weight_cache  # Preserve the exact original cached Qm.
    retained = retained_entries(weights, labels)
    different = (labels[:, None] != labels[None, :])[None].expand_as(retained)
    positive = weights > 0
    excluded = ~retained
    facade.denominator_retained = retained
    facade.alignment_loss = MethodType(alignment_loss, facade)
    stats = dict(rule='selected original Qm positives union all different-class panel rows',
        same_original_target_tensor=True, removed_target_mass=False, own_supervision_unchanged=True,
        coefficient=.05, temperature=.2, route_only=True, COMMON_symmetry_claim=False,
        total_scored_anchor_rows=int(retained.shape[0]*retained.shape[1]),
        retained_entries=int(retained.sum()), excluded_same_class_zero_target_entries=int(excluded.sum()),
        retained_selected_positive_entries=int(positive.sum()), retained_different_class_entries=int(different.sum()),
        retained_denominator_count_per_route_anchor=retained.sum(-1).cpu().tolist(),
        excluded_count_per_route_anchor=excluded.sum(-1).cpu().tolist(),
        same_class_positive_count_per_route_anchor=positive.sum(-1).cpu().tolist(),
        denominator_preparation_seconds=time.monotonic()-started,
        numerical_rule='stored target mass times retained logsumexp minus finite target-weighted scores',
        denominator_renormalizes_retained_cotangents=True, competence_guaranteed=False)
    return stats
