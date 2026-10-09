"""Prospective scalar/guard and cadence adapter over the unchanged source helper."""
from contextlib import contextmanager
from dataclasses import dataclass

ARMS = ('shared_own_only', 'native_pool_credit', 'source_view_supervision',
        'uncoupled_source_contrast', 'COMMON_cycle', 'assigned_source_supply')
FAMILIES = ('actor', 'director', 'keyword')
ASSIGNMENTS = (*FAMILIES, None)
WARMUP, CADENCE, MAX_EPOCHS = 10, 5, 200


def require(value, message):
    if not value:
        raise RuntimeError(message)


@dataclass(frozen=True)
class OpportunityConfig:
    """Explicit successor for transient COMMON assignments; never weaken base admission."""
    base: object
    assignments: tuple
    arm: str

    def __getattr__(self, name):
        return getattr(self.base, name)

    def require_enabled(self):
        self.base.require_enabled()
        require(self.arm in ARMS and self.base.assignments == ASSIGNMENTS, 'Prospective fixed source policy')
        require(self.assignments == ASSIGNMENTS or (self.arm == 'COMMON_cycle'
                and self.assignments[3] is None and len(set(self.assignments[:3])) == 1
                and self.assignments[0] in FAMILIES), 'Only declared COMMON repetition extends base assignment scope')


@contextmanager
def objective_scope(helper, arm):
    """Keep reference/replay/cone/trials; amend only the declared scalar and guards."""
    original = helper._terms
    def terms(outputs, peers, targets, assignments, config):
        result = original(outputs, peers, targets, assignments, config)
        if arm in ('assigned_source_supply', 'COMMON_cycle', 'shared_own_only'):
            return result
        active = [m for m, source in enumerate(assignments) if source is not None]
        if arm == 'native_pool_credit':
            # Complete TRAIN-mode factual pool, same reference tokens as the
            # source arms. The separate eval-mode full_pool risk is unchanged.
            objective = helper._mean_pool_nll([
                helper._observed_logp(outputs[helper.OutputKey('train_full', m)], targets, config)
                for m in range(len(assignments))], config)
        elif arm == 'source_view_supervision':
            objective = sum(result['probe_own:' + str(m)] for m in active) / len(active)
        elif arm == 'uncoupled_source_contrast':
            objective = sum(-helper._observed_logp(outputs[helper.OutputKey('train_full', m)], targets, config).mean()
                            - result['probe_own:' + str(m)] for m in active) / len(active)
        else:
            raise RuntimeError('Unknown prospectively declared control objective')
        result['original_source_J'] = result['J']
        result['J'] = objective
        for m in active:
            # Controls retain all11 factual/probe/absent anchors. Their own
            # global Armijo already implies this repeated global nonincrease;
            # no extra recipient-J restriction weakens a matched control.
            key = 'source_j:' + str(m)
            result['original_' + key] = result[key]
            result[key] = objective
        return result
    helper._terms = terms
    try:
        yield
    finally:
        helper._terms = original


class ScheduledSession:
    """Delegate the native fit_bank loop and actual BankSession; no new fitter."""
    def __init__(self, session, arm, arm_identity):
        require(arm in ARMS, 'One of six fixed arms')
        self.session, self.arm, self.arm_identity = session, arm, arm_identity
        self.completed_own_epochs, self.opportunities, self.scheduled_slots = 0, 0, 0
        self.accepted, self.rejected, self.zero = 0, 0, 0
        self.exposures = {family: 0 for family in FAMILIES}
        self.actual_dose, self.logs = 0., []

    def __getattr__(self, name):
        return getattr(self.session, name)

    def train_epoch(self):
        value = self.session.train_epoch()
        self.completed_own_epochs = self.session.counters['own_epochs']
        return value

    def correct(self, own_displacements):
        epoch = self.completed_own_epochs
        due = epoch >= 15 and epoch % CADENCE == 0
        self.scheduled_slots += int(due)
        if not due or self.arm == 'shared_own_only':
            return dict(status='own_only' if self.arm == 'shared_own_only' else 'not_due',
                        completed_own_epochs=epoch, scheduled_slot=due, private_L2_dose=0.,
                        shadow_reference_replay_or_proposal_work=False)
        assignment = ASSIGNMENTS
        if self.arm == 'COMMON_cycle':
            family = FAMILIES[self.opportunities % 3]
            assignment = (family, family, family, None)
        old = self.session.config
        config = OpportunityConfig(old, assignment, self.arm)
        config.require_enabled()
        self.session.config = config
        try:
            with objective_scope(self.helper, self.arm):
                value = self.session.correct(own_displacements)
        finally:
            self.session.config = old
        self.opportunities += 1
        for family in assignment[:3]:
            self.exposures[family] += 1
        self.accepted += int(value['status'] == 'accepted')
        self.rejected += int(value['status'] == 'rejected')
        self.zero += int(value['status'] == 'zero')
        self.actual_dose += value['private_L2_dose']
        result = dict(value, arm=self.arm, completed_own_epochs=epoch, opportunity=self.opportunities,
                      actual_assignments=list(assignment), cumulative_recipient_family_exposures=dict(self.exposures),
                      cumulative_actual_private_L2_dose=self.actual_dose)
        self.logs.append(result)
        return result

    def snapshot(self, epoch, scores, outputs, identity):
        saved = self.session.snapshot(epoch, scores, outputs, identity)
        saved['pilot_policy'] = dict(arm=self.arm, completed_own_epochs=self.completed_own_epochs,
            scheduled_slots=self.scheduled_slots, opportunities=self.opportunities, accepted=self.accepted,
            rejected=self.rejected, zero=self.zero, exposures=dict(self.exposures),
            actual_dose=self.actual_dose, actual_warm_updates_performed_in_this_branch=True,
            cross_arm_bitwise_warm_state_gate=False,
            selected_after_actual_source_opportunity=self.opportunities > 0,
            base_assignment_metadata=list(ASSIGNMENTS), COMMON_assignment_is_transient_per_opportunity=True)
        return saved
