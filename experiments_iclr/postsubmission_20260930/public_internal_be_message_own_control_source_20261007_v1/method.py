"""One audited message-own exception to the sealed I update; stdlib on import."""
import hashlib
import json
from pathlib import Path

METHOD_ID = 'be_init__allocation_I_message_own'
ALLOCATION_METHOD_SHA = 'c4f879f803e6ec3b2216b6cdadb400cede9beeee401ad5aa14324c091322d6cd'
AUDITOR_SHA = '81ee1208d5e0ac6c0325cf8ab5fd232e8f9914bcf802b586c123738a064b16fe'
CATALOG_SHA = '58e177ca2affc5695219498e1c2138f4fe433256828375bea661f127793c6dfb'
SUPERVISION = {'shared_own': 'own', 'boundary_private_own': 'own',
    'internal_private': 'J except the catalog-resolved existing message_factors tensor: own'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def require_admission(admission):
    """Explicit caller assertion of a later root decision; no automatic adoption."""
    require(isinstance(admission, dict) and set(admission) == {
        'fixed18_closed', 'message_question_warranted', 'admitted_by', 'decision_reference'},
        'Disabled source: an explicit later root admission record is required')
    require(admission['fixed18_closed'] is True and admission['message_question_warranted'] is True
        and admission['admitted_by'] == 'root' and isinstance(admission['decision_reference'], str)
        and bool(admission['decision_reference'].strip()),
        'Root may admit this control only after fixed18 closes and warrants the message question')
    return dict(admission)


class _View:
    def __init__(self, value):
        self.value = value

    def __getattr__(self, name):
        return getattr(self.value, name)


class _ObjectivesView(_View):
    def __init__(self, value, owner):
        super().__init__(value)
        self.owner = owner

    def alignment_loss(self, *args, **kwargs):
        result = self.value.alignment_loss(*args, **kwargs)
        require(self.owner.active and self.owner.alignment is None, 'One original alignment call per update required')
        self.owner.alignment = result
        return result


class _AutogradView(_View):
    def __init__(self, value, owner):
        super().__init__(value)
        self.owner = owner

    def grad(self, loss, inputs, **kwargs):
        return self.owner.reverse(loss, inputs, kwargs)


class _SessionView:
    """Only the delegate sees these views; the audited raw Session is unchanged."""
    __slots__ = ('raw', 'torch', 'core')

    def __init__(self, raw, owner):
        self.raw = raw
        self.torch = _View(raw.torch)
        self.torch.autograd = _AutogradView(raw.torch.autograd, owner)
        self.core = {**raw.core, 'objectives': _ObjectivesView(raw.core['objectives'], owner)}

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def __setattr__(self, name, value):
        if name in self.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self.raw, name, value)


class MessageOwnAdapter:
    """Delegate the complete I update, replacing one returned gradient by identity."""
    def __init__(self, session, audited_adapter, delegate_type, *, policy, lambda_value, admission):
        self.admission = require_admission(admission)
        require(session.task == 'molhiv' and session.arm == 'be_init' and policy == 'I'
            and type(lambda_value) in (int, float) and lambda_value == .5,
            'This one control is only original MolHIV be_init I at fixed lambda=.5')
        require(hashlib.sha256(Path(delegate_type.train_step.__code__.co_filename).read_bytes()).hexdigest()
            == ALLOCATION_METHOD_SHA, 'Exact sealed I update delegate required')
        # This calls the unchanged original auditor on the ORIGINAL Session.
        self.delegate = delegate_type(session, audited_adapter, policy='I', lambda_value=.5)
        self.session, self.partition = session, self.delegate.partition
        partition = audited_adapter.PrivateSteeringAdapter.__init__.__globals__['partition_parameters']
        source = Path(partition.__code__.co_filename)
        require(hashlib.sha256(source.read_bytes()).hexdigest() == AUDITOR_SHA, 'Exact original permission auditor required')
        catalog_path = source.parent / 'BACKBONE_AUDIT.json'
        require(hashlib.sha256(catalog_path.read_bytes()).hexdigest() == CATALOG_SHA, 'Exact original factor catalog required')
        spec = json.loads(catalog_path.read_text())['tasks'][session.task]
        require(len(spec['extra_private']) == 1, 'One explicitly catalogued extra-private MolHIV tensor required')
        (name, entry), = spec['extra_private'].items()
        parameter = dict(self.partition['all'])[name]
        require(entry['role'] == self.partition['roles'][name] == 'internal_private'
            and list(parameter.shape) == entry['shape'] and parameter is session.model.models[0].message_factors,
            'Catalog entry must resolve to the existing audited message tensor object')
        self.message_name, self.message_parameter = name, parameter
        self.own_inputs = tuple(p for n, p in self.partition['all'] if self.partition['roles'][n] != 'internal_private')
        self.phi_inputs = tuple(p for _, p in self.partition['internal_private'])
        self.message_index = next(i for i, p in enumerate(self.phi_inputs) if p is parameter)
        self.counters = self.delegate.counters
        self.active = False
        self.own = self.alignment = None
        self.delegate_collections = self.native_collections = 0
        self.delegate.session = _SessionView(session, self)

    def reverse(self, loss, inputs, kwargs):
        require(self.active, 'Scoped reverse bridge is inactive outside the delegated TRAIN update')
        self.delegate_collections += 1
        first = self.delegate_collections == 1
        require(self.delegate_collections in (1, 2), 'Exact two I delegate collections required')
        expected = self.own_inputs if first else self.phi_inputs
        require(len(inputs) == len(expected) and all(p is q for p, q in zip(inputs, expected))
            and kwargs == {'retain_graph': first, 'create_graph': False, 'allow_unused': True},
            'Pinned I collection inputs/order/options changed')
        require(self.alignment is not None, 'Original alignment must precede the I reverse collections')
        native = self.session.torch.autograd.grad
        self.native_collections += 1
        if first:
            self.own = loss
            return native(loss, inputs, **kwargs)
        # Keep the existing J+.05A phi collection, with graph retained for the
        # REAL third collection. All other returned phi gradients are unchanged.
        gradients = list(native(loss, inputs, **{**kwargs, 'retain_graph': True}))
        require(all(g is None or self.session.torch.isfinite(g).all() for g in gradients),
            'Nonfinite original I internal gradient collection')
        replacement_loss = self.own + .05 * self.alignment
        require(self.session.torch.isfinite(replacement_loss), 'Nonfinite message-own plus original alignment loss')
        self.counters['autograd_grad_calls'] += 1
        self.native_collections += 1
        replacement, = native(replacement_loss, (self.message_parameter,),
            retain_graph=False, create_graph=False, allow_unused=True)
        gradients[self.message_index] = replacement
        return tuple(gradients)

    def train_step(self, batch, labels):
        require(not self.active, 'No nested update')
        self.own = self.alignment = None
        self.delegate_collections = self.native_collections = 0
        self.active = True
        try:
            result = self.delegate.train_step(batch, labels)
            require(self.delegate_collections == 2 and self.native_collections == 3
                and result['charged_reverse_passes'] == 3, 'Three actual reverse collections required')
            return result
        finally:
            self.active = False
            self.own = self.alignment = None

    def metadata(self):
        return {**self.delegate.metadata(), 'scope': 'conditionally admitted MolHIV message-own control',
            'mode': 'message_own_control', 'method_identity': METHOD_ID, 'parent_policy': 'I',
            'supervision_by_role': dict(SUPERVISION), 'message_tensor_name': self.message_name,
            'message_tensor_shape': list(self.message_parameter.shape),
            'message_supervised_gradient': 'two-view mean-own', 'other_internal_supervised_gradient': 'J at fixed lambda=.5',
            'message_total_gradient': 'own + .05 alignment', 'alignment_on_message_unchanged': True,
            'all_alignment_permissions_unchanged': 'same audited phi only',
            'charged_reverse_calls_per_update': 3, 'additional_reverse_calls_over_I': 1,
            'original_I_message_gradient_computed_then_discarded': True,
            'cost_padding': False, 'equal_compute_with_I_claimed': False,
            'admission': dict(self.admission), 'admission_is_caller_assertion_not_verified_fixed18_evidence': True,
            'runtime_verified_at_preparation': False, 'global_scalar_objective_claimed': False}
