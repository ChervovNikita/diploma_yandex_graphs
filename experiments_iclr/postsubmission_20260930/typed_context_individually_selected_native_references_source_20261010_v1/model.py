"""One fresh complete body per fit; contextual width32 uses unchanged V3 primitives."""
from contextlib import contextmanager
from typed_label_context_factor_source_prototype_20261010_v3.caps import CLOSED
from typed_label_context_factor_source_prototype_20261010_v3 import model as v3_model
from typed_label_context_factor_source_prototype_20261010_v3.runtime_gate import PROJECT, require

FAMILIES = ("ordinary_native_independent4", "contextual_native_independent4")


def install(native_module, prototype, family, generator_seed, member_index, caps=CLOSED):
    caps.require("source_bound", "model", "runtime")
    require(family in FAMILIES and member_index in range(4), "Exact reference and fixed body index")
    torch = native_module.torch
    adapter = v3_model.load_adapter(PROJECT / "sehgnn_grouped_member_factor_adapter_source_20261009_v1/grouped_member_factors.py", caps)
    adapter._validate_binding(native_module, prototype)
    contextual = family == "contextual_native_independent4"
    if contextual:
        # The V3 single128 control is NOT used as a four-body width32 reference.
        one = adapter.install_member_bank(native_module, prototype, adapter.AdapterConfig(enabled=True, members=1))
        body = one.members[0]
        wrapper = v3_model._context_type(torch)
        for site in v3_model.CONTEXT_SITES:
            owner, key, previous = adapter._site(body, site)
            owner._modules[key] = wrapper(previous, 32, v3_model._seed(generator_seed, member_index, site), False)
    else:
        body = prototype  # Exact ordinary native model; no factors, H/U or mass input.

    class OneBody(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.members = torch.nn.ModuleList([body])
            self.shared, self.contextual, self.condition = False, contextual, family

        @contextmanager
        def context_bound(self, field):
            sites = [adapter._site(body, site)[2] for site in v3_model.CONTEXT_SITES] if contextual else []
            require(all(site._context_field is None for site in sites), "No stale reference context binding")
            try:
                for site in sites:
                    object.__setattr__(site, "_context_field", field)
                yield
            finally:
                for site in sites:
                    object.__setattr__(site, "_context_field", None)

        def forward_member(self, member, batch, features, labels, field, mask=None):
            require(member == 0 and mask is None and len(features) == 25 and len(labels) == 12
                    and set(features) == set(body.feat_keys) and set(labels) == set(body.label_feat_keys),
                    "Complete native37 channels; no reference mask/task shrink")
            with self.context_bound(field):
                return body(batch, features, labels, mask=None)

        def verify_ownership(self):
            parameters = tuple(self.parameters())
            storage = lambda value: (str(value.device), int(value.untyped_storage().data_ptr()))
            require(len({id(value) for value in parameters}) == len(parameters)
                    and len({storage(value) for value in parameters}) == len(parameters), "Every complete-body parameter has separate storage")
            require(all(value.requires_grad for value in parameters), "Ordinary native body/head and contextual routes all learn")
            return {"members_in_this_fit": 1, "shared_trainable_parameters": False,
                    "fresh_native_body": True, "contextual_generator_width": 32 if contextual else 0,
                    "ordinary_native_has_no_factors_or_mass_generator": not contextual}

        def verify_shared_state_dict(self, state):
            # One independently owned body has no duplicated shared aliases.
            require(set(state) == set(self.state_dict()), "Complete own-body selected state keys")

    bank = OneBody()
    bank.verify_ownership()
    return bank
