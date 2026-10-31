"""Lazy geometry-only bank; reuse pinned native sharing and affine helpers."""
import copy
from support import require


def make_bank(torch, adapter, native_factory):
    class GeometryOnlyBank(torch.nn.Module):
        def __init__(self):
            super().__init__()
            prototype = native_factory()
            require(type(prototype) is adapter.load_native_general_class(), 'Pinned original NSD class')
            require(prototype.layers == len(prototype.sheaf_learners), 'One original nonsparse incidence learner per layer')
            self.native_parameter_objects = len(list(prototype.parameters()))
            self.native_parameter_count = sum(value.numel() for value in prototype.parameters())
            clones = [prototype] + [copy.deepcopy(prototype, adapter._shared_copy_memo(prototype)) for _ in range(3)]
            self.members = torch.nn.ModuleList(clones)
            self.factor_paths = tuple('sheaf_learners.' + str(layer) + '.linear1' for layer in range(prototype.layers))
            for path in self.factor_paths:
                original = adapter._module_at(prototype, path)
                require(type(original) is torch.nn.Linear and original.bias is None
                        and original.in_features == 2*prototype.hidden_dim and original.out_features == prototype.d**2,
                        'Exact original bias-free ordered-incidence Linear')
                for member in clones:
                    adapter._replace(member, path, adapter.SharedBELinear(original))
            self.bound_edge_index = prototype.edge_index
            self.bound_graph_version = prototype.edge_index._version
            self.members_count = 4
            self.assert_ownership()

        def assert_ownership(self):
            names = [dict(member.named_parameters()) for member in self.members]
            fast_names = {path + suffix for path in self.factor_paths for suffix in ('.r', '.s')}
            require(all(set(value) == set(names[0]) for value in names), 'Same complete native parameter schema')
            slow_names = set(names[0])-fast_names
            for name in slow_names:
                require(len({id(value[name]) for value in names}) == 1, 'Shared native parameter identity: ' + name)
            factors = [value[name] for value in names for name in sorted(fast_names)]
            require(len({id(value) for value in factors}) == len(factors)
                    and len({(str(value.device), value.untyped_storage().data_ptr()) for value in factors}) == len(factors), 'Only private r/s, with disjoint factor storage')
            shared = {id(names[0][name]) for name in slow_names}
            require(not shared.intersection(id(value) for value in factors), 'Private factors disjoint from all slow parameters')
            require(len({id(member) for member in self.members}) == len({id(member.laplacian_builder) for member in self.members}) == 4,
                    'Four complete private native modules/builders')
            for member in self.members:
                require(member.edge_index is self.bound_edge_index and member.laplacian_builder.edge_index is self.bound_edge_index
                        and member.edge_index._version == self.bound_graph_version, 'Immutable shared topology, private native states')
                for path in self.factor_paths:
                    layer = adapter._module_at(member, path)
                    require(layer.bias is None and layer.r.shape == (layer.in_features,) and layer.s.shape == (layer.out_features,), 'Ordered incidence factor shapes/bias')
            require({id(value) for value in self.parameters()} == shared.union(id(value) for value in factors), 'No extra private head/feature parameters')
            require(len(shared) == self.native_parameter_objects and sum(names[0][name].numel() for name in slow_names) == self.native_parameter_count,
                    'Every original native parameter shared, none removed')
            require(sum(value.numel() for value in factors) == 4*self.members[0].layers*(2*self.members[0].hidden_dim+self.members[0].d**2),
                    'Exact geometry-only private factor count')
            return dict(shared_parameter_objects=len(shared), private_factor_objects=len(factors),
                shared_active_parameters=sum(names[0][name].numel() for name in slow_names),
                private_active_parameters=sum(value.numel() for value in factors),
                private_factors_only=list(sorted(fast_names)), original_constructors=1, complete_native_paths=4,
                factor_initialization='all ones, no RNG draw', incidence_bias=False)

        def parameter_groups(self, sheaf_decay, weight_decay):
            sheaf, other = [], []
            for name, value in self.named_parameters():
                (sheaf if 'sheaf_learners' in name else other).append(value)
            all_values = sheaf + other
            require(len({id(value) for value in all_values}) == len(all_values) == len(list(self.parameters())), 'One deduplicated optimizer entry per parameter')
            return [dict(params=sheaf, weight_decay=sheaf_decay), dict(params=other, weight_decay=weight_decay)]

        def forward(self, x):
            self.assert_ownership()
            return torch.log(torch.stack([member(x).exp() for member in self.members]).mean(0))

    return GeometryOnlyBank()
