"""Inactive literal raw-support source views for the exact native IMDB seam.

No numerical import, data loader, model, optimizer or orchestration is supplied.
The caller supplies the root-qualified native runtime and already-loaded contexts.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
SYMBOLS = ('M', 'D', 'A', 'K')
FAMILIES = {'director': ('D', ('MD', 'DM')),
            'actor': ('A', ('MA', 'AM')),
            'keyword': ('K', ('MK', 'KM'))}


class ViewContractError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise ViewContractError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pair_digest(pairs):
    digest = hashlib.sha256()
    for head, tail in pairs:
        digest.update(struct.pack('<qq', head, tail))
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def source_gate():
    seal = json.loads((HERE / 'SEAL.json').read_text())
    require(seal['source_only'] is True and seal['runtime_enabled'] is False
            and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Exact inactive source-view seal')
    for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed sealed view source')
    sources = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    for row in sources['files']:
        path = (HERE.parent / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE.parent) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed pinned native seam/helper source')
    return sources


@dataclass(frozen=True)
class ViewConfig:
    enabled: bool = False
    root_source_review_approved: bool = False
    native_qualification_binding: str | None = None
    full_view_binding: str | None = None
    role_binding: str | None = None
    source_seal_sha256: str | None = None

    def require_enabled(self):
        require(self.enabled is True and self.root_source_review_approved is True,
                'Inactive source views: explicit root-qualified enablement required')
        for value in (self.native_qualification_binding, self.full_view_binding, self.role_binding, self.source_seal_sha256):
            require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Exact root qualification/factual/role/source SHA256 bindings')
        require(self.source_seal_sha256 == sha(HERE / 'SEAL.json'), 'Exact reviewed source-view seal release')


@dataclass
class NativeFamilyView:
    source: str
    binding: str
    metadata: dict
    feats: dict
    label_feats: dict
    data_size: dict
    data: object
    seed: int
    targets: object
    targets_cuda: object
    train_index: object
    valid_index: object
    train_count: int
    valid_count: int

    def source_supply_descriptor(self, helper_module):
        """Describe a successfully rebuilt/checked cache to the unchanged helper."""
        require(self.metadata['construction_complete'] is True, 'Concrete native view must be built before declaration')
        return helper_module.SourceView(source=self.source, binding=self.binding, removed_families=(self.source,),
            required_reverse_relations_removed=True, other_families_retained=True,
            query_own_features_retained=True, native_support_and_paths_rebuilt=True)


def raw_support_plan(data, family):
    """Plan from actual typed semantic pairs; numeric raw relation IDs are read."""
    require(family in FAMILIES, 'Only the prospectively fixed three real source families')
    _, removed = FAMILIES[family]
    plan = []
    for row in data.relation_rows:
        head_type, tail_type = row['raw_head_type'], row['raw_tail_type']
        name = SYMBOLS[head_type] + SYMBOLS[tail_type]
        pairs = data.canonical_relation_pairs[row['raw_relation_id']]
        plan.append(dict(name=name, raw_relation_id=row['raw_relation_id'], raw_head_type=head_type,
                         raw_tail_type=tail_type, original_unique_pairs=len(pairs),
                         original_pair_sha256=pair_digest(pairs), remove=name in removed,
                         retained_unique_pairs=0 if name in removed else len(pairs),
                         retained_pair_sha256=pair_digest(()) if name in removed else pair_digest(pairs)))
    require([row['name'] for row in plan] == ['MD', 'DM', 'MA', 'AM', 'MK', 'KM']
            and {row['name'] for row in plan if row['remove']} == set(removed), 'Remove exactly both native raw directions, retaining the other four')
    return plan


def validate_context(rt, full_static, full_ctx, config, sources):
    torch = rt['torch']
    data = full_ctx.data
    require(data.counts == full_static.data.counts and data.offsets == full_static.data.offsets
            and data.input_bindings == full_static.data.input_bindings
            and data.features is full_static.data.features
            and data.canonical_relation_pairs is full_static.data.canonical_relation_pairs,
            'Exact once-loaded native RoleData/static/seed seam')
    require(tuple(data.counts) == (4932, 2393, 6124, 7971) and data.class_dim == 5,
            'Complete actual native population and five Bernoulli columns')
    require(sorted(full_ctx.feats) == rt['protocol']['feature_paths']
            and sorted(full_ctx.label_feats) == rt['protocol']['label_paths']
            and len(full_ctx.feats) == 25 and len(full_ctx.label_feats) == 12,
            'Complete native factual namespaces')
    require(full_ctx.train_count == len(data.train_ids) and full_ctx.valid_count == len(data.valid_ids)
            and full_ctx.train_count == len(full_ctx.train_index) and full_ctx.valid_count == len(full_ctx.valid_index)
            and full_ctx.train_index.tolist() == list(data.train_ids) and full_ctx.valid_index.tolist() == list(data.valid_ids),
            'Frozen complete native TRAIN/VALID order')
    require(full_ctx.targets.shape == (4932, 5) and full_ctx.targets_cuda.shape == (4932, 5)
            and torch.isfinite(full_ctx.targets[full_ctx.train_index]).all().item()
            and torch.isfinite(full_ctx.targets[full_ctx.valid_index]).all().item()
            and torch.isnan(full_ctx.targets[torch.LongTensor(data.unassigned_movie_ids)]).all().item(),
            'Known targets only; no unassigned target truth')
    require(torch.equal(full_ctx.targets[full_ctx.train_index], torch.FloatTensor(data.train_labels))
            and torch.equal(full_ctx.targets[full_ctx.valid_index], torch.FloatTensor(data.valid_labels)),
            'Factual targets still match frozen known-role labels')
    require(torch.equal(full_ctx.targets_cuda[full_ctx.train_index].cpu(), full_ctx.targets[full_ctx.train_index])
            and torch.equal(full_ctx.targets_cuda[full_ctx.valid_index].cpu(), full_ctx.targets[full_ctx.valid_index]),
            'Factual placed targets still match complete known roles')
    for key, value in full_ctx.feats.items():
        require(value.device.type == 'cpu' and value.dtype == torch.float32 and not value.requires_grad
                and list(value.shape) == list(full_static.raw_feats[key].shape), 'Native full factual CPU feature shape/dtype')
    for key, value in full_ctx.label_feats.items():
        require(value.device.type == 'cpu' and value.dtype == torch.float32 and not value.requires_grad
                and value.shape == (4932, 5), 'Native full factual TRAIN-label shape/dtype')
    require(torch.equal(full_ctx.feats['M'], full_static.raw_feats['M']), 'Factual movie-own input matches original loaded query attributes')
    require(rt['native'].__file__ and sha(rt['native'].__file__) == sources['native_helper_sha256'],
            'Unchanged exact native propagation helper module')
    require(sha(rt['model_module'].__file__) == sources['native_model_sha256'], 'Exact qualified native model module binding; no constructor called')
    require(rt['protocol']['source_commit'] == sources['native_author_commit'], 'Pinned native protocol')


def build_one(rt, full_static, full_ctx, family, costs, config, sources):
    """Literal edge removal BEFORE native aggregation and row normalization."""
    config.require_enabled()
    torch, np, dgl, SparseTensor, native = (rt[key] for key in ('torch', 'numpy', 'dgl', 'SparseTensor', 'native'))
    data = full_ctx.data
    symbol, removed = FAMILIES[family]
    first_cost_row = len(costs.rows)
    plan = raw_support_plan(data, family)
    graph_edges, adjs, raw_features = {}, {}, {}
    with costs.measure('source_' + family + '_raw_graph_features_and_native_normalization') as row:
        for type_id, name in enumerate(SYMBOLS):
            if data.features[type_id] is None:
                raw_features[name] = torch.eye(data.counts[type_id])
            else:
                values = np.frombuffer(data.features[type_id], dtype=np.float32).reshape(data.counts[type_id], -1)
                raw_features[name] = torch.FloatTensor(values)
        for item in plan:
            head_type, tail_type = item['raw_head_type'], item['raw_tail_type']
            # Both removed typed relations remain present, with genuinely empty raw edge arrays.
            pairs = () if item['remove'] else data.canonical_relation_pairs[item['raw_relation_id']]
            rows = torch.LongTensor([head - data.offsets[head_type] for head, _ in pairs])
            columns = torch.LongTensor([tail - data.offsets[tail_type] for _, tail in pairs])
            adj = SparseTensor(row=rows, col=columns, sparse_sizes=(data.counts[head_type], data.counts[tail_type]))
            require(adj.nnz() == item['retained_unique_pairs'], 'Literal retained sparse support, no reweighted channel mask')
            adjs[item['name']] = adj
            graph_edges[(SYMBOLS[tail_type], SYMBOLS[tail_type] + '-' + SYMBOLS[head_type], SYMBOLS[head_type])] = (columns.numpy(), rows.numpy())
        # Explicit sizes preserve isolated source nodes after edge removal; add no edges or self loops.
        graph = dgl.heterograph(graph_edges, num_nodes_dict={name: data.counts[i] for i, name in enumerate(SYMBOLS)})
        require(all(graph.num_nodes(name) == data.counts[i] for i, name in enumerate(SYMBOLS)), 'All original typed/query nodes retained')
        for name, value in raw_features.items():
            graph.nodes[name].data[name] = value
        for key in adjs:
            adjs[key].storage._value = None
            adjs[key].storage._value = torch.ones(adjs[key].nnz()) / adjs[key].sum(dim=-1)[adjs[key].storage.row()]
        for item in plan:
            edge = (SYMBOLS[item['raw_tail_type']], SYMBOLS[item['raw_tail_type']] + '-' + SYMBOLS[item['raw_head_type']], SYMBOLS[item['raw_head_type']])
            require(graph.num_edges(edge) == item['retained_unique_pairs'], 'Exact raw DGL removal in both directions')
        row.update(raw_feature_bytes=sum(value.numel() * value.element_size() for value in raw_features.values()),
                   raw_edge_pairs_retained=sum(item['retained_unique_pairs'] for item in plan),
                   removed_raw_relation_names=list(removed), all_original_node_features_present=True)
    with costs.measure('source_' + family + '_full_native_four_hop_features'):
        graph = native.hg_propagate_feat_dgl(graph, 'M', 4, 5, [], echo=True)
        propagated = {key: graph.nodes['M'].data.pop(key) for key in list(graph.nodes['M'].data.keys())}
        require(sorted(propagated) == rt['protocol']['feature_paths'], 'All25 native channels generated by unchanged DGL propagation, including empty supports')
    with costs.measure('source_' + family + '_native_role_feature_clone_and_TRAIN_labels'):
        feats = {key: propagated[key].detach().clone() for key in full_ctx.feats}
        label_source = torch.zeros((data.counts[0], data.class_dim))
        # Factual TRAIN labels and IDs only, in the identical frozen role context.
        label_source[full_ctx.train_index] = full_ctx.targets[full_ctx.train_index].float()
        require(torch.count_nonzero(label_source[full_ctx.valid_index]).item() == 0
                and torch.count_nonzero(label_source[torch.LongTensor(data.unassigned_movie_ids)]).item() == 0,
                'No VALID/unassigned propagation label input')
        products = native.hg_propagate_sparse_pyg(adjs, 'M', 4, 5, [], prop_feats=False, echo=True, prop_device='cpu')
        require(sorted(products) == rt['protocol']['label_paths'], 'All12 native return products generated, including empty products')
        require(all(tuple(value.sizes()) == (4932, 4932) for value in products.values()), 'Complete native movie-return product shapes')
        require(all(value.nnz() == 0 for key, value in products.items() if symbol in key), 'Native sparse products through removed raw support are empty')
        label_feats = {key: rt['remove_diag'](products[key]) @ label_source for key in full_ctx.label_feats}
    with costs.measure('source_' + family + '_complete_cache_and_raw_removal_checks') as row:
        for key, value in feats.items():
            require(value.shape == full_ctx.feats[key].shape and value.dtype == torch.float32 and value.device.type == 'cpu'
                    and not value.requires_grad and torch.isfinite(value).all().item(), 'Native feature key/shape/dtype/finiteness: ' + key)
            if symbol in key:
                require(torch.count_nonzero(value).item() == 0, 'Genuine native empty-support zero feature: ' + key)
        for key, value in label_feats.items():
            require(value.shape == full_ctx.label_feats[key].shape and value.dtype == torch.float32 and value.device.type == 'cpu'
                    and not value.requires_grad and torch.isfinite(value).all().item(), 'Native label key/shape/dtype/finiteness: ' + key)
            if symbol in key:
                require(torch.count_nonzero(value).item() == 0, 'Genuine native empty-product zero TRAIN-label channel: ' + key)
        require(torch.equal(feats['M'], full_ctx.feats['M']), 'Movie-own M query features exactly retained')
        require(tuple(feats) == tuple(full_ctx.feats) and tuple(label_feats) == tuple(full_ctx.label_feats), 'Exact separate native namespace key order')
        row.update(feature_cache_bytes=sum(value.numel() * value.element_size() for value in feats.values()),
                   label_cache_bytes=sum(value.numel() * value.element_size() for value in label_feats.values()),
                   dependent_feature_keys=[key for key in feats if symbol in key],
                   dependent_label_keys=[key for key in label_feats if symbol in key],
                   dependent_zeros_generated_by_native_operators=True, M_own_exact=True)
    metadata = dict(schema='literal-raw-family-native-IMDB-source-view-v1', construction_complete=True,
                    source=family, removed_raw_names=list(removed), raw_support_plan=plan,
                    native_node_counts={name: data.counts[i] for i, name in enumerate(SYMBOLS)},
                    all_original_typed_node_features_preserved=True, query_own_M_exact=True,
                    feature_keys=list(feats), label_keys=list(label_feats),
                    feature_shapes={key: list(value.shape) for key, value in feats.items()},
                    label_shapes={key: list(value.shape) for key, value in label_feats.items()},
                    TRAIN_ids=list(data.train_ids), VALID_ids=list(data.valid_ids), seed=full_ctx.seed,
                    input_files=data.input_bindings, role_binding=config.role_binding,
                    full_view_binding=config.full_view_binding, native_qualification_binding=config.native_qualification_binding,
                    source_seal_sha256=config.source_seal_sha256, native_helper_sha256=sources['native_helper_sha256'],
                    native_model_sha256=sources['native_model_sha256'], source_method='delete both raw typed edge arrays before normalization/aggregation; rebuild all native paths',
                    unaffected_channel_reuse=False, all_feature_and_label_channels_recomputed=True,
                    post_normalization_mask_or_zero_substitution=False, new_DataLoader_seed_or_model_call=False,
                    label_source_TRAIN_only=True, diagonal_removed_without_renormalization=True,
                    TEST_file_access=False, TEST_membership_known=False, all_full_scope_costs_charged=True)
    view_binding = canonical_digest(metadata)
    require(view_binding != config.full_view_binding, 'Probe and factual view bindings distinct')
    metadata['binding'] = view_binding
    metadata['cost_scope_receipts'] = [dict(row) for row in costs.rows[first_cost_row:]]
    metadata['cost_events_file'] = str((costs.folder / 'COST_EVENTS.jsonl').resolve())
    with costs.measure('source_' + family + '_release_transient_native_graph_and_products'):
        graph = adjs = propagated = products = raw_features = label_source = None
        gc.collect()
    metadata['cost_scope_receipts'] = [dict(row) for row in costs.rows[first_cost_row:]]
    with costs.measure('source_' + family + '_write_view_receipt'):
        receipt = costs.folder / ('SOURCE_VIEW_' + family + '.json')
        require(not receipt.exists(), 'Fresh source-view receipt, no silent cache reuse')
        write(receipt, metadata)
    return NativeFamilyView(family, view_binding, metadata, feats, label_feats, dict(full_ctx.data_size),
                           data, full_ctx.seed, full_ctx.targets, full_ctx.targets_cuda,
                           full_ctx.train_index, full_ctx.valid_index, full_ctx.train_count, full_ctx.valid_count)


def build_family_views(rt, full_static, full_ctx, costs, config=ViewConfig()):
    """Build the three complete views without consuming training streams or fitting."""
    config.require_enabled()
    require(not (costs.folder / 'SOURCE_VIEWS_COMPLETE.json').exists()
            and all(not (costs.folder / ('SOURCE_VIEW_' + family + '.json')).exists() for family in FAMILIES),
            'Fresh source-view receipt location; no overwrite or cache adoption')
    before = state = None
    views, completion = {}, {'status': 'started', 'complete': False, 'TEST_file_access': False}
    try:
        with costs.measure('all_three_literal_raw_source_views'):
            sources = source_gate()
            validate_context(rt, full_static, full_ctx, config, sources)
            state = rt.get('state')
            if state is None:
                # Native runner registers the exact state helper; no new numerical imports.
                import sys
                state = sys.modules.get('state_helpers')
            require(state is not None and sha(state.__file__) == sources['native_state_helper_sha256'], 'Exact registered native owned-stream helper')
            before = state.capture_rng(rt['numpy'], rt['torch'])
            try:
                for family in FAMILIES:
                    views[family] = build_one(rt, full_static, full_ctx, family, costs, config, sources)
                require(state.exact(rt['torch'], state.capture_rng(rt['numpy'], rt['torch']), before), 'No native training stream consumed by source preprocessing')
            finally:
                state.restore_rng(rt['numpy'], rt['torch'], before)
                require(state.exact(rt['torch'], state.capture_rng(rt['numpy'], rt['torch']), before), 'Source preprocessing preserves caller streams')
        completion.update(status='complete', complete=True, bindings={family: view.binding for family, view in views.items()},
                          no_model_or_fit=True, native_model_and_state_transactions_separately_required=True)
        return views
    except BaseException as error:
        completion.update(status='failed', complete=False, built_families=list(views),
                          failure={'type': type(error).__name__, 'message': str(error)},
                          partial_costs_and_view_receipts_preserved=True)
        raise
    finally:
        write(costs.folder / 'SOURCE_VIEWS_COMPLETE.json', completion)


if __name__ == '__main__':
    print(json.dumps({'inactive': True, 'data_provider_model_or_fit': False,
                      'entry': 'root-qualified caller uses build_family_views; no launcher'}))
