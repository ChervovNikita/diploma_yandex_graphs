"""Stdlib source/typed-walk audit; no graph, numerical providers or data execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def typed_walks():
    # Published four-type star algebra only; no node/edge fixture or model is constructed.
    neighbors = {'M': ('D', 'A', 'K'), 'D': ('M',), 'A': ('M',), 'K': ('M',)}
    paths, frontier = {'M'}, {'M'}
    for _ in range(4):
        frontier = {path + next_type for path in frontier for next_type in neighbors[path[-1]]}
        paths.update(frontier)
    return sorted(paths)


def check():
    sources = read(HERE / 'SOURCE_BINDINGS.json')
    for row in sources['files']:
        path = HERE.parent / row['path']
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Pinned unmodified origin ' + row['path'])
    source = (HERE / 'native_family_views.py').read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    numerical = {'torch', 'numpy', 'dgl', 'torch_sparse', 'sklearn'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            require(not any(alias.name.split('.')[0] in numerical for alias in node.names), 'No numerical import, including lazy import')
        elif isinstance(node, ast.ImportFrom):
            require((node.module or '').split('.')[0] not in numerical, 'No numerical provider import')
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
            require(name not in {'load_dataset', 'load', 'data_loader', 'check_acc', 'DataLoader',
                                 'set_random_seed', 'manual_seed', 'seed', 'shuffle', 'make_model',
                                 'train', 'evaluate', 'Adam', 'GradScaler', 'load_state_dict', 'masked_fill',
                                 'masked_fill_', 'zeros_like', 'gen_file_for_evaluate'}, 'No data/model/fit/channel-zero shortcut call ' + name)
    graph_calls = [node for node in ast.walk(functions['build_one']) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == 'heterograph']
    require(len(graph_calls) == 1 and {item.arg for item in graph_calls[0].keywords} == {'num_nodes_dict'}, 'Explicit full typed populations after raw edge removal')
    calls = [node for node in ast.walk(functions['build_one']) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)]
    feature = [node for node in calls if node.func.attr == 'hg_propagate_feat_dgl']
    label = [node for node in calls if node.func.attr == 'hg_propagate_sparse_pyg']
    require(len(feature) == len(label) == 1, 'One unchanged full native feature and label propagation per family')
    for node in (feature[0], label[0]):
        require([ast.literal_eval(value) for value in node.args[1:]] == ['M', 4, 5, []], 'Original full four-hop/native max-length propagation arguments')
    label_kwargs = {item.arg: ast.literal_eval(item.value) for item in label[0].keywords}
    require(label_kwargs == {'prop_feats': False, 'echo': True, 'prop_device': 'cpu'}, 'Native TRAIN-label CPU return-path product recipe')
    require('pairs = () if item[\'remove\'] else data.canonical_relation_pairs' in source
            and 'graph.num_edges(edge) == item[\'retained_unique_pairs\']' in source
            and "rt['remove_diag'](products[key]) @ label_source" in source
            and "torch.count_nonzero(value).item() == 0" in source
            and "value.nnz() == 0" in source
            and "torch.equal(feats['M'], full_ctx.feats['M'])" in source,
            'Raw removal, edge count, genuine native zeros, diagonal and query-own check seams')
    require('full_ctx.targets, full_ctx.targets_cuda,' in source
            and 'full_ctx.train_index, full_ctx.valid_index, full_ctx.train_count, full_ctx.valid_count' in source,
            'Factual target/index object seam retained; no source-label role invention')
    protocol = read(HERE / 'PROTOCOL.json')
    paths = typed_walks()
    labels = [key for key in paths if len(key) > 1 and key[-1] == 'M']
    require(paths == protocol['all_feature_paths'] and labels == protocol['all_label_paths']
            and len(paths) == 25 and len(labels) == 12, 'Complete published typed-walk algebra')
    counts = {}
    for family, row in protocol['families'].items():
        symbol = row['source_symbol']
        require(set(row['removed_raw_relation_names']) == {'M' + symbol, symbol + 'M'}, 'Both raw directions for ' + family)
        feature = [key for key in paths if symbol in key]
        label = [key for key in labels if symbol in key]
        require(feature == row['dependent_feature_paths'] and label == row['dependent_label_paths']
                and len(feature) == 12 and len(label) == 6 and 'M' not in feature,
                'Raw family dependence and mandatory M self channel')
        require([key for key in paths if symbol not in key] == row['unaffected_feature_paths']
                and [key for key in labels if symbol not in key] == row['unaffected_label_paths']
                and row['reuse_unaffected_channels'] is False, 'No uncounted unaffected-channel reuse')
        counts[family] = {'zero_feature_paths_after_native_removal': len(feature), 'zero_label_paths_after_native_removal': len(label)}
    release = read(HERE / 'RELEASE.disabled.json')
    require(release['enabled'] is False and release['root_source_review_approved'] is False
            and all(release[key] is None for key in ('native_qualification_binding', 'full_view_binding', 'role_binding', 'source_seal_sha256')),
            'Default inactive root-fill template')
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text())
    for path in HERE.glob('*.json'):
        read(path)
    return {'status': 'passed', 'scope': 'source_AST_hash_and_published_typed_walk_algebra_only',
            'native_driver_and_helpers_unmodified': True, 'literal_bidirectional_raw_removal_source': True,
            'explicit_populations_and_all37_channel_shape_checks': True, 'native_propagation_and_zero_checks_present': True,
            'typed_family_dependence': counts, 'numerical_graph_or_empty_kernel_behavior_qualified': False,
            'data_model_provider_or_fit_execution': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check()
    if args.output:
        require(args.output.resolve().parent == HERE and not args.output.exists(), 'Fresh source-only check receipt')
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
