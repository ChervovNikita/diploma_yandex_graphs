"""Pinned native GCN/GAT/SAGE wrapper port, loaded lazily after admission.

No numerical imports or provider payload reads occur at module import. Literal
Family construction/seed/RNG/forward and train primitives are extracted from
the frozen source, with only explicit provider feature/class shape bindings.
"""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

try:
    from .classifier import ClassifierAdapter
    from ..spec import normalized_id, ORIGINAL_PAPER_DATASETS
except ImportError:
    from adapters.classifier import ClassifierAdapter
    from spec import normalized_id, ORIGINAL_PAPER_DATASETS

SOURCES = {
    'family': ('common_wrapper_native_backbone_full_family_source_20261010_v1/run_family.py', 'def9e67b42e618f3b14fb98e616c145706e367d4347d3ce9d6251e390c9402c4'),
    'routes': ('shared_fast_graph_model_interface_20261010_v1/common_routes.py', '84ff13b28ef471ac0a4199e10d4284f24647d396e021646e2a21a06f96fb0773'),
    'factors': ('portable_internal_be_public_interface_20261007_v2/core/factors.py', '9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3'),
    'native': ('common_wrapper_native_source_20261010_v1/models.py', '07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303'),
    'driver': ('common_wrapper_native_source_20261010_v1/run_common.py', 'e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc'),
    'base_train': ('common_wrapper_native_source_20261010_v1/run_base.py', 'ca78469a7c1dcc9e98fba054ae1df714f6b19c3e91690ee7b48538005b4e52b1'),
    'gnnm_train': ('common_wrapper_native_source_20261010_v1/run_tabm.py', 'b1d748e50c854802b93f4d4fa7485f096daa1cd36d3dd3bd1c89fbfa63cbd124'),
}
ARMS = ('ordinary_M1', 'factorized_allmap_M1', 'shared4_unchanged', 'separable_equal_size', 'exchange', 'original_gnnm')
I4 = {'ordinary_I4': 'ordinary_M1', 'equipped_I4': 'factorized_allmap_M1'}
BACKBONES = ('GCN', 'GAT', 'SAGE')
DATASETS = {'wikics': 20, 'amazon-photo': 10}
SOURCE_ANCHOR = {'learning_rate': .001, 'dropout': .2}
CONTRACT = {
    'schema': 'common-native-wrapper-source-port-v1',
    'native': 'literal original Model GCN/GAT/SAGE, depth2 hidden128 FFN multiplier1 heads8 LayerNorm GELU residual stack',
    'site': 'after complete native residual_module0, with one continuing native residual module',
    'factors': 'literal FactorLinear: linear(x*r,W)*s+shared_bias; retained native W/b; only input_linear.r Rademacher; every later r/s and stem s one',
    'separable_equal_size': 'raw+relu(layer_norm(raw,eps1e-5)@B)@C; B Glorot C zero; shared 8192 added parameters',
    'exchange': 'raw+(softmax((zQ)(zK)^T/sqrt16 with diagonal masked)@(zV))@U; Q/K/V Glorot U zero; shared 8192 added parameters',
    'exchange_gradient': 'all four live node-aligned prefixes retained together; coupled block backward reaches every route; prefix/tail resume each same persistent member RNG stream',
    'ownership': 'one native body per packed bank; private fast rows; live shared native attention/norms; optimizer after wrapping owns all trainable parameters exactly once',
    'packed_objective': 'literal Family.fit bank branch packed mean own CE, one backward and one AdamW WD0 update',
    'single_objective': 'literal run_base.train_step(model,optimizer,data,False), within Family.scope(streams,0)',
    'original_gnnm': 'literal native TABMModel and run_tabm.train_step_tabm; four CE/4 backward calls, one AdamW step; original mean raw logits before softmax',
    'original_rng': 'one persistent source dropout stream around the unmodified four-forward train primitive; sequential shared stream is preserved',
    'serving': 'probability mean for ordinary/equipped M1, fresh genuine I4, shared unchanged, separable, exchange; raw-logit mean for original GNNM',
    'source_controller': {'maximum_updates': 1000, 'patience_updates': 300, 'metric': 'probability-pool VALID accuracy', 'ties': 'strict first', 'cadence': 1, 'checkpoint0': False},
    'prospective_controller': {'maximum_updates': 5000, 'policy': 'full_budget', 'early_termination': False, 'metric': 'SELECT accuracy using each serving law', 'ties': 'strict first', 'cadence': 1, 'checkpoint0': False},
    'source_anchor': SOURCE_ANCHOR,
    'source_exact_recipe': False,
    'adaptations': ['source input300/classes10 and stem sign(1,300) bound to provider features/classes',
        'new homogeneous factual GRAPH/TRAIN/SELECT/CAL/evaluate authorities and full20/10 mask rosters',
        'source WikiCS merged VALID5274 replaced by provider SELECT3505/CAL1769 at mask0',
        'nine LR/dropout settings; source anchor is one center setting, not an extra fit',
        '5000 full updates replaces historical1000/patience300 for all prospective procedures',
        'new fixed search/comparison seeds; four separately fitted fresh I4 bodies; no M1/member0 artifact reuse',
        'original GNNM shares new roles/HPO/controller; source train primitive and mean-logit serving unchanged',
        'common after-pool scalar CAL500; raw and calibrated results both retained',
        'AST extraction avoids historical CLI/dataset imports; learned/operator AST unchanged'],
    'domain_extension': 'GAT is explicitly source-compatible in this new packet; prior v8 practical GCN/SAGE coverage does not imply GAT was previously ported',
    'unsupported': ['binary/BCE', 'typed graphs/metapaths', 'heterogeneous providers', 'GT/GAT-sep/sheaf/bundle/Polynormer', 'running/static buffers', 'cached adjacency', 'lazy maps', 'tied aliases', 'already factorized native body'],
    'status': 'preparation candidate; scientific roster approval, source commit, resource admission, master integration and explicit GO pending',
}


def receipt(root, path):
    root = Path(root).resolve()
    path = Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError('Confined immutable file required')
    raw = path.read_bytes()
    return {'path': str(path.resolve().relative_to(root)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def bound(root, row):
    relative = Path(row['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Project-relative source/input required')
    path = Path(root).resolve() / relative
    if receipt(root, path) != row:
        raise ValueError('Pinned source/input differs')
    return path


def load_module(root, row):
    path = bound(root, row)
    name = 'common_wrapper_source_' + row['sha256'][:20]
    if name not in sys.modules:
        definition = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(definition)
        sys.modules[name] = module
        definition.loader.exec_module(module)
    return sys.modules[name]


def ast_hash(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def render_sources(root, bindings, features, classes):
    """Extract literal laws and allow exactly three provider-only AST changes."""
    if set(bindings) != set(SOURCES) or any(bindings[k]['sha256'] != row[1] for k, row in SOURCES.items()):
        raise ValueError('Exact seven-file source roster required')
    if type(features) is not int or features < 1 or type(classes) is not int or classes < 3:
        raise ValueError('Multiclass homogeneous feature/class provider required')
    paths = {key: bound(root, row) for key, row in bindings.items()}
    trees = {key: ast.parse(path.read_text()) for key, path in paths.items()}
    family = next(n for n in trees['family'].body if isinstance(n, ast.ClassDef) and n.name == 'Family')
    originals = [next(n for n in family.body if isinstance(n, ast.FunctionDef) and n.name == name)
                 for name in ('seeds', 'make', 'scope', 'logits')]
    methods = copy.deepcopy(originals)
    changes = []

    class ProviderDimensions(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self' and node.func.attr == 'Model':
                if (len(node.args) != 9 or not isinstance(node.args[2], ast.Constant) or node.args[2].value != 300
                        or not isinstance(node.args[4], ast.Constant) or node.args[4].value != 10):
                    raise ValueError('Native Model provider anchors changed')
                node.args[2] = ast.copy_location(ast.Constant(features), node.args[2])
                node.args[4] = ast.copy_location(ast.Constant(classes), node.args[4])
                changes.extend(['Model.input_dim:300->provider_features', 'Model.output_dim:10->provider_classes'])
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch' and node.func.attr == 'randint':
                shape = node.args[2]
                if not isinstance(shape, ast.Tuple) or [getattr(n, 'value', None) for n in shape.elts] != [1, 300]:
                    raise ValueError('First-factor sign shape anchor changed')
                shape.elts[1] = ast.copy_location(ast.Constant(features), shape.elts[1])
                changes.append('stem_sign_shape:(1,300)->(1,provider_features)')
            return node

    methods = [ProviderDimensions().visit(n) for n in methods]
    if changes != ['Model.input_dim:300->provider_features', 'Model.output_dim:10->provider_classes', 'stem_sign_shape:(1,300)->(1,provider_features)']:
        raise ValueError('Unexpected provider binding edits')
    # The source's complete packed bank train branch is copied without edits.
    fit = next(n for n in family.body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
    branches = [n for n in ast.walk(fit) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'bank']
    if len(branches) != 1 or len(branches[0].body) != 7:
        raise ValueError('Literal complete packed source train branch changed')
    branch = branches[0].body
    packed = ast.FunctionDef(name='packed_train_step', args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=n) for n in ('self', 'model', 'optimizer', 'streams', 'members')], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=copy.deepcopy(branch) + [ast.Return(value=ast.Name(id='train_loss', ctx=ast.Load()))], decorator_list=[])
    helpers = []
    for key, name in (('base_train', 'train_step'), ('gnnm_train', 'train_step_tabm')):
        helpers.append(copy.deepcopy(next(n for n in trees[key].body if isinstance(n, ast.FunctionDef) and n.name == name)))
    build = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='SourceBuild', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[]))
    train = ast.fix_missing_locations(ast.Module(body=[packed] + helpers, type_ignores=[]))
    return {'build': ast.unparse(build) + '\n', 'train': ast.unparse(train) + '\n', 'binding_changes': changes,
        'source_method_AST_hashes': {n.name: ast_hash(n) for n in originals},
        'bound_method_AST_hashes': {n.name: ast_hash(n) for n in methods},
        'train_primitive_AST_hashes': {n.name: ast_hash(n) for n in helpers},
        'packed_branch_AST_hash': ast_hash(ast.Module(body=branch, type_ignores=[])),
        'bound_features': features, 'bound_classes': classes}


def construct_source(torch, root, bindings, *, features, classes, seed, body, arm, backbone, dropout, learning_rate, device):
    if arm not in ARMS or backbone not in BACKBONES:
        raise ValueError('Explicit native source arm/backbone required')
    rendered = render_sources(root, bindings, features, classes)
    namespace = {'__name__': 'pinned_common_native_wrapper', 'F': torch.nn.functional, 'TABM_INITS': 4}
    exec(compile(rendered['build'], '<literal_family_provider_binding>', 'exec'), namespace)
    exec(compile(rendered['train'], '<literal_native_training_primitives>', 'exec'), namespace)
    source = namespace['SourceBuild']()
    source.torch, source.F = torch, torch.nn.functional
    source.device = torch.device(device)
    source.cuda = source.device.index or 0
    source.cfg = {'backbone': backbone, 'depth': 2, 'hidden': 128, 'rank': 16, 'dropout': dropout, 'learning_rate': learning_rate}
    native = load_module(root, bindings['native'])
    source.Model = native.Model
    source.set_seed = load_module(root, bindings['driver']).set_seed
    source.routes = load_module(root, bindings['routes'])
    source.factors = load_module(root, bindings['factors'])
    members = 4 if arm in ('shared4_unchanged', 'separable_equal_size', 'exchange', 'original_gnnm') else 1
    if arm != 'original_gnnm':
        kind = {'separable_equal_size': 'separable', 'exchange': 'exchange'}.get(arm, 'baseline')
        model, optimizer, streams = source.make(seed, body, kind, arm != 'ordinary_M1', members)
    else:
        native_seed, _, dropout_seed = source.seeds(seed, body)
        source.set_seed(native_seed)
        # Keep original TABM fast-factor allocation/draws on its declared
        # device, as run_tabm does. Fresh CPU construction is a wrapper rule.
        model = native.TABMModel(backbone, 2, features, 128, classes, 1, 8, 'LayerNorm', dropout, tabm_inits=4, device=device).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
        source.set_seed(dropout_seed)
        streams = [{'cpu': torch.get_rng_state(), **({'cuda': torch.cuda.get_rng_state(source.device)} if source.device.type == 'cuda' else {})}]
    return source, model, optimizer, streams, namespace, rendered


def make_adapter(*, torch, project_root, study, run, device):
    config = run['config']
    arm = config['wrapper_arm']
    dataset = study['dataset']
    if normalized_id(dataset) in ORIGINAL_PAPER_DATASETS or dataset not in DATASETS or config['backbone'] not in BACKBONES or arm not in ARMS:
        raise ValueError('Only declared new homogeneous WikiCS/Photo native source cells are supported')
    members = 4 if arm in ('shared4_unchanged', 'separable_equal_size', 'exchange', 'original_gnnm') else 1
    if (config['members'] != members or config['physical_slow_bodies'] != 1 or config['source_optimizer_count'] != 1
            or config['width'] != 128 or config['depth'] != 2 or config['rank'] != 16 or config['classes'] < 3
            or config['weight_decay'] != 0 or config['learning_rate'] not in (.0005, .001, .002) or config['dropout'] not in (.1, .2, .4)):
        raise ValueError('Fixed source capacity, ownership and frozen LR/dropout domain required')
    def load(row):
        return torch.load(bound(project_root, row), map_location='cpu', weights_only=True)
    graph = load(config['node_inputs']['graph'])
    train = load(config['node_inputs']['train'])
    select = load(config['node_inputs']['select'])
    if set(graph) != {'x', 'edge_index'} or set(train) != {'ids', 'y'} or set(select) != {'ids', 'y'}:
        raise ValueError('Exactly factual GRAPH/TRAIN/SELECT authorities required')
    x, edges = graph['x'].to(device), graph['edge_index'].to(device)
    metadata = config['provider_graph']
    if (x.dtype != torch.float32 or x.ndim != 2 or tuple(x.shape) != (metadata['nodes'], metadata['features'])
            or edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] != metadata['edges']
            or int(edges[0].eq(edges[1]).sum()) != metadata['nodes']):
        raise ValueError('Custodied float32 features/COO graph with one factual self per node required')
    for role in (train, select):
        ids, targets = role['ids'], role['y']
        if (ids.dtype != torch.long or targets.dtype != torch.long or ids.ndim != 1 or targets.shape != ids.shape
                or len(ids) == 0 or ids.min() < 0 or ids.max() >= len(x) or targets.min() < 0 or targets.max() >= config['classes']
                or len(torch.unique(ids)) != len(ids) or not torch.equal(ids, ids.sort().values)):
            raise ValueError('Sorted unique in-range int64 TRAIN/SELECT IDs and multiclass targets required')
    if torch.isin(train['ids'], select['ids']).any():
        raise ValueError('TRAIN/SELECT authorities overlap')
    train_ids, train_y = train['ids'].to(device), train['y'].to(device)
    select_ids, select_y = select['ids'].to(device), select['y'].to(device)
    source, model, optimizer, streams, primitives, rendered = construct_source(torch, project_root, config['wrapper_sources'],
        features=x.shape[1], classes=config['classes'], seed=run['seed'], body=run['body'], arm=arm, backbone=config['backbone'],
        dropout=config['dropout'], learning_rate=config['learning_rate'], device=device)
    labels = torch.zeros(len(x), dtype=torch.long, device=device)
    mask = torch.zeros(len(x), dtype=torch.bool, device=device)
    labels[train_ids], mask[train_ids] = train_y, True
    source.data = SimpleNamespace(x=x, edge_index=edges, y=labels, train_mask=mask)
    source.train_ids = train_ids
    source.batch = {'x': x, 'edge_index': edges, 'graph': SimpleNamespace(edge_index=edges)}
    bank = arm in ('shared4_unchanged', 'separable_equal_size', 'exchange')

    def logits_fn(current, training, step):
        if arm != 'original_gnnm':
            return source.logits(current, streams, bank, torch.arange(len(x), device=x.device))
        with source.scope(streams, 0):
            return torch.stack([current(source.data, x, tabm_seed=row) for row in range(4)])

    def custom_train(adapter, step):
        if bank:
            loss = primitives['packed_train_step'](source, model, optimizer, streams, members)
        else:
            with source.scope(streams, 0):
                loss = (primitives['train_step_tabm'](model, source.data, optimizer, False) if arm == 'original_gnnm'
                        else primitives['train_step'](model, optimizer, source.data, False))
        return {'objective': float(loss), 'ce': float(loss), 'regularization': 0.,
            'operation_counts': {'optimizer_updates': 1, 'backward_calls': 4 if arm == 'original_gnnm' else 1,
                'model_forwards': members, 'route_predictions': members, 'supervised_examples': len(train_ids),
                'native_graph_block_calls': 2 * members, 'coupled_route_block_calls': int(arm in ('separable_equal_size', 'exchange'))}}

    def capture():
        return {'streams': copy.deepcopy(streams), 'training_modes': {name: module.training for name, module in model.named_modules()},
            'factor_members': {name: module.member for name, module in model.named_modules() if isinstance(module, source.factors.FactorLinear)}}

    def restore(value):
        streams[:] = copy.deepcopy(value['streams'])
        modules = dict(model.named_modules())
        for name, mode in value['training_modes'].items():
            modules[name].training = mode
        for name, member in value['factor_members'].items():
            modules[name].member = member

    selector = {'metric': 'accuracy', 'direction': 'max', 'ties': 'first', 'patience_steps': 5000, 'save_every': 1}
    return ClassifierAdapter(torch, torch.nn.functional, model, optimizer, logits_fn=logits_fn,
        train_ids=train_ids, train_y=train_y, select_ids=select_ids, select_y=select_y,
        regularizer=lambda current: x.new_zeros(()), fusion='mean_logits' if arm == 'original_gnnm' else 'mean_probabilities',
        selector=selector, custom_train=custom_train, external_capture=capture, external_restore=restore,
        provenance={'run_id': run['run_id'], 'data': study['data'], 'source': study['source'], 'config': config,
            'actual_model_seed': source.seeds(run['seed'], run['body'])[0], 'source_contract': CONTRACT,
            'explicit_provider_binding_changes': rendered['binding_changes'], 'source_method_AST_hashes': rendered['source_method_AST_hashes'],
            'bound_method_AST_hashes': rendered['bound_method_AST_hashes'], 'train_primitive_AST_hashes': rendered['train_primitive_AST_hashes'],
            'packed_branch_AST_hash': rendered['packed_branch_AST_hash']})
