"""Inactive source gates, exact borrowed role/scoring helpers and native placement."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def require(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def source_checks():
    seal = read(HERE / 'SEAL.json')
    require(seal['source_only'] is True and seal['execution_enabled'] is False
            and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Exact inactive source seal')
    for row in read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed wrapper source')
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    for row in pins['files']:
        path = (PHASE / row['path']).resolve(strict=True)
        require(path.is_relative_to(PHASE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed borrowed or author source')
    return pins


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def borrowed(pins):
    root = PHASE / pins['nsd_runner_directory']
    common = module(root / 'common.py', '_bsnn_original_role_helpers')
    placement = module(root / 'native_placement.py', '_bsnn_original_placement_reference')
    absent = object()
    previous = {k: sys.modules.get(k, absent) for k in ('common', 'native_placement')}
    try:
        sys.modules.update(common=common, native_placement=placement)
        baseline = module(root / 'baseline_runner.py', '_bsnn_original_metric_rng_helpers')
    finally:
        for key, value in previous.items():
            if value is absent: sys.modules.pop(key, None)
            else: sys.modules[key] = value
    return common, baseline


def native_class(pins):
    root = (PHASE / pins['author_directory']).resolve(strict=True)
    require(not any(k == 'models' or k.startswith('models.') or k == 'lib' or k.startswith('lib.')
                    for k in sys.modules), 'Fresh native models/lib namespace required')
    sys.path.insert(0, str(root))
    value = importlib.import_module('models.bayes_disc_models')
    for name, item in list(sys.modules.items()):
        if name == 'models' or name.startswith('models.') or name == 'lib' or name.startswith('lib.'):
            require(Path(item.__file__).resolve().is_relative_to(root), 'Exact author namespace origin')
    return value.BayesBundleSheafDiffusion


def factory(torch, cls, cpu_edges, edges, args):
    """Original CPU constructor/index arithmetic; explicit plain tensor transfer."""
    device = torch.device(args['device'])
    require(cpu_edges.device.type == 'cpu' and edges.device == device and torch.equal(cpu_edges, edges.cpu()), 'Same complete canonical topology/order')
    version = edges._version
    def make():
        require(edges._version == version, 'Static support unchanged')
        model = cls(cpu_edges, dict(args, device='cpu'))
        require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()), 'Original CPU/float32 initialization')
        model.to(device)
        model.edge_index, model.device = edges, device
        model.time_range = model.time_range.to(device)
        builder = model.laplacian_builder
        for name in ('edge_index', 'full_left_right_idx', 'left_right_idx', 'vertex_tril_idx', 'diag_indices',
                     'tril_indices', 'deg', 'fixed_diag_indices', 'fixed_tril_indices'):
            value = getattr(builder, name, None)
            if value is not None: setattr(builder, name, edges if name == 'edge_index' else value.to(device))
        builder.device = device
        learner = getattr(model, 'weight_learner', None)
        if learner is not None: learner.full_left_right_idx = learner.full_left_right_idx.to(device)
        for owner in model.modules():
            for name, value in vars(owner).items():
                require(not isinstance(value, torch.Tensor) or value.device == device, 'Unplaced native plain tensor: ' + name)
        require(model.edge_index is builder.edge_index, 'Native graph object identity')
        return model
    return make


def numpy_pack(np, value):
    if isinstance(value, np.ndarray): return dict(array=value.tolist(), dtype=str(value.dtype), shape=list(value.shape))
    if isinstance(value, dict): return {k: numpy_pack(np, v) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(numpy_pack(np, v) for v in value)
    if isinstance(value, list): return [numpy_pack(np, v) for v in value]
    if isinstance(value, np.generic): return value.item()
    return value


def numpy_unpack(np, value):
    if isinstance(value, dict) and set(value) == {'array', 'dtype', 'shape'}:
        return np.asarray(value['array'], dtype=value['dtype']).reshape(value['shape'])
    if isinstance(value, dict): return {k: numpy_unpack(np, v) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(numpy_unpack(np, v) for v in value)
    if isinstance(value, list): return [numpy_unpack(np, v) for v in value]
    return value


def scipy_rng(model): return model.laplacian_builder.random_so.random_state


def capture(np, torch, helpers, model):
    value = helpers.capture_rng(np, torch)
    rng = scipy_rng(model)
    if isinstance(rng, np.random.RandomState):
        value['scipy'] = dict(kind='RandomState', state=numpy_pack(np, rng.get_state()))
    elif isinstance(rng, np.random.Generator):
        value['scipy'] = dict(kind='Generator', bit_generator=type(rng.bit_generator).__name__, state=numpy_pack(np, rng.bit_generator.state))
    else: raise TypeError('Unqualified SciPy sampler RNG type')
    return value


def restore(np, torch, helpers, model, state):
    helpers.restore_rng(np, torch, state)
    rng, saved = scipy_rng(model), state['scipy']
    if saved['kind'] == 'RandomState' and isinstance(rng, np.random.RandomState):
        rng.set_state(numpy_unpack(np, saved['state']))
    elif saved['kind'] == 'Generator' and isinstance(rng, np.random.Generator):
        require(type(rng.bit_generator).__name__ == saved['bit_generator'], 'Same native SciPy bit-generator type')
        rng.bit_generator.state = numpy_unpack(np, saved['state'])
    else: raise TypeError('Changed native SciPy sampler RNG kind')


def seed_scipy(np, model, seed):
    rng = scipy_rng(model)
    if isinstance(rng, np.random.RandomState): rng.seed(seed % 2**32)
    elif isinstance(rng, np.random.Generator): rng.bit_generator.state = type(rng.bit_generator)(seed).state
    else: raise TypeError('Unqualified SciPy sampler RNG type')


def seed_streams(np, torch, helpers, model, seed):
    helpers.seed_all(np, torch, seed)
    seed_scipy(np, model, seed)


def cpu_tree(torch, value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(torch, v) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(cpu_tree(torch, v) for v in value)
    if isinstance(value, list): return [cpu_tree(torch, v) for v in value]
    return value


def exact(torch, left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if type(left) is not type(right): return False
    if isinstance(left, dict): return set(left) == set(right) and all(exact(torch, v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)): return len(left) == len(right) and all(exact(torch, a, b) for a, b in zip(left, right))
    return left == right


def selector(scores, epoch):
    # Exact expression from the pinned current NSD baseline runner.
    return (scores['valid']['auroc'], -scores['valid']['nll'], -epoch)
