"""Genuine outer producer identity with an explicit unchanged helper view.

No model, data or Torch import occurs here. The parent helper checks are never
patched: they receive their exact original source identity, nested inside the
public binding of the producer that actually ran.
"""
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent.parent
RUNTIME = HERE / 'runtime'
RUNTIME_NAMES = ('run_fit.py', 'adapters.py', 'bookkeeping.py', 'common.py', 'controllers.py', 'source_rng.py')
SCHEMA = 'server51-composed-normalization-genuine-runtime-binding-v2'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def receipt(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path, name):
    path = Path(path).resolve()
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).resolve() == path, 'Source module identity conflict')
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_sources(root):
    root = Path(root).resolve()
    require(HERE.is_relative_to(root), 'Genuine source must be inside the actual project')
    manifest_pin = receipt(HERE / 'MANIFEST.json')
    manifest = json.loads((HERE / 'MANIFEST.json').read_text())
    require(manifest['schema'] == 'server51-composed-normalization-runtime-source-v4', 'Wrong producer source schema')
    for row in manifest['files']:
        path = (HERE / row['path']).resolve()
        require(path.is_relative_to(HERE), 'Source member escapes package')
        pin = receipt(path)
        require(all(pin[k] == row[k] for k in ('bytes', 'sha256')), 'Producer member changed')
    for row in json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())['files']:
        path = (EXPANSION.parent / row['path_relative_to_phase']).resolve()
        require(path.is_relative_to(root), 'Source dependency escapes project')
        pin = receipt(path)
        require(all(pin[k] == row[k] for k in ('bytes', 'sha256')), 'Frozen producer dependency changed')
    return manifest_pin


def _fresh_runtime_imports():
    for name in ('common', 'controllers', 'bookkeeping', 'source_rng', 'adapters', 'link_authority'):
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve().parent == RUNTIME,
                    'A fresh worker is required for the composed runtime')
    for name in ('host_binding', 'descriptor_binding', 'runtime_bridge'):
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve().parent == HERE,
                    'A fresh worker is required for the composed host/runtime authority')
    if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
    if str(RUNTIME) not in sys.path: sys.path.insert(0, str(RUNTIME))


def _helper_binding(adapter, family):
    folder = EXPANSION / 'portable_dispatcher_v3' / ('runtime_link' if family == 'link' else 'runtime_node_molecular')
    return {'runtime_sha256': {name: receipt(folder / name)['sha256'] for name in RUNTIME_NAMES},
            'policy_sha256': None if family != 'link' else adapter.policy_receipt['sha256'],
            'callback_sha256': adapter.callback_receipt['sha256'],
            'extension_sha256': None if adapter.extension_receipt is None else adapter.extension_receipt['sha256'],
            'dependency_sha256': {pin['path']: pin['sha256'] for pin in adapter.dependency_receipts}}


def build_runtime_binding(adapter, root):
    require(getattr(adapter, 'composed_producer', False), 'The genuine composed producer is required')
    source = verify_sources(root)
    helper = _helper_binding(adapter.normalized, adapter.job['family'])
    require(helper == adapter.helper_runtime, 'Unchanged helper source identity changed')
    return {'schema': SCHEMA,
            'runtime_sha256': {name: receipt(RUNTIME / name)['sha256'] for name in RUNTIME_NAMES},
            'policy_sha256': helper['policy_sha256'], 'callback_sha256': helper['callback_sha256'],
            'extension_sha256': helper['extension_sha256'], 'dependency_sha256': copy.deepcopy(helper['dependency_sha256']),
            'composition_producer': {'source_manifest': source, 'entrypoint': receipt(RUNTIME / 'run_fit.py'),
                'normalization_source': receipt(EXPANSION / 'composable_normalization_v2/MANIFEST.json'),
                'link_extension_authority': receipt(RUNTIME / 'link_authority.py') if adapter.job['family'] == 'link' else None,
                'unchanged_helper_runtime': copy.deepcopy(helper)}}


class ProducerAdapter:
    composed_producer = True

    def __init__(self, normalized, job, root, source):
        self.normalized, self.job, self.root = normalized, copy.deepcopy(job), Path(root).resolve()
        self.helper_runtime = _helper_binding(normalized, job['family'])
        # Constructed before any update or restore. This becomes part of the
        # genuine complete checkpoint envelope; old V2 states cannot acquire it.
        self.normalized.binding['genuine_runtime_producer'] = {'source_manifest': copy.deepcopy(source),
            'entrypoint': receipt(RUNTIME / 'run_fit.py'), 'unchanged_helper_runtime': copy.deepcopy(self.helper_runtime)}
        self._selected_public = None

    def __getattr__(self, name):
        return getattr(self.normalized, name)

    def _public(self, value):
        require(value == build_runtime_binding(self, self.root), 'Exact current outer producer runtime required')

    def train_epoch(self):
        self.normalized.owner.idle(); self._selected_public = None
        return self.normalized.train_epoch()

    def train_update(self):
        self.normalized.owner.idle(); self._selected_public = None
        return self.normalized.train_update()

    def load_state(self, value):
        self.normalized.owner.idle(); self._selected_public = None
        return self.normalized.load_state(value)

    def bind_loaded_selection(self, payload, *, selected_checkpoint, job_hash, runtime_binding):
        self._public(runtime_binding)
        require(payload.get('runtime_binding') == runtime_binding, 'Selected outer runtime differs')
        helper_payload = dict(payload)
        helper_payload['runtime_binding'] = copy.deepcopy(self.helper_runtime)
        self.normalized.bind_loaded_selection(helper_payload, selected_checkpoint=selected_checkpoint,
            job_hash=job_hash, runtime_binding=self.helper_runtime)
        self._selected_public = {'job_hash': job_hash, 'selected_epoch': payload['selector']['selected_epoch'],
            'selected_checkpoint': copy.deepcopy(selected_checkpoint), 'runtime_binding': copy.deepcopy(runtime_binding)}

    def selected_mean_export(self, actual, *, job_hash, selected_epoch, selected_checkpoint, runtime_binding):
        self._public(runtime_binding)
        expected = {'job_hash': job_hash, 'selected_epoch': selected_epoch,
            'selected_checkpoint': selected_checkpoint, 'runtime_binding': runtime_binding}
        require(self._selected_public == expected, 'Bind this exact loaded outer selected state before export')
        result = self.normalized.selected_mean_export(actual, job_hash=job_hash, selected_epoch=selected_epoch,
            selected_checkpoint=selected_checkpoint, runtime_binding=self.helper_runtime)
        if result is None: return None
        result = dict(result)
        result['unchanged_helper_runtime_binding'] = copy.deepcopy(self.helper_runtime)
        result['runtime_binding'] = copy.deepcopy(runtime_binding)
        result['normalization_runtime_producer'] = copy.deepcopy(runtime_binding['composition_producer'])
        return result

    def state_signature(self):
        """Read-only model/bank fingerprint, excluding loader and optimizer RNG."""
        self.normalized.assert_identity()
        with self.normalized.owner.state('export'):
            return self.normalized._state_signature({'model': self.model.state_dict(),
                                                     'private_BN_banks': self.manager.state_dict()})

    @contextmanager
    def inference_operation(self, role):
        # The inner owner checks pointer identity and exact buffer contents.
        with self.normalized.inference_operation(role):
            yield

    def metadata(self):
        return {**self.normalized.metadata(), 'genuine_runtime_binding': build_runtime_binding(self, self.root)}


def create_adapter(torch, job, policy, root):
    _fresh_runtime_imports()
    source = verify_sources(root)
    require(job.get('family') in ('ogb', 'link') and job.get('normalization_composition'),
            'This producer is for declared native-BN molecular/BUDDY compositions only')
    host = load(HERE / 'host_binding.py', 'host_binding')
    host.validate_host_job(job, root)
    family = 'runtime_link' if job['family'] == 'link' else 'runtime_node_molecular'
    if job['family'] == 'link':
        # The frozen helper imports this genuine outer producer's explicit registry.
        # Its own adapter source/factory remains the original pinned portable3 file.
        load(RUNTIME / 'link_authority.py', 'link_authority')
    parent = load(EXPANSION / 'portable_dispatcher_v3' / family / 'adapters.py', '_composed_pinned_' + family)
    normalization = load(EXPANSION / 'composable_normalization_v2/adapter.py', '_composed_normalization_v2')
    normalized = normalization.create_adapter(torch, job, policy, root, parent_factory=parent.create_adapter)
    try:
        return ProducerAdapter(normalized, job, root, source)
    except BaseException:
        normalized.close()
        raise


def load_runtime(spec, job, root):
    """Serving loader using exactly the producer used by training, with no fit."""
    _fresh_runtime_imports()
    entry = Path(spec['runtime_entrypoint']['path']).resolve()
    require(entry == RUNTIME / 'run_fit.py' and receipt(entry) == spec['runtime_entrypoint'],
            'Exact genuine composed producer entrypoint required')
    runtime = load(entry, '_composed_genuine_run_fit')
    runtime.bound_project_root(root); runtime.validate_host_job(job, root)
    policy = runtime.validate_job(job, root)
    torch = sys.modules['torch']
    torch.set_num_threads(int(job.get('torch_threads', 2)))
    runtime.seed_all(torch, job['seed'], device=job.get('device', 'cuda:0'))
    adapter = create_adapter(torch, job, policy, root)
    return runtime, adapter, build_runtime_binding(adapter, root)
