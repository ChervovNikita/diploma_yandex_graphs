"""Pinned source loading and native RNG custody; stdlib until explicit call."""
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PUBLIC_NAME = 'portable_internal_be_public_interface_20261007_v2'
CORE_NAME = 'label_only_private_corrector_core_source_20261008_v1'
PUBLIC_MANIFEST_SHA = '190940ca9f8141ac45f739d064cb1ef76aaf1965ba8c91adb544ad8e38fef724'
CORE_MANIFEST_SHA = '5311420414866ad7159185fd12d4e14cac3e935f36bc4f6b2c8786e03c669dd9'
CORE_PROGRAM_SHA = '41e0165829d091e73ec08a7520f5a830cb4fb37006a27d26684cb2ad2b8d675d'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(root, expected):
    root = Path(root).resolve(strict=True)
    require(sha(root / 'MANIFEST.json') == expected, 'Exact original source manifest required')
    for row in json.loads((root / 'MANIFEST.json').read_text())['files']:
        path = (root / row['path']).resolve(strict=True)
        require(path.is_relative_to(root) and path.stat().st_size == row['bytes']
                and sha(path) == row['sha256'], 'Changed sealed source: ' + row['path'])
    return root


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value  # Required by the unchanged core's dataclass.
    spec.loader.exec_module(value)
    return value


@contextmanager
def aliases(values):
    missing = object(); old = {name: sys.modules.get(name, missing) for name in values}
    try:
        sys.modules.update(values)
        yield
    finally:
        for name, value in old.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def dependencies():
    """Fresh private source namespaces; no original module or source mutation."""
    public_root = sealed(PHASE / PUBLIC_NAME, PUBLIC_MANIFEST_SHA)
    core_root = sealed(PHASE / CORE_NAME, CORE_MANIFEST_SHA)
    require(sha(core_root / 'core.py') == CORE_PROGRAM_SHA, 'Unchanged sealed corrector core required')
    public = module(public_root / 'portable.py', '_label_integrated_public_portable')
    with aliases({'portable': public}):
        data = module(public_root / 'data_interface.py', '_label_integrated_public_data')
        with aliases({'data_interface': data}):
            driver = module(public_root / 'train.py', '_label_integrated_original_native_driver')
    core = module(core_root / 'core.py', '_label_integrated_unchanged_corrector_core')
    return public, data, driver, core, public_root


def clone_streams(streams):
    return [{name: value.clone() for name, value in stream.items()} for stream in streams]


def native_rng_state(native):
    """Read current states without drawing from any native/ambient stream."""
    torch = native.torch
    return dict(streams=clone_streams(native.streams), cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(native.cuda_index) if native.cuda_index is not None else None,
        python=random.getstate(), numpy=copy.deepcopy(native.np.random.get_state()))


def require_same_native_rng(native, expected, operation):
    torch = native.torch; actual = native_rng_state(native)
    streams = actual['streams']; old = expected['streams']
    require(len(streams) == len(old) and all(left.keys() == right.keys() and
            all(torch.equal(left[name], right[name]) for name in left)
            for left, right in zip(streams, old)), 'Native private stream changed during ' + operation)
    require(torch.equal(actual['cpu'], expected['cpu'])
            and (expected['cuda'] is None or torch.equal(actual['cuda'], expected['cuda']))
            and actual['python'] == expected['python']
            and actual['numpy'][0] == expected['numpy'][0]
            and native.np.array_equal(actual['numpy'][1], expected['numpy'][1])
            and actual['numpy'][2:] == expected['numpy'][2:],
            'Ambient native RNG changed during ' + operation)


def integration_identity():
    return dict(source_program_sha256=sha(HERE / 'train.py'),
        capture_program_sha256=sha(HERE / 'capture.py'),
        common_program_sha256=sha(__file__),
        integration_manifest_sha256=sha(HERE / 'MANIFEST.json'),
        public_manifest_sha256=PUBLIC_MANIFEST_SHA,
        unchanged_core_manifest_sha256=CORE_MANIFEST_SHA,
        unchanged_core_program_sha256=CORE_PROGRAM_SHA)


def control_module(binding):
    """Only an exact separately sealed shared-backbone control source."""
    require(isinstance(binding, dict) and binding.get('factory') == 'make_shared_controls',
            'Explicit bound shared-backbone control-core protocol required')
    root = (PHASE / binding['directory']).resolve(strict=True)
    require(root.is_relative_to(PHASE), 'Control source stays inside this project phase')
    sealed(root, binding['manifest_sha256'])
    path = (root / binding['program']).resolve(strict=True)
    require(path.is_relative_to(root) and sha(path) == binding['program_sha256'], 'Exact control program')
    return module(path, '_label_integrated_shared_backbone_controls')


def tensor_digest(value):
    """Digest actual shape/dtype/ordered bytes; no invented role metadata."""
    value = value.detach().cpu().contiguous()
    result = hashlib.sha256()
    result.update(json.dumps(dict(shape=list(value.shape), dtype=str(value.dtype)),
                             sort_keys=True).encode())
    result.update(value.numpy().tobytes(order='C'))
    return result.hexdigest()


def context_identity(native, train):
    nodes = native.torch.arange(len(train['x']), dtype=native.torch.long, device='cpu')
    return dict(node_order_sha256=tensor_digest(nodes),
        observed_edges_sha256=tensor_digest(train['edge_index']),
        TRAIN_ids_sha256=tensor_digest(train['ids']), TRAIN_labels_sha256=tensor_digest(train['y']),
        context_policy_sha256=sha(PHASE / CORE_NAME / 'MASKING_UPDATE_CONTRACT.json'))


def backbone_id(native):
    return 'wikics_split0_native_single_seed' + str(native.seed) + '_' + native.native_provenance['polynormer_model_sha256']


def selection_policy(condition):
    value = dict(metric_name='complete_development_union_split0_accuracy', role='VALID',
        native_local_restore='each_own_native_local_selector',
        corrector_local_restore='same_own_native_selected_epoch', end_local_streams='live_no_rewind',
        final_selector='mean_probability_family' if condition == 'shared_backbone_untied_correctors4'
                       else 'each_own_corrected_predictor')
    value['policy_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value
