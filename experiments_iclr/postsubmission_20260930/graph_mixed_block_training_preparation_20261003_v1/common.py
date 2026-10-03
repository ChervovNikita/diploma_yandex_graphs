"""Source custody and the immutable native HGB readers/builders; no eager Torch."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
PACKET = Path(__file__).resolve().parent
PHASE = PACKET.parent
SEEDS = (131, 137, 139, 149, 151)
DATASETS = ('HGB-DBLP', 'HGB-ACM')
POLICIES = ('own/own', 'pool/pool', 'pool/own', 'own/pool')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(record, base=PHASE):
    path = Path(record['path'])
    if not path.is_absolute():
        path = base / path
    require(sha(path) == record['sha256'], 'Fingerprint mismatch: ' + str(path))
    if 'bytes' in record:
        require(path.stat().st_size == record['bytes'], 'Byte length mismatch: ' + str(path))
    return path


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def packet_guard():
    manifest_sha = sha(PACKET / 'MANIFEST.json')
    require(json.loads((PACKET / 'SEAL.json').read_text())['manifest_sha256'] == manifest_sha, 'Mixed packet seal mismatch')
    for row in json.loads((PACKET / 'MANIFEST.json').read_text())['payload']:
        verify(row, PACKET)
    provenance = json.loads((PACKET / 'PROVENANCE.json').read_text())
    for row in provenance['source_records']:
        verify(row)
    return provenance, manifest_sha


def admission(freeze_path, release_path, mode):
    provenance, manifest_sha = packet_guard()
    freeze_bytes, release_bytes = Path(freeze_path).read_bytes(), Path(release_path).read_bytes()
    frozen, release = json.loads(freeze_bytes), json.loads(release_bytes)
    freeze_sha = hashlib.sha256(freeze_bytes).hexdigest()
    require(frozen['schema'] == 'HGB_mixed_block_executable_freeze_v1'
            and tuple(frozen['datasets']) == DATASETS and tuple(frozen['seeds']) == SEEDS
            and tuple(frozen['policies']) == POLICIES and frozen['model'] == 'global_BE'
            and frozen['members'] == 4 and frozen['CP_excluded'] is True
            and frozen['test_labels_closed'] is True and frozen['terminals'] == 40,
            'Complete independently justified two-graph/four-policy freeze required')
    require(frozen['scientific_design']['sha256'] == provenance['scientific_design']['sha256'], 'Root mixed scientific authority differs')
    verify(frozen['scientific_design'])
    require(release['prepared_manifest_sha256'] == manifest_sha and release['study_freeze_sha256'] == freeze_sha
            and release['scientific_design_sha256'] == frozen['scientific_design']['sha256']
            and release['device'] == 'cpu' and release['threads'] == 1,
            'Exact source/freeze/science/CPU release required')
    if mode == 'training':
        require(release['execution_authorized'] is True and release['root_observed_qualification'] is True,
                'Root training release and observed focused qualification required')
        qualified = json.loads(verify(release['qualification']).read_text())
        require(qualified['status'] == 'qualified' and qualified['prepared_manifest_sha256'] == manifest_sha
                and qualified['study_freeze_sha256'] == freeze_sha
                and qualified['scientific_design_sha256'] == frozen['scientific_design']['sha256']
                and qualified['synthetic_same_state_witness_passed'] is True
                and [(r['dataset'], r['policy']) for r in qualified['rows']] == [(d, p) for d in DATASETS for p in POLICIES]
                and all(r['status'] == 'qualified' and r['consecutive_updates'] == 6
                        and r['next_step_complete_state_replay'] is True for r in qualified['rows'])
                and qualified['originals_preserved'] is True and qualified['validation_or_test_scored'] is False,
                'Complete source-bound two-graph resource/replay qualification required')
    else:
        require(mode == 'qualification' and release['qualification_authorized'] is True
                and release['execution_authorized'] is False, 'Separate score-free qualification release required')
    require([r['dataset'] for r in frozen['source_freezes']] == list(DATASETS), 'Both immutable graph authorities required')
    sources = {}
    for row, authority in zip(frozen['source_freezes'], provenance['graph_authorities']):
        require(row['record']['sha256'] == authority['record']['sha256'], 'Graph freeze authority differs')
        graph_freeze = json.loads(verify(row['record']).read_text())
        require(graph_freeze['dataset'] == row['dataset'] and tuple(graph_freeze['seeds']) == SEEDS
                and [r['seed'] for r in graph_freeze['splits']] == list(SEEDS)
                and graph_freeze['scope'] == 'complete_release_development_only'
                and graph_freeze['test_labels_closed'] is True, 'Closed complete development graph freeze required')
        sources[row['dataset']] = graph_freeze
    require(frozen['baseline_reuse']['mode'] in ('fresh_all40', 'reuse_all10'), 'All-ten-or-none baseline policy required')
    return provenance, frozen, sources, release, dict(prepared_manifest_sha256=manifest_sha,
        study_freeze_sha256=freeze_sha, scientific_design_sha256=frozen['scientific_design']['sha256'],
        root_release_sha256=hashlib.sha256(release_bytes).hexdigest(), device='cpu', threads=1,
        test_labels_closed=True), (freeze_bytes, release_bytes)


def runtime():
    require(platform.system() == 'Linux' and sys.version_info[:3] == (3, 11, 14), 'Exact qualified Linux/Python runtime required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU execution requires CUDA hidden before import')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key) == '1', 'One-thread preimport environment required: ' + key)
    import torch
    require(torch.__version__ == '2.1.2+cu118', 'Exact baseline Torch runtime required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'One native CPU thread required')
    require(torch.get_default_dtype() == torch.float32, 'Original FP32 constructor default required')
    return torch, dict(torch=torch.__version__, python=sys.version, platform=platform.platform(),
        device='cpu', CPU_threads=1, interop_threads=1, CUDA_VISIBLE_DEVICES='',
        default_dtype=str(torch.get_default_dtype()), deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def modules(provenance):
    return {key: load('mixed_immutable_' + key, PHASE / path) for key, path in provenance['modules'].items()}


def graph_inputs(mods, dataset, frozen, implementation, torch):
    inputs = mods['dblp_inputs'] if dataset == 'HGB-DBLP' else mods['acm_inputs']
    schema, attributes, edges = inputs.stream_schema(frozen['archive'], frozen['members'])
    require(schema['member_sha256'] == frozen['member_sha256']
            and schema['node_counts'] == frozen['expected_node_counts']
            and schema['input_dims'] == frozen['expected_input_dims'], 'Complete graph bytes/counts/features differ')
    require({str(r['raw_id']): [r['source'], r['target']] for r in schema['relations']} == frozen['expected_relations'],
            'Raw relation geometry differs')
    if dataset == 'HGB-ACM':
        require(schema['provided_attribute_widths'] == frozen['expected_feature_widths'], 'ACM provided attributes differ')
        stats = {str(r['raw_id']): {k: r[k] for k in ('raw_records', 'support_edges', 'raw_self_records', 'duplicates_coalesced')}
                 for r in schema['relations']}
        require(stats == frozen['expected_relation_statistics'], 'ACM released relation support differs')
    development = inputs.read_development_labels(frozen['development_labels'], frozen['archive']['sha256'])
    source_hash = development['source_label_member_sha256'] if dataset == 'HGB-DBLP' else development['source_member_sha256']
    require(source_hash == frozen['source_label_member_sha256'], 'Development source label bytes differ')
    graph, features, schema = inputs.materialize(schema, attributes, edges, implementation, frozen['relation_row_order'], torch.device('cpu'))
    return inputs, graph, features, schema, development


def build(mods, dataset, graph, schema, classes, seed):
    args = (graph, schema['input_dims'], classes, seed, seed + 900001, 'cpu', ['global_BE'])
    if dataset == 'HGB-DBLP':
        return mods['families'].build(mods['implementation'], *args)['global_BE']
    return mods['acm_families'].build(mods['implementation'], mods['families'], *args)['global_BE']


def tensor_sha(value):
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(dtype=str(value.dtype), shape=list(value.shape)), sort_keys=True).encode())
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(tensor_sha(value).encode())
    return digest.hexdigest()


def memory():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key = line.split(':', 1)[0]
        if key in ('VmSize', 'VmPeak', 'VmRSS', 'VmHWM'):
            values[key + '_bytes'] = int(line.split()[1]) * 1024
    return values


def preserved(provenance, frozen, sources, freeze_path, release_path, original_bytes):
    packet_guard()
    require((Path(freeze_path).read_bytes(), Path(release_path).read_bytes()) == original_bytes, 'Mixed freeze/release mutated')
    for row in frozen['source_freezes']:
        verify(row['record'])
    verify(frozen['scientific_design'])
    for value in sources.values():
        for row in [value['archive'], value['development_labels']] + [r['descriptor'] for r in value['splits']]:
            verify(row)
