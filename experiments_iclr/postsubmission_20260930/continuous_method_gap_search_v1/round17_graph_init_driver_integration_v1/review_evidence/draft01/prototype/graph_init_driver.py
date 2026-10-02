"""Round17 versioned graph-initialization CLI. SOURCE ONLY, never run by author.

The stdlib admission/source guards precede scientific imports. qualify uses only
TRAIN labels and disposable source-only updates. warm/initialize/fit require an
explicit scientific-phase admission. compare closes all30 cfg0 cells before the
separate once-only report command can open compact final-pool labels.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
PACKET = Path(__file__).resolve().parents[1]
RESEARCH = Path(__file__).resolve().parents[3]
ARMS = ('graph', 'common_only', 'random_tangent', 'topology_permuted', 'warm_copy')
CELLS = (('Squirrel', 'polyformer_mono'), ('Photo', 'polynormer_r'))
SEEDS = (17, 29, 43)
PREPARATION_SHA = '376e8b779d8d71cf21785a62b8543ecadc9c09c9db8174dc230b55b8d0dbd253'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def verified(record):
    require(isinstance(record, dict) and set(record) == {'path', 'sha256'}, 'Exact path/hash descriptor required')
    require(isinstance(record['path'], str) and isinstance(record['sha256'], str)
            and len(record['sha256']) == 64, 'Incomplete descriptor')
    path = Path(record['path']).resolve()
    require(sha(path) == record['sha256'], 'Fingerprint mismatch: '+str(path))
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {'unserialized_type': type(value).__name__}


def write_json(path, value):
    with open(path, 'x') as handle:
        json.dump(json_safe(value), handle, indent=2, allow_nan=False)
        handle.write('\n')


def append_trace(path, value):
    with open(path, 'a') as handle:
        handle.write(json.dumps(json_safe(value), allow_nan=False)+'\n')


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def tree_records(root, exclude=()):
    return [dict(path=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size)
            for path in sorted(root.rglob('*')) if path.is_file() and str(path.relative_to(root)) not in exclude]


def verify_tree(root, records, exact=False, exclude=()):
    names = []
    for row in records:
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe payload path')
        path = root/relative
        require(path.is_file() and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'],
                'Frozen payload changed: '+str(path))
        names.append(str(relative))
    require(len(names) == len(set(names)), 'Duplicate payload identity')
    if exact:
        require({row['path'] for row in tree_records(root, exclude)} == set(names),
                'Frozen directory file inventory changed')


def source_guard(context):
    """All code fingerprints verified without importing its scientific bodies."""
    require(verified(context['source_bindings']) == PACKET/'SOURCE_BINDINGS.json', 'Wrong local source binding')
    require(verified(context['protocol']) == PACKET/'PROTOCOL.json', 'Wrong local protocol')
    bindings = read_json(PACKET/'SOURCE_BINDINGS.json')
    require(bindings['sealed'] is True, 'Source packet is not sealed')
    require(read_json(PACKET/'PROTOCOL.json')['arms'] == list(ARMS), 'Fixed arm list differs')
    for relative, expected in bindings['local_implementation_sha256'].items():
        require(sha(PACKET/relative) == expected, 'Local operation source changed')
    method = bindings['unchanged_method']
    require(sha(RESEARCH/method['path']) == method['sha256'] ==
            sha(PACKET/'prototype/graph_band_route_initializer.py'), 'Round15 operation must be byte identical')
    modern = bindings['modern_packet']
    modern_root = RESEARCH/modern['path']
    require(sha(modern_root/'MANIFEST.json') == modern['manifest_sha256'], 'Modern v2 seal identity differs')
    require(sha(modern_root/'SEAL.json') == modern['seal_sha256'], 'Modern v2 seal record differs')
    manifest = read_json(modern_root/'MANIFEST.json')
    verify_tree(modern_root, manifest['payload'])
    for name, expected in bindings['modern_model_sha256'].items():
        require(sha(modern_root/'prototype'/name) == expected, 'Pinned modern model/preprocess source changed')
    # The author/native recipe is fixed; only cfg0 is admitted in this smaller amendment.
    require(sha(modern_root/'TEACHER_PROTOCOL.json') == bindings['native_protocol_sha256'],
            'Native schedule recipe changed')
    return modern_root/'prototype'


def context_guard(context):
    expected_keys = {'schema', 'graph', 'backbone', 'seed', 'source_split_index', 'config',
        'role_freeze', 'graph_input', 'source_labels', 'environment', 'device', 'source_bindings',
        'protocol', 'source_label_binding', 'modern_qualification_certificate',
        'roles_frozen_before_label_extraction', 'prior_exposure_disclosure',
        'independent_of_stage1_outcomes', 'resource_forecast'}
    require(set(context) == expected_keys and context['schema'] == 'graph-init-cell-context-v1',
            'Unexpected/incomplete cell context; final labels are forbidden here')
    require((context['graph'], context['backbone']) in CELLS and context['seed'] in SEEDS
            and context['config'] == 0 and context['source_split_index'] == SEEDS.index(context['seed']),
            'Only two exact graphs, three paired seeds and cfg0 are admitted')
    require(set(context['source_labels']) == {'train', 'validation'}, 'Exactly compact train/validation descriptors')
    require(context['roles_frozen_before_label_extraction'] is True and
            context['independent_of_stage1_outcomes'] is True and
            isinstance(context['prior_exposure_disclosure'], str) and context['prior_exposure_disclosure'].strip(),
            'Complete source provenance and exposure disclosure required')
    require(isinstance(context['resource_forecast'], dict) and context['resource_forecast'].get('admitted') is True
            and isinstance(context['resource_forecast'].get('forecast'), dict)
            and isinstance(context['resource_forecast'].get('evidence'), list)
            and bool(context['resource_forecast']['evidence']),
            'A concrete whole-graph resource forecast must be admitted')
    for record in context['resource_forecast']['evidence']:
        verified(record)
    require(context['environment']['deterministic_algorithms'] is True and
            (context['device'] == 'cpu' or context['environment']['cublas_workspace_config'] == ':4096:8'),
            'Uniform admitted deterministic runtime required before qualification')
    source_directory = source_guard(context)
    role_path = verified(context['role_freeze'])
    role = read_json(role_path)
    require(role['role_derivation_version'] == 'derived_roles_v2' and
            role['preparation_driver_sha256'] == PREPARATION_SHA and
            role['graph'] == context['graph'] and role['seed'] == context['seed'] and
            role['source_split_index'] == context['source_split_index'] and role['labels_read'] is False,
            'Immutable label-blind derived-role provenance differs')
    require(role['graph_input'] == context['graph_input'], 'Graph input must match exact role acquisition')
    verify_tree(role_path.parent, role['payload'])
    graph_input = read_json(verified(context['graph_input']))
    require(graph_input['graph'] == context['graph'], 'Graph input/name mismatch')
    verified(graph_input['features']); verified(graph_input['edges'])
    for record in context['source_labels'].values():
        verified(record)  # Hash only; qualification never opens validation labels.
    modern_certificate_guard(context, source_directory, role, graph_input)
    return source_directory


def modern_certificate_guard(context, source_directory, role, graph_input):
    """Exact v2 certificate, source extraction bundle and supplemented audits.

    V1 passed-only receipts and unrelated short qualifiers cannot admit a cell.
    Modern numerical tests on seed17 and audited target-role coverage remain
    distinct; round17 independently tests each actual target/warm function.
    """
    custody = load_module('graph_init_modern_custody', source_directory/'modern_custody.py')
    modern_admission = dict(graph=context['graph'], backbone=context['backbone'], family='gnnm_boundary_4',
        config=0, environment=context['environment'], role_freeze=context['role_freeze'],
        source_labels=context['source_labels'], source_label_binding=context['source_label_binding'],
        implementation_sha256=custody.implementations())
    identity = custody.source_identity(modern_admission, role, graph_input)
    certificate = read_json(verified(context['modern_qualification_certificate']))
    require(certificate['schema'] == 'modern-teacher-qualification-certificate-v2' and
            certificate['admission_eligible'] is True and certificate['report_eligible'] is False and
            certificate['qualified_config'] == 0 and certificate['backbone'] == context['backbone'] and
            certificate['family'] == 'gnnm_boundary_4' and certificate['environment'] == context['environment'] and
            certificate['input_identity'] == certificate['authorized_target_input'] == identity and
            certificate['numerical_tested_seeds'] == [17] and
            certificate['tested_seed17_input']['seed'] == 17 and
            certificate['numerical_tests_on_target_seed'] is (context['seed'] == 17) and
            certificate['model_sha256'] == custody.models() and
            certificate['implementation_sha256'] == custody.implementations() and
            certificate['preprocessing'] == custody.preprocessing(modern_admission, identity) and
            certificate['source_seal'] == descriptor(source_directory.parent/'SEAL.json') and
            certificate['full_schedule_feasibility_separately_required'] is True,
            'Exact modern cfg0 graph/seed/role/labels/runtime supplemented certificate required')
    for key in ('partial_receipt', 'external_audit', 'source_seal', 'tested_source_seal',
                'tested_admission', 'request', 'preprocessing_evidence', 'coverage_authorization'):
        verified(certificate[key])
    tested = read_json(verified(certificate['tested_admission']))
    tested_identity = custody.source_identity(dict(tested, source_label_binding=context['source_label_binding']),
        read_json(verified(tested['role_freeze'])), read_json(verified(context['graph_input'])))
    require(tested_identity == certificate['tested_seed17_input'] and
            tested['environment'] == context['environment']
            and tested['backbone'] == context['backbone'] and tested['family'] == 'gnnm_boundary_4'
            and tested['config'] == 0, 'Modern certificate tested a different native graph/family/config')
    coverage = read_json(verified(certificate['coverage_authorization']))
    require(coverage['schema'] == 'modern-exact-role-label-coverage-v2' and
            coverage['tested_seed17_input'] == tested_identity and
            coverage['backbone'] == context['backbone'] and coverage['family'] == 'gnnm_boundary_4' and
            coverage['environment'] == context['environment'] and coverage['model_sha256'] == custody.models() and
            set(coverage['checks']) == set(custody.COVERAGE_CHECKS) and
            all(value is True for value in coverage['checks'].values()) and
            coverage['numerical_tests_on_untested_seeds'] is False and
            len(coverage['authorized_target_inputs']) == 3 and identity in coverage['authorized_target_inputs'] and
            {case['seed'] for case in coverage['authorized_target_inputs']} == set(SEEDS),
            'Explicit exact-target coverage must remain distinct from tested seed17')
    for case in coverage['authorized_target_inputs']:
        case_admission = dict(modern_admission, role_freeze=case['role_freeze'], source_labels=case['source_labels'],
                              source_label_binding=case['source_label_binding'])
        case_role = read_json(verified(case['role_freeze']))
        require(custody.source_identity(case_admission, case_role, read_json(verified(case_role['graph_input']))) == case and
                all(case[key] == tested_identity[key] for key in ('graph', 'backbone', 'graph_input',
                    'source_label_binding', 'provider_row_identity', 'dimensions')), 'Coverage changes graph/provider/extraction')
    require(isinstance(coverage['evidence'], list) and bool(coverage['evidence']), 'Missing coverage evidence')
    for evidence in coverage['evidence']:
        verified(evidence)
    external = read_json(verified(certificate['external_audit']))
    require(external['schema'] == 'modern-external-runtime-audit-v2' and
            external['tested_seed17_input'] == tested_identity and
            external['coverage_authorization'] == certificate['coverage_authorization'] and
            external['partial_receipt'] == certificate['partial_receipt'] and
            set(external['checks']) == set(custody.AUDITS) and all(value is True for value in external['checks'].values())
            and external['frozen_tolerances'] == dict(logit_rtol=1e-5, logit_atol=1e-6,
                gradient_rtol=1e-4, gradient_atol=1e-6)
            and external['failed_attempts_and_fresh_version_retries_retained'] is True,
            'Modern external native optimizer/member/restore evidence incomplete')
    for key in ('attempt_history', 'instrumentation_sources', 'evidence'):
        require(isinstance(external[key], list) and bool(external[key]), 'Missing modern qualification evidence')
        for record in external[key]:
            verified(record)
    return certificate


def load_module(name, path):
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).resolve() == path.resolve(), 'Stale conflicting module: '+name)
        return sys.modules[name]
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def runtime(device):
    import numpy as np
    import scipy
    import torch
    import torch_geometric
    require(device == 'cpu' or (device.startswith('cuda') and torch.cuda.is_available()), 'Device unavailable')
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    require(device == 'cpu' or os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA runtime initialization')
    environment = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        torch=torch.__version__, torch_geometric=torch_geometric.__version__, cuda=torch.version.cuda,
        device=device, gpu=torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    return np, torch, environment


def load_runtime(context, source_directory):
    np, torch, environment = runtime(context['device'])
    require(environment == context['environment'], 'Frozen execution environment changed')
    for name in ('native_polyformer', 'native_polyformer_outer', 'native_polyformer_preprocess',
                 'native_polynormer', 'backbone_boundary_adapter', 'modern_teacher_adapter'):
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve() == (source_directory/(name+'.py')).resolve(),
                    'Unadmitted modern module already loaded: '+name)
    sys.path.insert(0, str(source_directory))
    boundary = load_module('backbone_boundary_adapter', source_directory/'backbone_boundary_adapter.py')
    adapter = load_module('modern_teacher_adapter', source_directory/'modern_teacher_adapter.py')
    integration = load_module('graph_init_training_adapter', PACKET/'prototype/graph_init_training_adapter.py')
    method = load_module('graph_band_route_initializer', PACKET/'prototype/graph_band_route_initializer.py')
    return SimpleNamespace(np=np, torch=torch, adapter=adapter, boundary=boundary, integration=integration,
                           method=method, device=context['device'], environment=environment)


def sync(rt):
    if rt.device.startswith('cuda'):
        rt.torch.cuda.synchronize(rt.device)


class Ledger:
    def __init__(self, out):
        self.out, self.costs = out, []

    def sink(self, name, value):
        append_trace(self.out/'interface_trace.jsonl', dict(event=name, receipt=value))

    def measured(self, rt, name, function):
        sync(rt)
        if rt.device.startswith('cuda'):
            rt.torch.cuda.reset_peak_memory_stats(rt.device)
        start = time.perf_counter()
        status = 'failed'
        try:
            result = function()
            status = 'completed'
            return result
        finally:
            sync(rt)
            cost = dict(operation=name, status=status, seconds=time.perf_counter()-start,
                peak_allocated_bytes=rt.torch.cuda.max_memory_allocated(rt.device)
                    if rt.device.startswith('cuda') else None,
                peak_reserved_bytes=rt.torch.cuda.max_memory_reserved(rt.device)
                    if rt.device.startswith('cuda') else None,
                process_maxrss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cpu_peak_scope='process lifetime, not isolated phase peak')
            self.costs.append(cost)
            append_trace(self.out/'cost_trace.jsonl', cost)


def canonical_graph(rt, manifest):
    np = rt.np
    x = np.load(verified(manifest['features']), allow_pickle=False)
    edges = np.load(verified(manifest['edges']), allow_pickle=False)
    n, f, c = manifest['num_nodes'], manifest['num_features'], manifest['num_classes']
    require(x.dtype == np.float32 and x.shape == (n, f) and np.isfinite(x).all(), 'Finite full FP32 features required')
    require(edges.dtype == np.int64 and edges.shape == (2, manifest['num_edges']) and c >= 2,
            'Canonical full edge shape/dtype differs')
    require((edges >= 0).all() and (edges < n).all() and (edges[0] < edges[1]).all(), 'Require0<=u<v<N')
    require(np.unique(edges.T, axis=0).shape[0] == edges.shape[1] and
            np.array_equal(np.lexsort((edges[1], edges[0])), np.arange(edges.shape[1])),
            'Canonical edges must be sorted unique undirected pairs')
    return x, edges


def load_labels(rt, record, expected_nodes, classes):
    with rt.np.load(verified(record), allow_pickle=False) as pack:
        require(set(pack.files) == {'nodes', 'labels'}, 'Compact pack needs exactly nodes/labels')
        nodes, labels = pack['nodes'], pack['labels']
        require(nodes.dtype == labels.dtype == rt.np.int64 and rt.np.array_equal(nodes, expected_nodes)
                and labels.shape == nodes.shape and ((labels >= 0) & (labels < classes)).all(),
                'Label pack must match exact committed sorted role identities/classes')
        return SimpleNamespace(nodes=rt.torch.from_numpy(nodes.copy()).to(rt.device),
                               labels=rt.torch.from_numpy(labels.copy()).to(rt.device))


def source_inputs(rt, context, ledger, validation):
    manifest = read_json(verified(context['graph_input']))
    x_np, e_np = ledger.measured(rt, 'verified_graph_load', lambda: canonical_graph(rt, manifest))
    x, edges = rt.torch.from_numpy(x_np.copy()).to(rt.device), rt.torch.from_numpy(e_np.copy()).to(rt.device)
    graph = ledger.measured(rt, 'full_native_preprocessing', lambda: rt.adapter.prepare_graph(
        x, edges, context['backbone'], context['environment'],
        read_json(verified(context['modern_qualification_certificate']))['preprocessing']['input_binding']))
    require(graph.preprocessing == read_json(verified(context['modern_qualification_certificate']))['preprocessing'],
            'Actual native preprocessing differs from per-cell qualified identity')
    native = rt.adapter.NATIVE[context['backbone']]
    require((manifest['num_nodes'], manifest['num_features'], manifest['num_classes']) ==
            (native['nodes'], native['features'], native['classes']), 'Exact full native dimensions required')
    cell = verified(context['role_freeze']).parent
    ids = {name: rt.np.load(cell/(name+'_nodes.npy'), allow_pickle=False)
           for name in ('train', 'validation', 'pool')}
    for value in ids.values():
        require(value.dtype == rt.np.int64 and value.ndim == 1 and len(value) > 0 and
                rt.np.all(value[1:] > value[:-1]) and value.min() >= 0 and value.max() < native['nodes'],
                'Roles must contain nonempty sorted unique whole-graph IDs')
    require(len(rt.np.unique(rt.np.concatenate(list(ids.values())))) == sum(map(len, ids.values())),
            'Train/validation/final roles overlap')
    train = load_labels(rt, context['source_labels']['train'], ids['train'], graph.classes)
    val = load_labels(rt, context['source_labels']['validation'], ids['validation'], graph.classes) if validation else None
    return graph, edges, train, val


def save_logits(rt, out, name, logits):
    require(bool(rt.torch.isfinite(logits).all()) and logits.ndim == 3, 'Finite member logits [K,N,C] required')
    rt.np.save(out/(name+'_member_logits.npy'), logits.detach().cpu().numpy(), allow_pickle=False)


def phase_freeze(out, phase, context, admission_record, ledger, **metadata):
    write_json(out/'COMPLETED.json', dict(phase=phase, completed=True, costs=ledger.costs,
                                        scientific_execution_by_author=False, **metadata))
    freeze = dict(schema='graph-init-phase-freeze-v1', phase=phase, output_root=str(out),
        context=context, context_sha256=object_hash(context), admission=admission_record,
        qualification_only=phase == 'qualify', source_labels_read=['train'] if phase in ('qualify', 'initialize') else ['train', 'validation'],
        final_labels_read=False, costs=ledger.costs, completed=True, **metadata,
        payload=tree_records(out, exclude=('FREEZE.json',)))
    write_json(out/'FREEZE.json', freeze)
    return freeze


def verify_freeze(record, phase=None, context=None):
    path = verified(record)
    freeze = read_json(path)
    require(path.name == 'FREEZE.json' and freeze['schema'] == 'graph-init-phase-freeze-v1'
            and freeze['completed'] is True and freeze['output_root'] == str(path.parent)
            and freeze['final_labels_read'] is False, 'Unfinished/changed phase freeze')
    require(freeze['context_sha256'] == object_hash(freeze['context']), 'Context digest differs')
    if phase is not None:
        require(freeze['phase'] == phase, 'Wrong dependency phase')
    if context is not None:
        require(freeze['context'] == context, 'Dependency belongs to a different common cell context')
    verify_tree(path.parent, freeze['payload'], exact=True, exclude=('FREEZE.json',))
    return freeze, path.parent


def expected_attempts(contexts, anchor):
    rows = []
    for ctx in contexts:
        cell = ctx['graph']+'_seed'+str(ctx['seed'])
        for phase in ('qualify', 'warm', 'initialize', 'fit'):
            for arm in ARMS if phase in ('initialize', 'fit') else (None,):
                key = object_hash(dict(context_sha256=object_hash(ctx), phase=phase, arm=arm))
                output = anchor/phase/cell if arm is None else anchor/phase/cell/arm
                rows.append(dict(key=key, context_sha256=object_hash(ctx), phase=phase,
                                 arm=arm, output=str(output)))
    return sorted(rows, key=lambda row: row['key'])


def register(args):
    """Metadata-only prospective fixed72-phase registry; no model/runtime import."""
    request_record = descriptor(args.request)
    request = read_json(args.request)
    require(set(request) == {'schema', 'registration_authorized', 'study_id', 'contexts',
                            'anchor_directory', 'prior_attempt_registries'} and
            request['schema'] == 'graph-init-registry-request-v1' and request['registration_authorized'] is True
            and isinstance(request['study_id'], str) and request['study_id'].strip(), 'Explicit study registry request required')
    contexts = request['contexts']
    require(len(contexts) == 6 and {(ctx['graph'], ctx['seed']) for ctx in contexts} ==
            {(graph, seed) for graph, _ in CELLS for seed in SEEDS}, 'Complete six-cell source cohort required')
    for context in contexts:
        context_guard(context)
    require(isinstance(request['prior_attempt_registries'], list), 'Prior failure/version disclosure required')
    for prior in request['prior_attempt_registries']:
        verified(prior)
    anchor = Path(request['anchor_directory']).resolve()
    require(Path(args.output).resolve() == anchor/'GRAPH_INIT_ATTEMPT_REGISTRY.json', 'Registry must stay at declared study anchor')
    anchor.mkdir(parents=True, exist_ok=False)
    (anchor/'claims').mkdir(); (anchor/'terminals').mkdir()
    write_json(anchor/'GRAPH_INIT_ATTEMPT_REGISTRY.json', dict(schema='graph-init-attempt-registry-v1',
        study_id=request['study_id'], request=request_record, anchor_directory=str(anchor), contexts=contexts,
        source_bindings=descriptor(PACKET/'SOURCE_BINDINGS.json'), protocol=descriptor(PACKET/'PROTOCOL.json'),
        prior_attempt_registries=request['prior_attempt_registries'],
        attempts=expected_attempts(contexts, anchor), silent_retry_forbidden=True,
        failure_blocks_this_study=True, final_labels_accessible_only_after_all72_completed=True))


def registry_guard(record):
    path = verified(record)
    registry = read_json(path)
    require(registry['schema'] == 'graph-init-attempt-registry-v1' and
            path == Path(registry['anchor_directory'])/'GRAPH_INIT_ATTEMPT_REGISTRY.json' and
            registry['source_bindings'] == descriptor(PACKET/'SOURCE_BINDINGS.json') and
            registry['protocol'] == descriptor(PACKET/'PROTOCOL.json') and
            registry['silent_retry_forbidden'] is True and registry['failure_blocks_this_study'] is True,
            'Wrong/moved study attempt registry')
    contexts = registry['contexts']
    require(len(contexts) == 6 and {(ctx['graph'], ctx['seed']) for ctx in contexts} ==
            {(graph, seed) for graph, _ in CELLS for seed in SEEDS}, 'Changed registered source cohort')
    require(registry['attempts'] == expected_attempts(contexts, path.parent), 'Changed registered phase/arm inventory')
    verified(registry['request'])
    for prior in registry['prior_attempt_registries']:
        verified(prior)
    return registry, path.parent


def claim_attempt(registry_record, admission_record, context, phase, arm, out):
    registry, anchor = registry_guard(registry_record)
    require(context in registry['contexts'], 'Unregistered source context')
    key = object_hash(dict(context_sha256=object_hash(context), phase=phase, arm=arm))
    rows = [row for row in registry['attempts'] if row['key'] == key]
    require(len(rows) == 1 and rows[0]['output'] == str(out), 'Use exact prospectively registered output directory')
    claim = anchor/'claims'/(key+'.json')
    terminal = anchor/'terminals'/(key+'.json')
    require(not terminal.exists(), 'Registered attempt already ended; no replacement in a new output directory')
    write_json(claim, dict(schema='graph-init-attempt-claim-v1', attempt_registry=registry_record,
        admission=admission_record, attempt=rows[0], started_unix=time.time(), automatic_retry=False))
    return descriptor(claim), terminal


def registry_closure(record):
    registry, anchor = registry_guard(record)
    terminals = []
    for row in registry['attempts']:
        claim_path, terminal_path = anchor/'claims'/(row['key']+'.json'), anchor/'terminals'/(row['key']+'.json')
        claim, terminal = read_json(claim_path), read_json(terminal_path)
        require(claim['attempt'] == row and claim['attempt_registry'] == record and
                terminal['schema'] == 'graph-init-attempt-terminal-v1' and terminal['completed'] is True and
                terminal['claim'] == descriptor(claim_path) and terminal['final_labels_read'] is False,
                'Incomplete/failed registered attempt blocks source closure')
        freeze, _ = verify_freeze(terminal['freeze'], row['phase'])
        require(freeze['context_sha256'] == row['context_sha256'] and
                freeze.get('arm') == row['arm'] and freeze['attempt_claim'] == descriptor(claim_path)
                and freeze['attempt_registry'] == record, 'Terminal/source-freeze custody differs')
        terminals.append(dict(attempt=row, claim=descriptor(claim_path), terminal=descriptor(terminal_path),
                              freeze=terminal['freeze']))
    require(len(terminals) == 72, 'All72 fixed phase attempts must complete')
    return registry, terminals


def guard_dependencies(admission, phase):
    context, deps = admission['context'], admission['dependencies']
    if phase == 'qualify':
        require(deps == {}, 'Qualification may not consume a fitted checkpoint')
        return {}
    expected = {'warm': {'qualification'}, 'initialize': {'warm'}, 'fit': {'initialization'}}[phase]
    require(set(deps) == expected, 'Exact phase dependencies required')
    if phase == 'warm':
        qualifier, directory = verify_freeze(deps['qualification'], 'qualify', context)
        require(qualifier['passed'] is True and qualifier['qualification_only'] is True,
                'Whole-graph disposable qualification must pass before warm fit')
        return dict(qualification=qualifier, qualification_directory=directory)
    if phase == 'initialize':
        warm, directory = verify_freeze(deps['warm'], 'warm', context)
        qualifier, _ = verify_freeze(warm['qualification_freeze'], 'qualify', context)
        require(qualifier['passed'] is True, 'Qualified source context required')
        return dict(warm=warm, warm_directory=directory)
    initialization, directory = verify_freeze(deps['initialization'], 'initialize', context)
    warm, warm_directory = verify_freeze(initialization['warm_freeze'], 'warm', context)
    qualifier, _ = verify_freeze(warm['qualification_freeze'], 'qualify', context)
    require(initialization['passed'] is True and qualifier['passed'] is True
            and initialization['arm'] == admission['arm'], 'Exact qualified initialized arm required')
    return dict(initialization=initialization, initialization_directory=directory,
                warm=warm, warm_directory=warm_directory)


def bind_and_qualify(rt, native, graph, train, ledger):
    k1 = ledger.measured(rt, 'K1_common_clone', lambda: rt.integration.clone_boundary(native, rt.boundary, 1))
    k4 = ledger.measured(rt, 'K4_identity_clone', lambda: rt.integration.clone_boundary(native, rt.boundary, 4))
    ledger.sink('identity_logits', ledger.measured(rt, 'native_K1_K4_logit_equality',
        lambda: rt.integration.identity_logits_audit(native, k1, k4, graph)))
    k1.eval(); k4.eval()
    theta0, closure, binding = rt.method.bind_common_model(k1,
        rt.integration.DISPLAY[graph.teacher_backbone], rt.integration.raw_arguments(graph, graph.teacher_backbone),
        lambda result: result[0])
    counter = {'calls_started': 0, 'calls_completed': 0}

    def observed(theta):
        counter['calls_started'] += 1
        ledger.sink('closure_forward_started', dict(call=counter['calls_started']))
        result = closure(theta)
        counter['calls_completed'] += 1
        return result

    qualification = ledger.measured(rt, 'actual_common_function_gradient_qualification', lambda:
        rt.method.qualify_gradient_interface(observed, theta0, train.nodes, train.labels,
                                             native.specification['seed']+90000))
    ledger.sink('gradient_qualification', qualification)
    require(qualification['passed'] is True, 'Actual predictive-function AD qualification failed')
    return k1, k4, theta0, observed, binding, qualification, counter


def normalized_S(rt, graph, edges, ledger):
    return ledger.measured(rt, 'normalized_filter_adjacency', lambda: rt.method.symmetric_normalized_adjacency(
        graph.teacher_input.shape[0], edges, graph.teacher_input.dtype, rt.device))


def qualify_body(rt, admission, out, ledger):
    context = admission['context']
    graph, edges, train, _ = source_inputs(rt, context, ledger, validation=False)
    spec = rt.adapter.specification(context['backbone'], 'single_author', 0, context['seed'])
    native = ledger.measured(rt, 'disposable_native_construction', lambda: rt.adapter.TeacherFamily(spec).to(rt.device))
    optimizer = rt.adapter.optimizer_for(native)
    native.eval()
    # Source-only disposable steps populate genuine nonzero Adam moments. Both
    # Photo stages are exercised; no useful state/checkpoint is exported.
    stages = (False, True) if context['backbone'] == 'polynormer_r' else (False,)
    for global_stage in stages:
        native.set_global_stage(global_stage)
        native.eval()
        def populate():
            optimizer.zero_grad(set_to_none=True)
            logits = native(graph)[0]
            loss = rt.torch.nn.functional.cross_entropy(logits[train.nodes], train.labels)
            require(bool(rt.torch.isfinite(loss)), 'Nonfinite disposable qualification loss')
            loss.backward(); optimizer.step()
            return dict(global_stage=global_stage, train_loss=float(loss.detach()), scientific_result=False)
        ledger.sink('disposable_Adam_population', ledger.measured(rt, 'qualification_state_population', populate))
    frozen = rt.integration.named_optimizer_snapshot(native, optimizer)
    equivalence = ledger.measured(rt, 'disposable_one_step_native_K4_Adam_equivalence', lambda:
        rt.integration.optimizer_equivalence_audit(native, frozen, rt.boundary, graph, train, ledger.sink))
    k1, k4, theta0, closure, binding, ad, counter = bind_and_qualify(rt, native, graph, train, ledger)
    S = normalized_S(rt, graph, edges, ledger)
    target_nodes = rt.torch.arange(len(graph.teacher_input), device=rt.device, dtype=rt.torch.int64)
    slices, stats = ledger.measured(rt, 'disposable_fixed_graph_initializer_qualification', lambda:
        rt.method.initialize_four_routes(closure, theta0, S, target_nodes, train.nodes, train.labels))
    rt.method.install_factor_slices(k4, slices, rt.integration.DISPLAY[context['backbone']])
    with rt.torch.no_grad():
        actual = k4(*rt.integration.raw_arguments(graph, context['backbone']))
        reference = rt.torch.stack([closure(row) for row in slices])
    installation = rt.integration.difference(actual, reference, 1e-6, 1e-5)
    ledger.sink('disposable_initializer', stats); ledger.sink('installed_K4_vs_closure', installation)
    require(installation['passed'], 'K4 installation/closure correspondence failed')
    write_json(out/'QUALIFICATION.json', dict(passed=True, equivalence=equivalence, ad=ad,
        binding=binding, installation=installation, initializer=stats, closure_count=counter,
        whole_graph=True, label_scope=['train'], disposable_only=True, scientific_result=False,
        predictive_stage='global' if context['backbone'] == 'polynormer_r' else 'native'))
    return dict(passed=True, whole_graph=True, nonzero_Adam_history_checked=True,
                predictive_stage='global' if context['backbone'] == 'polynormer_r' else 'native')


def warm_body(rt, admission, dependency, out, ledger):
    context = admission['context']
    graph, _, train, validation = source_inputs(rt, context, ledger, validation=True)
    spec = rt.adapter.specification(context['backbone'], 'single_author', 0, context['seed'])
    native, optimizer, checkpoint = ledger.measured(rt, 'complete_fixed_native_warm', lambda:
        rt.integration.warm_native(rt.adapter, spec, graph, train, validation,
            lambda row: append_trace(out/'warm_trace.jsonl', row),
            lambda local: rt.torch.save(local, out/'native_local_transition.pt')))
    rt.torch.save(checkpoint, out/'warm_checkpoint.pt')
    logits, value = ledger.measured(rt, 'warm_endpoint_logits', lambda:
                                   rt.integration.evaluate(native, graph, validation))
    save_logits(rt, out, 'warm', logits)
    write_json(out/'WARM_RECEIPT.json', dict(specification=spec, metadata=checkpoint['metadata'],
        primary_validation_nll=value, preprocessing=graph.preprocessing,
        storage=rt.integration.storage_receipt(native), warm_checkpoint=descriptor(out/'warm_checkpoint.pt'),
        R_S_identity_and_fixed=True, native_shared_weights_and_biases_trained=True))
    return dict(qualification_freeze=admission['dependencies']['qualification'],
        warm_checkpoint=descriptor(out/'warm_checkpoint.pt'), fixed_common_warm=True,
        warm_updates=checkpoint['metadata']['actual_updates'], preprocessing=graph.preprocessing)


def initialize_body(rt, admission, dependency, out, ledger):
    context, arm = admission['context'], admission['arm']
    graph, edges, train, _ = source_inputs(rt, context, ledger, validation=False)
    warm_path = dependency['warm_directory']/'warm_checkpoint.pt'
    checkpoint = rt.torch.load(warm_path, map_location=rt.device, weights_only=True)
    native, restored_optimizer = ledger.measured(rt, 'warm_native_restore', lambda:
                               rt.integration.restore_native(rt.adapter, checkpoint, rt.device))
    del restored_optimizer
    require(checkpoint['specification'] == rt.adapter.specification(context['backbone'], 'single_author', 0,
            context['seed']) and graph.preprocessing == dependency['warm']['preprocessing'], 'Warm context differs')
    equivalence = ledger.measured(rt, 'actual_warm_one_step_native_K4_Adam_equivalence', lambda:
        rt.integration.optimizer_equivalence_audit(native, checkpoint['optimizer'], rt.boundary, graph, train, ledger.sink))
    k1, k4, theta0, closure, binding, qualification, counter = bind_and_qualify(rt, native, graph, train, ledger)
    optimizer, transport = ledger.measured(rt, 'named_warm_Adam_transport', lambda:
        rt.integration.transport_optimizer(native, checkpoint['optimizer'], k4))
    if arm == 'warm_copy':
        slices = theta0.detach().repeat(4, 1)
        stats = dict(operation='unchanged_warm_copy_v1', status='unchanged_warm_copy',
            reason='prospective_cloning_null', attempts=[], vjp_calls=0, jvp_calls=0,
            graph_sparse_products=0, line_search_forward_calls=0, accepted_alpha=0.0)
    else:
        # The immutable API requires sparse S even for common_only; no graph
        # normalization/permutation/band/JVP work is added to that cheapest arm.
        if arm == 'common_only':
            n = len(graph.teacher_input)
            S = rt.torch.sparse_coo_tensor(rt.torch.empty((2, 0), dtype=rt.torch.int64, device=rt.device),
                rt.torch.empty(0, dtype=theta0.dtype, device=rt.device), (n, n)).coalesce()
        else:
            S = normalized_S(rt, graph, edges, ledger)
        permutation_record = None
        if arm == 'topology_permuted':
            S, permutation = ledger.measured(rt, 'node_topology_permutation', lambda:
                rt.method.permute_topology_nodes(S, context['seed']+80000))
            rt.np.save(out/'topology_permutation.npy', permutation.cpu().numpy(), allow_pickle=False)
            permutation_record = descriptor(out/'topology_permutation.npy')
        targets = rt.torch.arange(len(graph.teacher_input), dtype=rt.torch.int64, device=rt.device)
        mode = 'random_tangent' if arm == 'random_tangent' else 'common_only' if arm == 'common_only' else 'graph'
        slices, stats = ledger.measured(rt, 'fixed_initializer_'+arm, lambda: rt.method.initialize_four_routes(
            closure, theta0, S, targets, train.nodes, train.labels, tangent_mode=mode,
            control_seed=context['seed']+70000 if arm == 'random_tangent' else None))
        stats['topology_permutation'] = permutation_record
    ledger.sink('all_initializer_attempts_and_fallbacks', stats)
    rt.method.install_factor_slices(k4, slices, rt.integration.DISPLAY[context['backbone']])
    k4.eval()
    with rt.torch.no_grad():
        actual = ledger.measured(rt, 'actual_K4_installed_logits', lambda:
            k4(*rt.integration.raw_arguments(graph, context['backbone'])))
        expected = ledger.measured(rt, 'four_installed_slice_reference_forwards', lambda:
                                  rt.torch.stack([closure(row) for row in slices]))
    installation = rt.integration.difference(actual, expected, 1e-6, 1e-5)
    ledger.sink('actual_installed_K4_vs_common_closure', installation)
    require(installation['passed'], 'Actual K4 installed factor functions differ from qualified closure')
    save_logits(rt, out, 'initialized', actual)
    rt.np.save(out/'installed_slices.npy', slices.cpu().numpy(), allow_pickle=False)
    # All arms resume the exact post-warm RNG AFTER cloning/qualification/AD.
    # K4 sequential full trajectories consume independent dropout draws, while
    # every matched arm begins with the same RNG arrays and member order.
    rt.integration.rng_restore(checkpoint['rng'])
    initialized = dict(schema='graph-init-initialized-checkpoint-v1', state=rt.integration.cpu_copy(k4.state_dict()),
        optimizer=rt.integration.named_optimizer_snapshot(k4, optimizer), rng=rt.integration.rng_snapshot(),
        arm=arm, context_sha256=object_hash(context), warm_checkpoint=descriptor(warm_path),
        global_stage=native.global_stage, factor_binding=binding, transport_operation=rt.integration.TRANSPORT)
    rt.torch.save(initialized, out/'initialized_checkpoint.pt')
    write_json(out/'INITIALIZATION.json', dict(arm=arm, method=stats, actual_warm_qualification=qualification,
        actual_warm_optimizer_equivalence=equivalence, installation=installation, transport=transport,
        factor_binding=binding, closure_count=counter, exact_warm_RNG_restored=True,
        storage=dict(native=rt.integration.storage_receipt(native), K1=rt.integration.storage_receipt(k1),
            K4=rt.integration.storage_receipt(k4), detached_functional_snapshots_additional=True),
        label_scope_for_initializer=['train'], validation_used_only_for_endpoint_diagnostics=False))
    return dict(arm=arm, passed=True, warm_freeze=admission['dependencies']['warm'],
        initialized_checkpoint=descriptor(out/'initialized_checkpoint.pt'), initializer_status=stats['status'],
        fallback_reason=stats.get('reason'), same_native_warm_RNG=True,
        gradient_interface_and_optimizer_equivalence_passed=True, preprocessing=graph.preprocessing)


def fit_body(rt, admission, dependency, out, ledger):
    context = admission['context']
    graph, _, train, validation = source_inputs(rt, context, ledger, validation=True)
    warm_path = dependency['warm_directory']/'warm_checkpoint.pt'
    warm = rt.torch.load(warm_path, map_location=rt.device, weights_only=True)
    native, restored_optimizer = ledger.measured(rt, 'warm_donor_restore_for_continuation', lambda:
                               rt.integration.restore_native(rt.adapter, warm, rt.device))
    raw = ledger.measured(rt, 'K4_continuation_construction', lambda:
                          rt.integration.clone_boundary(native, rt.boundary, 4))
    init_path = dependency['initialization_directory']/'initialized_checkpoint.pt'
    initial = rt.torch.load(init_path, map_location=rt.device, weights_only=True)
    require(initial['schema'] == 'graph-init-initialized-checkpoint-v1' and initial['arm'] == admission['arm']
            and initial['context_sha256'] == object_hash(context) and initial['warm_checkpoint'] == descriptor(warm_path)
            and initial['transport_operation'] == rt.integration.TRANSPORT, 'Initialized checkpoint custody differs')
    raw.load_state_dict(initial['state'])
    if context['backbone'] == 'polynormer_r':
        require(initial['global_stage'] is True, 'Photo continuation must use final global branch')
        raw.set_global_stage(True)
    optimizer = rt.integration.restore_named_optimizer(raw, initial['optimizer'])
    rt.integration.rng_restore(initial['rng'])
    # Delete donor before long continuation; construction peak was charged above.
    del native, restored_optimizer
    best, selection = ledger.measured(rt, 'complete_native_budget_continuation', lambda:
        rt.integration.continuation(raw, optimizer, graph, train, validation,
            lambda row: append_trace(out/'continuation_trace.jsonl', row),
            lambda name, logits: save_logits(rt, out, name, logits)))
    rt.torch.save(dict(schema='graph-init-selected-checkpoint-v1', arm=admission['arm'],
        context_sha256=object_hash(context), global_stage=context['backbone'] == 'polynormer_r', **best),
        out/'selected_checkpoint.pt')
    write_json(out/'SELECTION.json', dict(selection=selection, storage=rt.integration.storage_receipt(raw),
        preprocessing=graph.preprocessing, selected_checkpoint=descriptor(out/'selected_checkpoint.pt'),
        initialized_from=admission['dependencies']['initialization'], warm_checkpoint=descriptor(warm_path),
        label_scope=['train', 'validation'], final_pool_labels_read=False, secondary_pool_selected=False))
    return dict(arm=admission['arm'], initialization_freeze=admission['dependencies']['initialization'],
        warm_freeze=dependency['initialization']['warm_freeze'], selection=selection,
        selected_checkpoint=descriptor(out/'selected_checkpoint.pt'),
        selected_logits=descriptor(out/'selected_member_logits.npy'), preprocessing=graph.preprocessing,
        report_eligible=True, complete_declared_continuation=True)


def failure_locals(error):
    """Retain in-flight method stats/epochs from exceptions without editing method."""
    result, frame = [], error.__traceback__
    while frame is not None:
        values = frame.tb_frame.f_locals
        keep = {key: json_safe(values[key]) for key in ('stats', 'records', 'counter', 'epoch', 'completed',
                'total_updates', 'stage', 'attempt', 'branch', 'best_epoch') if key in values}
        if keep:
            result.append(dict(function=frame.tb_frame.f_code.co_name, line=frame.tb_lineno, partial=keep))
        frame = frame.tb_next
    return result


def phase_run(args):
    started = time.perf_counter()
    admission_record = descriptor(args.admission)
    admission = read_json(args.admission)
    require(set(admission) == {'schema', 'execution_authorized', 'authorized_phase', 'context', 'dependencies',
                               'arm', 'attempt_registry'}
            and admission['schema'] == 'graph-init-phase-admission-v1'
            and admission['execution_authorized'] is True and admission['authorized_phase'] == args.command,
            'Complete explicit admission for this exact phase required')
    require(admission['arm'] in ARMS if args.command in ('initialize', 'fit') else admission['arm'] is None,
            'Fixed arm required only at initialize/fit')
    source_directory = context_guard(admission['context'])
    dependency = guard_dependencies(admission, args.command)
    require(descriptor(args.admission) == admission_record, 'Admission changed during guarding')
    out = Path(args.output).resolve()
    claim_record, terminal_path = claim_attempt(admission['attempt_registry'], admission_record,
        admission['context'], args.command, admission['arm'], out)
    ledger = Ledger(out)
    out_created = False
    try:
        out.mkdir(parents=True, exist_ok=False)
        out_created = True
        write_json(out/'ATTEMPT_STARTED.json', dict(admission=admission_record, command=args.command,
            context_sha256=object_hash(admission['context']), claim=claim_record,
            attempt_registry=admission['attempt_registry'], started_unix=time.time(), automatic_retry=False))
        rt = load_runtime(admission['context'], source_directory)
        if args.command == 'qualify':
            metadata = qualify_body(rt, admission, out, ledger)
        elif args.command == 'warm':
            metadata = warm_body(rt, admission, dependency, out, ledger)
        elif args.command == 'initialize':
            metadata = initialize_body(rt, admission, dependency, out, ledger)
        else:
            metadata = fit_body(rt, admission, dependency, out, ledger)
        metadata['phase_wall_seconds_including_runtime_and_IO'] = time.perf_counter()-started
        metadata['attempt_registry'], metadata['attempt_claim'] = admission['attempt_registry'], claim_record
        phase_freeze(out, args.command, admission['context'], admission_record, ledger, **metadata)
        write_json(terminal_path, dict(schema='graph-init-attempt-terminal-v1', completed=True,
            claim=claim_record, freeze=descriptor(out/'FREEZE.json'), final_labels_read=False,
            phase_wall_seconds=time.perf_counter()-started))
    except Exception as error:
        failed = dict(error_type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), partial_method_and_epoch_receipts=failure_locals(error),
            completed_nested_costs=ledger.costs, phase_wall_seconds=time.perf_counter()-started,
            automatic_retry=False, final_labels_read=False)
        if out_created and not (out/'FAILED_ATTEMPT.json').exists():
            write_json(out/'FAILED_ATTEMPT.json', failed)
        if not terminal_path.exists():
            write_json(terminal_path, dict(schema='graph-init-attempt-terminal-v1', completed=False,
                claim=claim_record, failure=failed, final_labels_read=False))
        raise


def compare(args):
    admission_record = descriptor(args.admission)
    admission = read_json(args.admission)
    require(set(admission) == {'schema', 'execution_authorized', 'authorized_phase', 'expected_contexts',
                               'fit_freezes', 'attempt_registry'}
            and admission['schema'] == 'graph-init-comparison-admission-v1'
            and admission['execution_authorized'] is True and admission['authorized_phase'] == 'compare',
            'Exact comparison admission required')
    contexts = admission['expected_contexts']
    registry, terminals = registry_closure(admission['attempt_registry'])
    require(contexts == registry['contexts'], 'Comparison must close the registered source cohort')
    require(len(contexts) == 6 and len(admission['fit_freezes']) == 30, 'All six paired contexts and thirty arms required')
    expected = {(graph, seed, arm) for graph, _ in CELLS for seed in SEEDS for arm in ARMS}
    require({(ctx['graph'], ctx['seed']) for ctx in contexts} == {(g, s) for g, _ in CELLS for s in SEEDS},
            'Expected source cohort differs')
    for context in contexts:
        context_guard(context)
    by_key = {(ctx['graph'], ctx['seed']): ctx for ctx in contexts}
    rows, seen, warm_by_cell = [], set(), {}
    for record in admission['fit_freezes']:
        fit, directory = verify_freeze(record, 'fit')
        require(any(terminal['freeze'] == record and terminal['attempt']['phase'] == 'fit'
                    for terminal in terminals), 'Fit outside registered completed study')
        ctx = fit['context']
        require(ctx == by_key[(ctx['graph'], ctx['seed'])] and fit['report_eligible'] is True
                and fit['complete_declared_continuation'] is True, 'Ineligible source fit')
        key = (ctx['graph'], ctx['seed'], fit['arm'])
        require(key in expected and key not in seen, 'Duplicate/unregistered arm cell')
        seen.add(key)
        init, _ = verify_freeze(fit['initialization_freeze'], 'initialize', ctx)
        warm, _ = verify_freeze(fit['warm_freeze'], 'warm', ctx)
        qualification, _ = verify_freeze(warm['qualification_freeze'], 'qualify', ctx)
        require(init['passed'] is True and qualification['passed'] is True and
                init['warm_freeze'] == fit['warm_freeze'] and init['arm'] == fit['arm'], 'Changed dependency chain')
        cell = (ctx['graph'], ctx['seed'])
        warm_by_cell.setdefault(cell, fit['warm_freeze'])
        require(warm_by_cell[cell] == fit['warm_freeze'], 'Matched arms did not branch from the same exact warm freeze')
        require(fit['selected_logits'] == descriptor(directory/'selected_member_logits.npy') and
                fit['selected_checkpoint'] == descriptor(directory/'selected_checkpoint.pt'), 'Selection artifact custody differs')
        rows.append(dict(graph=ctx['graph'], backbone=ctx['backbone'], seed=ctx['seed'], arm=fit['arm'],
            fit_freeze=record, context_sha256=fit['context_sha256'], selected_logits=fit['selected_logits'],
            selected_checkpoint=fit['selected_checkpoint'], primary_validation_nll=fit['selection']['primary_validation_nll'],
            selection=fit['selection'], initialization_status=init['initializer_status'],
            fallback_reason=init['fallback_reason'], warm_freeze=fit['warm_freeze'],
            costs=dict(shared_qualification=qualification['costs'], shared_warm=warm['costs'],
                       initialization=init['costs'], continuation=fit['costs']),
            acquisition_record=ctx['role_freeze']))
    require(seen == expected, 'Incomplete narrow comparison; final labels stay closed')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows.sort(key=lambda row: (row['graph'], row['seed'], ARMS.index(row['arm'])))
    write_json(out/'SOURCE_COMPARISON.json', dict(rows=rows, source_selection_closed=True,
        configurations=[0], arms=list(ARMS), cells=30, all_attempts_must_be_retained=True,
        prior_failed_attempts_disclosure='The admissions/source study log must retain any earlier failed versions; no silent retry.',
        scope='cfg0 matched initialization mechanism comparison; strong native-family extension claims remain external'))
    write_json(out/'COMPARISON_FREEZE.json', dict(schema='graph-init-comparison-freeze-v1', output_root=str(out),
        admission=admission_record, expected_contexts=contexts, rows=rows, source_selection_closed=True,
        attempt_registry=admission['attempt_registry'], terminal_closure=terminals,
        complete_cells=30, final_labels_read=False, configs=[0], arms=list(ARMS),
        payload=tree_records(out, exclude=('COMPARISON_FREEZE.json',))))


def verify_comparison(record):
    path = verified(record)
    frozen = read_json(path)
    require(path.name == 'COMPARISON_FREEZE.json' and frozen['schema'] == 'graph-init-comparison-freeze-v1'
            and frozen['output_root'] == str(path.parent) and frozen['source_selection_closed'] is True
            and frozen['complete_cells'] == 30 and frozen['final_labels_read'] is False
            and frozen['configs'] == [0] and frozen['arms'] == list(ARMS), 'Incomplete final-report source freeze')
    verify_tree(path.parent, frozen['payload'], exact=True, exclude=('COMPARISON_FREEZE.json',))
    registry, terminals = registry_closure(frozen['attempt_registry'])
    require(terminals == frozen['terminal_closure'] and registry['contexts'] == frozen['expected_contexts'],
            'Registered phase closure changed after source freeze')
    require(len(frozen['rows']) == 30 and len(frozen['expected_contexts']) == 6, 'Changed comparison inventory')
    expected = {(g, s, a) for g, _ in CELLS for s in SEEDS for a in ARMS}
    require({(row['graph'], row['seed'], row['arm']) for row in frozen['rows']} == expected,
            'Missing/duplicate report source cells')
    for ctx in frozen['expected_contexts']:
        context_guard(ctx)
    for row in frozen['rows']:
        fit, _ = verify_freeze(row['fit_freeze'], 'fit')
        require(fit['context_sha256'] == row['context_sha256'] and fit['selection'] == row['selection']
                and fit['selected_logits'] == row['selected_logits'] and fit['report_eligible'] is True,
                'Changed fitted source selection')
        init, _ = verify_freeze(fit['initialization_freeze'], 'initialize', fit['context'])
        warm, _ = verify_freeze(fit['warm_freeze'], 'warm', fit['context'])
        qualifier, _ = verify_freeze(warm['qualification_freeze'], 'qualify', fit['context'])
        require(init['passed'] and qualifier['passed'], 'Unqualified source chain')
    return frozen, path.parent


def numpy_metrics(np, logits, nodes, labels):
    z = logits[:, nodes].astype(np.float64)
    require(np.isfinite(z).all(), 'Nonfinite frozen logits')
    def probabilities_and_nll(values):
        shifted = values-values.max(axis=-1, keepdims=True)
        log_normalizer = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        logp = shifted-log_normalizer
        return np.exp(logp), -logp[..., np.arange(len(labels)), labels].mean(axis=-1), logp
    member_p, member_nll, member_logp = probabilities_and_nll(z)
    pooled_p, pooled_nll, _ = probabilities_and_nll(z.mean(0))
    prediction = pooled_p.argmax(-1)
    errors = z.argmax(-1) != labels[None, :]
    residual = member_p-np.eye(z.shape[-1])[labels][None, :, :]
    flat = residual.reshape(len(z), -1)
    flat = flat-flat.mean(axis=1, keepdims=True)
    covariance = flat@flat.T/max(flat.shape[1]-1, 1)
    sd = np.sqrt(np.diag(covariance))
    denominator = sd[:, None]*sd[None, :]
    correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
    secondary = member_p.mean(0)
    true_logp = member_logp[:, np.arange(len(labels)), labels]
    log_max = true_logp.max(0)
    secondary_nll = float(-(log_max+np.log(np.exp(true_logp-log_max).mean(0))).mean())
    return dict(primary_nll=float(pooled_nll), primary_accuracy=float((prediction == labels).mean()),
        member_nll=np.asarray(member_nll).tolist(), member_accuracy=(~errors).mean(1).tolist(),
        weakest_member_accuracy=float((~errors).mean(1).min()),
        error_overlap_rates=(errors.astype(np.float64)@errors.T/len(labels)).tolist(),
        class_residual_covariance=covariance.tolist(), class_residual_correlation=correlation.tolist(),
        residual_zero_variance_members=(sd == 0).tolist(),
        residual_definition='member softmax minus one-hot label; flattened node/class covariance; zero-variance correlation stored0',
        secondary_mean_probability_nll=secondary_nll,
        secondary_mean_probability_accuracy=float((secondary.argmax(-1) == labels).mean()),
        secondary_selected=False, pool_nodes=len(labels)), prediction


def report(args):
    started = time.perf_counter()
    record = descriptor(args.admission)
    admission = read_json(args.admission)
    require(set(admission) == {'schema', 'execution_authorized', 'authorized_phase', 'comparison_freeze', 'final_labels'}
            and admission['schema'] == 'graph-init-final-report-admission-v1'
            and admission['execution_authorized'] is True and admission['authorized_phase'] == 'report',
            'Separate exact final-report authorization required')
    frozen, comparison_root = verify_comparison(admission['comparison_freeze'])
    keys = {(g, s) for g, _ in CELLS for s in SEEDS}
    packs = admission['final_labels']
    require(len(packs) == 6 and {(row['graph'], row['seed']) for row in packs} == keys,
            'Exactly six compact final-pool packs required')
    for row in packs:
        require(set(row) == {'graph', 'seed', 'labels'}, 'Final descriptor keys differ')
        verified(row['labels'])  # All code/cohort/source-selection guards precede any label open.
    import numpy as np
    for ctx in frozen['expected_contexts']:
        require(np.__version__ == ctx['environment']['numpy'] and
                platform.python_version() == ctx['environment']['python'], 'Report NumPy/Python environment changed')
    # Validate every saved logit tensor BEFORE opening even the first final pack.
    for row in frozen['rows']:
        ctx = next(c for c in frozen['expected_contexts'] if c['graph'] == row['graph'] and c['seed'] == row['seed'])
        graph_meta = read_json(verified(ctx['graph_input']))
        logits = np.load(verified(row['selected_logits']), allow_pickle=False, mmap_mode='r')
        require(logits.dtype == np.float32 and logits.shape == (4, graph_meta['num_nodes'], graph_meta['num_classes'])
                and np.isfinite(logits).all(), 'Saved selected logits must satisfy complete K4 contract')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    claim = comparison_root.parent/(comparison_root.name+'.final-report-claim-'+admission['comparison_freeze']['sha256']+'.json')
    write_json(claim, dict(comparison_freeze=admission['comparison_freeze'], report_admission=record,
                           report_output=str(out), started_unix=time.time(), retry_allowed=False))
    try:
        final_by_cell = {}
        for ctx in frozen['expected_contexts']:
            key = (ctx['graph'], ctx['seed'])
            pool = np.load(verified(ctx['role_freeze']).parent/'pool_nodes.npy', allow_pickle=False)
            label_record = next(row['labels'] for row in packs if (row['graph'], row['seed']) == key)
            with np.load(verified(label_record), allow_pickle=False) as pack:
                require(set(pack.files) == {'nodes', 'labels'}, 'Compact final pack must contain exact nodes/labels')
                nodes, labels = pack['nodes'], pack['labels']
                classes = read_json(verified(ctx['graph_input']))['num_classes']
                require(nodes.dtype == labels.dtype == np.int64 and np.array_equal(nodes, pool)
                        and labels.shape == nodes.shape and ((labels >= 0) & (labels < classes)).all(),
                        'Final labels must match exact frozen pool')
                final_by_cell[key] = (nodes.copy(), labels.copy())
        rows, predictions = [], {}
        for source in frozen['rows']:
            key = (source['graph'], source['seed'])
            nodes, labels = final_by_cell[key]
            logits = np.load(verified(source['selected_logits']), allow_pickle=False)
            metrics, prediction = numpy_metrics(np, logits, nodes, labels)
            predictions[key+(source['arm'],)] = prediction
            fit, fit_root = verify_freeze(source['fit_freeze'], 'fit')
            init, init_root = verify_freeze(fit['initialization_freeze'], 'initialize', fit['context'])
            warm, warm_root = verify_freeze(fit['warm_freeze'], 'warm', fit['context'])
            timepoints = {}
            for name, path in [('warm', warm_root/'warm_member_logits.npy'),
                               ('initialized', init_root/'initialized_member_logits.npy'),
                               ('halfway', fit_root/'halfway_member_logits.npy')]:
                if path.exists():
                    timepoints[name] = numpy_metrics(np, np.load(path, allow_pickle=False), nodes, labels)[0]
                else:
                    require(name == 'halfway' and fit['selection']['halfway_saved'] is False,
                            'Missing prescribed frozen diagnostic')
                    timepoints[name] = {'absent_reason': fit['selection']['halfway_absent_reason']}
            rows.append(dict(**source, metrics=metrics, diagnostic_timepoints=timepoints))
        paired = []
        for key in sorted(keys):
            _, labels = final_by_cell[key]
            graph_pred = predictions[key+('graph',)]
            graph_row = next(row for row in rows if (row['graph'], row['seed'], row['arm']) == key+('graph',))
            classes = np.load(verified(graph_row['selected_logits']), allow_pickle=False, mmap_mode='r').shape[2]
            for arm in ARMS[1:]:
                other_pred = predictions[key+(arm,)]
                other = next(row for row in rows if (row['graph'], row['seed'], row['arm']) == key+(arm,))
                paired.append(dict(graph=key[0], seed=key[1], comparator=arm,
                    graph_minus_control_nll=graph_row['metrics']['primary_nll']-other['metrics']['primary_nll'],
                    graph_minus_control_accuracy=graph_row['metrics']['primary_accuracy']-other['metrics']['primary_accuracy'],
                    wrong_to_correct=int(((other_pred != labels) & (graph_pred == labels)).sum()),
                    correct_to_wrong=int(((other_pred == labels) & (graph_pred != labels)).sum()),
                    classwise_wrong_to_correct=[int(((labels == c) & (other_pred != labels) &
                        (graph_pred == labels)).sum()) for c in range(classes)],
                    classwise_correct_to_wrong=[int(((labels == c) & (other_pred == labels) &
                        (graph_pred != labels)).sum()) for c in range(classes)]))
        aggregate = []
        for graph, _ in CELLS:
            for arm in ARMS:
                chosen = [row for row in rows if row['graph'] == graph and row['arm'] == arm]
                aggregate.append(dict(graph=graph, arm=arm, paired_seeds=3,
                    mean_primary_nll=sum(row['metrics']['primary_nll'] for row in chosen)/3,
                    mean_primary_accuracy=sum(row['metrics']['primary_accuracy'] for row in chosen)/3,
                    inference='descriptive only; no significance or native-family threshold claim'))
        write_json(out/'FINAL_REPORT.json', dict(schema='graph-init-narrow-final-report-v1',
            comparison_freeze=admission['comparison_freeze'], report_admission=record, claim=descriptor(claim),
            rows=rows, aggregate_by_graph_arm=aggregate, paired_graph_differences=paired,
            pooling='softmax(mean raw member logits)',
            scientific_claim_scope='exploratory cfg0 matched initializer screen; no originality or broad extension-utility claim',
            strong_native_families_external=True, all_fallbacks_retained=True, final_labels_selected_nothing=True,
            final_pool_estimand='derived_roles_v2 mixed final pool, not published test reproduction',
            costs='shared acquisition/qualification/warm once in cohort; also disclose standalone full cost per arm',
            report_seconds=time.perf_counter()-started))
    except Exception as error:
        write_json(out/'FAILED_FINAL_REPORT.json', dict(error_type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), report_seconds=time.perf_counter()-started,
            once_only_claim_retained=True, automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    env = sub.add_parser('environment')
    env.add_argument('--device', required=True)
    env.add_argument('--output', required=True)
    registry = sub.add_parser('register')
    registry.add_argument('--request', required=True)
    registry.add_argument('--output', required=True)
    for phase in ('qualify', 'warm', 'initialize', 'fit', 'compare', 'report'):
        command = sub.add_parser(phase)
        command.add_argument('--admission', required=True)
        command.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'environment':
        _, _, environment = runtime(args.device)
        write_json(Path(args.output).resolve(), environment)
    elif args.command == 'register':
        register(args)
    elif args.command == 'compare':
        compare(args)
    elif args.command == 'report':
        report(args)
    else:
        phase_run(args)


if __name__ == '__main__':
    main()
