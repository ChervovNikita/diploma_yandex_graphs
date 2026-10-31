"""Stdlib exact-registry continuation custody. Never opens arrays/checkpoints."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REMOTE_REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
REMOTE_PHASE = REMOTE_REPO/'experiments_iclr/postsubmission_20260930'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
GPU_UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
R17_REL = 'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v3_precision'
MODERN_REL = 'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3'
R17_MANIFEST_SHA = '5e8250d9fbe219593bb2aa074665e30cc5686d6ed227fc549f83809dcf89f830'
R17_SEAL_SHA = '457cf3bd3275cad315e3d57b5036683577929369f03ea346a718b4f8ffad10f2'
MODERN_MANIFEST_SHA = 'e2fd767c4d08e93ca7ec9c9c526440c4cc5b8f4e7d8bb345de90308e9c57302b'
MODERN_SEAL_SHA = '9589bd1a41ec59fc9259d72c833c5e550e6887196eb773b86c75c6d03b67645d'
SOURCE_BINDING_SHA = 'f6d8ff959204ccc793212b0ff0560213f08de7d92b74e162441888058262bc48'
PROTOCOL_SHA = '91f20afb455c234ce113a5e3d3035955bc42b412ff7f389e5f511c7de47de446'
PREPARATION_SHA = '376e8b779d8d71cf21785a62b8543ecadc9c09c9db8174dc230b55b8d0dbd253'
ARMS = ('graph', 'common_only', 'random_tangent', 'topology_permuted', 'warm_copy')
CELLS = (('Squirrel', 'polyformer_mono'), ('Photo', 'polynormer_r'))
SEEDS = (17, 29, 43)
TEXT_EXTENSIONS = {'.json', '.py', '.md', '.csv', '.txt', '.sh', '.toml', '.yaml', '.yml',
                   '.diff', '.patch', '.log', '.jsonl'}
DISCLOSURE = 'This precision qualification amendment was designed after the known Photo17 v2 finite-difference failure and an exactly replayed arithmetic diagnostic. Photo and Squirrel families were previously exposed; this remains exploratory and outcome-aware. independent_of_stage1_outcomes describes source-pack custody only: no reuse, binding or selection of Stage1 fitted outputs or labels as this study evidence. It does not describe independence of idea/cohort choice or an unseen confirmatory cohort. The failed v2 registry, all old attempts and all diagnostic lineage remain preserved. No old checkpoint, phase output, qualification or optimizer history is inherited. All six cold and every actual-warm state require new v3 qualification with retained original FP32 diagnostics and unchanged thresholds.'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def confined(path, base=PHASE):
    path = Path(path)
    require(path.is_absolute() and path == Path(str(path)), 'Absolute path required')
    require('..' not in path.parts and path.is_relative_to(base) and path != base,
            'Path must stay inside the research phase')
    cursor = base
    for part in path.relative_to(base).parts:
        cursor /= part
        require(not cursor.is_symlink(), 'Symlink path forbidden: '+str(cursor))
    return path


def mirror(path):
    """Only text metadata are mirrored for local drafting; runtime paths stay exact."""
    path = Path(path)
    if path.is_relative_to(REMOTE_PHASE) and PHASE != REMOTE_PHASE:
        path = PHASE/path.relative_to(REMOTE_PHASE)
    return confined(path)


def text_path(path):
    path = mirror(path)
    require(path.suffix in TEXT_EXTENSIONS, 'Metadata tool cannot open array/model artifact: '+str(path))
    require(path.is_file(), 'Missing metadata file: '+str(path))
    return path


def sha(path):
    return hashlib.sha256(text_path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(text_path(path).read_text())


def remote(path):
    path = Path(path)
    if path.is_relative_to(PHASE):
        return str(REMOTE_PHASE/path.relative_to(PHASE))
    require(path.is_relative_to(REMOTE_PHASE), 'Remote phase path required')
    return str(path)


def descriptor(path):
    return {'path': remote(path), 'sha256': sha(path)}


def bound(record):
    require(isinstance(record, dict) and {'path', 'sha256'} <= set(record) <= {'path', 'sha256', 'bytes'},
            'Exact path/SHA descriptor with optional bytes required')
    require(isinstance(record['sha256'], str) and len(record['sha256']) == 64,
            'SHA256 required')
    path = text_path(record['path'])
    require(sha(path) == record['sha256'], 'Bound metadata changed: '+str(path))
    if 'bytes' in record:
        require(type(record['bytes']) is int and record['bytes'] >= 0 and path.stat().st_size == record['bytes'],
                'Descriptor byte length differs')
    return path


def write(path, value):
    path = confined(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_descriptors():
    r17 = REMOTE_PHASE/R17_REL
    modern = REMOTE_PHASE/MODERN_REL
    return [
        {'path': str(r17/'MANIFEST.json'), 'sha256': R17_MANIFEST_SHA},
        {'path': str(r17/'SEAL.json'), 'sha256': R17_SEAL_SHA},
        {'path': str(modern/'MANIFEST.json'), 'sha256': MODERN_MANIFEST_SHA},
        {'path': str(modern/'SEAL.json'), 'sha256': MODERN_SEAL_SHA}]


def verify_sources():
    """Hash sealed text/source payloads only; do not import scientific modules."""
    records = source_descriptors()
    for manifest_record, seal_record in zip(records[::2], records[1::2]):
        path = bound(manifest_record)
        seal = read(bound(seal_record))
        require(seal['manifest_sha256'] == manifest_record['sha256'], 'Seal/manifest mismatch')
        manifest = read(path)
        names = set()
        for row in manifest['payload']:
            rel = Path(row['path'])
            require(not rel.is_absolute() and '..' not in rel.parts and str(rel) not in names,
                    'Unsafe or repeated sealed payload')
            names.add(str(rel))
            payload = text_path(path.parent/rel)
            require(sha(payload) == row['sha256'] and payload.stat().st_size == row['bytes'],
                    'Sealed text/source payload changed: '+str(payload))
    require(sha(PHASE/R17_REL/'SOURCE_BINDINGS.json') == SOURCE_BINDING_SHA,
            'R17 source binding differs')
    require(sha(PHASE/R17_REL/'PROTOCOL.json') == PROTOCOL_SHA, 'R17 protocol differs')
    ancestry = read(PHASE/R17_REL/'SOURCE_BINDINGS.json')['unchanged_method']
    require(ancestry['sha256'] == '1a8036c7bbf9f2b831747f99d3aa2dfdabc31cb636cd41ccad9457206a70cbcf' and
            sha(PHASE/ancestry['path']) == ancestry['sha256'], 'Required unchanged Round15 deployment source differs or is absent')
    return records


def bindings():
    return ({'path': str(REMOTE_PHASE/R17_REL/'SOURCE_BINDINGS.json'), 'sha256': SOURCE_BINDING_SHA},
            {'path': str(REMOTE_PHASE/R17_REL/'PROTOCOL.json'), 'sha256': PROTOCOL_SHA})


def expected_attempts(contexts, anchor):
    rows = []
    for context in contexts:
        digest = object_hash(context)
        cell = context['graph']+'_seed'+str(context['seed'])
        for phase in ('qualify', 'warm', 'initialize', 'fit'):
            for arm in ARMS if phase in ('initialize', 'fit') else (None,):
                out = Path(anchor)/phase/cell
                if arm is not None:
                    out /= arm
                rows.append({'key': object_hash({'context_sha256': digest, 'phase': phase, 'arm': arm}),
                    'context_sha256': digest, 'phase': phase, 'arm': arm, 'output': str(out)})
    rows.sort(key=lambda row: row['key'])
    require(len(rows) == 72 and len({row['key'] for row in rows}) == 72 and
            len({row['output'] for row in rows}) == 72, 'Exactly72 unique attempts required')
    return rows


PHASES = ('qualify', 'warm', 'initialize', 'fit')
FIRST_CELL = ('Squirrel', 17)


def own_sources():
    for name in ('MANIFEST.json', 'SEAL.json'):
        require((HERE/name).is_file(), 'Continuation packet must be sealed before decisions')
    return [descriptor(HERE/'MANIFEST.json'), descriptor(HERE/'SEAL.json')]


def deployment_ancillary():
    ancestry = read(PHASE/R17_REL/'SOURCE_BINDINGS.json')['unchanged_method']
    record = {'path': str(REMOTE_PHASE/ancestry['path']), 'sha256': ancestry['sha256']}
    bound(record)
    return [record]


def verify_own_sources(manifest_record, seal_record):
    require(manifest_record['path'] == remote(HERE/'MANIFEST.json') and
            seal_record['path'] == remote(HERE/'SEAL.json'), 'Exact continuation packet required')
    manifest_path = bound(manifest_record)
    seal = read(bound(seal_record))
    require(seal['manifest_sha256'] == manifest_record['sha256'], 'Continuation seal differs')
    names = set()
    for row in read(manifest_path)['payload']:
        rel = Path(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts and str(rel) not in names,
                'Unsafe continuation payload')
        names.add(str(rel))
        path = text_path(HERE/rel)
        require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Continuation source changed')


def registry_guard(record):
    path = bound(record)
    registry = read(path)
    require(registry['schema'] == 'graph-init-attempt-registry-v1' and
            record['path'] == str(Path(registry['anchor_directory'])/'GRAPH_INIT_ATTEMPT_REGISTRY.json') and
            registry['source_bindings'] == bindings()[0] and registry['protocol'] == bindings()[1] and
            registry['silent_retry_forbidden'] is True and registry['failure_blocks_this_study'] is True and
            registry['final_labels_accessible_only_after_all72_completed'] is True,
            'Exact immutable current R17 registry required')
    require(registry['study_id'] == 'graph_init_cfg0_outcome_aware_precision_v1' and
            registry['anchor_directory'] == str(REMOTE_PHASE/'graph_init_precision_execution_root_v1/study_v1'),
            'Exact new precision study/anchor required')
    contexts = registry['contexts']
    require(len(contexts) == 6 and {(c['graph'], c['backbone'], c['seed'], c['source_split_index'], c['config'])
            for c in contexts} == {(g, b, s, i, 0) for g, b in CELLS for i, s in enumerate(SEEDS)},
            'Exact six registered contexts required')
    for context in contexts:
        require(context['resource_forecast']['admitted'] is True and
                context['source_bindings'] == bindings()[0] and context['protocol'] == bindings()[1] and
                set(context['source_labels']) == {'train', 'validation'}, 'Registered source/resource context differs')
        caps = context['resource_forecast']['forecast']['wall_seconds_by_phase']
        require(set(caps) == set(PHASES) and all(type(v) in (int, float) and 5 < v <= 28800 for v in caps.values()),
                'Every whole-phase cap must equal a bounded registered forecast')
    require(registry['attempts'] == expected_attempts(contexts, registry['anchor_directory']),
            'Canonical72attempt inventory differs')
    request = read(bound(registry['request']))
    lineage = read(bound(registry['lineage_authorization']))
    require(request['registration_authorized'] is True and request['contexts'] == contexts and
            request['anchor_directory'] == registry['anchor_directory'] and request['study_id'] == registry['study_id'] and
            request['lineage_authorization'] == registry['lineage_authorization'] and
            request['prior_attempt_registries'] == registry['prior_attempt_registries'] and
            lineage['approved'] is True and lineage['predecessor_history_complete'] is True and
            lineage['contexts_sha256'] == object_hash(contexts) and
            lineage['source_bindings'] == bindings()[0] and lineage['protocol'] == bindings()[1] and
            lineage['anchor_directory'] == registry['anchor_directory'] and lineage['study_id'] == registry['study_id'] and
            lineage['prior_attempt_registries'] == registry['prior_attempt_registries'], 'Exact registry request/lineage differs')
    for evidence in lineage['evidence']:
        bound(evidence)
    for prior in registry['prior_attempt_registries']:
        bound(prior)
    return registry


def context_for(registry, row):
    contexts = [c for c in registry['contexts'] if object_hash(c) == row['context_sha256']]
    require(len(contexts) == 1, 'One exact registered context required')
    return contexts[0]


def row_for(registry, context, phase, arm=None):
    key = object_hash({'context_sha256': object_hash(context), 'phase': phase, 'arm': arm})
    rows = [r for r in registry['attempts'] if r['key'] == key]
    require(len(rows) == 1, 'Exact canonical attempt required')
    return rows[0]


def finite_plan(registry):
    """All six fresh qualifications first, then all warm, all initializer arms, all fits."""
    rows = []
    for phase in PHASES:
        for graph, _ in CELLS:
            for seed in SEEDS:
                context = [c for c in registry['contexts'] if (c['graph'], c['seed']) == (graph, seed)][0]
                for arm in ARMS if phase in ('initialize', 'fit') else (None,):
                    row = copy.deepcopy(row_for(registry, context, phase, arm))
                    row['whole_cap_seconds'] = context['resource_forecast']['forecast']['wall_seconds_by_phase'][phase]
                    rows.append(row)
    require(len(rows) == 72 and len({r['key'] for r in rows}) == 72, 'Exactly72 fresh attempts required')
    return rows


def completed_phase(registry_record, registry, row):
    """Validate exact canonical claim/terminal/FREEZE metadata. Runtime verifies all payload bytes."""
    anchor = Path(registry['anchor_directory'])
    claim_path = mirror(anchor/'claims'/(row['key']+'.json'))
    terminal_path = mirror(anchor/'terminals'/(row['key']+'.json'))
    require(claim_path.is_file() and terminal_path.is_file(), 'Canonical dependency is not completed')
    claim_record = descriptor(claim_path)
    claim, terminal = read(claim_path), read(terminal_path)
    require(terminal['schema'] == 'graph-init-attempt-terminal-v1' and terminal['completed'] is True and
            terminal['claim'] == claim_record and terminal['final_labels_read'] is False and
            claim['attempt_registry'] == registry_record and claim['attempt'] == row and
            claim['automatic_retry'] is False, 'Failed or cross-registry canonical dependency')
    freeze_record = terminal['freeze']
    require(freeze_record['path'] == str(Path(row['output'])/'FREEZE.json'), 'Dependency must use canonical phase output')
    freeze = read(bound(freeze_record))
    context = context_for(registry, row)
    require(freeze['schema'] == 'graph-init-phase-freeze-v1' and freeze['completed'] is True and
            freeze['phase'] == row['phase'] and freeze['context'] == context and
            freeze['context_sha256'] == row['context_sha256'] and freeze['output_root'] == row['output'] and
            freeze['attempt_registry'] == registry_record and freeze['attempt_claim'] == claim_record and
            freeze['final_labels_read'] is False and freeze['admission'] == claim['admission'],
            'Canonical dependency freeze identity differs')
    admission = read(bound(freeze['admission']))
    require(admission['execution_authorized'] is True and admission['context'] == context and
            admission['authorized_phase'] == row['phase'] and admission['arm'] == row['arm'] and
            admission['attempt_registry'] == registry_record, 'Dependency admission differs')
    if row['phase'] in ('qualify', 'initialize'):
        require(freeze['passed'] is True, 'Qualification/initializer numerical gates must pass')
    if row['phase'] == 'qualify':
        require(freeze['qualification_only'] is True and freeze['whole_graph'] is True and
                freeze['nonzero_Adam_history_checked'] is True and freeze['source_labels_read'] == ['train'],
                'Whole-graph TRAIN-only cold qualifier required')
    if row['phase'] == 'warm':
        require(freeze['fixed_common_warm'] is True, 'Exact fixed native warm required')
    if row['phase'] == 'initialize':
        require(freeze['arm'] == row['arm'] and freeze['same_native_warm_RNG'] is True and
                freeze['gradient_interface_and_optimizer_equivalence_passed'] is True,
                'Actual-warm AD/Adam/install gates and exact RNG required')
    if row['phase'] != 'qualify':
        prior_phase = {'warm': 'qualify', 'initialize': 'warm', 'fit': 'initialize'}[row['phase']]
        prior_arm = row['arm'] if row['phase'] == 'fit' else None
        prior_row = row_for(registry, context, prior_phase, prior_arm)
        prior_record, _, prior_freeze = completed_phase(registry_record, registry, prior_row)
        dependency_name = {'warm': 'qualification', 'initialize': 'warm', 'fit': 'initialization'}[row['phase']]
        freeze_name = {'warm': 'qualification_freeze', 'initialize': 'warm_freeze', 'fit': 'initialization_freeze'}[row['phase']]
        require(admission['dependencies'] == {dependency_name: prior_record} and freeze[freeze_name] == prior_record,
                'Dependency chain must close through exact canonical current-registry metadata')
        if row['phase'] == 'fit':
            require(freeze['warm_freeze'] == prior_freeze['warm_freeze'] and
                    freeze['arm'] == row['arm'] and freeze['complete_declared_continuation'] is True,
                    'Fit must complete its exact same-arm initialization and shared warm chain')
    else:
        require(admission['dependencies'] == {}, 'Cold qualifier cannot consume a fitted dependency')
    # Hash text receipts only. Never open .npy/.npz/.pt, even for hashing.
    names = set()
    for item in freeze['payload']:
        rel = Path(item['path'])
        require(not rel.is_absolute() and '..' not in rel.parts and str(rel) not in names and
                isinstance(item['sha256'], str) and len(item['sha256']) == 64 and type(item['bytes']) is int and item['bytes'] >= 0,
                'Unsafe or malformed phase payload metadata')
        names.add(str(rel))
        if rel.suffix in TEXT_EXTENSIONS:
            path = text_path(Path(row['output'])/rel)
            require(sha(path) == item['sha256'] and path.stat().st_size == item['bytes'], 'Dependency text receipt changed')
    return freeze_record, descriptor(terminal_path), freeze


def scan_state(registry_record, registry):
    """Every failed or unresolved claim blocks another admission; unknown identities fail."""
    anchor = mirror(registry['anchor_directory'])
    known = {r['key']: r for r in registry['attempts']}
    claims = {p.stem for p in (anchor/'claims').glob('*.json')}
    terminals = {p.stem for p in (anchor/'terminals').glob('*.json')}
    require(claims <= set(known) and terminals <= claims and claims == terminals,
            'Unknown or unresolved registry claim blocks this study')
    completed = {}
    for key in sorted(terminals):
        completed[key] = completed_phase(registry_record, registry, known[key])
    return completed


def decision_guard(path):
    decision = read(path)
    require(decision['schema'] == 'graph-init-finite-continuation-root-decision-v1' and
            decision['approved'] is True and decision['scientific_phases_authorized'] is True and
            decision['approved_by'].strip() and decision['approved_utc'] and
            decision['independent_R17_source_audit_accepted'] is True and
            decision['continuation_source_review_accepted'] is True and
            decision['allowed_phases'] == list(PHASES) and decision['compare_authorized'] is False and
            decision['report_authorized'] is False and decision['final_labels_authorized'] is False and
            decision['automatic_retry_authorized'] is False and decision['fixed_native_protocol_unchanged'] is True,
            'One explicit reviewed finite scientific allowance is required')
    require(decision['source_seals'] == source_descriptors(), 'Exact R17/modern source seals required')
    verify_sources()
    verify_own_sources(decision['continuation_source_manifest'], decision['continuation_source_seal'])
    bound(decision['independent_R17_source_audit'])
    bound(decision['continuation_source_review'])
    for record in decision['protected_files']:
        bound(record)
    registry = registry_guard(decision['attempt_registry'])
    plan = finite_plan(registry)
    require(decision['study_id'] == registry['study_id'] and decision['anchor_directory'] == registry['anchor_directory'] and
            decision['contexts_sha256'] == object_hash(registry['contexts']) and decision['finite_plan'] == plan and
            decision['finite_plan_sha256'] == object_hash(plan), 'Signed exact72attempt schedule/caps required')
    from lineage_support import verify_lineage, prior_registries
    verify_lineage()
    require(registry['prior_attempt_registries'] == prior_registries(), 'All failed v2 registries must be preserved')
    launch = read(bound(decision['registration_launch_request']))
    whole = read(bound(decision['registration_whole_supervision_terminal']))
    root = read(bound(decision['registration_root_terminal']))
    require(launch['action'] == 'register' and launch['root_admitted'] is True and
            launch['phase_payload'] == registry['request'] and launch['output'] == decision['attempt_registry']['path'] and
            launch['source_seals'] == source_descriptors() and
            whole['complete'] is True and whole['within_whole_cap'] is True and
            whole['root_request_unchanged'] is True and whole['whole_cap_seconds'] == 600 and
            root['completed'] is True and root['child_exit_code'] == 0 and root['action'] == 'register' and
            root['output'] == decision['attempt_registry']['path'] and
            root['request_sha256'] == decision['registration_launch_request']['sha256'],
            'Fresh successful registration root/whole terminals required')
    return decision, registry, plan


def expected_admission(decision, registry, row):
    completed = scan_state(decision['attempt_registry'], registry)
    require(row['key'] not in completed, 'No completed attempt can be retried')
    context = context_for(registry, row)
    # No warm starts until every qualifier is completed, including the first.
    if row['phase'] != 'qualify':
        require(all(r['key'] in completed for r in registry['attempts'] if r['phase'] == 'qualify'),
                'All six cold qualifications must pass first')
    deps = {}
    if row['phase'] != 'qualify':
        dependency_phase = {'warm': 'qualify', 'initialize': 'warm', 'fit': 'initialize'}[row['phase']]
        dep = row_for(registry, context, dependency_phase, row['arm'] if row['phase'] == 'fit' else None)
        require(dep['key'] in completed, 'Exact same-registry canonical dependency incomplete')
        deps[{'warm': 'qualification', 'initialize': 'warm', 'fit': 'initialization'}[row['phase']]] = completed[dep['key']][0]
    return {'schema': 'graph-init-phase-admission-v1', 'execution_authorized': True,
        'authorized_phase': row['phase'], 'context': context, 'dependencies': deps,
        'arm': row['arm'], 'attempt_registry': decision['attempt_registry']}
