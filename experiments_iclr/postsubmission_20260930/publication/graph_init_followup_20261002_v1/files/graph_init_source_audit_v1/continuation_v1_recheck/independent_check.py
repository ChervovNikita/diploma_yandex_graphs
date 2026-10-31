"""Bounded independent stdlib metadata/source audit; supplied code is never run."""
from pathlib import Path
import ast
import datetime
import hashlib
import json

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PACKET = BASE / 'graph_init_execution_continuation_v1'
ANCHOR = BASE / 'graph_init_execution_root_v1/study_v1'
REMOTE = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930'
TEXT = {'.json', '.py', '.md', '.csv', '.txt', '.sh', '.toml', '.yaml', '.yml', '.diff', '.patch', '.log', '.jsonl'}
CHECKS, HASHES = [], {}


def sha(path):
    if path.suffix not in TEXT:
        raise ValueError('Non-text payload is outside audit scope')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def obj(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def check(name, value):
    CHECKS.append(dict(check=name, ok=bool(value)))


def read(path):
    return json.loads(path.read_text())


def local(path):
    return BASE / path.split('/postsubmission_20260930/', 1)[1] if '/postsubmission_20260930/' in path else Path(path)


def descriptor(path):
    return dict(path=REMOTE + '/' + str(path.relative_to(BASE)), sha256=sha(path))


def bound(record):
    path = local(record['path'])
    check('bound_text:' + record['path'], sha(path) == record['sha256']
          and ('bytes' not in record or type(record['bytes']) is int and path.stat().st_size == record['bytes']))
    HASHES[record['path']] = dict(descriptor=record, actual_sha256=sha(path), actual_bytes=path.stat().st_size)
    return path


rp = ANCHOR / 'GRAPH_INIT_ATTEMPT_REGISTRY.json'
registry_record = descriptor(rp)
registry = read(rp)
canonical = []
arms = ['graph', 'common_only', 'random_tangent', 'topology_permuted', 'warm_copy']
for c in registry['contexts']:
    for phase in ['qualify', 'warm', 'initialize', 'fit']:
        for arm in arms if phase in ['initialize', 'fit'] else [None]:
            identity = dict(context_sha256=obj(c), phase=phase, arm=arm)
            output = registry['anchor_directory'] + '/' + phase + '/' + c['graph'] + '_seed' + str(c['seed'])
            if arm is not None:
                output += '/' + arm
            canonical.append(dict(key=obj(identity), **identity, output=output))
canonical.sort(key=lambda row: row['key'])
check('actual_registry_exact_72_canonical_attempts', canonical == registry['attempts'] and len(canonical) == 72)
request = read(bound(registry['request']))
lineage = read(bound(registry['lineage_authorization']))
check('actual_registry_context_request_lineage', request['contexts'] == registry['contexts']
      and request['registration_authorized'] is True and request['lineage_authorization'] == registry['lineage_authorization']
      and lineage['contexts_sha256'] == obj(registry['contexts']) and lineage['approved'] is True
      and lineage['predecessor_history_complete'] is True and lineage['anchor_directory'] == registry['anchor_directory'])

plan = []
for phase in ['qualify', 'warm', 'initialize', 'fit']:
    for graph in ['Squirrel', 'Photo']:
        for seed in [17, 29, 43]:
            if (phase, graph, seed) == ('qualify', 'Squirrel', 17):
                continue
            c = next(c for c in registry['contexts'] if (c['graph'], c['seed']) == (graph, seed))
            for arm in arms if phase in ['initialize', 'fit'] else [None]:
                row = next(r for r in canonical if (r['context_sha256'], r['phase'], r['arm']) == (obj(c), phase, arm))
                plan.append(dict(row, whole_cap_seconds=c['resource_forecast']['forecast']['wall_seconds_by_phase'][phase]))
check('independent_finite_plan_71_counts_caps', len(plan) == len({r['key'] for r in plan}) == 71
      and {p: sum(r['phase'] == p for r in plan) for p in ['qualify', 'warm', 'initialize', 'fit']}
      == {'qualify': 5, 'warm': 6, 'initialize': 30, 'fit': 30}
      and all(5 < r['whole_cap_seconds'] <= 28800 for r in plan))

first_context = next(c for c in registry['contexts'] if (c['graph'], c['seed']) == ('Squirrel', 17))
first = next(r for r in canonical if (r['context_sha256'], r['phase'], r['arm']) == (obj(first_context), 'qualify', None))
claim_path = ANCHOR / 'claims' / (first['key'] + '.json')
terminal_path = ANCHOR / 'terminals' / (first['key'] + '.json')
claim, terminal = read(claim_path), read(terminal_path)
check('only_first_claim_and_terminal_in_current_state', {p.stem for p in (ANCHOR/'claims').glob('*.json')}
      == {p.stem for p in (ANCHOR/'terminals').glob('*.json')} == {first['key']})
check('initial_canonical_claim_terminal', claim['attempt'] == first and claim['attempt_registry'] == registry_record
      and claim['automatic_retry'] is False and terminal['completed'] is True and terminal['claim'] == descriptor(claim_path)
      and terminal['final_labels_read'] is False)
freeze = read(bound(terminal['freeze']))
admission = read(bound(freeze['admission']))
check('initial_freeze_context_registry_claim_admission', freeze['context'] == first_context
      and freeze['context_sha256'] == first['context_sha256'] and freeze['attempt_registry'] == registry_record
      and freeze['attempt_claim'] == descriptor(claim_path) and freeze['admission'] == claim['admission']
      and freeze['phase'] == 'qualify' and freeze['completed'] is True and freeze['final_labels_read'] is False
      and terminal['freeze']['path'] == first['output'] + '/FREEZE.json')
check('initial_cold_numeric_metadata_scope', freeze['passed'] is True and freeze['qualification_only'] is True
      and freeze['whole_graph'] is True and freeze['nonzero_Adam_history_checked'] is True
      and freeze['source_labels_read'] == ['train'] and admission['context'] == first_context
      and admission['execution_authorized'] is True and admission['authorized_phase'] == 'qualify'
      and admission['dependencies'] == {} and admission['arm'] is None and admission['attempt_registry'] == registry_record)
for row in freeze['payload']:
    p = local(first['output']) / row['path']
    if p.suffix in TEXT:
        bound(dict(path=first['output'] + '/' + row['path'], sha256=row['sha256'], bytes=row['bytes']))
check('actual_next_registered_key', plan[0]['phase'] == 'qualify' and plan[0]['arm'] is None
      and plan[0]['output'].endswith('/qualify/Squirrel_seed29') and plan[0]['whole_cap_seconds'] == 1800)

draft_path = PACKET/'ROOT_FINITE_SCIENCE_DECISION_TEMPLATE.json'
if draft_path.is_file():
    draft = read(draft_path)
    plan_draft = read(PACKET/'FINITE_71_PLAN_DRAFT.json')
    check('sealed_draft_exact_registry_context_plan', draft['attempt_registry'] == registry_record
          and draft['contexts_sha256'] == obj(registry['contexts']) and draft['finite_plan'] == plan
          and draft['finite_plan_sha256'] == obj(plan) and plan_draft['attempts'] == plan
          and plan_draft['plan_sha256'] == obj(plan)
          and plan_draft['remaining_whole_cap_budget_seconds'] == sum(r['whole_cap_seconds'] for r in plan) == 423900)
    check('draft_all_scientific_approvals_false', draft['approved'] is False
          and draft['scientific_phases_authorized'] is False
          and draft['independent_R17_source_audit_accepted'] is False
          and draft['continuation_source_review_accepted'] is False
          and draft['compare_authorized'] is False and draft['report_authorized'] is False
          and draft['final_labels_authorized'] is False and draft['automatic_retry_authorized'] is False)
    check('draft_initial_actual_qualifier_descriptors', draft['initial_qualification_freeze'] == terminal['freeze']
          and draft['initial_qualification_terminal'] == descriptor(terminal_path))
    whole = read(bound(draft['initial_whole_supervision_terminal']))
    root = read(bound(draft['initial_root_terminal']))
    check('initial_whole_root_success_metadata', whole['complete'] is True and whole['within_whole_cap'] is True
          and whole['root_request_unchanged'] is True and root['completed'] is True
          and root['child_exit_code'] == 0 and root['output'] == first['output'] and root['action'] == 'qualify')
    initial_request_path = BASE/'graph_init_execution_root_v1/admitted_qualify_v1/ROOT_SQUIRREL17_QUALIFY_REQUEST.json'
    initial_request = read(initial_request_path)
    check('initial_root_request_exact_phase_binding', root['request_sha256'] == sha(initial_request_path)
          and initial_request['phase_payload'] == freeze['admission']
          and initial_request['output'] == first['output'] and initial_request['whole_cap_seconds'] == 1800)
    initial_start = read(local(draft['initial_whole_supervision_terminal']['path']).with_name('START.json'))
    check('initial_whole_start_exact_request_and_cap', initial_start['root_request']['sha256'] == sha(initial_request_path)
          and initial_start['whole_cap_seconds'] == 1800
          and whole['START_sha256'] == sha(local(draft['initial_whole_supervision_terminal']['path']).with_name('START.json')))

asts = []
for path in sorted(PACKET.glob('*.py')):
    tree = ast.parse(path.read_text(), filename=str(path))
    asts.append(dict(path=path.name, sha256=sha(path), functions=[n.name for n in tree.body if isinstance(n, ast.FunctionDef)]))

sealed = (PACKET/'MANIFEST.json').is_file() and (PACKET/'SEAL.json').is_file()
payload = []
expected_manifest = 'ef80429d0b15e1c2bc6de136537444fb470ffb9c9e38e3121b600509a03ff5c0'
expected_seal = 'ea5510fc3954f985c25723b8db6bc6d924db31f69328ee53bcb0dac43590e24c'
reviewed_source = {
    'check_source.py': '55d326a46ac0ebef5e17d5b405da4c873ea8482b19d8c7c3b404350521932ca4',
    'build_continuation.py': '9788993ca680e25b67770ee50af14e2e28fb8487bae74fe2812fa1e2deb01bc3',
    'continuation_support.py': '3eaedac8fcac9ca4e6d5ab9240adfdbd46df314046159035eb3b058a4d846d28',
    'continuation_entry.py': '0ed5f656837432207e3aeac585591cfd0f9eb709847b48dcc5f9d1726113688a',
    'finite_coordinator.py': 'e50421b8cca211ee44043c691d9e6c69b77d0ba6aac0890019cce60e25301051',
}
if sealed:
    manifest, seal = read(PACKET/'MANIFEST.json'), read(PACKET/'SEAL.json')
    check('continuation_seal_manifest', seal['manifest_sha256'] == sha(PACKET/'MANIFEST.json'))
    for row in manifest['payload']:
        path = PACKET / row['path']
        ok = sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
        check('continuation_payload:' + row['path'], ok)
        payload.append(dict(**row, actual_sha256=sha(path), actual_bytes=path.stat().st_size, ok=ok))
    check('continuation_manifest_exact_root_pin', sha(PACKET/'MANIFEST.json') == expected_manifest)
    check('continuation_seal_exact_root_pin', sha(PACKET/'SEAL.json') == expected_seal)
    check('reviewed_source_unchanged_during_sealing', {r['path']: r['sha256'] for r in asts} == reviewed_source)

result = dict(schema='independent-finite-continuation-bounded-source-metadata-audit-v1',
              utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), packet_sealed=sealed,
              manifest_sha256=sha(PACKET/'MANIFEST.json') if sealed else None,
              seal_sha256=sha(PACKET/'SEAL.json') if sealed else None,
              expected_manifest_sha256=expected_manifest, expected_seal_sha256=expected_seal,
              reviewed_source_unchanged_during_sealing={r['path']: r['sha256'] for r in asts} == reviewed_source,
              checks=CHECKS, AST_source_receipts=asts, payload_receipts=payload,
              actual_initial_registry=registry_record, actual_initial_freeze=terminal['freeze'],
              actual_initial_terminal=descriptor(terminal_path), independently_rebuilt_plan=plan,
              independently_rebuilt_plan_sha256=obj(plan), actual_next_registered_attempt=plan[0],
              text_metadata_receipts=list(HASHES.values()),
              summary=dict(predicates=len(CHECKS), failed=sum(not c['ok'] for c in CHECKS),
                           wrapper_payloads=len(payload), source_ASTs=len(asts), text_metadata_receipts=len(HASHES)),
              scope='Source/JSON metadata only. No supplied code import/execution, scientific import, array/checkpoint read/hash, SSH/GPU, or sealed edits. Initial numerical success is a checked metadata prerequisite, not a rerun or performance claim.')
(OUT/'RECEIPTS.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(packet_sealed=sealed, **result['summary'], next_attempt=plan[0])))
for c in CHECKS:
    if not c['ok']:
        print(json.dumps(c))
