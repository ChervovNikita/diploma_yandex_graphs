from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

PHASE = Path(__file__).resolve().parents[2]
ROOT = PHASE / 'amazon_polynormer_paired_family_execution_root_20261003_v3'
RECEIPTS = Path(__file__).resolve().parent
PACKET = ROOT / 'v6_execution_metadata_preparation_v1/after_runtime_disabled_v1'
OPAQUE = {'.pt', '.npz', '.npy', '.pth'}
verified = {}


def read(path):
    return json.loads(path.read_text())


def desc(path):
    data = path.read_bytes()
    return dict(path=path.relative_to(PHASE).as_posix(),
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def verify(row):
    path = PHASE / row['path']
    assert not Path(row['path']).is_absolute()
    assert path.resolve().is_relative_to(PHASE)
    assert path.suffix not in OPAQUE, row['path']
    actual = desc(path)
    assert actual == {k: row[k] for k in ('path', 'sha256', 'bytes')}, (row, actual)
    verified[actual['path']] = actual
    return path


def descriptors(obj):
    if isinstance(obj, dict):
        if all(k in obj for k in ('path', 'sha256', 'bytes')):
            yield {k: obj[k] for k in ('path', 'sha256', 'bytes')}
        for value in obj.values():
            yield from descriptors(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from descriptors(value)


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return desc(path)


q = read(PACKET / 'disabled_releases/qualification.json')
plan = read(PACKET / 'PLAN.json')
attempt = read(verify(q['attempt_registry']))
stage = read(RECEIPTS / 'STAGE_RESULT.json')
fetch = read(RECEIPTS / 'FETCH_RESULT.json')
publication = read(RECEIPTS / 'PUBLICATION_RESULT.json')
source = q['source']
source_dir = verify(source['manifest']).parent
assert read(verify(source['seal']))['manifest'] == source['manifest']
for row in read(source_dir / 'MANIFEST.json')['payload']:
    verify({**row, 'path': (source_dir / row['path']).relative_to(PHASE).as_posix()})
for entry in read(PACKET / 'MANIFEST.json')['payload']:
    verify(entry['descriptor'])
assert read(PACKET / 'SEAL.json')['manifest'] == desc(PACKET / 'MANIFEST.json')
for row in fetch['fetched_descriptors']:
    verify(row)
opaque = {row['path']: row for row in stage['opaque_existing_files_stat_checked_not_opened']}
custody = {row['path']: row for row in q['custody_inputs']}
assert len(custody) == len(q['custody_inputs'])
opaque_custody = []
for row in q['custody_inputs']:
    if Path(row['path']).suffix in OPAQUE:
        assert opaque[row['path']] == row
        opaque_custody.append(row)
    else:
        verify(row)
assert all(custody[row['path']] == row for row in attempt['prior_attempt_artifacts'])
assert q['execution_authorized'] is False and q['automatic_retry_authorized'] is False
assert q['test_labels_authorized'] is False and q['kind'] == 'qualify' and q['device'] == 'cuda:0'
assert q['caps'] == dict(wall_seconds=3600, rss_bytes=32 * 1024**3,
                         cuda_peak_allocated_bytes=75 * 1024**3,
                         cuda_peak_reserved_bytes=75 * 1024**3)
assert attempt['execution_authorized'] is False and attempt['scientific_fits_started'] == 0
assert attempt['new_source_numerically_qualified'] is False
assert attempt['fresh_V6_qualification_pending'] is True
gate_review = read(verify(q['source_review']))
review = read(verify(plan['independent_source_review_evidence']))
assert gate_review['source'] == source and gate_review['status'] == 'passed'
assert gate_review['review_outcome'] == review['status'] == 'PASS_SOURCE_ONLY'
assert gate_review['independent_review_report'] == plan['independent_source_review_evidence']
for obj in (gate_review, review):
    for key in ('execution_authorized', 'training_authorized',
                'resource_admission_established', 'numerical_qualification_established'):
        assert obj[key] is False
consumer = read(verify(q['consumer_release']))
assert consumer['source'] == source and consumer['execution_authorized'] is True
assert consumer['test_labels_authorized'] is False
old_consumer = read(ROOT / 'CONSUMER_RELEASE.json')
assert {k: v for k, v in consumer.items() if k not in ('source', 'execution_authorized')} == {
    k: v for k, v in old_consumer.items() if k not in ('source', 'execution_authorized')}
old_consumer_descriptors = {row['path']: row for row in descriptors(old_consumer)}
consumer_opaque = []
for row in descriptors(consumer):
    if Path(row['path']).suffix in OPAQUE:
        assert row == old_consumer_descriptors[row['path']]
        if row['path'] in opaque:
            assert row == opaque[row['path']]
        consumer_opaque.append(dict(descriptor=row, predecessor_exact=True,
                                   remote_stat_in_this_stage=row['path'] in opaque))
    elif not Path(row['path']).is_absolute():
        verify(row)
cohort = read(source_dir / 'COHORT_SOURCE_BINDING.json')
registry = read(verify(q['registry']))
assert q['registry'] == cohort['registry'] == plan['original_registry']
assert registry['source'] == cohort['registered_source']
assert registry['master_source_claim'] == cohort['master_source_claim'] == plan['original_master_source_claim']
verify(cohort['master_source_claim'])
assert len(registry['physical_fits']) == 15 and len(registry['families']) == 9
assert [r['row'] for r in registry['physical_fits']] == read(source_dir / 'DESIGN.json')['physical_fit_schedule']
for row, candidate in zip(registry['physical_fits'], plan['original15fit_candidates']):
    value = read(verify(candidate['candidate']))
    assert value['execution_authorized'] is False and value['fit_id'] == row['id']
    assert value['output'] == row['output'] and value['self_path'] == row['release_path']
    assert value['registered_claim_path'] == row['claim_path']
    assert value['qualification_freeze'] is None and value['resource_admission'] is None
for entry in publication['published']:
    candidate = read(verify(entry['candidate']))
    published = read(verify(entry['published']))
    assert candidate['execution_authorized'] is False and published['execution_authorized'] is True
    assert published == {**candidate, 'execution_authorized': True}
    assert entry['changed_fields'] == ['execution_authorized']
costs = {}
for mode, name, device in [('runtime_cpu', 'v6_runtime_cpu_v1', 'cpu'),
                           ('runtime_gpu', 'v6_runtime_cuda0_v1', 'cuda:0')]:
    path = ROOT / name
    freeze = read(path / 'FREEZE.json')
    assert freeze['source'] == source and freeze['status'] == 'success'
    for entry in freeze['files']:
        verify(entry['descriptor'])
    terminal = read(path / 'TERMINAL.json')
    runtime = read(path / 'RUNTIME.json')
    assert terminal['physical_exit_code'] == 0 and terminal['status'] == 'success'
    assert terminal['admission'] == freeze['admission']
    assert runtime['source'] == source and runtime['runtime']['hardware']['device'] == device
    assert runtime['runtime']['versions'] == q['expected_versions']
    client = read(RECEIPTS / (mode + '_RESULT.json'))
    assert client['physical_supervisor_exit_code'] == 0 and client['automatic_retry'] is False
    assert client['qualifier_fit_or_scoring_launched'] is False
    costs[mode] = dict(freeze=desc(path / 'FREEZE.json'), runtime=desc(path / 'RUNTIME.json'),
                       terminal=desc(path / 'TERMINAL.json'),
                       whole_process_wall_seconds=terminal['whole_process_wall_seconds'],
                       aggregate_peak_observed_rss_bytes=terminal['aggregate_peak_observed_rss_bytes'])
assert q['runtime_receipt'] == costs['runtime_gpu']['runtime']
assert fetch['all15_registered_fit_paths_absent'] is True
assert fetch['qualification_release_and_output_absent'] is True
verification = write(RECEIPTS / 'LOCAL_METADATA_VERIFICATION.json', dict(
    schema='amazon_polynormer_V6_runtime_metadata_verification_v1',
    UTC=datetime.now(timezone.utc).isoformat(), status='passed',
    text_descriptors_verified=list(verified.values()), opaque_custody_descriptors_not_opened=opaque_custody,
    opaque_consumer_descriptors_preserved_without_opening=consumer_opaque,
    source_payload_count=36, qualification_custody_count=len(custody), original_fit_count=15,
    exact_original_registry_and_master_claim_preserved=True,
    published_changes_only_execution_authorized=True,
    all_candidate_execution_authorizations_false=True, costs=costs,
    remote_absence_observation_UTC=fetch['UTC'],
    numerical_qualification_or_resource_admission_established=False,
    arrays_checkpoints_or_runtime_binary_values_read=False))
closure = write(RECEIPTS / 'EXECUTION_CLOSURE.json', dict(
    schema='amazon_polynormer_V6_runtime_execution_closure_v1',
    UTC=datetime.now(timezone.utc).isoformat(), status='success', route=fetch['route'], source=source,
    source_review=q['source_review'], independent_review=plan['independent_source_review_evidence'],
    staging=dict(created_files=len(stage['created_exact_files']),
                 existing_identical_files=len(stage['existing_exact_files']),
                 opaque_stat_only_files=len(stage['opaque_existing_files_stat_checked_not_opened']),
                 files_overwritten=False), publication=publication['published'], costs=costs,
    local_metadata_verification=verification,
    disabled_qualification_candidate=desc(PACKET / 'disabled_releases/qualification.json'),
    after_runtime_packet=dict(plan=desc(PACKET / 'PLAN.json'), manifest=desc(PACKET / 'MANIFEST.json'),
                              seal=desc(PACKET / 'SEAL.json')),
    actual_consumer=q['consumer_release'], extended_attempt_registry=q['attempt_registry'],
    original_registry=q['registry'], original_master_claim=cohort['master_source_claim'],
    fit_qualification_scoring_or_registration_launched=False,
    remote_absence_observation_UTC=fetch['UTC'],
    full15_fit_paths_absent_at_observation=True, qualifier_release_output_absent_at_observation=True,
    qualification_numerical_pass_pending=True, execution_authorized=False))
inventory = write(RECEIPTS / 'MANIFEST.json', dict(
    schema='amazon_polynormer_V6_runtime_execution_receipts_manifest_v1',
    UTC=datetime.now(timezone.utc).isoformat(), execution_authorized=False,
    payload=[dict(relative=p.relative_to(RECEIPTS).as_posix(), descriptor=desc(p))
             for p in sorted(RECEIPTS.rglob('*')) if p.is_file() and p.name not in ('MANIFEST.json', 'SEAL.json')]))
seal = write(RECEIPTS / 'SEAL.json', dict(
    schema='amazon_polynormer_V6_runtime_execution_receipts_seal_v1',
    UTC=datetime.now(timezone.utc).isoformat(), manifest=inventory, execution_authorized=False))
print(json.dumps(dict(status='passed', verified_text_files=len(verified),
                      opaque_custody_count=len(opaque_custody), verification=verification,
                      closure=closure, manifest=inventory, seal=seal), indent=2))
