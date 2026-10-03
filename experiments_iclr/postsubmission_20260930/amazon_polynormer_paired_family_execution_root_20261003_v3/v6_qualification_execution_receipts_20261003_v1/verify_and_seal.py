"""Verify fetched text receipts; carry checkpoint descriptors without opening them."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
ROOT = HERE.parent
OUTPUT = ROOT / 'v6_qualification_block0_cuda0_v1'
PACKET = ROOT / 'v6_execution_metadata_preparation_v1/after_qualification_disabled_v1'
OPAQUE = {'.pt', '.npz', '.npy', '.pth'}


def read(p):
    return json.loads(p.read_text())


def desc(p):
    assert p.suffix not in OPAQUE
    raw = p.read_bytes()
    return dict(path=p.relative_to(PHASE).as_posix(), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def verify(row):
    p = PHASE / row['path']
    assert not Path(row['path']).is_absolute() and p.resolve().is_relative_to(PHASE)
    assert desc(p) == row
    return p


def write(p, value):
    with p.open('x') as h:
        json.dump(value, h, indent=2, allow_nan=False); h.write('\n')
    return desc(p)


published = read(HERE / 'publish_RESULT.json')
release = read(verify(published['published']))
candidate = read(verify(published['candidate']))
assert release == dict(candidate, execution_authorized=True)
run = read(HERE / 'run_RESULT.json')
assert run['physical_supervisor_exit_code'] == 0 and run['automatic_retry'] is False
assert run['ordinary_supervision'] is True
assert run['predictive_fits_TEST_control_scoring_or_new_registration'] is False
for row in read(HERE / 'fetch_RESULT.json')['fetched_descriptors']:
    verify(row)
freeze, terminal, result, ready = [read(OUTPUT / n) for n in
                                  ('FREEZE.json', 'TERMINAL.json', 'RESULT.json', 'READY.json')]
assert terminal['status'] == freeze['status'] == 'success' and terminal['physical_exit_code'] == 0
assert freeze['admission'] == terminal['admission'] == ready['admission'] == published['published']
assert freeze['source'] == ready['source'] == result['source'] == release['source']
assert result['status'] == 'passed' and result['report_eligible'] is False
assert result['full_study_authorized'] is False and len(result['forms']) == 5
assert result['runtime_receipt'] == release['runtime_receipt']
assert result['consumer_release'] == release['consumer_release']
assert result['prior_attempt_registry'] == release['attempt_registry']
assert ready['kind'] == 'qualify' and ready['result'] == desc(OUTPUT / 'RESULT.json')
opaque = []
for row in freeze['files']:
    if Path(row['descriptor']['path']).suffix in OPAQUE:
        opaque.append(row['descriptor'])
    else:
        verify(row['descriptor'])
assert [f['row'] for f in result['forms']] == read(
    PHASE / 'amazon_polynormer_paired_family_source_preparation_20261003_v6/DESIGN.json')['physical_fit_schedule'][:5]
forms = []
for f in result['forms']:
    assert f['report_eligible'] is False
    assert f['local_replay']['bitwise_full_next_step'] and f['global_replay']['bitwise_full_next_step']
    assert f['nontrivial_scratch_model_Adam_counter_rollback']
    assert f['selected_local_final_restore_after_global']
    rp = f['retirement_probe']
    assert rp['live_model_Adam_grad_modes_stage_and_RNG_bitwise_unchanged']
    assert rp['selected_local_and_global_probe_still_available']
    assert rp['scientific_selector_or_training_updates_added'] == 0
    verify(rp['journal'])
    forms.append(dict(row=f['row'], body_seconds=f['body_seconds'], memory=f['memory'],
                      checkpoint_io=f['checkpoint_io'], state_bytes=f['state_bytes'],
                      charged_actual_train_updates=f['charged_actual_train_updates'],
                      complete_member_trajectory_updates=f['complete_member_trajectory_updates'],
                      retirement_probe=rp, local_and_global_full_next_step_bitwise=True))
after = read(HERE / 'after-qualification_RESULT.json')
assert after['exit_code'] == 0 and after['execution_authorized'] is False and after['metadata_only']
for row in after['fetched_descriptors']:
    verify(row)
plan = read(PACKET / 'PLAN.json')
assert plan['stage'] == 'after-qualification' and plan['execution_authorized'] is False
assert plan['qualification_freeze'] == desc(OUTPUT / 'FREEZE.json')
assert plan['resource_admission'] is None and plan['registered_release_output_claim_paths_created'] is False
for row in read(PACKET / 'MANIFEST.json')['payload']:
    value = read(verify(row['descriptor']))
    if 'execution_authorized' in value:
        assert value['execution_authorized'] is False
assert read(PACKET / 'SEAL.json')['manifest'] == desc(PACKET / 'MANIFEST.json')
attempt = read(verify(plan['attempt_registry_disabled']))
assert attempt['new_source_numerically_qualified'] is True
assert attempt['execution_authorized'] is False and attempt['scientific_fits_started'] == 0
closure = write(HERE / 'EXECUTION_CLOSURE.json', dict(
    schema='amazon_polynormer_V6_qualification_execution_closure_v1',
    UTC=datetime.now(timezone.utc).isoformat(), status='success', route=run['route'],
    source=release['source'], admission=published['published'],
    disabled_predecessor=published['candidate'], changed_fields=['execution_authorized'],
    resources_before_launch=run['resources_before_launch'], caps=release['caps'],
    terminal=desc(OUTPUT / 'TERMINAL.json'), result=desc(OUTPUT / 'RESULT.json'),
    freeze=desc(OUTPUT / 'FREEZE.json'),
    whole_process_wall_seconds=terminal['whole_process_wall_seconds'],
    aggregate_peak_observed_rss_bytes=terminal['aggregate_peak_observed_rss_bytes'],
    qualification_cost=result['cost'], forms=forms,
    retained_checkpoint_descriptors_not_opened=opaque,
    disabled_after_qualification_packet=dict(plan=desc(PACKET / 'PLAN.json'),
        manifest=desc(PACKET / 'MANIFEST.json'), seal=desc(PACKET / 'SEAL.json')),
    disabled_attempt_registry=plan['attempt_registry_disabled'],
    original_registry=release['registry'], consumer=release['consumer_release'],
    report_eligible=False, predictive_fits_started=0, TEST_control_or_scoring_launched=False,
    automatic_retry=False, new_registration=False, resource_admission_established=False,
    original_registered_output_claim_release_paths_created=False,
    checkpoint_or_array_states_fetched_or_decoded=False, execution_authorized=False))
manifest = write(HERE / 'MANIFEST.json', dict(
    schema='amazon_polynormer_V6_qualification_execution_receipts_manifest_v1',
    UTC=datetime.now(timezone.utc).isoformat(), execution_authorized=False,
    payload=[dict(relative=p.relative_to(HERE).as_posix(), descriptor=desc(p))
             for p in sorted(HERE.rglob('*')) if p.is_file() and p.name not in ('MANIFEST.json', 'SEAL.json')]))
seal = write(HERE / 'SEAL.json', dict(
    schema='amazon_polynormer_V6_qualification_execution_receipts_seal_v1',
    UTC=datetime.now(timezone.utc).isoformat(), manifest=manifest, execution_authorized=False))
print(json.dumps(dict(status='success', closure=closure, manifest=manifest, seal=seal,
                      freeze=desc(OUTPUT / 'FREEZE.json'), result=desc(OUTPUT / 'RESULT.json'),
                      terminal=desc(OUTPUT / 'TERMINAL.json'), wall_seconds=terminal['whole_process_wall_seconds'],
                      peak_RSS_bytes=terminal['aggregate_peak_observed_rss_bytes'],
                      cost=result['cost'], after_qualification_packet=dict(plan=desc(PACKET / 'PLAN.json'),
                      manifest=desc(PACKET / 'MANIFEST.json'), seal=desc(PACKET / 'SEAL.json'))), indent=2))
