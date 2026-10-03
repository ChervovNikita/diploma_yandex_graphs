"""Prepare ordinary root metadata; no packet execution or numerical imports."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
OLD = PHASE / 'amazon_polynormer_paired_family_execution_root_20261003_v1'
PACKET = PHASE / 'amazon_polynormer_paired_family_source_preparation_20261003_v5'
V2 = PHASE / 'amazon_polynormer_paired_family_source_preparation_20261003_v2'
REVIEW = PHASE / 'amazon_polynormer_v5_adam_image_ownership_source_review_20261003_v1/REVIEW.json'
LAUNCH = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python'


def read(p):
    return json.loads(p.read_text())


def desc(p):
    b = p.read_bytes()
    return dict(path=str(p.relative_to(PHASE)), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def verify(row):
    p = PHASE / row['path']
    assert not Path(row['path']).is_absolute() and '..' not in p.parts
    assert p.suffix not in ('.npy', '.npz', '.pt', '.pth', '.ckpt')
    assert desc(p) == row
    return p


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def descriptors(value):
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256', 'bytes'}:
            yield value
        for child in value.values():
            yield from descriptors(child)
    elif isinstance(value, list):
        for child in value:
            yield from descriptors(child)


source = dict(manifest=desc(PACKET/'MANIFEST.json'), seal=desc(PACKET/'SEAL.json'))
assert source['manifest']['sha256'] == '25606a662be16d39219c9ef1fb75f13433f6d76b43559624b8607d1a5a5642bb'
assert source['seal']['sha256'] == '1ca33815317fa40069b43e2fc7bc853c4802d91c648a9564dc28ab0f8e10071b'
assert read(REVIEW)['status'] == 'passed' and read(REVIEW)['source'] == source
assert desc(REVIEW)['sha256'] == '6d458f272b878e50ac38cab9b14a0c101a8e6589477467869bb03ccb36b6f2a6'
for row in read(PACKET/'MANIFEST.json')['payload']:
    actual = desc(PACKET/row['path'])
    assert (actual['sha256'], actual['bytes']) == (row['sha256'], row['bytes'])
for row in read(PACKET/'SOURCE_BINDINGS.json')['read_or_hashed_files']:
    verify(row)
now = datetime.now(timezone.utc).isoformat()
old_consumer = read(OLD/'CONSUMER_RELEASE.json')
assert old_consumer['execution_authorized'] is True and old_consumer['test_labels_authorized'] is False
identical = [desc(PACKET/p) for p in ('DESIGN.json', 'common.py', 'worker.py')]
assert all((PACKET/p).read_bytes() == (V2/p).read_bytes() for p in ('DESIGN.json', 'common.py', 'worker.py'))
design = read(PACKET/'DESIGN.json')['data_projection']
assert old_consumer['existing_data_manifest'] == design['existing_manifest']
data = read(verify(old_consumer['existing_data_manifest']))
assert old_consumer['producer_release'] == data['producer_release']
assert old_consumer['producer_packet'] == data['packet_manifest']
assert old_consumer['raw_descriptor'] == data['raw_release']
assert data['public_graph'] == design['public_graph_binding_from_manifest']
assert data['train_labels'] == design['official_TRAIN_bindings_from_manifest']
assert data['validation_labels'] == design['official_VAL_bindings_from_manifest']
adoption = read(OLD/'ROLE_COMPARISON_ADOPTION.json')
assert all(r['ordered_FIT_control_VAL_identity_passed'] is True for r in adoption['comparisons'])
for role in old_consumer['preprocessing_identity']['roles']:
    prior = read(verify(old_consumer['prior_role_receipts'][str(role['split'])]))
    assert prior['role_record'] == role and prior['prior_artifact_comparison_satisfied'] is True
    assert prior['data_manifest'] == old_consumer['existing_data_manifest']
    verify(prior['source_role'])
consumer = copy.deepcopy(old_consumer)
consumer['source'] = source
assert {k:v for k,v in consumer.items() if k != 'source'} == {k:v for k,v in old_consumer.items() if k != 'source'}
write(ROOT/'CONSUMER_RELEASE.json', consumer)
reuse = dict(schema='amazon_polynormer_consumer_source_rebind_v1', UTC=now,
    predecessor=desc(OLD/'CONSUMER_RELEASE.json'), successor=desc(ROOT/'CONSUMER_RELEASE.json'),
    predecessor_role_adoption=desc(OLD/'ROLE_COMPARISON_ADOPTION.json'),
    predecessor_role_transport=desc(OLD/'ROLE_COMPARISON_TRANSPORT.json'), source=source,
    design_preprocessing_and_role_operation_byte_identity=identical,
    only_consumer_field_changed='source', every_other_consumer_field_identical=True,
    old_verified_data_preprocessing_and_prior_roles_retained=True,
    role_operation='common.py exact byte identity to approved v2; existing successful ordered FIT/control/VAL comparisons retained',
    role_regeneration_performed=False, arrays_or_checkpoints_opened_or_hashed=False,
    numerical_imports=False, scientific_fits_started=0,
    source_rebind_does_not_adopt_old_runtime_or_old_failed_qualification=True)
write(ROOT/'CONSUMER_REUSE_NOTE.json', reuse)
preserved=[]
for doc in ('SOURCE_BINDINGS.json','PRESERVED_FAILURES.json','REPAIR_NOTE.json','REUSE_AND_DEVIATIONS.json'):
    preserved += list(descriptors(read(PACKET/doc)))
prior_artifacts = read(OLD/'ATTEMPT_REGISTRY_v1.json')['prior_attempt_artifacts']
prior_artifacts += [desc(OLD/p) for p in ('qualification_block0_cuda0/LAUNCH.json',
    'qualification_block0_cuda0/FAILURE.json','qualification_block0_cuda0/TERMINAL.json',
    'qualification_block0_cuda0/stderr.txt','qualification_block0_cuda0/OUTER_TRACEBACK.txt',
    'registry/REGISTRY.json','ATTEMPT_REGISTRY_v1.json','releases/qualify.json')]
prior_artifacts += [row for row in preserved if 'v3_clone_repair_source_review' in row['path'] or '/v3/' in row['path']]
prior_artifacts += [desc(PHASE/'amazon_polynormer_paired_family_source_preparation_20261003_v3'/p) for p in ('MANIFEST.json','SEAL.json')]
# Preserve every V4 execution cost/registry/capture/client record without inspecting values.
PREVIOUS_EXECUTION = PHASE/'amazon_polynormer_paired_family_execution_root_20261003_v2'
prior_artifacts += read(PREVIOUS_EXECUTION/'ATTEMPT_REGISTRY_BEFORE_QUALIFIER_v1.json')['prior_attempt_artifacts']
prior_artifacts += [desc(p) for p in PREVIOUS_EXECUTION.rglob('*') if p.is_file() and p.suffix in ('.json','.py','.txt','.log','.md','.sh')]
for previous in (OLD,PREVIOUS_EXECUTION):
    registry=read(previous/'registry/REGISTRY.json')
    assert len(registry['physical_fits'])==15 and not any((PHASE/r['output']).exists() for r in registry['physical_fits'])
    prior_artifacts += [registry['master_source_claim']]
prior_artifacts = list({row['path']:row for row in prior_artifacts}.values())
for row in prior_artifacts:
    verify(row)
attempts = dict(schema='amazon_polynormer_attempt_registry_v2', source=source,
    preserved_failures=desc(PACKET/'PRESERVED_FAILURES.json'), prior_attempt_artifacts=prior_artifacts,
    all_prior_failed_incomplete_and_superseded_attempts_disclosed=True,
    scientific_fits_started=0, new_source_numerically_qualified=False,
    v2_and_v4_qualifier_failed_and_v3_rejected_source_preserved=True,
    v2_and_v4_registered_physical_fit_outputs_absent=True,
    runtime_captures_not_yet_executed=True)
write(ROOT/'ATTEMPT_REGISTRY_PRE_RUNTIME_v1.json', attempts)
authorization = dict(schema='amazon_polynormer_root_runtime_preparation_authorization_v1', UTC=now,
    source=source, source_review=desc(REVIEW), route_probe=desc(ROOT/'ROUTE_PREWRITE_PROBE.json'),
    root_instruction='Prepare/upload exact V5 source, dependencies and metadata only. Root conditionally authorizes CPU/GPU runtime captures after passed source review, then registration and full qualifier after authoritative runtime/registration success; launch nothing in preparation.',
    runtime_capture_authorized=['cpu','cuda:0'], runtime_caps=dict(wall_seconds=900,rss_bytes=16*2**30,
        cuda_peak_allocated_bytes=16*2**30,cuda_peak_reserved_bytes=16*2**30),
    qualifier_authorized=False, predictive_fits_authorized=False, registration_launch_authorized=False,
    register_authorized_after_runtime_physical_success=True,qualifier_authorized_after_runtime_and_registration_physical_success=True,
    heldout_control_scoring_authorized=False, test_labels_authorized=False,
    existing_route_only=True, installs_or_environment_changes_authorized=False)
write(ROOT/'ROOT_PREPARATION_AUTHORIZATION.json', authorization)
base=read(OLD/'releases/runtime_gpu_v2.json')
assert base['interpreter']['path'] == LAUNCH
assert base['expected_versions'] == read(PACKET/'DEPENDENCIES.json')['admissible_existing_profiles']['original_photo_qualified']['versions']
base.update(source=source, source_review=desc(REVIEW), caps=authorization['runtime_caps'])
base['custody_inputs'] = [desc(ROOT/'ROOT_PREPARATION_AUTHORIZATION.json'), desc(ROOT/'ROUTE_PREWRITE_PROBE.json'),
    desc(ROOT/'ATTEMPT_REGISTRY_PRE_RUNTIME_v1.json'), *prior_artifacts]
for mode in ('cpu','gpu'):
    value=copy.deepcopy(base)
    value.update(kind='runtime_capture', execution_authorized=True, independent_source_review_passed=True,
        device='cpu' if mode=='cpu' else 'cuda:0', output=str((ROOT/('runtime_'+mode)).relative_to(PHASE)),
        self_path=str((ROOT/'releases'/('runtime_'+mode+'.json')).relative_to(PHASE)))
    write(ROOT/'releases'/('runtime_'+mode+'.json'), value)
    spec=copy.deepcopy(value)
    for key in ('source','self_path','source_review','expected_versions','interpreter'):
        spec.pop(key)
    spec.update(source_review_descriptor_path=str(REVIEW.relative_to(PHASE)), interpreter_path=LAUNCH,
                existing_runtime_profile='original_photo_qualified')
    write(ROOT/'release_specs'/('runtime_'+mode+'.json'), spec)
register=copy.deepcopy(base)
register.update(kind='register', execution_authorized=False, device='cpu',
    output=str((ROOT/'registry').relative_to(PHASE)), self_path=str((ROOT/'releases/register.json').relative_to(PHASE)),
    physical_output_root=str((ROOT/'fits').relative_to(PHASE)),
    fit_release_root=str((ROOT/'releases/fits').relative_to(PHASE)), claim_root=str((ROOT/'claims').relative_to(PHASE)))
register['custody_inputs'] += [desc(ROOT/'CONSUMER_RELEASE.json'),desc(ROOT/'CONSUMER_REUSE_NOTE.json')]
write(ROOT/'releases/register.json', register)
spec=copy.deepcopy(register)
for key in ('source','self_path','source_review','expected_versions','interpreter'):
    spec.pop(key)
spec.update(source_review_descriptor_path=str(REVIEW.relative_to(PHASE)), interpreter_path=LAUNCH,
            existing_runtime_profile='original_photo_qualified')
write(ROOT/'release_specs/register.json', spec)
qualify=copy.deepcopy(spec)
for key in ('physical_output_root','fit_release_root','claim_root'):
    qualify.pop(key)
qualify.update(kind='qualify', execution_authorized=False, device='cuda:0',
    output=str((ROOT/'qualification_block0_cuda0').relative_to(PHASE)),
    runtime_receipt_descriptor_path=str((ROOT/'runtime_gpu/RUNTIME.json').relative_to(PHASE)),
    consumer_release_descriptor_path=str((ROOT/'CONSUMER_RELEASE.json').relative_to(PHASE)),
    attempt_registry_descriptor_path=str((ROOT/'ATTEMPT_REGISTRY_AFTER_RUNTIME_PENDING.json').relative_to(PHASE)),
    caps=dict(wall_seconds=None,rss_bytes=None,cuda_peak_allocated_bytes=None,cuda_peak_reserved_bytes=None))
write(ROOT/'release_specs/qualify.template.json', qualify)
write(ROOT/'QUALIFIER_TEMPLATE_NOTE.json',dict(UTC=now, execution_authorized=False,
    template=desc(ROOT/'release_specs/qualify.template.json'),
    requires=['successful exact-V5 CPU/GPU runtime captures and physical freezes',
        'new attempt registry extending all prior artifacts with these captures',
        'root conditional authorization and fixed 3600s/32GiB/75GiB caps','fixed registration under approved V5'],
    old_gpu_runtime_not_reused=True, qualifier_not_launched=True, predictive_fit_releases_not_prepared=True))
print(json.dumps(dict(runtime_cpu=desc(ROOT/'releases/runtime_cpu.json'),runtime_gpu=desc(ROOT/'releases/runtime_gpu.json'),
    consumer=desc(ROOT/'CONSUMER_RELEASE.json'),reuse_note=desc(ROOT/'CONSUMER_REUSE_NOTE.json'),
    registration=desc(ROOT/'releases/register.json'),qualifier_template=desc(ROOT/'release_specs/qualify.template.json'))))

# Root conditional authorization is recorded now, but these stages materialize only after physical gates.
conditional=dict(schema='amazon_polynormer_conditional_register_qualifier_spec_v1',UTC=now,
    source=source,source_review=desc(REVIEW),interpreter=base['interpreter'],expected_versions=base['expected_versions'],
    runtime_releases={m:desc(ROOT/'releases'/('runtime_'+m+'.json')) for m in ('cpu','gpu')},
    consumer_release=desc(ROOT/'CONSUMER_RELEASE.json'),consumer_reuse_note=desc(ROOT/'CONSUMER_REUSE_NOTE.json'),
    pre_runtime_attempt_registry=desc(ROOT/'ATTEMPT_REGISTRY_PRE_RUNTIME_v1.json'),
    metadata_helper=desc(ROOT/'enable_register_qualifier_after_runtime_v1.py'),
    register_authorized_after_exact_runtime_physical_success=True,
    qualifier_authorized_after_exact_runtime_physical_success=True,
    qualifier_release_materialization_requires_registration_physical_success=True,
    registration_output=str((ROOT/'registry').relative_to(PHASE)),
    qualification_output=str((ROOT/'qualification_block0_cuda0').relative_to(PHASE)),
    caps=dict(wall_seconds=3600,rss_bytes=32*2**30,cuda_peak_allocated_bytes=75*2**30,cuda_peak_reserved_bytes=75*2**30),
    root_instruction='Source-approved V5 CPU/GPU captures900s/16GiB, then register/full qualifier3600s/32GiB/75GiB after authoritative physical successes; no numerical launch in preparation.',
    automatic_retry_authorized=False,predictive_fits_authorized=False,control_predictive_scoring_authorized=False,
    test_labels_authorized=False,installs_or_environment_changes_authorized=False,no_capture_restart_authorized=True)
write(ROOT/'CONDITIONAL_REGISTER_QUALIFIER_SPEC_v1.json',conditional)
disabled=copy.deepcopy(base)
disabled.update(kind='qualify',execution_authorized=False,device='cuda:0',
    output=str((ROOT/'qualification_block0_cuda0').relative_to(PHASE)),
    self_path=str((ROOT/'releases/qualify.disabled.json').relative_to(PHASE)),
    caps=conditional['caps'],runtime_receipt=None,consumer_release=conditional['consumer_release'],registry=None,attempt_registry=None)
write(ROOT/'releases/qualify.disabled.json',disabled)
print(json.dumps(dict(conditional_policy=desc(ROOT/'CONDITIONAL_REGISTER_QUALIFIER_SPEC_v1.json'),disabled_qualifier=desc(ROOT/'releases/qualify.disabled.json'))))
