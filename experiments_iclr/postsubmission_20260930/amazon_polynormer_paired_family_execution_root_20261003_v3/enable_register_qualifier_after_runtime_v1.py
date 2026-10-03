"""Enable separately authorized root releases after authoritative physical gates.

Only source, JSON and compact log metadata are read. No numerical package,
packet module, dataset, checkpoint or runtime binary is opened or imported.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
OLD = PHASE / 'amazon_polynormer_paired_family_execution_root_20261003_v1'
POLICY = ROOT / 'CONDITIONAL_REGISTER_QUALIFIER_SPEC_v1.json'


def confined(value):
    p = Path(value)
    p = p if p.is_absolute() else PHASE / p
    assert p.is_relative_to(PHASE) and '..' not in p.parts
    assert not any(q.is_symlink() for q in (p, *p.parents) if q.is_relative_to(PHASE))
    return p


def read(p):
    return json.loads(confined(p).read_text())


def desc(p):
    p = confined(p)
    assert p.suffix.lower() in ('.json', '.py', '.md', '.txt', '.log', '.sh')
    h = hashlib.sha256()
    size = 0
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
            size += len(b)
    return dict(path=str(p.relative_to(PHASE)), sha256=h.hexdigest(), bytes=size)


def verify(row):
    assert set(row) == {'path', 'sha256', 'bytes'} and desc(row['path']) == row
    return confined(row['path'])


def write(p, value):
    p = confined(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
    return desc(p)


def unique(rows):
    return list({row['path']:row for row in rows}.values())


def verify_source(policy):
    source = policy['source']
    packet = verify(source['manifest']).parent
    verify(source['seal'])
    assert packet.name == 'amazon_polynormer_paired_family_source_preparation_20261003_v5'
    manifest = read(packet/'MANIFEST.json')
    assert read(packet/'SEAL.json')['manifest'] == source['manifest']
    files = sorted(str(p.relative_to(packet)) for p in packet.rglob('*') if p.is_file())
    assert files == sorted([r['path'] for r in manifest['payload']] + ['MANIFEST.json','SEAL.json'])
    for row in manifest['payload']:
        d = desc(packet/row['path'])
        assert d['sha256'] == row['sha256'] and d['bytes'] == row['bytes']
    for row in read(packet/'SOURCE_BINDINGS.json')['read_or_hashed_files']:
        verify(row)
    review = read(verify(policy['source_review']))
    assert review['status'] == 'passed' and review['source'] == source
    return packet


def physical_success(output, release_record, kind, source):
    output = confined(output)
    assert not (output/'FAILURE.json').exists() and not (output/'OUTER_TRACEBACK.txt').exists()
    freeze = read(output/'FREEZE.json')
    terminal = read(output/'TERMINAL.json')
    ready = read(output/'READY.json')
    result = read(output/'RESULT.json')
    assert freeze['schema'] == 'amazon_polynormer_success_freeze_v2' and freeze['status'] == 'success'
    assert freeze['source'] == source and freeze['admission'] == release_record
    assert terminal['schema'] == 'amazon_polynormer_physical_terminal_v2'
    assert terminal['status'] == 'success' and terminal['physical_exit_code'] == 0
    assert terminal['admission'] == release_record
    assert ready['schema'] == 'amazon_polynormer_body_ready_v2' and ready['kind'] == kind
    assert ready['source'] == source and ready['admission'] == release_record
    assert ready['result'] == desc(output/'RESULT.json') and result['source'] == source
    declared = {r['relative']:r['descriptor'] for r in freeze['files']}
    actual = sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.name != 'FREEZE.json')
    assert actual == sorted(declared)
    for rel, row in declared.items():
        assert desc(output/rel) == row
    assert declared['TERMINAL.json'] == desc(output/'TERMINAL.json')
    return unique([desc(output/'FREEZE.json'), *declared.values()])


def verify_consumer(policy, packet):
    row = policy['consumer_release']
    consumer = read(verify(row))
    note = read(verify(policy['consumer_reuse_note']))
    predecessor = read(verify(note['predecessor']))
    assert consumer['source'] == policy['source'] and consumer['execution_authorized'] is True
    assert consumer['test_labels_authorized'] is False
    assert {k:v for k,v in consumer.items() if k != 'source'} == {k:v for k,v in predecessor.items() if k != 'source'}
    for same in note['design_preprocessing_and_role_operation_byte_identity']:
        verify(same)
    assert note['successor'] == row and note['every_other_consumer_field_identical'] is True
    assert read(verify(note['predecessor_role_adoption']))['consumer'] == note['predecessor']
    data = read(verify(consumer['existing_data_manifest']))
    design = read(packet/'DESIGN.json')['data_projection']
    assert consumer['existing_data_manifest'] == design['existing_manifest']
    assert data['public_graph'] == design['public_graph_binding_from_manifest']
    assert data['train_labels'] == design['official_TRAIN_bindings_from_manifest']
    assert data['validation_labels'] == design['official_VAL_bindings_from_manifest']
    assert data['test_label_artifacts'] == []
    rows = [row,policy['consumer_reuse_note'],consumer['existing_data_manifest'],data['producer_release'],data['packet_manifest'],
        data['public_graph'],*data['train_labels'].values(),*data['validation_labels'].values(),*consumer['prior_role_receipts'].values(),
        note['predecessor'],note['predecessor_role_adoption'],note['predecessor_role_transport']]
    # Payload descriptors are retained exactly; metadata is verified here.
    # The admitted qualifier source gate verifies the original data payload bytes.
    for artifact in rows:
        if Path(artifact['path']).suffix == '.json':
            verify(artifact)
    for role in consumer['preprocessing_identity']['roles']:
        prior = read(verify(consumer['prior_role_receipts'][str(role['split'])]))
        assert prior['prior_artifact_comparison_satisfied'] is True and prior['role_record'] == role
        assert prior['data_manifest'] == consumer['existing_data_manifest']
        verify(prior['source_role'])
        rows += [prior['source_role']]
    return unique(rows)


def runtime_successes(policy):
    artifacts = []
    releases = {}
    for mode in ('cpu','gpu'):
        release_row = policy['runtime_releases'][mode]
        release = read(verify(release_row))
        assert release['source'] == policy['source'] and release['source_review'] == policy['source_review']
        assert release['kind'] == 'runtime_capture' and release['execution_authorized'] is True
        assert release['device'] == ('cpu' if mode == 'cpu' else 'cuda:0')
        assert release['interpreter'] == policy['interpreter'] and release['expected_versions'] == policy['expected_versions']
        for row in release['custody_inputs']:
            verify(row)
        output = confined(release['output'])
        artifacts += [release_row,*physical_success(output,release_row,'runtime_capture',policy['source'])]
        receipt_row = desc(output/'RUNTIME.json')
        receipt = read(output/'RUNTIME.json')
        assert receipt['schema'] == 'amazon_polynormer_runtime_v2' and receipt['source'] == policy['source']
        assert receipt['runtime']['versions'] == policy['expected_versions']
        assert read(output/'RESULT.json')['runtime_receipt'] == receipt_row
        releases[mode] = release
    return releases, unique(artifacts)


def base_attempts(policy, packet, runtime_artifacts):
    predecessor = read(verify(policy['pre_runtime_attempt_registry']))
    assert predecessor['source'] == policy['source']
    assert predecessor['all_prior_failed_incomplete_and_superseded_attempts_disclosed'] is True
    rows = unique([*predecessor['prior_attempt_artifacts'],policy['pre_runtime_attempt_registry'],*runtime_artifacts])
    for row in rows:
        verify(row)
    for old_root in (OLD, PHASE/'amazon_polynormer_paired_family_execution_root_20261003_v2'):
        old_registry = read(old_root/'registry/REGISTRY.json')
        assert len(old_registry['physical_fits']) == 15
        assert not any(confined(row['output']).exists() for row in old_registry['physical_fits'])
        verify(old_registry['master_source_claim'])
        rows = unique([*rows,old_registry['master_source_claim'],*[desc(old_root/'registry'/p) for p in
            ('FREEZE.json','TERMINAL.json','LAUNCH.json','READY.json','RESULT.json','stderr.txt','stdout.txt')]])
    return dict(schema='amazon_polynormer_attempt_registry_v2',source=policy['source'],
        preserved_failures=desc(packet/'PRESERVED_FAILURES.json'),prior_attempt_artifacts=rows,
        all_prior_failed_incomplete_and_superseded_attempts_disclosed=True,scientific_fits_started=0,
        old_v2_and_v4_runtime_and_qualifier_failed_costs_preserved=True,old_v2_and_v4_registries_zero_fit=True,
        v3_rejection_preserved=True,exact_v5_runtime_physical_successes_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('after-runtime','after-register'))
    args = parser.parse_args()
    policy = read(POLICY)
    assert policy['schema'] == 'amazon_polynormer_conditional_register_qualifier_spec_v1'
    assert policy['register_authorized_after_exact_runtime_physical_success'] is True
    assert policy['qualifier_authorized_after_exact_runtime_physical_success'] is True
    assert policy['automatic_retry_authorized'] is False and policy['predictive_fits_authorized'] is False
    assert policy['source']['manifest']['sha256'] == '25606a662be16d39219c9ef1fb75f13433f6d76b43559624b8607d1a5a5642bb'
    assert policy['source']['seal']['sha256'] == '1ca33815317fa40069b43e2fc7bc853c4802d91c648a9564dc28ab0f8e10071b'
    assert policy['source_review']['sha256'] == '6d458f272b878e50ac38cab9b14a0c101a8e6589477467869bb03ccb36b6f2a6'
    assert policy['caps'] == dict(wall_seconds=3600,rss_bytes=32*2**30,cuda_peak_allocated_bytes=75*2**30,cuda_peak_reserved_bytes=75*2**30)
    packet = verify_source(policy)
    releases, runtime_artifacts = runtime_successes(policy)
    consumer_custody = verify_consumer(policy,packet)
    attempts = base_attempts(policy,packet,runtime_artifacts)
    base = copy.deepcopy(releases['gpu'])
    base['caps'] = policy['caps']
    base['custody_inputs'] = unique([*base['custody_inputs'],desc(POLICY),*runtime_artifacts,*consumer_custody])
    now = datetime.now(timezone.utc).isoformat()
    if args.stage == 'after-runtime':
        targets = [ROOT/'ATTEMPT_REGISTRY_AFTER_RUNTIME_v1.json',ROOT/'releases/register_authorized_v1.json',ROOT/'REGISTER_RELEASE_ADMISSION_v1.json']
        assert not any(p.exists() for p in targets), 'Already materialized; preserve original attempt'
        attempts_row = write(targets[0],attempts)
        register = copy.deepcopy(base)
        register.update(kind='register',device='cpu',execution_authorized=True,
            output=policy['registration_output'],self_path=str(targets[1].relative_to(PHASE)),
            physical_output_root=str((ROOT/'fits').relative_to(PHASE)),
            fit_release_root=str((ROOT/'releases/fits').relative_to(PHASE)),claim_root=str((ROOT/'claims').relative_to(PHASE)))
        register['custody_inputs'] = unique([*register['custody_inputs'],attempts_row,*attempts['prior_attempt_artifacts']])
        release_row = write(targets[1],register)
        note_row = write(targets[2],dict(UTC=now,source=policy['source'],policy=desc(POLICY),release=release_row,
            authoritative_runtime_freezes_and_terminals_verified=True,source_and_custody_unchanged=True,
            old_failure_costs_and_zero_fit_registry_preserved=True,numerical_work_executed=False))
        print(json.dumps(dict(register_release=release_row,attempt_registry=attempts_row,admission_note=note_row)))
    else:
        register_row = desc(ROOT/'releases/register_authorized_v1.json')
        register = read(verify(register_row))
        assert register['kind']=='register' and register['source']==policy['source'] and register['execution_authorized'] is True
        for row in register['custody_inputs']:
            if Path(row['path']).suffix in ('.json','.txt','.log','.py','.md','.sh'):
                verify(row)
        registration_artifacts = physical_success(policy['registration_output'],register_row,'register',policy['source'])
        registry_row = desc(confined(policy['registration_output'])/'REGISTRY.json')
        registry = read(verify(registry_row))
        assert registry['source']==policy['source'] and registry['distinct_physical_fits']==15 and registry['complete_family_records']==9
        assert registry['automatic_retry_authorized'] is False
        verify(registry['master_source_claim'])
        attempts['prior_attempt_artifacts'] = unique([*attempts['prior_attempt_artifacts'],
            desc(ROOT/'ATTEMPT_REGISTRY_AFTER_RUNTIME_v1.json'),register_row,*registration_artifacts,registry['master_source_claim']])
        targets = [ROOT/'ATTEMPT_REGISTRY_BEFORE_QUALIFIER_v1.json',ROOT/'releases/qualify_authorized_v1.json',ROOT/'QUALIFIER_RELEASE_ADMISSION_v1.json']
        assert not any(p.exists() for p in targets), 'Already materialized; preserve original attempt'
        attempts_row = write(targets[0],attempts)
        qualify = copy.deepcopy(base)
        qualify.update(kind='qualify',device='cuda:0',execution_authorized=True,
            output=policy['qualification_output'],self_path=str(targets[1].relative_to(PHASE)),
            runtime_receipt=desc(ROOT/'runtime_gpu/RUNTIME.json'),consumer_release=policy['consumer_release'],
            registry=registry_row,attempt_registry=attempts_row)
        qualify['custody_inputs'] = unique([*qualify['custody_inputs'],qualify['runtime_receipt'],qualify['consumer_release'],
            registry_row,attempts_row,*attempts['prior_attempt_artifacts']])
        release_row = write(targets[1],qualify)
        note_row = write(targets[2],dict(UTC=now,source=policy['source'],policy=desc(POLICY),release=release_row,
            authoritative_runtime_and_registration_physical_successes_verified=True,
            source_and_custody_unchanged=True,consumer_rebind=policy['consumer_release'],
            qualifier_authorized=True,predictive_fits_authorized=False,numerical_work_executed=False))
        print(json.dumps(dict(qualifier_release=release_row,attempt_registry=attempts_row,admission_note=note_row)))


if __name__ == '__main__':
    main()
