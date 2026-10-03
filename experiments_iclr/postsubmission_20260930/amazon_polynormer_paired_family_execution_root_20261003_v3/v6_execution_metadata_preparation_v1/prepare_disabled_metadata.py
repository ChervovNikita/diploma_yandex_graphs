"""Source/JSON-only V6 execution preparation; never authorize or launch.

Each invocation writes a fresh metadata packet under this helper directory.
Actual root releases/output/claims remain untouched. Numerical bodies, arrays,
checkpoint contents and runtime binaries are never opened or imported here.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE = ROOT.parent
SOURCE = PHASE / 'amazon_polynormer_paired_family_source_preparation_20261003_v6'
FORECAST = ROOT / 'fit_schedule_resource_candidate_v2_v6_bounded_retention'
TEXT = {'.json', '.jsonl', '.py', '.md', '.txt', '.log', '.sh', '.html', '.patch'}
ROOT_REL = str(ROOT.relative_to(PHASE))
OUTPUTS = {k: ROOT_REL + '/' + v for k, v in {
    'runtime_cpu': 'v6_runtime_cpu_v1', 'runtime_gpu': 'v6_runtime_cuda0_v1',
    'qualification': 'v6_qualification_block0_cuda0_v1',
    'closure': 'v6_closure_v1', 'all': 'v6_full_schedule_v1'}.items()}
RELEASES = {k: ROOT_REL + '/v6_releases/' + k + '_v1.json' for k in OUTPUTS}
RUNTIME_CAPS = dict(wall_seconds=900, rss_bytes=16*2**30,
    cuda_peak_allocated_bytes=16*2**30, cuda_peak_reserved_bytes=16*2**30)
QUALIFIER_CAPS = dict(wall_seconds=3600, rss_bytes=32*2**30,
    cuda_peak_allocated_bytes=75*2**30, cuda_peak_reserved_bytes=75*2**30)


def confined(value):
    p = Path(value)
    p = p if p.is_absolute() else PHASE/p
    assert p.is_relative_to(PHASE) and '..' not in p.parts
    assert not any(q.is_symlink() for q in (p, *p.parents) if q.is_relative_to(PHASE))
    return p


def read(path):
    p = confined(path)
    assert p.suffix == '.json'
    return json.loads(p.read_text())


def desc(path):
    p = confined(path)
    assert p.suffix in TEXT
    b = p.read_bytes(); b.decode('utf8')
    return dict(path=str(p.relative_to(PHASE)), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def verify(row):
    assert set(row) == {'path', 'sha256', 'bytes'} and desc(row['path']) == row
    return confined(row['path'])


def unique(rows):
    result = {}
    for r in rows:
        assert r['path'] not in result or result[r['path']] == r
        result[r['path']] = r
    return list(result.values())


def success(kind, output, source):
    p = confined(output)
    release = desc(RELEASES[kind])
    admission = read(verify(release))
    assert admission['execution_authorized'] is True and admission['source'] == source
    expected_kind = 'runtime_capture' if kind.startswith('runtime_') else 'qualify'
    assert admission['kind'] == expected_kind
    assert admission['output'] == output and admission['self_path'] == RELEASES[kind]
    assert not (p/'FAILURE.json').exists() and not (p/'OUTER_TRACEBACK.txt').exists()
    f, t, r, ready = [read(p/name) for name in ('FREEZE.json', 'TERMINAL.json', 'RESULT.json', 'READY.json')]
    assert f['schema'] == 'amazon_polynormer_success_freeze_v2' and f['status'] == 'success'
    assert f['source'] == source and f['admission'] == release
    assert t['status'] == 'success' and t['physical_exit_code'] == 0 and t['admission'] == release
    assert r['source'] == source and ready['source'] == source and ready['admission'] == release
    assert ready['kind'] == expected_kind
    assert ready['result'] == desc(p/'RESULT.json')
    actual = sorted(str(q.relative_to(p)) for q in p.rglob('*') if q.is_file() and q.name != 'FREEZE.json')
    assert actual == sorted(v['relative'] for v in f['files'])
    records = [release, desc(p/'FREEZE.json')]
    for row in f['files']:
        path = p/row['relative']
        assert confined(row['descriptor']['path']) == path
        if path.suffix in TEXT:
            assert desc(path) == row['descriptor']
        else:
            assert path.suffix == '.pt'  # Descriptor only; no checkpoint-body read.
        records.append(row['descriptor'])
    return r, unique(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('initial', 'after-runtime', 'after-qualification', 'after-fit-releases'), required=True)
    parser.add_argument('--packet-name', required=True)
    parser.add_argument('--review')
    parser.add_argument('--runtime-gate-review')
    parser.add_argument('--resource-admission')
    parser.add_argument('--consumer-release')
    args = parser.parse_args()
    assert Path(args.packet_name).name == args.packet_name and args.packet_name not in ('', '.', '..')
    packet = HERE/args.packet_name
    assert not packet.exists()
    source = dict(manifest=desc(SOURCE/'MANIFEST.json'), seal=desc(SOURCE/'SEAL.json'))
    assert source['manifest']['sha256'] == 'd4160d8aeec2b770274875f9a1fb137facc9f93efd4eca6ad91e58802521ff44'
    assert source['seal']['sha256'] == 'b70beb8868d09c7f066c82cd9c0d1c582229cdfe0124485b32e12d7a7069f373'
    assert read(SOURCE/'SEAL.json')['manifest'] == source['manifest']
    for row in read(SOURCE/'MANIFEST.json')['payload']:
        p = SOURCE/row['path']; d = desc(p)
        assert (d['sha256'], d['bytes']) == (row['sha256'], row['bytes'])
        if p.suffix == '.py': compile(ast.parse(p.read_text()), str(p), 'exec')
    cohort = read(SOURCE/'COHORT_SOURCE_BINDING.json')
    registry = read(verify(cohort['registry']))
    assert registry['source'] == cohort['registered_source'] and len(registry['physical_fits']) == 15
    assert [r['row'] for r in registry['physical_fits']] == read(SOURCE/'DESIGN.json')['physical_fit_schedule']
    assert not registry['automatic_retry_authorized']
    assert read(verify(cohort['master_source_claim']))['no_second_registry_or_replacement'] is True
    assert not any(confined(r[k]).exists() for r in registry['physical_fits'] for k in ('output','claim_path'))
    if args.stage != 'after-fit-releases':
        assert not any(confined(r['release_path']).exists() for r in registry['physical_fits'])
    review = desc(args.review) if args.review else None
    runtime_review = None
    if review:
        value = read(verify(review)); assert value['status'] in ('passed','PASS_SOURCE_ONLY') and value['source'] == source
        if value['status'] == 'passed': runtime_review = review
    if args.runtime_gate_review:
        runtime_review = desc(args.runtime_gate_review)
        value = read(verify(runtime_review)); assert value['status'] == 'passed' and value['source'] == source
    if args.stage != 'initial': assert review is not None
    if args.stage != 'initial': assert runtime_review is not None
    forecast_manifest = desc(FORECAST/'MANIFEST.json')
    forecast_seal = desc(FORECAST/'SEAL.json')
    assert read(FORECAST/'SEAL.json')['manifest'] == forecast_manifest
    if review:
        assert read(verify(review))['forecast'] == dict(manifest=forecast_manifest,seal=forecast_seal)
    for row in read(FORECAST/'MANIFEST.json')['payload']:
        assert desc(FORECAST/row['relative']) == row['descriptor']
    old_consumer = read(ROOT/'CONSUMER_RELEASE.json')
    data = read(verify(old_consumer['existing_data_manifest']))
    design = read(SOURCE/'DESIGN.json')['data_projection']
    assert old_consumer['existing_data_manifest'] == design['existing_manifest']
    assert data['public_graph'] == design['public_graph_binding_from_manifest']
    assert data['train_labels'] == design['official_TRAIN_bindings_from_manifest']
    assert data['validation_labels'] == design['official_VAL_bindings_from_manifest'] and data['test_label_artifacts'] == []
    consumer_inputs = [old_consumer['existing_data_manifest'], data['producer_release'], data['packet_manifest'],
        data['public_graph'], *data['train_labels'].values(), *data['validation_labels'].values(),
        *old_consumer['prior_role_receipts'].values()]
    for row in consumer_inputs:
        if confined(row['path']).suffix in TEXT: verify(row)
        else: assert confined(row['path']).suffix == '.npz'  # No array read/hash.
    for name in ('common.py','DESIGN.json'):
        assert (SOURCE/name).read_bytes() == (PHASE/'amazon_polynormer_paired_family_source_preparation_20261003_v5'/name).read_bytes()
    consumer = copy.deepcopy(old_consumer)
    consumer.update(source=source, execution_authorized=False)
    assert {k:v for k,v in consumer.items() if k not in ('source','execution_authorized')} == {k:v for k,v in old_consumer.items() if k not in ('source','execution_authorized')}
    admitted_consumer = None
    if args.consumer_release:
        admitted_consumer = desc(args.consumer_release)
        current_consumer = read(verify(admitted_consumer))
        assert current_consumer['source'] == source and current_consumer['execution_authorized'] is True
        assert current_consumer['test_labels_authorized'] is False
        assert {k:v for k,v in current_consumer.items() if k not in ('source','execution_authorized')} == {k:v for k,v in old_consumer.items() if k not in ('source','execution_authorized')}
    if args.stage != 'initial': assert admitted_consumer is not None
    attempts = copy.deepcopy(read(FORECAST/'ATTEMPT_REGISTRY_CANDIDATE.json'))
    attempts.update(source=source, preserved_failures=desc(SOURCE/'PRESERVED_FAILURES.json'),
        execution_authorized=False, new_source_numerically_qualified=False)
    artifacts = [*attempts['prior_attempt_artifacts'], *source.values(), cohort['registry'], cohort['master_source_claim'],
        desc(ROOT/'CONSUMER_RELEASE.json'), forecast_manifest, forecast_seal, desc(FORECAST/'prepare_v6_metadata_candidate.py')]
    if review: artifacts.append(review)
    if runtime_review: artifacts.append(runtime_review)
    if admitted_consumer: artifacts.append(admitted_consumer)
    runtime_rows = {}
    if args.stage in ('after-runtime','after-qualification','after-fit-releases'):
        for mode in ('runtime_cpu','runtime_gpu'):
            result, rows = success(mode, OUTPUTS[mode], source)
            runtime = read(confined(OUTPUTS[mode])/'RUNTIME.json')
            assert runtime['schema'] == 'amazon_polynormer_runtime_v2' and runtime['source'] == source
            assert runtime['runtime']['hardware']['device'] == ('cpu' if mode == 'runtime_cpu' else 'cuda:0')
            runtime_rows[mode] = desc(confined(OUTPUTS[mode])/'RUNTIME.json')
            artifacts += rows
    qualification = resource = None
    if args.stage in ('after-qualification','after-fit-releases'):
        q, rows = success('qualification', OUTPUTS['qualification'], source)
        assert q['status'] == 'passed' and q['report_eligible'] is False and len(q['forms']) == 5
        assert q['runtime_receipt'] == runtime_rows['runtime_gpu'] and q['consumer_release'] == admitted_consumer
        assert all(f[s]['bitwise_full_next_step'] for f in q['forms'] for s in ('local_replay','global_replay'))
        assert all(f['retirement_probe']['live_model_Adam_grad_modes_stage_and_RNG_bitwise_unchanged'] and
            f['retirement_probe']['selected_local_and_global_probe_still_available'] for f in q['forms'])
        qualification = desc(confined(OUTPUTS['qualification'])/'FREEZE.json')
        artifacts += rows
        attempts['new_source_numerically_qualified'] = True
        if args.resource_admission:
            resource = desc(args.resource_admission); admitted = read(verify(resource))
            assert admitted['schema'] == 'amazon_polynormer_resource_admission_v3'
            assert admitted['source'] == source and admitted['qualification_freeze'] == qualification
            assert admitted['registry'] == cohort['registry'] and admitted['execution_authorized'] is True
            assert admitted['full_15_fit_schedule_authorized'] is True
            assert admitted['retained_local_and_final_checkpoint_replays_costed'] is True
            assert admitted['all_selected_checkpoint_replays_required'] is False
            assert all(v is not None for v in admitted['forecast_to_fill_from_actual_probe'].values())
            artifacts.append(resource)
    if args.stage == 'after-fit-releases': assert resource is not None
    attempts['prior_attempt_artifacts'] = unique(artifacts)
    for row in attempts['prior_attempt_artifacts']:
        if confined(row['path']).suffix in TEXT: verify(row)
        else: assert confined(row['path']).suffix in ('.pt','.npz','.npy','.pth')
    packet.mkdir()
    def write(name, value):
        p = packet/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as h:json.dump(value,h,indent=2,allow_nan=False);h.write('\n')
        return desc(p)
    consumer_row = write('CONSUMER_RELEASE_DISABLED.json', consumer)
    attempt_row = write('ATTEMPT_REGISTRY_DISABLED.json', attempts)
    write('CONSUMER_REBIND_NOTE.json', dict(source=source, predecessor=desc(ROOT/'CONSUMER_RELEASE.json'),
        successor_disabled=consumer_row, unchanged_preprocessing_role_sources=['common.py','DESIGN.json'],
        exact_other_fields_preserved=True, fresh_source_rebind_only=True, root_authorization_required=True,
        arrays_or_labels_or_numeric_execution=False))
    base = copy.deepcopy(read(ROOT/'releases/runtime_gpu.json'))
    base.update(source=source, source_review=runtime_review, independent_source_review_passed=bool(runtime_review),
        execution_authorized=False, automatic_retry_authorized=False, test_labels_authorized=False)
    base['custody_inputs'] = unique([*attempts['prior_attempt_artifacts'], *source.values(),
        consumer_row, attempt_row, *consumer_inputs, *([review] if review else []), *([runtime_review] if runtime_review else [])])
    releases = {}
    for mode in ('runtime_cpu','runtime_gpu'):
        value = copy.deepcopy(base)
        value.update(kind='runtime_capture', device='cpu' if mode == 'runtime_cpu' else 'cuda:0',
            output=OUTPUTS[mode],self_path=RELEASES[mode],caps=RUNTIME_CAPS)
        if args.stage == 'initial': assert not confined(OUTPUTS[mode]).exists() and not confined(RELEASES[mode]).exists()
        releases[mode] = write('disabled_releases/'+mode+'.json', value)
    qualifier = copy.deepcopy(base)
    qualifier.update(kind='qualify',device='cuda:0',output=OUTPUTS['qualification'],self_path=RELEASES['qualification'],
        caps=QUALIFIER_CAPS,registry=cohort['registry'],runtime_receipt=runtime_rows.get('runtime_gpu'),
        consumer_release=admitted_consumer or consumer_row,attempt_registry=attempt_row)
    if args.stage in ('initial','after-runtime'): assert not confined(OUTPUTS['qualification']).exists() and not confined(RELEASES['qualification']).exists()
    qualifier['custody_inputs'] = unique([*qualifier['custody_inputs'], *runtime_rows.values(), cohort['registry']])
    releases['qualification'] = write('disabled_releases/qualification.json', qualifier)
    fit_candidates = []
    admitted_caps = read(verify(resource))['caps'] if resource else read(FORECAST/'RESOURCE_ADMISSION_CANDIDATE.json')['caps']
    for row in registry['physical_fits']:
        value = copy.deepcopy(qualifier)
        value.update(kind='fit',fit_id=row['id'],output=row['output'],self_path=row['release_path'],
            registered_claim_path=row['claim_path'],qualification_freeze=qualification,
            resource_admission=resource,caps=admitted_caps)
        value['custody_inputs'] = unique([*value['custody_inputs'], *([qualification] if qualification else []), *([resource] if resource else [])])
        record = write('disabled_fit_releases/'+row['id']+'.json',value)
        fit_candidates.append(dict(fit_id=row['id'],candidate=record,registered_output=row['output'],
            registered_release_path=row['release_path'],registered_claim_path=row['claim_path']))
    close = copy.deepcopy(qualifier)
    close.update(kind='close',device='cpu',output=OUTPUTS['closure'],self_path=RELEASES['closure'],caps=dict(admitted_caps,wall_seconds=21600))
    releases['closure'] = write('disabled_releases/closure.json',close)
    queue = copy.deepcopy(qualifier)
    queue.update(kind='all',output=OUTPUTS['all'],self_path=RELEASES['all'],qualification_freeze=qualification,
        resource_admission=resource,caps=dict(admitted_caps,wall_seconds=15*admitted_caps['wall_seconds']+25200),
        fit_releases=[dict(path=r['registered_release_path'],sha256=None,bytes=None) for r in fit_candidates],
        closure_release=dict(path=RELEASES['closure'],sha256=None,bytes=None))
    if args.stage == 'after-fit-releases':
        actual_releases = []
        for row in registry['physical_fits']:
            record = desc(row['release_path']); actual = read(verify(record))
            assert actual['execution_authorized'] is True and actual['kind'] == 'fit'
            assert actual['source'] == source and actual['source_review'] == runtime_review
            assert actual['fit_id'] == row['id'] and actual['self_path'] == row['release_path'] and actual['output'] == row['output']
            assert actual['registry'] == cohort['registry'] and actual['runtime_receipt'] == runtime_rows['runtime_gpu']
            assert actual['consumer_release'] == admitted_consumer and actual['qualification_freeze'] == qualification
            assert actual['resource_admission'] == resource and actual['caps'] == admitted_caps
            actual_releases.append(record)
        actual_close = desc(RELEASES['closure']); closed = read(verify(actual_close))
        assert closed['execution_authorized'] is True and closed['kind'] == 'close' and closed['source'] == source
        assert closed['source_review'] == runtime_review and closed['registry'] == cohort['registry']
        assert closed['runtime_receipt'] == runtime_rows['runtime_gpu'] and closed['consumer_release'] == admitted_consumer
        assert closed['output'] == OUTPUTS['closure'] and closed['self_path'] == RELEASES['closure']
        queue.update(fit_releases=actual_releases,closure_release=actual_close)
        queue['custody_inputs'] = unique([*queue['custody_inputs'], qualification, resource, *actual_releases, actual_close])
    releases['all'] = write('disabled_releases/all.json',queue)
    policy = dict(schema='amazon_polynormer_V6_disabled_execution_metadata_v1',UTC=datetime.now(timezone.utc).isoformat(),
        stage=args.stage,source=source,independent_source_review_evidence=review,
        source_review_for_runtime_gate=runtime_review,source_review_pending=review is None,
        runtime_gate_review_status_compatibility_pending=runtime_review is None,
        metadata_helper=desc(Path(__file__)),
        original_registry=cohort['registry'],original_master_source_claim=cohort['master_source_claim'],
        second_registration_forbidden=True,outputs=OUTPUTS,root_release_paths=RELEASES,
        source_only_forecast_helper=desc(FORECAST/'prepare_v6_metadata_candidate.py'),
        forecast_helper_independent_source_review=review,
        forecast_packet_manifest=forecast_manifest,forecast_packet_seal=forecast_seal,
        forecast_helper_review_or_measured_resource_admission_not_inferred=True,
        consumer_disabled=consumer_row,admitted_consumer_release=admitted_consumer,attempt_registry_disabled=attempt_row,releases_disabled=releases,
        runtime_receipts=runtime_rows,qualification_freeze=qualification,resource_admission=resource,
        original15fit_candidates=fit_candidates,execution_authorized=False,root_release_materialization_required=True,
        registered_release_output_claim_paths_created=False,scientific_fits_control_or_TEST_launches=0,
        real_arrays_checkpoint_or_runtime_binary_values_read=False,
        exact_consumer_release_and_fit_queue_descriptors_require_root_promotion_then_rebinding=True,
        all_fit_and_queue_candidates_remain_disabled_even_after_actual_probe=True)
    policy_row = write('PLAN.json',policy)
    files=sorted(p for p in packet.rglob('*') if p.is_file())
    manifest=write('MANIFEST.json',dict(schema='amazon_polynormer_V6_disabled_execution_metadata_manifest_v1',
        execution_authorized=False,payload=[dict(relative=str(p.relative_to(packet)),descriptor=desc(p)) for p in files]))
    seal=write('SEAL.json',dict(schema='amazon_polynormer_V6_disabled_execution_metadata_seal_v1',manifest=manifest,execution_authorized=False))
    print(json.dumps(dict(plan=policy_row,manifest=manifest,seal=seal,execution_authorized=False,stage=args.stage),indent=2))


if __name__ == '__main__':
    main()
