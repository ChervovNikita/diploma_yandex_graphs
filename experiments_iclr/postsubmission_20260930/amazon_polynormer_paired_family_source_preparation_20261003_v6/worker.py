"""Worker body; only the CPU supervisor may publish physical success."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import common as c


def register(a, output):
    c.require(False, 'V6 inherits the exact existing fifteen-fit registry; a second registration is forbidden')
    fits_root = c.confined(a['physical_output_root'])
    releases_root = c.confined(a['fit_release_root'])
    claims_root = c.confined(a['claim_root'])
    c.require(not any(p.is_relative_to(c.PACKET) or p.is_relative_to(output) for p in
                      (fits_root, releases_root, claims_root)), 'Registry outputs/claims/releases must be outside sealed files')
    master_claim = c.PHASE / 'amazon_polynormer_source_once_only_claims_v2' / (a['source']['manifest']['sha256'] + '.json')
    c.write(master_claim, {'schema': 'amazon_polynormer_one_cohort_per_source_v2', 'source': a['source'],
                           'registration_release': c.record(a['self_path']), 'registry_output': a['output'],
                           'no_second_registry_or_replacement': True})
    rows = []
    for row in c.schedule():
        ident = c.fit_id(row)
        rows.append({'id': ident, 'row': row, 'output': str((fits_root / ident).relative_to(c.PHASE)),
                     'release_path': str((releases_root / (ident + '.json')).relative_to(c.PHASE)),
                     'claim_path': str((claims_root / (ident + '.json')).relative_to(c.PHASE))})
    families = []
    for split, seed in c.BLOCKS:
        selected = [v for v in rows if v['row']['split'] == split]
        native = [v['id'] for v in selected if v['row']['kind'] == 'native_independent']
        boundary = [v['id'] for v in selected if v['row']['kind'] == 'gnnm_boundary_4']
        c.require(len(native) == 4 and len(boundary) == 1, 'Registry family coverage invalid')
        for name, references in (('single_author', native[:1]), ('gnnm_boundary_4', boundary),
                                  ('independent_author_4_same_width', native)):
            families.append({'split': split, 'block_seed': seed, 'family': name,
                             'physical_fit_references': references,
                             'native_single_is_exact_member0_alias': name == 'single_author'})
    registry = {'schema': 'amazon_polynormer_registry_v2', 'source': a['source'],
                'physical_fits': rows, 'families': families, 'automatic_retry_authorized': False,
                'master_source_claim': c.record(master_claim),
                'distinct_physical_fits': 15, 'complete_family_records': 9,
                'scientific_optimizer_updates': 40500, 'complete_member_training_trajectory_updates': 64800}
    record = c.write(output / 'REGISTRY.json', registry)
    c.write(output / 'RESULT.json', {'schema': 'amazon_polynormer_registered_v2', 'status': 'complete',
                                    'source': a['source'], 'registry': record})


def prepare_consumer(a, output):
    import native_training as n
    x, edge, blocks, preprocessing = c.load_data(a, 'cpu', prepare=True)
    d = c.read(c.verify(a['existing_data_manifest']))
    roles = {}
    for split, _ in c.BLOCKS:
        roles[str(split)] = c.write(output / ('ROLE_SPLIT%d.json' % split),
            {'schema': 'amazon_fixed_role_identity_v1', 'data_manifest': a['existing_data_manifest'],
             'role_record': blocks[split]['role_record'], 'fresh_before_fit_derivation': True,
             'prior_artifact_comparison_satisfied': False})
    proposed = {'schema': 'amazon_polynormer_consumer_release_v2', 'source': a['source'],
                'execution_authorized': False, 'test_labels_authorized': False,
                'existing_data_manifest': a['existing_data_manifest'], 'producer_release': d['producer_release'],
                'producer_packet': d['packet_manifest'], 'raw_descriptor': d['raw_release'],
                'new_pre_fit_role_receipts': roles, 'prior_role_receipts': None,
                'preprocessing_identity': preprocessing,
                'required_review': 'Independently compare to genuine preserved prior role identities and fill prior_role_receipts; this proposal cannot admit qualification/fits.'}
    proposal = c.write(output / 'PROPOSED_CONSUMER.json', proposed)
    c.write(output / 'RESULT.json', {'schema': 'amazon_polynormer_consumer_prepare_v2', 'status': 'complete',
            'source': a['source'], 'proposal': proposal, 'role_receipts': roles,
            'TRAIN_control_predictive_outcomes_computed': False, 'TEST_labels_opened': False,
            'memory': n.memory('cpu'), 'consumer_release_authorized': False})


def close(a, output):
    import study
    registry = study.registered_cohort(a)
    c.verify(registry['master_source_claim'])
    c.require([q['row'] for q in registry['physical_fits']] == c.schedule(),
              'Complete exact registry required')
    results = []
    for row in registry['physical_fits']:
        fit_path = c.confined(row['output'])
        freeze_record = c.record(fit_path / 'FREEZE.json')
        _, frozen = study.freeze(freeze_record)
        value = c.read(frozen / 'RESULT.json')
        import evaluate
        evaluate.check_trace(value)
        c.require(value['bindings']['fit_id'] == row['id'] and value['bindings']['row'] == row['row'] and
                  value['bindings']['source'] == a['source'] and value['bindings']['registry'] == a['registry'] and
                  value['bindings']['runtime_receipt'] == a['runtime_receipt'] and
                  value['bindings']['consumer_release'] == a['consumer_release'] and value['report_eligible'] is True,
                  'Physical exact source/data/runtime cohort differs')
        claim = c.read(row['claim_path'])
        c.require(claim['fit_id'] == row['id'] and claim['registry'] == a['registry'] and
                  claim['release'] == value['bindings']['admission'] and
                  claim['output'] == row['output'], 'Once-only physical claim differs')
        c.verify(claim['release'])
        results.append({'id': row['id'], 'freeze': freeze_record, 'result': c.record(frozen / 'RESULT.json'),
                        'claim': c.record(row['claim_path']), 'cost': value['cost']})
    c.require(len(results) == 15 and len(registry['families']) == 9, 'Partial cohort cannot close')
    c.write(output / 'RESULT.json', {'schema': 'amazon_polynormer_cohort_closure_v2', 'status': 'complete',
            'source': a['source'], 'registry': a['registry'], 'runtime_receipt': a['runtime_receipt'],
            'consumer_release': a['consumer_release'], 'physical_fits': results, 'families': registry['families'],
            'distinct_physical_fits': 15, 'complete_family_records': 9,
            'actual_scientific_optimizer_updates': 40500, 'complete_member_training_trajectory_updates': 64800,
            'control_predictive_outcomes_computed': False})


def all_fits(a, output):
    import supervise
    import study
    registry = study.registered_cohort(a)
    c.require([q['row'] for q in registry['physical_fits']] == c.schedule() and
              len(a['fit_releases']) == 15, 'Exact fifteen-fit launch coverage required')
    receipts = []
    for row, release_record in zip(registry['physical_fits'], a['fit_releases']):
        release_path = c.verify(release_record)
        release = c.read(release_path)
        c.require(release_record in a['custody_inputs'] and release['kind'] == 'fit' and
                  release['fit_id'] == row['id'] and release['self_path'] == row['release_path'] and
                  release['output'] == row['output'] and release['registry'] == a['registry'] and
                  release['qualification_freeze'] == a['qualification_freeze'] and
                  release['resource_admission'] == a['resource_admission'] and
                  release['runtime_receipt'] == a['runtime_receipt'] and
                  release['consumer_release'] == a['consumer_release'], 'Launch admission coverage differs')
        receipt = supervise.run_one('fit', release_path, c.confined(row['output']))
        c.require(receipt['status'] == 'success', 'Physical fit failure stops full cohort; no automatic retry')
        receipts.append(receipt)
        c.write(output / ('FIT_EXIT_%02d.json' % len(receipts)), receipt)
    close_path = c.verify(a['closure_release'])
    close_release = c.read(close_path)
    c.require(a['closure_release'] in a['custody_inputs'] and close_release['registry'] == a['registry'] and
              close_release['runtime_receipt'] == a['runtime_receipt'] and
              close_release['consumer_release'] == a['consumer_release'], 'Explicit cohort closure release differs')
    closure = supervise.run_one('close', close_path, c.confined(close_release['output']))
    c.require(closure['status'] == 'success', 'Full cohort closure failed')
    c.write(output / 'RESULT.json', {'schema': 'amazon_polynormer_full_schedule_exit_v2', 'status': 'complete',
            'source': a['source'], 'registry': a['registry'], 'physical_fit_exits': receipts,
            'closure_freeze': closure['freeze'], 'control_predictive_scoring_performed': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True, choices=('register', 'runtime_capture', 'consumer_prepare', 'qualify', 'fit', 'close', 'all', 'evaluate'))
    parser.add_argument('--release', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = c.confined(args.output)
    a, admission_record = c.gate(args.release, args.kind, output)
    c.require(Path(sys.executable).resolve() == Path(a['interpreter']['path']).resolve(), 'Worker interpreter differs')
    numeric = args.kind in ('runtime_capture', 'consumer_prepare', 'qualify', 'fit', 'evaluate')
    if args.kind == 'fit':
        import study
        study.science_gate(a)  # Exact qualification/resource gate before any numerical import.
    if args.kind == 'evaluate':
        import evaluate
        evaluate.cohort(a)  # Complete physical closure before any numerical import.
    before = c.runtime(a['device'], a, capture=args.kind == 'runtime_capture') if numeric else None
    if args.kind == 'runtime_capture':
        c.write(output / 'RUNTIME.json', {'schema': 'amazon_polynormer_runtime_v2', 'source': a['source'], 'runtime': before})
        c.write(output / 'RESULT.json', {'schema': 'amazon_polynormer_runtime_capture_exit_v2', 'status': 'complete',
                'source': a['source'], 'runtime_receipt': c.record(output / 'RUNTIME.json')})
    elif args.kind == 'register':
        register(a, output)
    elif args.kind == 'consumer_prepare':
        prepare_consumer(a, output)
    elif args.kind == 'close':
        close(a, output)
    elif args.kind == 'all':
        all_fits(a, output)
    else:
        module = __import__({'qualify': 'qualify', 'fit': 'study', 'evaluate': 'evaluate'}[args.kind])
        module.run(a, admission_record, output)
    if numeric:
        after = c.runtime(a['device'], a, capture=args.kind == 'runtime_capture', configure=False)
        c.require(after == before, 'Runtime binary/source/settings/device custody changed during body')
    c.preserve(a, admission_record)
    c.write(output / 'READY.json', {'schema': 'amazon_polynormer_body_ready_v2', 'kind': args.kind,
            'source': a['source'], 'admission': admission_record, 'result': c.record(output / 'RESULT.json'),
            'body_ready_is_not_physical_success': True})


if __name__ == '__main__':
    main()
