"""Materialize explicit root JSON specs; never authorize or execute numerical work."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import common as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    spec = c.read(args.spec)
    c.require(spec['schema'] == 'amazon_polynormer_root_release_v2' and
              type(spec['execution_authorized']) is bool and spec['automatic_retry_authorized'] is False and
              spec['test_labels_authorized'] is False, 'Explicit root release spec required')
    result = dict(spec)
    result['source'] = c.verify_sources()
    result['self_path'] = str(c.confined(args.output).relative_to(c.PHASE))
    for key in tuple(result):
        if key.endswith('_descriptor_path'):
            name = key[:-len('_descriptor_path')]
            p = c.confined(result.pop(key))
            c.require(p.suffix == '.json', 'Release helper hashes metadata JSON only')
            result[name] = c.record(p)
    profile = result.pop('existing_runtime_profile')
    profiles = c.read(c.PACKET / 'DEPENDENCIES.json')['admissible_existing_profiles']
    c.require(profile in profiles, 'Declared existing runtime profile required')
    result['expected_versions'] = profiles[profile]['versions']
    path = result.pop('interpreter_path')
    result['interpreter'] = {k: v for k, v in c.runtime_file(path, 'binary').items() if k != 'kind'}
    launch = Path(path)
    c.require(launch.is_absolute() and '..' not in launch.parts, 'Exact absolute interpreter launch path required')
    # Hash resolved executable bytes, but launch the virtual-environment entry
    # point so Python retains the admitted package search prefix.
    result['interpreter']['path'] = str(launch)
    rows = list(result.get('custody_inputs', []))
    for key in ('runtime_receipt', 'consumer_release', 'registry', 'qualification_freeze',
                'resource_admission', 'closure_freeze', 'attempt_registry', 'closure_release'):
        if key in result:
            rows.append(result[key])
    if 'consumer_release' in result:
        consumer = c.read(c.verify(result['consumer_release']))
        manifest_row = consumer['existing_data_manifest']
        d = c.read(c.verify(manifest_row))
        rows += [manifest_row, d['producer_release'], d['packet_manifest'], d['public_graph'],
                 *d['train_labels'].values(), *d['validation_labels'].values(), *consumer['prior_role_receipts'].values()]
    if result['kind'] == 'consumer_prepare':
        d = c.read(c.verify(result['existing_data_manifest']))
        rows += [result['existing_data_manifest'], d['producer_release'], d['packet_manifest'], d['public_graph'],
                 *d['train_labels'].values(), *d['validation_labels'].values()]
    if 'attempt_registry' in result:
        attempts = c.read(c.verify(result['attempt_registry']))
        rows += attempts['prior_attempt_artifacts']
    if 'fit_releases' in result:
        rows += result['fit_releases']
    result['custody_inputs'] = list({c.object_sha(row): row for row in rows}.values())
    # Metadata promotion is conditional on the explicit supplied authorization.
    # The helper does not promote review/role/resource/qualification status.
    if result['execution_authorized']:
        review = c.read(c.verify(result['source_review']))
        c.require(result['independent_source_review_passed'] is True and review['status'] == 'passed' and
                  review['source'] == result['source'], 'Exact independently passed source review required')
    row = c.write(args.output, result)
    print(json.dumps({'root_release': row, 'execution_authorized': result['execution_authorized'],
                      'numerical_work_executed': False}, sort_keys=True))


if __name__ == '__main__':
    main()
