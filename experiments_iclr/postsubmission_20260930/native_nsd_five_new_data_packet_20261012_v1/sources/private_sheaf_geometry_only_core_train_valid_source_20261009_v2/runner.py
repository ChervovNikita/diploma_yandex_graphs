"""Inactive regularizer-off shared geometry and complete independent NSD controls."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import time
import independent
import shared_fit
import native_fit
from support import HERE, PHASE, borrowed, admission, read, require, sha, source_checks

VERSIONS = ('torch', 'numpy', 'scipy', 'torch-geometric', 'torch-sparse', 'torch-scatter', 'torch-householder', 'scikit-learn')


def key(record): return (record['kind'], record['base_seed'], record.get('member', -1))


def run(args):
    pins, protocol = source_checks(), read(HERE / 'PROTOCOL.json')
    release, receipt, config, output = admission(args, pins, protocol)
    require(release['action'] == 'post_screen_geometry_core', 'Scientific core release action required')
    require(set(release['expected_runtime_versions']) == set(VERSIONS), 'Complete recorded existing runtime versions')
    started, usage0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    old, common, helpers, placement = borrowed(pins)
    output.mkdir(parents=True, exist_ok=False)
    records, failure = [], None
    schedule = [dict(kind='shared_fit', base_seed=base) for base in protocol['seeds']]
    schedule += [dict(kind='independent_member', base_seed=base, member=member) for base in protocol['seeds'] for member in range(4)]
    schedule += [dict(kind='independent_pool', base_seed=base) for base in protocol['seeds']]
    def retain(record):
        require(key(record) not in {key(value) for value in records}, 'No duplicate/replacement attempts')
        records.append(record); common.append_jsonl(output / 'ALL_RECORDS.jsonl', record)
    try:
        common.write_json(output / 'SCHEDULE.json', dict(protocol=protocol, root_release=release, admission_receipt=receipt, components=schedule, automatic_retry=False))
        import numpy as np
        import torch
        from sklearn.metrics import roc_auc_score
        versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        require(versions == release['expected_runtime_versions'] and str(torch.__version__) == '2.1.2+cu118'
                and np.__version__ == '1.26.4', 'Exact qualified existing runtime')
        device = torch.device(release['device'])
        require(device == torch.device('cuda:0') and torch.cuda.device_count() == 1, 'One root-owned visible cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(release['deterministic_algorithms'])
        arrays, role_meta, role_sha = common.read_roles(np, args.roles)
        require(role_meta.get('exposure_classification') == 'original_paper_benchmark_exploratory', 'Corrected original benchmark exposure classification')
        common.write_json(output / 'ROLE_METADATA.json', role_meta)
        data = {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}
        data['cpu_edge_index'] = torch.from_numpy(arrays['edge_index'])
        native_fit.bind(helpers)
        adapter = common.load_adapter()
        identity = dict(source_seal_sha256=sha(HERE / 'SEAL.json'), manifest_sha256=read(HERE / 'SEAL.json')['manifest_sha256'],
            execution_source_commit=release['execution_source_commit'], admission_receipt_sha256=sha(args.admission),
            role_archive_sha256=release['roles_sha256'], role_metadata_sha256=role_sha,
            native_commit='11e21b561d884713ab1a18a521a7dc2fb26b9361', runtime_versions=versions,
            configuration=config, optimizer=receipt['optimizer'], root_release_sha256=sha(args.release),
            root_decision_sha256=receipt['root_decision_sha256'], experiment_admission=receipt['admission_mode'],
            original_native15_status='incomplete_no_freeze', original_native15_protocol_pass_claimed=False,
            new_base_seeds=protocol['seeds'], reuse_native_member0=False, new_seeds_are_independent_confirmation=False,
            replay_policy=protocol['replay_policy'], device=str(device), deterministic_algorithms=release['deterministic_algorithms'], allow_tf32=False,
            context_regularizer=0.0, TEST_truth_present=False, fresh_dataset_or_unused_split_claim=False)
        common.write_json(output / 'RUN_IDENTITY.json', identity)
        for base in protocol['seeds']:
            retain(shared_fit.fit(np, torch, roc_auc_score, old, common, helpers, placement, adapter, receipt['optimizer'], config,
                                  data, role_meta, identity, output, base))
            independent.run(np, torch, roc_auc_score, old, common, helpers, placement, adapter, receipt['optimizer'], config,
                            data, identity, receipt, release, output, base, retain)
    except BaseException as error:
        failure = common.failure_record(error, 'post_screen_exploratory_core_panel')
        raise
    finally:
        for item in schedule:
            if key(item) in {key(value) for value in records}: continue
            base = item['base_seed']
            if item['kind'] == 'shared_fit': path = output / 'shared' / ('seed'+str(base)) / 'RESULT.json'
            elif item['kind'] == 'independent_pool': path = output / 'independent' / ('base'+str(base)) / 'POOL_RESULT.json'
            else: path = output / 'independent' / ('base'+str(base)) / ('MEMBER'+str(item['member'])+'.json')
            if path.exists():
                record = read(path)
                if record['status'] == 'started': record.update(status='interrupted', failure=failure)
            else:
                record = dict(item, status='failed_before_component', failure=failure, actual_fit_started=False, automatic_retry=False)
                if item['kind'] == 'independent_member':
                    seed = base+1000003*item['member']
                    native_path = output / 'independent' / ('base'+str(base)) / ('native_single__'+config['id']+'__seed'+str(seed)) / 'RESULT.json'
                    if native_path.exists(): record.update(status='interrupted', native_partial_record=read(native_path), actual_fit_started=True)
            records.append(record)
        complete = failure is None and len(records) == 18 and {key(value) for value in records} == {key(value) for value in schedule}
        complete = complete and all(value['status'] == 'complete' for value in records)
        common.write_json(output / 'ALL_RECORDS.json', records)
        if complete:
            common.write_json(output / 'PANEL_SUMMARY.json', dict(complete=True, configuration=config['id'],
                shared_scores=[value['scores'] for value in records if value['kind'] == 'shared_fit'],
                shared_member_scores=[value['member_scores'] for value in records if value['kind'] == 'shared_fit'],
                independent_pooled_scores=[value['scores'] for value in records if value['kind'] == 'independent_pool'],
                independent_member_scores=[value['member_scores'] for value in records if value['kind'] == 'independent_pool'],
                competence='requires_root_member_learning_curve_and_reference_review',
                context_regularizer=0.0, comparative_opening_authorized=False, TEST_truth_present=False))
        usage = resource.getrusage(resource.RUSAGE_SELF)
        common.write_json(output / 'COMPLETE.json', dict(complete=complete, required_components=schedule, records=records, failure=failure,
            complete_new_execution_seconds=time.perf_counter()-started, CPU_user_seconds=usage.ru_utime-usage0.ru_utime,
            CPU_system_seconds=usage.ru_stime-usage0.ru_stime, process_RSS_high_water_cumulative_bytes=common.process_peak_rss_bytes(),
            all_independent_bodies_fresh=True, historical_member0_reuse=False, all_components_required_before_comparison=True,
            automatic_retry=False, configuration_search=False, comparative_opening_authorized=False, TEST_truth_present=False))
    require(complete, 'All paired shared/individual/reference attempts required; no survivor report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--roles', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(inactive=True, architecture='post-screen exploratory fixed d4_f16_L4; source/admission required', members=4,
                              protocol=read(HERE / 'PROTOCOL.json'), numeric_model_or_role_import=False)))
        return
    require(args.release and args.admission and args.roles and args.output, 'Explicit root release/admission/roles/fresh output required')
    run(args)


if __name__ == '__main__': main()
