"""Disabled complete36 server-only raw assembly over the unchanged Q/K readout."""
import argparse
import importlib.util
from pathlib import Path
import resource
import socket
import sys
import time

HERE = Path(__file__).resolve().parent


def run(release_path, release_sha256):
    spec = importlib.util.spec_from_file_location('_qk36_original_assembly_gate', HERE / 'gate_route.py')
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    gate.require(gate.sha(release_path) == release_sha256, 'Exact separately authorized server-only analysis release')
    cfg, pins = gate.read(release_path), gate.read(HERE / 'SOURCE_BINDINGS.json')
    g = gate.helpers(pins)
    gate.require(cfg['schema'] == 'qk36-original-route-assembly-release-v1'
        and all(cfg[key] is True for key in ('enabled', 'root_execution_authorized', 'source_review_approved',
            'whole36_route_collection_custody_closed', 'all9_baselines_globally_sealed', 'raw_archives_server_only', 'finite_external_owner_confirmed'))
        and all(cfg[key] is False for key in ('training', 'backward', 'reselection', 'calibration', 'TEST_access', 'automatic_retry')),
        'Inactive complete36 raw-only readout; no model calls or scientific gate changes')
    gate.require(cfg['readout_manifest_sha256'] == gate.sha(HERE / 'MANIFEST.json'), 'Reviewed exact original-route source')
    route = pins['routes'][cfg['analysis_route_id']]
    gate.require(socket.gethostname() == route['hostname'] and gate.PHASE == Path(route['phase']).resolve()
        and Path(sys.executable).resolve() == Path(route['python']).resolve(), 'Explicit admitted server-only analysis runtime; no Mac raw access')
    supervision = gate.read(g.bound(cfg['external_supervision']))
    gate.require(supervision['enabled'] is True and supervision['finite_owned_bound_confirmed'] is True
        and supervision['entry_program'] == g.binding(HERE / 'assemble_readout.py')
        and supervision['custody_program'] == pins['reuse']['custody_program']
        and supervision['release_argument_path'] == str(Path(release_path).resolve())
        and 0 < supervision['active_seconds'] <= 7200 and 0 < supervision['cleanup_seconds'] <= 15
        and supervision['hard_seconds'] == supervision['active_seconds'] + supervision['cleanup_seconds']
        and 0 <= supervision['owned_GPU_bytes'] <= pins['plan']['root_resource_limits'][cfg['analysis_route_id']]['max_owned_GPU_bytes']
        and 0 < supervision['owned_RSS_bytes'] <= pins['plan']['root_resource_limits'][cfg['analysis_route_id']]['max_owned_RSS_bytes']
        and 0 < supervision['maximum_output_bytes'] <= 4 * 1024**3, 'Existing finite owner/cost boundary for raw-only analysis')
    g.verify(dict(path=str((HERE / 'MANIFEST.json').relative_to(gate.PHASE)), sha256=cfg['readout_manifest_sha256']))
    for row in pins['source_files']: g.bound(row)
    for row in pins['source_manifests']: g.verify(row)
    freeze = gate.read(g.bound(cfg['global_native_baseline_freeze']))
    gate.require(freeze['complete'] is True and freeze['root_frozen_before_any_candidate_calls'] is True
        and freeze['plan'] == pins['global_plan'] and freeze['readout_manifest_sha256'] == cfg['readout_manifest_sha256'], 'Same all9 pre-candidate global cohort seal')
    staging = gate.read(g.bound(cfg['server_to_server_archive_custody_and_cost']))
    gate.require(staging['complete'] is True and staging['exact_archive_bytes_preserved'] is True
        and staging['raw_archives_server_only'] is True and staging['Mac_raw_copies'] is False
        and staging['plan'] == pins['global_plan'] and set(cfg['phase2_routes']) == set(pins['routes']), 'Actual raw-only staging/cost custody, all original routes')
    collection = dict(schema='qk36-selected-state-predictions-v1', cells=[],
        cohorts={kind: {} for kind in gate.KINDS}, all9_native_tied_baselines_frozen_before_candidates=True,
        status='complete', global_native_baseline_freeze=cfg['global_native_baseline_freeze'],
        original_training_routes_preserved=True, raw_arrays_server_only=True)
    collection['actual_phase2_terminal_custody'] = {}
    for route_id, seed in zip(pins['plan']['route_ids'], gate.SEEDS):
        evidence = cfg['phase2_routes'][route_id]
        local = gate.read(g.bound(evidence['collection']))
        costs = gate.read(g.bound(evidence['phase1_cost'])), gate.read(g.bound(evidence['phase2_cost']))
        baseline = freeze['routes'][route_id]
        baseline_collection = gate.read(g.bound(baseline['phase1_collection']))
        gate.require(evidence['phase1_cost'] == baseline['phase1_cost'], 'Same actual globally frozen baseline cost')
        gate.phase_terminal(g, pins, baseline['actual_phase1_owner_terminal_custody'], route_id, 'native_baselines',
            baseline['phase1_collection'], baseline['phase1_cost'], baseline_collection, costs[0], cfg['readout_manifest_sha256'])
        terminal = gate.phase_terminal(g, pins, evidence['actual_phase2_owner_terminal_custody'], route_id, 'candidates',
            evidence['collection'], evidence['phase2_cost'], local, costs[1], cfg['readout_manifest_sha256'])
        gate.require(terminal['release']['global_native_baseline_freeze'] == cfg['global_native_baseline_freeze'],
            'Actual phase2 release used the same successful global9 baseline barrier')
        collection['actual_phase2_terminal_custody'][route_id] = terminal
        if not terminal['phase_owner_succeeded']:
            collection['status'] = 'complete_with_retained_phase2_owner_failures'
        gate.require(local['readout_route_id'] == route_id and local['collection_phase'] == 'candidates'
            and local['global_native_baseline_freeze'] == cfg['global_native_baseline_freeze']
            and len(local['cells']) == 12 and all(row['seed'] == seed and row['original_training_route'] == route_id for row in local['cells'])
            and costs[0]['attempted_member_forwards'] == costs[0]['completed_member_forwards'] == 9
            and costs[1]['attempted_member_forwards'] <= 27
            and all(cost['TRAIN_updates'] == cost['backward_calls'] == cost['Adam_steps'] == 0 for cost in costs), 'All twelve original bank slots and actual phase costs, including failures')
        for row in local['cells']:
            record = dict(row)
            if record['collection_status'] == 'complete':
                staged = staging['prediction_archives'][record['cell']]
                gate.require(staged['original_route_id'] == route_id and staged['original_archive'] == record['raw_prediction_archive']
                    and all(staged['staged_archive'][key] == record['raw_prediction_archive'][key] for key in ('sha256', 'bytes')), 'Exact source prediction archive, original provenance retained')
                g.bound(staged['staged_archive'])
                record.update(original_archive=record['raw_prediction_archive'], staged_archive=staged['staged_archive'], raw_archive=staged['staged_archive'])
            collection['cells'].append(record)
        for kind in gate.KINDS:
            original = local['cohorts'][kind][str(seed)]
            staged = staging['cohort_archives'][route_id][kind]
            sealed = gate.read(g.bound(freeze['routes'][route_id]['baseline_receipt']))['cohorts'][kind][str(seed)]
            gate.require(original == sealed and original['available'] is True
                and staged['original_archive'] == original['archive']
                and all(staged['staged_archive'][key] == original['archive'][key] for key in ('sha256', 'bytes')), 'All9 exact original baseline/cohort archive seals')
            g.bound(staged['staged_archive'])
            collection['cohorts'][kind][str(seed)] = dict(original, original_archive=original['archive'],
                staged_archive=staged['staged_archive'], archive=staged['staged_archive'])
    gate.require(len(collection['cells']) == len({row['cell'] for row in collection['cells']}) == 36
        and sum(row['members'] for row in collection['cells']) == 108, 'Complete36 custody, no survivor subset')
    collection['phase2_owners_all_succeeded'] = all(row['phase_owner_succeeded'] for row in collection['actual_phase2_terminal_custody'].values())
    collection['original_per_bank_availability_unchanged'] = True
    output = g.inside(cfg['output_directory'])
    gate.require(not output.exists() and output.parent.is_dir(), 'Fresh server-only analysis directory')
    output.mkdir(mode=0o700); (output / 'compact').mkdir(mode=0o700)
    started, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    cost = dict(schema='qk36-original-route-assembly-cost-v1', raw_archive_load_attempts=0,
        raw_archive_bytes_submitted_to_decoder=0, member_forwards=0, TRAIN_updates=0, backward_calls=0,
        root_staging_cost=cfg['server_to_server_archive_custody_and_cost'], original_route_costs=cfg['phase2_routes'],
        actual_analysis_server=route)
    writer = g.module(g.bound(pins['reuse']['variable_member_collect']), '_qk36_assembly_original_writer')
    try:
        import numpy as np
        cost['actual_analysis_numpy_version'] = str(np.__version__)
        gate.require(cost['actual_analysis_numpy_version'] == cfg['actual_analysis_numpy_version'], 'Bound actual raw-analysis NumPy provider')
        readout = g.module(HERE / 'readout36.py', '_qk36_original_unchanged_readout')
        analysis, contract, arrays_for, analyse, adapted = readout.configure(g, pins)
        gate.require(all(adapted[key] == pins['adapted_AST_sha256'][key] for key in adapted), 'Unchanged frozen numerical readout/co-primary/sharing/full-QK rules')
        original_load = contract.load
        def charged_load(np_arg, path, expected):
            cost['raw_archive_load_attempts'] += 1; cost['raw_archive_bytes_submitted_to_decoder'] += expected['bytes']
            return original_load(np_arg, path, expected)
        contract.load = charged_load
        decision = analyse(np, g, pins, output, collection, analysis, contract)
        writer.write(output / 'compact/COLLECTION.json', collection)
        gate.require(sum(path.stat().st_size for path in output.rglob('*') if path.is_file()) <= supervision['maximum_output_bytes'], 'Owned analysis output cap')
        return dict(output=str(output / 'compact'), pilot_decision=decision)
    except Exception as error:
        collection.update(status='retained_assembly_failure', failure=dict(type=type(error).__name__, message=str(error), automatic_retry=False))
        writer.write(output / 'compact/COLLECTION.json', collection)
        raise
    finally:
        end = resource.getrusage(resource.RUSAGE_SELF)
        cost.update(inclusive_wall_seconds=time.monotonic() - started, CPU_user_seconds=end.ru_utime - usage.ru_utime,
            CPU_system_seconds=end.ru_stime - usage.ru_stime, peak_RSS_bytes=int(end.ru_maxrss * (1024 if sys.platform.startswith('linux') else 1)),
            actual_outer_owner_terminal_and_compact_mirroring_cost=None)
        writer.write(output / 'compact/COST.json', cost)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); print(run(args.release, args.release_sha256)['output'])


if __name__ == '__main__': main()
