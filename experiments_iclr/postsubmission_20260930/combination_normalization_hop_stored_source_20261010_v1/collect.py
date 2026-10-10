"""Disabled VALID stored-output diagnostic; reuse existing closed-family readers."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

ENABLED = False
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
FULL = PHASE/'private_hop_credit_pubmed_fullfit_source_20261010_v1'
NORM = PHASE/'pubmed_own4_M_normalization_source_20261010_v1'
OLD = PHASE/'combination_three_bank_stored_error_source_20261010_v1'
I4 = PHASE/'pubmed_factorized_I4_reference_source_20261010_v1'
RID = 'fixed_normalization_private_hop_VALID_diagnostic'
SEEDS = (9101, 9203, 9307)
LIMITS = dict(external_active_seconds=600, external_cleanup_seconds=10,
              GPU_bytes=0, RSS_bytes=1073741824, output_bytes=8388608,
              log_bytes=2097152)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def admit(args):
    entry = load('_normhop_full_entry', FULL/'run.py')
    entry.qualification_surface().route(owner=True)
    base = entry.load('_normhop_reference', PHASE/'pubmed_factor1_controls_source_20261010_v1/run.py')
    ic = entry.load('_normhop_I4_checks', I4/'common.py')
    path = args.release.resolve(strict=True)
    if ENABLED is not False or not path.is_relative_to(HERE) or entry.sha(path) != args.release_sha256:
        raise ValueError('Disabled disk source and exact new root release required')
    spec = json.loads(path.read_text())
    if spec.get('schema') != 'normalization-hop-stored-release-v1' or spec.get('record_id') != RID:
        raise ValueError('One fixed diagnostic record required')
    for key in ('enabled', 'root_authorized', 'source_review_approved', 'norm3_closed',
                'full18_closed', 'I4_closed', 'external_finite_bound_confirmed', 'VALID_access'):
        if spec.get(key) is not True:
            raise ValueError('Disabled diagnostic: '+key)
    for key in ('TEST_access', 'TRAIN_access', 'automatic_retry', 'HPO', 'arm_selection',
                'coefficient_selection', 'oracle_training_target', 'new_fits'):
        if spec.get(key) is not False:
            raise ValueError('Closed stored-output scope: '+key)
    if any(key in spec for key in ('test_bundle', 'test_ids', 'test_y', 'train_bundle',
                                  'train_ids', 'train_y', 'full_y', 'features', 'checkpoint')):
        raise ValueError('Bound VALID stored-output scope only')
    ic.verify(HERE, spec['source_manifest_sha256'])
    bindings = json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for row in bindings['files']:
        file = (PHASE/row['path']).resolve(strict=True)
        if not file.is_relative_to(PHASE) or entry.sha(file) != row['sha256'] or file.stat().st_size != row['bytes']:
            raise ValueError('Existing source/interface changed')
    for folder, key in ((NORM, 'normalization_manifest_sha256'), (FULL, 'full18_manifest_sha256'),
                        (OLD, 'original_diagnostic_manifest_sha256')):
        ic.verify(folder, bindings[key])
    fixed = json.loads((HERE/'FIXED_SPEC.json').read_text())
    if entry.sha(HERE/'FIXED_SPEC.json') != bindings['fixed_spec_sha256'] or spec['fixed_spec_sha256'] != bindings['fixed_spec_sha256']:
        raise ValueError('Exact prospective pair and eligibility rule required')
    review = json.loads(base.bind(spec['root_review']).read_text())
    if review.get('approved') is not True or review.get('source_manifest_sha256') != spec['source_manifest_sha256'] or review.get('fixed_spec_sha256') != spec['fixed_spec_sha256']:
        raise ValueError('Exact root source/spec review required')
    data = json.loads((NORM/'DATA_AND_RUNTIME.json').read_text())
    full_data = json.loads((FULL/'DATA_AND_RUNTIME.json').read_text())
    if spec['limits'] != LIMITS or spec['runtime'] != data['runtime']:
        raise ValueError('Unchanged existing finite CPU bound and native runtime required')
    if entry.sha(data['runtime']['python']['path']) != data['runtime']['python']['sha256'] or Path(sys.executable).resolve() != Path(data['runtime']['python']['path']).resolve() or os.environ.get('PYTHONPATH', '') != data['runtime']['PYTHONPATH']:
        raise ValueError('Exact admitted native Python/PYTHONPATH required')
    normread = entry.load('_normhop_normalization_reader', NORM/'run.py')
    diag = entry.load('_normhop_original_reader', OLD/'collect.py')
    # Existing loaders own all normalization/full18 owner and output custody.
    norms = normread.closed_science(entry, base, dict(
        science_owner_plan=spec['normalization_science_owner_plan'],
        source_manifest_sha256=bindings['normalization_manifest_sha256'],
        selector=entry.M4_SELECTOR, frozen_providers=data['frozen_providers']))
    full = diag.closed18(entry, base, spec['full18_owner_plan'], bindings['full18_manifest_sha256'], full_data)
    _bound, idata, roster, _reuse, im = ic.frozen()
    ib = json.loads((OLD/'I4_BINDING.json').read_text())
    if im != ib['source_manifest_sha256']:
        raise ValueError('Exact admitted I4 source required')
    for stage in ('admission', 'qualification', 'science', 'assembly', 'comparison'):
        ic.closed_stage(stage, roster, im)
    receipt = json.loads(ic.stage_result('comparison', 'factorized_I4_complete_comparison').read_text())
    comparison = json.loads(base.bind(ib['comparison']).read_text())
    if receipt.get('complete') is not True or receipt.get('comparison') != ib['comparison'] or receipt.get('source_manifest_sha256') != im or comparison.get('complete_family') is not True or comparison.get('three_complete_banks') is not True or comparison.get('TEST_access') is not False:
        raise ValueError('Exact complete I4 comparison closure required')
    for key in ('valid_bundle', 'validation_custody', 'runtime', 'frozen_providers'):
        if any(other[key] != data[key] for other in (full_data, idata)):
            raise ValueError('Same exact task/role/providers required')
    custody = json.loads(base.bind(data['validation_custody']).read_text())
    output = Path(spec['output']).resolve()
    if not output.is_relative_to(HERE) or output.exists():
        raise ValueError('Fresh confined diagnostic output required')
    return entry, base, ic, diag, data, custody, fixed, spec, norms, full, output


def persist(torch, ids, seed_masks):
    out = {}
    for name in seed_masks[0]:
        support = torch.stack([row[name] for row in seed_masks]).sum(0)
        out[name] = dict(seed_instance_count=int(support.sum().item()),
                         unique_node_count=int(support.gt(0).sum().item()),
                         support_counts={str(k): int(support.eq(k).sum().item()) for k in (1, 2, 3)},
                         node_ids_by_support={str(k): ids[support.eq(k)].tolist() for k in (1, 2, 3)})
    return out


def eligibility(rows, persistence, fixed):
    gate = fixed['eligibility']
    counts = gate['minimum_nodes_each_seed']
    each = lambda name: all(row['counts'][name] >= counts for row in rows)
    stable = lambda name: sum(persistence[name]['support_counts'][str(k)] for k in range(gate['minimum_persistent_support'], 4)) >= gate['minimum_persistent_nodes']
    changes = [row['deltas']['A'] for row in rows]
    A = (all(v['mean_member_accuracy_pp'] >= gate['A_mean_member_floor_pp'] and v['worst_member_accuracy_pp'] >= gate['A_worst_member_floor_pp'] for v in changes)
         and sum(v['mean_member_accuracy_pp'] for v in changes)/3 > 0
         and sum(v['pooled_NLL'] for v in changes)/3 <= gate['A_mean_pooled_NLL_delta_max'])
    changes = [row['deltas']['B'] for row in rows]
    B_competence = (all(v['mean_member_accuracy_pp'] >= gate['B_mean_member_floor_pp'] and v['worst_member_accuracy_pp'] >= gate['B_worst_member_floor_pp'] for v in changes)
                    and all(v >= gate['B_seed_class_floor_pp'] for row in changes for v in row['class_accuracy_pp'])
                    and all(sum(row['class_accuracy_pp'][c] for row in changes)/3 >= gate['B_mean_class_floor_pp'] for c in range(3)))
    B_graph = all(each(name) and stable(name) for name in ('B_acquired_correct_member', 'B_coverage_absent_both_controls'))
    exclusive = all(each(name) and stable(name) for name in ('A_only', 'B_only'))
    decisions = dict(A_competence_and_NLL=A, B_competence=B_competence,
                     B_controlled_acquisition=B_graph, complementary_served_repairs=exclusive)
    return dict(passed=all(decisions.values()), requirements=decisions,
                thresholds=gate, eligibility_only=True, does_not_release_fits=True,
                not_statistical_significance=True)


def execute(context, release_sha256, started):
    entry, base, ic, diag, data, custody, fixed, spec, norms, full, output = context
    # Every family/owner/source/role closure above completes before numerical imports.
    import numpy as np
    import torch
    providers = dict(numpy=str(np.__version__), torch=str(torch.__version__))
    for name in ('scipy', 'torch-geometric', 'torch-scatter', 'torch-sparse'):
        providers[name] = importlib.metadata.version(name)
    if providers != data['frozen_providers']:
        raise ValueError('Exact admitted providers required')
    torch.set_num_threads(2)
    count = lambda mask: int(mask.sum().item())
    with base.reference_surface() as (_unused_engine, metrics, _unused_factory):
        from source import fingerprint
        with np.load(base.bind(data['valid_bundle']), allow_pickle=False) as archive:
            if set(archive.files) != {'valid_ids', 'valid_y'}:
                raise ValueError('Projected VALID only')
            arrays = {name: archive[name].copy() for name in archive.files}
        if {k: fingerprint(v) for k, v in arrays.items()} != custody['array_fingerprints']:
            raise ValueError('Exact VALID identity required')
        ids, truth = torch.from_numpy(arrays['valid_ids']), torch.from_numpy(arrays['valid_y'])
        if ids.dtype != torch.int64 or truth.dtype != torch.int64 or ids.shape != (3942,) or truth.shape != ids.shape or len(np.unique(arrays['valid_ids'])) != 3942 or np.bincount(arrays['valid_y'], minlength=3).tolist() != [820, 1547, 1575]:
            raise ValueError('Complete exact VALID population required')
        old_common, old_path = sys.modules.get('common'), list(sys.path)
        sys.modules['common'] = ic
        try:
            reader = entry.load('_normhop_existing_masks', I4/'run.py')
        finally:
            sys.path[:] = old_path
            if old_common is None:
                sys.modules.pop('common', None)
            else:
                sys.modules['common'] = old_common
        paired = entry.load('_normhop_existing_errors', FULL/'error_changes.py')
        rows, seed_masks = [], []
        for seed in SEEDS:
            banks, expected, inputs = {}, {}, {}
            sources = {'A': norms[seed]}
            for name, condition in (('baseline', 'shared4_own'), ('B', 'private_missinghop'),
                                    ('full_aux', 'full_aux'), ('common_nonfull', 'common_nonfull')):
                sources[name] = full[f'seed{seed}__{condition}']
            for name, (value, folder) in sources.items():
                payload = value['complete_saved_member_logits']
                path = (folder/payload['path']).resolve(strict=True)
                if not path.is_relative_to(folder) or payload['factual_shape'] != [4, 19717, 3]:
                    raise ValueError('Exact selected factual bank required')
                inputs[name] = dict(path=str(path), sha256=payload['sha256'], bytes=payload['bytes'])
                expected[name] = value['selected_readouts']['VALID']
                banks[name] = diag.selected(np, torch, metrics, base, inputs[name], expected[name], ids, truth)
            masks = {name: reader.masks(torch, metrics, bank, {'VALID': (ids, truth)})['VALID'] for name, bank in banks.items()}
            g, a, b = (masks[name]['pool_correct'] for name in ('baseline', 'A', 'B'))
            ug, ua, ub = (masks[name]['coverage'] for name in ('baseline', 'A', 'B'))
            ra, rb = ~g & a, ~g & b
            flows = dict(R_A=ra, R_B=rb, A_only=ra & ~rb, B_only=rb & ~ra,
                         shared_repairs=ra & rb, harms_A=g & ~a, harms_B=g & ~b,
                         shared_harms=g & ~a & ~b,
                         acquisition_complement_A=ra & ~rb & ua & ~ub,
                         acquisition_complement_B=rb & ~ra & ub & ~ua,
                         serving_complement_A=ra & ~rb & ub & ~b,
                         serving_complement_B=rb & ~ra & ua & ~a,
                         B_coverage_absent_both_controls=ub & ~masks['full_aux']['coverage'] & ~masks['common_nonfull']['coverage'])
            for name, other in (('A', 'B'), ('B', 'A')):
                flows[name+'_only_pooled_only_rescue'] = flows[name+'_only'] & ~masks[name]['coverage'] & ~masks[other]['coverage']
                if count(flows[name+'_only']) != sum(count(flows[key+name]) for key in ('acquisition_complement_', 'serving_complement_')) + count(flows[name+'_only_pooled_only_rescue']):
                    raise ValueError('Exclusive repair partition must reconcile')
            for name in banks:
                u, p = masks[name]['coverage'], masks[name]['pool_correct']
                flows[name+'_pool_lost_correct'] = u & ~p
                flows[name+'_all_wrong_pool_rescue'] = ~u & p
                if name == 'baseline':
                    continue
                harm = g & ~p
                flows[name+'_acquired_correct_member'] = ~ug & u
                flows[name+'_acquired_served'] = ~ug & u & p
                flows[name+'_acquired_lost'] = ~ug & u & ~p
                flows[name+'_introduced_all_wrong'] = ug & ~u
                flows[name+'_introduced_pool_lost'] = u & ~p & ~(ug & ~g)
                flows[name+'_harm_no_correct_member'] = harm & ~u
                flows[name+'_harm_correct_alternative_lost'] = harm & u
                for before in (False, True):
                    for after in (False, True):
                        flows[f'{name}_harm_U0_{int(before)}_U1_{int(after)}'] = harm & (ug if before else ~ug) & (u if after else ~u)
                if count(harm) != count(flows[name+'_harm_no_correct_member']) + count(flows[name+'_harm_correct_alternative_lost']):
                    raise ValueError('Harm origins must reconcile')
            route_flows = {}
            for name in ('A', 'B'):
                route_flows[name] = []
                for route in range(4):
                    old, new = masks['baseline']['member_correct'][route], masks[name]['member_correct'][route]
                    route_flows[name].append(dict(route=route, repairs=count(~old & new), harms=count(old & ~new),
                        repair_node_ids=ids[~old & new].tolist(), harm_node_ids=ids[old & ~new].tolist()))
            bits = g.to(torch.int64)*4 + a.to(torch.int64)*2 + b.to(torch.int64)
            patterns = [dict(pattern=f'{k:03b}', count=count(bits.eq(k)), node_ids=ids[bits.eq(k)].tolist(),
                class_counts=[count(bits.eq(k) & truth.eq(c)) for c in range(3)]) for k in range(8)]
            availability = {}
            for name in banks:
                k = masks[name]['member_correct'].sum(0)
                availability[name] = [dict(correct_members=i, pool_correct=bool(j), count=count(k.eq(i) & masks[name]['pool_correct'].eq(bool(j)))) for i in range(5) for j in range(2)]
                if sum(v['count'] for v in availability[name]) != 3942:
                    raise ValueError('Member availability table must cover every VALID node')
            baseline = expected['baseline']['classification']
            deltas = {}
            for name in ('A', 'B', 'full_aux', 'common_nonfull'):
                c = expected[name]['classification']
                # Derive accuracy changes from verified integer counts, including zero boundaries.
                deltas[name] = dict(pooled_accuracy_pp=100*(c['pooled']['correct']-baseline['pooled']['correct'])/3942,
                    pooled_NLL=c['pooled']['NLL']-baseline['pooled']['NLL'],
                    mean_member_accuracy_pp=100*(sum(v['correct'] for v in c['members'])-sum(v['correct'] for v in baseline['members']))/(4*3942),
                    worst_member_accuracy_pp=100*(min(v['correct'] for v in c['members'])-min(v['correct'] for v in baseline['members']))/3942,
                    class_accuracy_pp=[100*(c['classes'][i]['correct']-baseline['classes'][i]['correct'])/c['classes'][i]['count'] for i in range(3)])
            rows.append(dict(seed=seed, counts={name: count(mask) for name, mask in flows.items()},
                node_ids={name: ids[mask].tolist() for name, mask in flows.items()},
                per_class=[dict(class_id=c, counts={name: count(mask & truth.eq(c)) for name, mask in flows.items()}) for c in range(3)],
                joint_pool_correctness=patterns, member_availability=availability,
                fixed_route_changes=route_flows, selected_readouts=expected, deltas=deltas,
                inputs=inputs, existing_paired_flows={name: paired.compare(torch, metrics, banks['baseline'], banks[name], ids, truth) for name in ('A', 'B', 'full_aux', 'common_nonfull')}))
            seed_masks.append(flows)
            if time.monotonic()-started >= LIMITS['external_active_seconds']:
                raise TimeoutError('Existing finite CPU envelope')
        persistence = persist(torch, ids, seed_masks)
        gate = eligibility(rows, persistence, fixed)
        result = dict(schema='normalization-hop-stored-complete-v1', complete=True, record_id=RID,
            source_manifest_sha256=spec['source_manifest_sha256'], fixed_spec_sha256=spec['fixed_spec_sha256'],
            release_sha256=release_sha256, rows=rows, persistence=persistence, gate=gate,
            role='VALID', role_count=3942, seeds=list(SEEDS),
            norm3_full18_I4_closed_before_payloads=True, I4_payloads_loaded=False,
            models=0, checkpoint_loads=0, new_fits=0, model_forwards=0, backwards=0, Adam_steps=0,
            TRAIN_access=False, TEST_access=False, oracle_training_target=False, automatic_retry=False,
            server_only_exact_node_evidence=True, does_not_release_fits=True,
            providers=providers, limits=LIMITS)
        base.write(output/'JOINT_ERROR_DIAGNOSTIC.json', result)
        summary = dict(complete=True, record_id=RID, gate=gate, does_not_release_fits=True, TEST_access=False,
            per_seed=[dict(seed=row['seed'], counts=row['counts'], deltas=row['deltas'],
                classification={name: value['classification'] for name, value in row['selected_readouts'].items()}) for row in rows],
            persistence={name: {key: value[key] for key in ('seed_instance_count', 'unique_node_count', 'support_counts')} for name, value in persistence.items()})
        base.write(output/'SUMMARY.json', summary)
        lines = ['# Normalization and private hop complementary repairs', '',
                 '| Seed | A only | B only | Shared | A harms | B harms | B acquired | B absent both view controls |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            c = row['counts']
            lines.append(f"| {row['seed']} | {c['A_only']} | {c['B_only']} | {c['shared_repairs']} | {c['harms_A']} | {c['harms_B']} | {c['B_acquired_correct_member']} | {c['B_coverage_absent_both_controls']} |")
        lines += ['', f"Prospective eligibility: **{'PASS' if gate['passed'] else 'FAIL'}**.",
                  'This supports only a conditional interaction rationale and releases no fits.',
                  'SUMMARY.json retains NLL, mean/worst/per-member/class competence, harms and persistence.',
                  'Exact server-side node evidence is in JOINT_ERROR_DIAGNOSTIC.json. TEST stayed closed.']
        (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
        output_bytes = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if output_bytes > LIMITS['output_bytes'] or rss > LIMITS['RSS_bytes'] or time.monotonic()-started >= LIMITS['external_active_seconds']:
            raise RuntimeError('Existing finite diagnostic envelope exceeded')
        complete = {key: value for key, value in result.items() if key not in ('rows', 'persistence')}
        complete.update(inclusive_seconds=time.monotonic()-started,
                        CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                        CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                        peak_RSS_bytes=rss, output_bytes_before_complete=output_bytes,
                        completion_cost_includes_diagnostic_summary_writes=True)
        base.write(output/'COMPLETE.json', complete)
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > LIMITS['output_bytes']:
            raise RuntimeError('Completion output cap exceeded')
    print(json.dumps(dict(complete=True, record_id=RID, eligibility_passed=gate['passed'], does_not_release_fits=True)))


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    context = admit(args)
    output = context[-1]
    output.mkdir(parents=True, exist_ok=False)
    try:
        execute(context, args.release_sha256, started)
    except BaseException as error:
        context[1].write(output/'FAILURE.json', dict(complete=False, error_type=type(error).__name__, error=str(error),
            inclusive_seconds=time.monotonic()-started, TEST_access=False, automatic_retry=False,
            models=0, model_forwards=0, new_fits=0))
        raise


if __name__ == '__main__':
    main()
