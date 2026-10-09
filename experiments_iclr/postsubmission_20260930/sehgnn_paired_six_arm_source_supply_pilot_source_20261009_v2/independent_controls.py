"""Separate genuine independent4 matched groups, reusing the sealed body fitter."""
import gc
import json
from pathlib import Path
import resource
import sys
import time

from family_driver import (PilotConfig, PAIRS, CANDIDATE, source_gate, validate_modules, validate_roles,
    require, write, sha, paired_identity, persist_diagnostic, persist_comparison)
from selected_diagnostics import collect_selected_logits, analyze

VARIANTS = ('plain_native', 'untied_same_six_factors')


def fit_independent_group(rt, engine, adapter, metadata, module, ctx, views, config, sources,
                          full_binding, role_seed, base_seed, variant, folder):
    """Four fresh full bodies, individual selection, safe fresh restore and assay."""
    require(variant in VARIANTS, 'One of both predeclared independent4 controls')
    folder.mkdir(exist_ok=False)
    torch = rt['torch']
    costs = engine.Costs(folder, torch, rt['device'])
    torch.cuda.reset_peak_memory_stats(rt['device'])
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    spec = dict(variant=variant, role_binding=config.role_bindings[role_seed], full_view_binding=full_binding,
        role_seed=role_seed, body_seeds=[1000 * base_seed + m + 1 for m in range(4)], body_order=[0, 1, 2, 3],
        selection=module.SELECTION, cost_policy=module.COST_POLICY)
    release = module.Independent4Config(enabled=True, root_source_review_approved=True,
        source_seal_sha256=sources['independent_seal_sha256'], native_qualification_binding=config.native_qualification_binding,
        committee_binding=module.digest(metadata.plain_metadata(spec)))
    record = dict(status='started', complete=False, variant=variant, role_seed=role_seed, base_seed=base_seed,
        specification=spec, committee_binding=release.committee_binding,
        no_subset_of_native_five_reference=True, source_supply_untied_correction=False, TEST_file_access=False)
    committee = group = served = diagnostic = body_result = None
    write(folder / 'RESULT.json', record)
    try:
        committee = module.Independent4(rt, engine, adapter, metadata, ctx, costs, spec, release)
        # The sealed fit() is this fixed fit_body loop plus restore. Persist
        # every completed body's actual selected state/history immediately so
        # a later body failure cannot erase already-completed body evidence.
        group = dict(schema='owned_native_independent4_control_v1', committee_binding=release.committee_binding,
            specification=committee.spec, bodies=[], source_supply_untied_correction=False)
        for member in spec['body_order']:
            body_result = committee.fit_body(member)
            group['bodies'].append(body_result)
            with costs.measure('independent_body_' + str(member) + '_completed_selected_state_and_history_write', gpu=True):
                torch.save(metadata.checkpoint_tree(torch, body_result), folder / ('BODY' + str(member) + '_SELECTED.pt'))
        group['ownership'] = committee.verify_independence()
        committee.restore(group)
        record.update(ownership=group['ownership'], selected_epochs=[row['selected_epoch'] for row in group['bodies']],
            selected_member_VALID_BCE=[row['selected_VALID_BCE'] for row in group['bodies']],
            final_counters=[dict(body.counters) for body in committee.bodies])
        checkpoint = folder / 'SELECTED_INDEPENDENT4_STATE.pt'
        with costs.measure('independent4_owned_all_body_selected_checkpoint_write', gpu=True):
            torch.save(metadata.checkpoint_tree(torch, group), checkpoint)
        committee = group = body_result = None
        gc.collect()
        torch.cuda.empty_cache()
        with costs.measure('independent4_fresh_four_native_bodies_owned_restore', gpu=True):
            group = torch.load(checkpoint, map_location='cpu', weights_only=True)
            committee = module.Independent4(rt, engine, adapter, metadata, ctx, costs, spec, release)
            committee.restore(group)
        served = committee.serve()
        values = collect_selected_logits(rt, engine, metadata, [body.model for body in committee.bodies],
            [body.rng for body in committee.bodies], ctx, views, costs)
        diagnostic = analyze(torch, values, ctx.targets[ctx.valid_index], ctx.data.valid_ids, costs)
        diagnostic['paired_identity'] = paired_identity(config, role_seed, base_seed, full_binding, ctx.data, metadata)
        diagnostic['control_identity'] = dict(variant=variant, committee_binding=release.committee_binding,
            source_supply_untied_correction=False)
        persist_diagnostic(torch, metadata, diagnostic, folder, costs)
        record.update(status='complete', complete=True, fresh_metrics=served['metrics'],
            checkpoint_sha256=sha(checkpoint), exact_owned_body_state_restore=True, output_bitwise_gate=False,
            complete_4_by_3_VALID_diagnostics=True, same_original_native_body_and_six_factor_sites=True)
    except BaseException as error:
        record.update(status='failed', complete=False, failure=dict(type=type(error).__name__, message=str(error)))
        if committee is not None:
            record['partial_body_counters'] = [dict(body.counters) for body in committee.bodies]
        raise
    finally:
        try:
            torch.cuda.synchronize(rt['device'])
            record.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(rt['device']),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(rt['device']))
            committee = group = served = diagnostic = body_result = None
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException as error:
            record.update(status='failed', complete=False, cleanup_failure=dict(type=type(error).__name__, message=str(error)))
        end = resource.getrusage(resource.RUSAGE_SELF)
        record.update(seconds=time.perf_counter() - started, CPU_user_seconds=end.ru_utime - usage.ru_utime,
            CPU_system_seconds=end.ru_stime - usage.ru_stime,
            cumulative_process_RSS_peak_bytes=int(end.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)))
        write(folder / 'RESULT.json', record)
        write(folder / 'COMPLETE.json', dict(status=record['status'], complete=record['complete'], TEST_file_access=False))
    return record


def run_independent_family(rt, engine, adapter, metadata, independent_module, view_module, data, frozen_roles,
                           folder, config=PilotConfig()):
    """Six separate fixed committees; all24 actual body fits and setup charged."""
    config.require_reference_fit_enabled()
    sources = source_gate()
    validate_modules(rt, engine, adapter, view_module, sources, independent_module=independent_module, metadata_module=metadata)
    validate_roles(config, frozen_roles)
    folder = Path(folder)
    require(not folder.exists(), 'Fresh separate independent4 family output')
    folder.mkdir(parents=True, exist_ok=False)
    costs = engine.Costs(folder, rt['torch'], rt['device'])
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    summary = dict(status='started', complete=False, pairs=[list(pair) for pair in PAIRS], variants=list(VARIANTS), cells=[],
        genuine_fresh_own_selected_independent4=True, independently_rebuilt_common_setup_charged=True,
        independent4_qualification_binding=config.independent4_qualification_binding,
        native_and_independent_reference_competence_pending=not config.native_and_independent_reference_competence_adopted,
        source_supply_untied_correction=False, untied_plus_J_required_before_sharing_advantage_claim=True, TEST_file_access=False)
    write(folder / 'FAMILY_REPORT.json', summary)
    try:
        static = engine.prepare_static(rt, data, costs)
        for role_seed, base_seed in PAIRS:
            pair_folder = folder / ('pair' + str(role_seed))
            pair_folder.mkdir()
            setup_costs = engine.Costs(pair_folder, rt['torch'], rt['device'])
            ctx = engine.prepare_seed(rt, static, frozen_roles[role_seed], role_seed, setup_costs)
            full_binding = metadata.digest(dict(study=config.study_binding, role_binding=config.role_bindings[role_seed],
                seed=role_seed, inputs=data.input_bindings, feature_shapes=static.feature_shapes,
                label_shapes={key: list(value.shape) for key, value in ctx.label_feats.items()}))
            views = view_module.build_family_views(rt, static, ctx, setup_costs, view_module.ViewConfig(enabled=True,
                root_source_review_approved=True, native_qualification_binding=config.native_qualification_binding,
                full_view_binding=full_binding, role_binding=config.role_bindings[role_seed], source_seal_sha256=sources['view_seal_sha256']))
            for variant in VARIANTS:
                cell = pair_folder / variant
                row = dict(role_seed=role_seed, base_seed=base_seed, variant=variant, output=str(cell.resolve()), status='started', complete=False)
                summary['cells'].append(row)
                try:
                    fit_independent_group(rt, engine, adapter, metadata, independent_module, ctx, views, config, sources,
                        full_binding, role_seed, base_seed, variant, cell)
                finally:
                    if (cell / 'RESULT.json').exists():
                        final = json.loads((cell / 'RESULT.json').read_text())
                        row.update(status=final['status'], complete=final['complete'])
                    write(folder / 'FAMILY_REPORT.json', summary)
                require(final['complete'], 'Every declared independent4 group retained')
            ctx = views = None
            gc.collect()
            rt['torch'].cuda.empty_cache()
        summary.update(status='complete', complete=True, complete_groups=6, actual_body_fits=24, no_subset_selection=True)
        return summary
    except BaseException as error:
        summary.update(status='failed', complete=False, failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        end = resource.getrusage(resource.RUSAGE_SELF)
        summary.update(seconds=time.perf_counter() - started, CPU_user_seconds=end.ru_utime - usage.ru_utime,
            CPU_system_seconds=end.ru_stime - usage.ru_stime,
            cumulative_process_RSS_peak_bytes=int(end.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)),
            RSS_peak_is_process_lifetime_highwater_not_incremental=True)
        write(folder / 'FAMILY_REPORT.json', summary)


def compare_independent_controls(rt, engine, metadata, shared_folder, independent_folder, folder, config=PilotConfig()):
    """Every paired candidate versus both independently selected committees."""
    config.require_enabled()
    sources = source_gate()
    require(sha(engine.__file__) == sources['native_engine_sha256'] and sha(metadata.__file__) == sources['independent_metadata_sha256'],
            'Exact arithmetic/serialization dependencies')
    shared_folder, independent_folder, folder = map(Path, (shared_folder, independent_folder, folder))
    require(not folder.exists(), 'Fresh comparison output')
    for root in (shared_folder, independent_folder):
        require(json.loads((root / 'FAMILY_REPORT.json').read_text())['complete'], 'Both full families must be complete')
    folder.mkdir(parents=True, exist_ok=False)
    costs = engine.Costs(folder, rt['torch'], rt['device'])
    for role_seed, _ in PAIRS:
        pair = folder / ('pair' + str(role_seed))
        pair.mkdir()
        with costs.measure('safe_paired_candidate_diagnostic_read'):
            candidate = rt['torch'].load(shared_folder / pair.name / CANDIDATE / 'SELECTED_VALID_DIAGNOSTICS.pt', map_location='cpu', weights_only=True)
        for variant in VARIANTS:
            with costs.measure('safe_paired_independent_diagnostic_read'):
                reference = rt['torch'].load(independent_folder / pair.name / variant / 'SELECTED_VALID_DIAGNOSTICS.pt', map_location='cpu', weights_only=True)
            persist_comparison(rt['torch'], metadata, reference, candidate, variant, pair, costs)
    write(folder / 'COMPLETE.json', dict(complete=True, matched_comparisons=6, success_assessed=False,
        source_supply_untied_correction=False, TEST_file_access=False))


if __name__ == '__main__':
    print(json.dumps(dict(inactive=True, root_callable='run_independent_family', data_model_host_or_quality_execution=False)))
