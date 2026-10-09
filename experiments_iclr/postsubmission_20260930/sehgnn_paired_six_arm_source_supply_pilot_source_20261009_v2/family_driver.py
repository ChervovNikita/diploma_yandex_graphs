"""Inactive paired six-arm pilot; actual native bank fitting, no provider/loader/host launcher."""
from __future__ import annotations
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

from control_policy import ARMS, ASSIGNMENTS, FAMILIES, ScheduledSession
from selected_diagnostics import collect_selected_logits, analyze, compare_selected

HERE = Path(__file__).resolve().parent
PAIRS = ((1, 1), (2, 2), (3, 3))  # (frozen outer role seed, native base seed)
CANDIDATE = 'assigned_source_supply'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def source_gate():
    seal = json.loads((HERE / 'SEAL.json').read_text())
    require(seal['runtime_disabled'] is True and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Exact inactive pilot source seal')
    for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed pilot payload')
    sources = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    require(sources['backend_bindings_pending'] is False, 'Exact reviewed compressed backend must be bound before pilot release')
    for row in sources['files']:
        path = (HERE.parent / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE.parent) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed frozen pilot dependency')
    return sources


def validate_modules(rt, engine, adapter, view_module, sources, helper=None, bank_module=None, independent_module=None, metadata_module=None):
    modules = [(engine, 'native_engine_sha256'), (adapter, 'adapter_source_sha256'),
               (view_module, 'view_source_sha256'), (rt['model_module'], 'native_model_sha256'),
               (rt['native'], 'native_helpers_sha256')]
    modules += [(module, key) for module, key in [(helper, 'helper_source_sha256'), (bank_module, 'bank_source_sha256'),
                (independent_module, 'independent_source_sha256'), (metadata_module, 'independent_metadata_sha256')] if module is not None]
    require(all(sha(module.__file__) == sources[key] for module, key in modules)
            and sha(engine.capture_rng.__code__.co_filename) == sources['native_state_sha256'], 'Exact supplied native/policy/backend modules')
    require(rt['model_class'] is rt['model_module'].SeHGNN and rt['device'].type == 'cuda', 'Already qualified ordinary native CUDA runtime')


def validate_roles(config, frozen_roles):
    require(set(frozen_roles) == {1, 2, 3}, 'Exactly three prospective paired roles')
    for seed in (1, 2, 3):
        path = Path(config.role_files[seed])
        require(path.name == 'seed' + str(seed) + '.json' and sha(path) == config.role_bindings[seed]
                and json.loads(path.read_text()) == frozen_roles[seed] and frozen_roles[seed]['seed'] == seed,
                'Actual unchanged root-frozen role descriptor supplied')


@dataclass(frozen=True)
class PilotConfig:
    enabled: bool = False
    root_source_review_approved: bool = False
    root_scientific_release_approved: bool = False
    source_seal_sha256: str | None = None
    native_qualification_binding: str | None = None
    integration_qualification_binding: str | None = None
    independent4_qualification_binding: str | None = None
    native_and_independent_reference_competence_adopted: bool = False
    role_bindings: object = None
    role_files: object = None
    study_binding: str | None = None

    def require_reference_fit_enabled(self):
        require(self.enabled is True and self.root_source_review_approved is True and self.root_scientific_release_approved is True,
                'Inactive reference fits: root source/qualification/study release required')
        for value in (self.source_seal_sha256, self.native_qualification_binding, self.integration_qualification_binding,
                      self.independent4_qualification_binding, self.study_binding):
            require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Exact root study/source/qualification bindings')
        require(self.source_seal_sha256 == sha(HERE / 'SEAL.json')
                and isinstance(self.role_bindings, dict) and set(self.role_bindings) == {1, 2, 3}
                and isinstance(self.role_files, dict) and set(self.role_files) == {1, 2, 3}, 'Frozen three-pair pilot release')
        require(all(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
                    for value in self.role_bindings.values()), 'Frozen role SHA identities')

    def require_enabled(self):
        self.require_reference_fit_enabled()
        require(self.native_and_independent_reference_competence_adopted is True,
                'Inactive shared candidates: real native and genuine independent4 reference competence must first be adopted')


def identity(config, role_seed, base_seed, arm, data, bank_module):
    return bank_module.plain_metadata(dict(study_binding=config.study_binding, source_seal_sha256=config.source_seal_sha256,
        native_qualification_binding=config.native_qualification_binding, integration_qualification_binding=config.integration_qualification_binding,
        independent4_qualification_binding=config.independent4_qualification_binding,
        role_seed=role_seed, base_seed=base_seed, arm=arm, role_binding=config.role_bindings[role_seed], input_files=data.input_bindings,
        member_rng_seeds=[1000 * base_seed + member + 1 for member in range(4)], candidate=CANDIDATE,
        schedule=dict(warmup_actual_own_epochs=10, first_correction_after=15, cadence=5, native_maximum=200, patience=50),
        COMMON_transient_assignment='actor/director/keyword deterministic cycle, three active recipients same source per opportunity'))


def paired_identity(config, role_seed, base_seed, full_binding, data, metadata):
    return metadata.plain_metadata(dict(study_binding=config.study_binding, role_seed=role_seed, base_seed=base_seed,
        role_binding=config.role_bindings[role_seed], full_view_binding=full_binding, input_files=data.input_bindings))


def new_session(rt, engine, adapter, helper, bank_module, ctx, views, costs, config, full_binding, base_seed, role_seed):
    # The shared bank is constructed from a single native prototype in this
    # matched cell; no previously fitted native singles are assembled here.
    rt['native'].set_random_seed(base_seed)
    prototype = engine.make_model(rt, ctx, costs)
    scalar = rt['torch'].cuda.amp.GradScaler()
    sources = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    cfg = bank_module.BankConfig(enabled=True, root_source_review_approved=True,
        source_seal_sha256=sources['bank_seal_sha256'], native_qualification_binding=config.native_qualification_binding,
        full_view_binding=full_binding, role_binding=config.role_bindings[role_seed],
        member_rng_seeds=tuple(1000 * base_seed + member + 1 for member in range(4)),
        assignments=ASSIGNMENTS, source_corrections=True)
    return bank_module.BankSession(rt, engine, adapter, helper, ctx, views, prototype, scalar, costs, cfg)


def numeric_summary(diagnostic):
    result = {key: diagnostic[key] for key in ('family_order', 'member_order', 'U', 'D', 'assigned_specificity', 'full_pool_BCE',
                                             'full_member_BCE', 'full_member_F1', 'full_pool_F1')}
    result['families'] = {family: dict(all_absent_BCE=details['all_absent_BCE'], members=[
        dict(member=item['member'], only_member_full_BCE=item['only_member_full_BCE'], only_member_absent_BCE=item['only_member_absent_BCE'],
             absent_peer_context={key: value for key, value in item['absent_peer_context'].items() if not key.endswith('mask')},
             full_peer_context={key: value for key, value in item['full_peer_context'].items() if not key.endswith('mask')})
        for item in details['members']]) for family, details in diagnostic['family_details'].items()}
    return result


def persist_diagnostic(torch, metadata, diagnostic, folder, costs):
    with costs.measure('complete_VALID_array_and_summary_write'):
        torch.save(metadata.checkpoint_tree(torch, diagnostic), folder / 'SELECTED_VALID_DIAGNOSTICS.pt')
        write(folder / 'SELECTED_VALID_SUMMARY.json', numeric_summary(diagnostic))


def persist_comparison(torch, metadata, reference, candidate, name, folder, costs):
    result = compare_selected(torch, reference, candidate, costs, name)
    with costs.measure('complete_VALID_deployed_pool_comparison_write'):
        torch.save(metadata.checkpoint_tree(torch, result), folder / (name + '_POOL_CHANGES.pt'))
        summary = dict(result)
        summary['deployed_full_input_pool_events'] = {key: value for key, value in result['deployed_full_input_pool_events'].items() if not key.endswith('mask')}
        write(folder / (name + '_POOL_CHANGES.json'), summary)
    return summary


def compare_shared_pair(torch, metadata, pair_folder, costs):
    """All five prespecified controls; no screening or selection from U/D."""
    with costs.measure('paired_safe_diagnostic_array_reads'):
        candidate = torch.load(pair_folder / CANDIDATE / 'SELECTED_VALID_DIAGNOSTICS.pt', map_location='cpu', weights_only=True)
    folder = pair_folder / 'PAIRED_POOL_COMPARISONS'
    folder.mkdir(exist_ok=False)
    for arm in ARMS:
        if arm == CANDIDATE:
            continue
        with costs.measure('paired_safe_diagnostic_array_read_' + arm):
            reference = torch.load(pair_folder / arm / 'SELECTED_VALID_DIAGNOSTICS.pt', map_location='cpu', weights_only=True)
        persist_comparison(torch, metadata, reference, candidate, arm, folder, costs)


def fit_one(rt, engine, adapter, helper, bank_module, ctx, views, config, full_binding, role_seed, base_seed, arm, folder):
    folder.mkdir(exist_ok=False)
    torch = rt['torch']
    costs = engine.Costs(folder, torch, rt['device'])
    torch.cuda.reset_peak_memory_stats(rt['device'])
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    record = dict(status='started', complete=False, role_seed=role_seed, base_seed=base_seed, arm=arm,
        declared_candidate=arm == CANDIDATE, TEST_file_access=False, shadow_work=False,
        own_warmup_retrained_and_charged=True, bitwise_cross_arm_warm_parity_required=False)
    session = policy = saved = fresh = None
    write(folder / 'RESULT.json', record)
    try:
        owned_identity = identity(config, role_seed, base_seed, arm, ctx.data, bank_module)
        session = new_session(rt, engine, adapter, helper, bank_module, ctx, views, costs, config, full_binding, base_seed, role_seed)
        policy = ScheduledSession(session, arm, owned_identity)
        # Exact reviewed native 200/patience50/complete-VALID BCE outer loop.
        fitted = bank_module.fit_bank(policy, owned_identity)
        saved = fitted.pop('selected_state')
        record.update(selected_epoch=fitted['selected_epoch'], selected_VALID_BCE=fitted['selected_VALID_BCE'],
            final_counters=dict(session.counters), scheduled_slots=policy.scheduled_slots, actual_opportunities=policy.opportunities,
            accepted=policy.accepted, rejected=policy.rejected, zero=policy.zero,
            exposures=dict(policy.exposures), actual_private_L2_dose=policy.actual_dose,
            selected_policy=saved['pilot_policy'], native_selection=fitted['selection'],
            selected_before_steering=not saved['pilot_policy']['selected_after_actual_source_opportunity'])
        with (folder / 'HISTORY.jsonl').open('x') as stream:
            for row in fitted['history']:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        checkpoint = folder / 'SELECTED_BANK_STATE.pt'
        with costs.measure('pilot_selected_owned_checkpoint_write', gpu=True):
            torch.save(saved, checkpoint)
        caller = engine.capture_rng(rt['numpy'], torch)
        session = policy = saved = None
        gc.collect()
        torch.cuda.empty_cache()
        try:
            with costs.measure('pilot_fresh_selected_native_bank_reconstruction', gpu=True):
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                fresh = new_session(rt, engine, adapter, helper, bank_module, ctx, views, costs, config, full_binding, base_seed, role_seed)
                fresh.restore_selected(saved, owned_identity)
            scores, outputs = fresh.evaluate()
            drift = {role: dict(max_abs_probability=float((outputs[role]['probabilities'] - saved['outputs'][role]['probabilities']).abs().max().item()),
                max_abs_member_logit=max(float((x-y).abs().max().item()) for x, y in zip(outputs[role]['member_logits'], saved['outputs'][role]['member_logits'])),
                prediction_changes=int(((outputs[role]['probabilities'] > .5) != (saved['outputs'][role]['probabilities'] > .5)).sum().item())) for role in ('TRAIN', 'VALID')}
            selected_logits = collect_selected_logits(rt, engine, bank_module, fresh.bank.members, fresh.member_rng, ctx, views, costs)
            diagnostic = analyze(torch, selected_logits, ctx.targets[ctx.valid_index], ctx.data.valid_ids, costs)
            diagnostic['paired_identity'] = paired_identity(config, role_seed, base_seed, full_binding, ctx.data, bank_module)
            require(fresh.config.full_view_binding == full_binding and saved['pilot_policy']['arm'] == arm, 'Selected method/source provenance preserved')
            persist_diagnostic(torch, bank_module, diagnostic, folder, costs)
            record.update(fresh_scores=scores, descriptive_reconstruction_drift=drift, output_bitwise_gate=False,
                          checkpoint_sha256=sha(checkpoint), complete_4_by_3_VALID_diagnostics=True,
                          correct_positive_and_negative_label_events_retained=True)
        finally:
            fresh = saved = None
            gc.collect()
            torch.cuda.empty_cache()
            engine.restore_rng(rt['numpy'], torch, caller)
            require(engine.exact(torch, engine.capture_rng(rt['numpy'], torch), caller), 'Fresh selected/diagnostic work preserves caller stream')
        record.update(status='complete', complete=True)
    except BaseException as error:
        record.update(status='failed', complete=False, failure={'type': type(error).__name__, 'message': str(error)})
        if policy is not None:
            record.update(final_counters=dict(session.counters), actual_opportunities=policy.opportunities,
                scheduled_slots=policy.scheduled_slots, accepted=policy.accepted, rejected=policy.rejected, zero=policy.zero,
                exposures=dict(policy.exposures), actual_private_L2_dose=policy.actual_dose, partial_correction_logs=policy.logs)
        raise
    finally:
        try:
            torch.cuda.synchronize(rt['device'])
            record.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(rt['device']), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(rt['device']))
            session = policy = saved = fresh = None
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException as error:
            record.update(status='failed', complete=False, cleanup_failure={'type': type(error).__name__, 'message': str(error)})
        end = resource.getrusage(resource.RUSAGE_SELF)
        record.update(seconds=time.perf_counter() - started, CPU_user_seconds=end.ru_utime - usage.ru_utime,
            CPU_system_seconds=end.ru_stime - usage.ru_stime, cumulative_RSS_peak_bytes=int(end.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)))
        write(folder / 'RESULT.json', record)
        write(folder / 'COMPLETE.json', dict(status=record['status'], complete=record['complete'], TEST_file_access=False))
    return record


def run_shared_family(rt, engine, adapter, helper, view_module, bank_module, data, frozen_roles, folder, config=PilotConfig()):
    """Complete18 paired shared fits; no dataset/provider loader or execution host action."""
    config.require_enabled()
    sources = source_gate()
    validate_modules(rt, engine, adapter, view_module, sources, helper=helper, bank_module=bank_module)
    validate_roles(config, frozen_roles)
    folder = Path(folder)
    require(not folder.exists(), 'Fresh prospectively bound study output')
    folder.mkdir(parents=True, exist_ok=False)
    costs = engine.Costs(folder, rt['torch'], rt['device'])
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    summary = dict(status='started', complete=False, candidate=CANDIDATE, pairs=[list(pair) for pair in PAIRS], arms=list(ARMS), cells=[],
        results_unpublished=True, TEST_file_access=False, independent4_groups_are_separate=True,
        common_static_and_per_role_source_setup_charged=True, cell_costs_do_not_include_common_setup=True,
        whole_family_inclusive_costs_include_setup_fits_selection_serving_diagnostics=True,
        backend_identity=sources['backend_identity'])
    summary['setup_cuda_peak_observations'] = []
    write(folder / 'FAMILY_REPORT.json', summary)
    try:
        static = engine.prepare_static(rt, data, costs)
        for role_seed, base_seed in PAIRS:
            pair_folder = folder / ('pair' + str(role_seed))
            pair_folder.mkdir()
            setup_costs = engine.Costs(pair_folder, rt['torch'], rt['device'])
            ctx = engine.prepare_seed(rt, static, frozen_roles[role_seed], role_seed, setup_costs)
            full_binding = bank_module.digest(dict(study=config.study_binding, role_binding=config.role_bindings[role_seed],
                seed=role_seed, inputs=data.input_bindings, feature_shapes=static.feature_shapes,
                label_shapes={key: list(value.shape) for key, value in ctx.label_feats.items()}))
            views = view_module.build_family_views(rt, static, ctx, setup_costs, view_module.ViewConfig(enabled=True,
                root_source_review_approved=True, native_qualification_binding=config.native_qualification_binding,
                full_view_binding=full_binding, role_binding=config.role_bindings[role_seed], source_seal_sha256=sources['view_seal_sha256']))
            summary['setup_cuda_peak_observations'].append(dict(role_seed=role_seed,
                allocated=rt['torch'].cuda.max_memory_allocated(rt['device']), reserved=rt['torch'].cuda.max_memory_reserved(rt['device'])))
            for arm in ARMS:
                cell = pair_folder / arm
                row = dict(role_seed=role_seed, base_seed=base_seed, arm=arm, output=str(cell.resolve()), complete=False, status='started')
                summary['cells'].append(row)
                try:
                    fit_one(rt, engine, adapter, helper, bank_module, ctx, views, config, full_binding, role_seed, base_seed, arm, cell)
                finally:
                    if (cell / 'RESULT.json').exists():
                        final = json.loads((cell / 'RESULT.json').read_text())
                        row.update(status=final['status'], complete=final['complete'], peak_cuda_allocated_bytes=final.get('peak_cuda_allocated_bytes'),
                                   peak_cuda_reserved_bytes=final.get('peak_cuda_reserved_bytes'))
                    write(folder / 'FAMILY_REPORT.json', summary)
                require(final['complete'] is True, 'Every fixed shared arm and seed remains in family report')
            compare_shared_pair(rt['torch'], bank_module, pair_folder, setup_costs)
            ctx = views = None
            gc.collect()
            rt['torch'].cuda.empty_cache()
        summary.update(status='complete', complete=True, complete_cells=18, candidate_not_selected_best_of_six=True)
        return summary
    except BaseException as error:
        summary.update(status='failed', complete=False, failure={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        end = resource.getrusage(resource.RUSAGE_SELF)
        summary.update(seconds=time.perf_counter() - started, CPU_user_seconds=end.ru_utime - usage.ru_utime,
            CPU_system_seconds=end.ru_stime - usage.ru_stime,
            cumulative_process_RSS_peak_bytes=int(end.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)),
            RSS_peak_is_process_lifetime_highwater_not_incremental=True)
        summary['observed_family_peak_cuda_allocated_bytes'] = max([rt['torch'].cuda.max_memory_allocated(rt['device'])]
            + [row['allocated'] for row in summary['setup_cuda_peak_observations']]
            + [row['peak_cuda_allocated_bytes'] for row in summary['cells'] if row.get('peak_cuda_allocated_bytes') is not None])
        summary['observed_family_peak_cuda_reserved_bytes'] = max([rt['torch'].cuda.max_memory_reserved(rt['device'])]
            + [row['reserved'] for row in summary['setup_cuda_peak_observations']]
            + [row['peak_cuda_reserved_bytes'] for row in summary['cells'] if row.get('peak_cuda_reserved_bytes') is not None])
        write(folder / 'FAMILY_REPORT.json', summary)


if __name__ == '__main__':
    print(json.dumps({'inactive': True, 'root_callable': 'run_shared_family', 'data_model_host_or_quality_execution': False}))
