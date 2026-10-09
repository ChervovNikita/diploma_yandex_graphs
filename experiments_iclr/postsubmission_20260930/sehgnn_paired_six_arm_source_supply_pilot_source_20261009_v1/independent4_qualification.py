"""Default-disabled fixed one-epoch genuine independent4 resource/state witness."""
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import resource
import sys
import time

from family_driver import source_gate, require, sha, write
from independent_controls import VARIANTS


@dataclass(frozen=True)
class QualificationConfig:
    enabled: bool = False
    root_source_review_approved: bool = False
    source_seal_sha256: str | None = None
    native_qualification_binding: str | None = None
    role_binding: str | None = None
    role_file: str | None = None
    qualification_binding: str | None = None

    def require_enabled(self):
        require(self.enabled is True and self.root_source_review_approved is True, 'Inactive independent4 qualification: exact root release required')
        for value in (self.source_seal_sha256, self.native_qualification_binding, self.role_binding, self.qualification_binding):
            require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Literal reviewed source/native/role/qualification bindings')
        require(self.source_seal_sha256 == sha(Path(__file__).resolve().parent / 'SEAL.json'), 'Exact sealed qualification source')


def factual_pool(rt, engine, metadata, committee, ctx, costs):
    """All known factual rows, no labels/quality metric/selector consultation."""
    torch = rt['torch']
    caller = engine.capture_rng(rt['numpy'], torch)
    outputs = []
    caches = tuple((id(mapping), tuple((key, id(value), int(value._version)) for key, value in mapping.items()))
                   for mapping in (ctx.feats, ctx.label_feats))
    try:
        with costs.measure('qualification_four_current_full_input_FP32_known_role_forwards_and_probability_pool', gpu=True), torch.no_grad():
            for body in committee.bodies:
                parameters = tuple((id(p), int(p._version)) for p in body.model.parameters())
                buffers = [(name, value, value.detach().cpu().clone()) for name, value in body.model.named_buffers(remove_duplicate=False)]
                values = []
                with metadata.scratch_native_state(rt, engine, body.model, 'eval', body.rng), torch.cuda.amp.autocast(enabled=False):
                    for batch, features, labels, mask in ctx.eval_batches:
                        output = body.model(batch.to(rt['device']), {key: value.to(rt['device']) for key, value in features.items()},
                            {key: value.to(rt['device']) for key, value in labels.items()}, mask).float()
                        require(torch.isfinite(output).all().item(), 'Finite complete native factual output')
                        values.append(output.detach().cpu().clone())
                logits = torch.cat(values, dim=0)
                require(logits.shape == (ctx.train_count + ctx.valid_count, 5), 'All known rows and five native logits')
                require(parameters == tuple((id(p), int(p._version)) for p in body.model.parameters()), 'Factual serving preserves learned parameters')
                require([(name, id(value)) for name, value in body.model.named_buffers(remove_duplicate=False)] == [(name, id(value)) for name, value, _ in buffers]
                    and all(torch.equal(value.detach().cpu(), old) for _, value, old in buffers), 'Actual original buffers preserved')
                outputs.append(logits)
            probability = torch.stack([torch.sigmoid(value) for value in outputs]).mean(dim=0)
            require(torch.isfinite(probability).all().item(), 'Finite actual per-label full-input four-body pool')
        committee.verify_independence()
        require(caches == tuple((id(mapping), tuple((key, id(value), int(value._version)) for key, value in mapping.items()))
            for mapping in (ctx.feats, ctx.label_feats)), 'Original canonical native caches preserved')
        return dict(member_logits=outputs, probabilities=probability)
    finally:
        engine.restore_rng(rt['numpy'], torch, caller)
        require(engine.exact(torch, engine.capture_rng(rt['numpy'], torch), caller), 'Factual qualification serving preserves caller stream')


def qualify(rt, engine, adapter, metadata, independent_module, data, frozen_role, folder, config=QualificationConfig()):
    """Both complete variants, four actual own updates each; no accuracy fit."""
    config.require_enabled()
    sources = source_gate()
    for module, key in ((engine, 'native_engine_sha256'), (adapter, 'adapter_source_sha256'),
        (metadata, 'independent_metadata_sha256'), (independent_module, 'independent_source_sha256'),
        (rt['model_module'], 'native_model_sha256'), (rt['native'], 'native_helpers_sha256')):
        require(sha(module.__file__) == sources[key], 'Exact already-qualified ordinary native independent4 dependency')
    require(sha(engine.capture_rng.__code__.co_filename) == sources['native_state_sha256']
        and rt['model_class'] is rt['model_module'].SeHGNN and rt['device'].type == 'cuda', 'Ordinary exact native CUDA runtime')
    path = Path(config.role_file)
    require(path.name == 'seed1.json' and sha(path) == config.role_binding and json.loads(path.read_text()) == frozen_role
        and frozen_role['seed'] == 1, 'Exact previously frozen role1, never resplit')
    folder = Path(folder)
    require(not folder.exists(), 'Fresh root-owned qualification output')
    folder.mkdir(parents=True, exist_ok=False)
    torch = rt['torch']
    costs = engine.Costs(folder, torch, rt['device'])
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    caller = engine.capture_rng(rt['numpy'], torch)
    torch.cuda.reset_peak_memory_stats(rt['device'])
    report = dict(status='started', complete=False, qualification_passed=False, role_seed=1,
        qualification_binding=config.qualification_binding, source_seal_sha256=config.source_seal_sha256,
        fixed_one_epoch_per_body=True, accuracy_fit=False, quality_scoring=False, VALID_checkpoint_selection=False,
        native_TRAIN_helper_metrics_computed_but_not_reported_or_used=True, TEST_file_access=False, variants=[])
    committee = group = fixed = fresh = initial = body = entry = saved = None
    write(folder / 'QUALIFICATION_REPORT.json', report)
    try:
        static = engine.prepare_static(rt, data, costs)
        ctx = engine.prepare_seed(rt, static, frozen_role, 1, costs)
        full_binding = metadata.digest(dict(qualification=config.qualification_binding, role_binding=config.role_binding,
            input_files=data.input_bindings, feature_shapes=static.feature_shapes,
            label_shapes={key: list(value.shape) for key, value in ctx.label_feats.items()}))
        for variant in VARIANTS:
            cell = folder / variant
            cell.mkdir()
            row = dict(variant=variant, status='started', complete=False, body_updates=[])
            report['variants'].append(row)
            local = engine.Costs(cell, torch, rt['device'])
            spec = dict(variant=variant, role_binding=config.role_binding, full_view_binding=full_binding, role_seed=1,
                body_seeds=[1001, 1002, 1003, 1004], body_order=[0, 1, 2, 3], selection=independent_module.SELECTION,
                cost_policy=independent_module.COST_POLICY)
            release = independent_module.Independent4Config(enabled=True, root_source_review_approved=True,
                source_seal_sha256=sources['independent_seal_sha256'], native_qualification_binding=config.native_qualification_binding,
                committee_binding=independent_module.digest(metadata.plain_metadata(spec)))
            committee = independent_module.Independent4(rt, engine, adapter, metadata, ctx, local, spec, release)
            group = dict(schema='owned_native_independent4_control_v1', committee_binding=release.committee_binding,
                specification=committee.spec, bodies=[], qualification_fixed_one_epoch_state=True, source_supply_untied_correction=False)
            for member, body in enumerate(committee.bodies):
                with local.measure('qualification_body_' + str(member) + '_actual_native_preupdate_parameter_copy', gpu=True):
                    initial = {name: value.detach().cpu().clone() for name, value in body.model.named_parameters()}
                engine.restore_rng(rt['numpy'], torch, body.rng)
                with engine.observe_update(rt, body.model, body.optimizer, body.counters), local.measure('qualification_body_' + str(member) + '_one_native_AMP_TRAIN_epoch', gpu=True):
                    loss, accuracy = rt['native'].train(body.model, ctx.feats, ctx.label_feats, ctx.targets_cuda,
                        torch.nn.BCEWithLogitsLoss(), body.optimizer, ctx.train_loader, rt['native'].evaluator, scalar=body.scalar)
                    require(math.isfinite(loss), 'Finite complete native TRAIN loss')
                body.counters['epochs_completed'] += 1
                body.rng = engine.capture_rng(rt['numpy'], torch)
                with local.measure('qualification_body_' + str(member) + '_actual_change_and_fixed_owned_CPU_state_write', gpu=True):
                    changed = sum(not torch.equal(value.detach().cpu(), initial[name]) for name, value in body.model.named_parameters())
                    require(changed > 0 and all(body.counters[key] == 1 for key in ('epochs_completed', 'TRAIN_member_forwards', 'TRAIN_backwards', 'actual_Adam_steps')),
                        'One real full native own update changes actual learned parameters')
                    saved = metadata.checkpoint_tree(torch, engine.snapshot(rt, body.model, body.optimizer, body.scalar, ctx, 0, {}, {}, body.identity))
                    entry = dict(body=member, body_seed=1001+member, selected_state=saved,
                        fixed_qualification_state_after_completed_own_epochs=1, no_quality_selection=True)
                    group['bodies'].append(entry)
                    torch.save(entry, cell / ('BODY' + str(member) + '_FIXED_STATE.pt'))
                row['body_updates'].append(dict(member=member, changed_parameter_objects=changed, counters=dict(body.counters)))
                initial = saved = loss = accuracy = None
                write(folder / 'QUALIFICATION_REPORT.json', report)
            group['ownership'] = committee.verify_independence()
            committee.restore(group)
            fixed = factual_pool(rt, engine, metadata, committee, ctx, local)
            checkpoint = cell / 'FIXED_INDEPENDENT4_STATE.pt'
            with local.measure('qualification_owned_group_fixed_state_write', gpu=True):
                torch.save(metadata.checkpoint_tree(torch, group), checkpoint)
            committee = group = body = entry = saved = None
            gc.collect()
            torch.cuda.empty_cache()
            with local.measure('qualification_fresh_four_complete_native_body_constructors_and_exact_owned_restore', gpu=True):
                group = torch.load(checkpoint, map_location='cpu', weights_only=True)
                committee = independent_module.Independent4(rt, engine, adapter, metadata, ctx, local, spec, release)
                ownership = committee.restore(group)
            fresh = factual_pool(rt, engine, metadata, committee, ctx, local)
            row.update(status='complete', complete=True, ownership=ownership, fixed_checkpoint_sha256=sha(checkpoint),
                fresh_exact_owned_state_restore=True, full_input_known_role_rows=ctx.train_count+ctx.valid_count,
                actual_full_input_pool=True, output_bitwise_or_tiny_mismatch_gate=False,
                descriptive_max_abs_probability_drift=float((fresh['probabilities']-fixed['probabilities']).abs().max()),
                descriptive_max_abs_member_logit_drift=max(float((a-b).abs().max()) for a,b in zip(fresh['member_logits'],fixed['member_logits'])))
            committee = group = fixed = fresh = body = entry = saved = None
            gc.collect()
            torch.cuda.empty_cache()
            write(folder / 'QUALIFICATION_REPORT.json', report)
        report.update(status='complete', complete=True, qualification_passed=True, real_native_body_updates=8,
            common_static_and_role_setup_included=True, source_views_or_corrections_not_part_of_this_witness=True,
            native_fit_competence_and_independent4_accuracy_reference_still_pending=True)
    except BaseException as error:
        report.update(status='failed', complete=False, qualification_passed=False, failure=dict(type=type(error).__name__, message=str(error)))
        if committee is not None:
            report['partial_body_counters'] = [dict(body.counters) for body in committee.bodies]
        raise
    finally:
        try:
            torch.cuda.synchronize(rt['device'])
            report.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(rt['device']),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(rt['device']))
            committee = group = fixed = fresh = initial = body = entry = saved = None
            gc.collect()
            torch.cuda.empty_cache()
            engine.restore_rng(rt['numpy'], torch, caller)
            require(engine.exact(torch, engine.capture_rng(rt['numpy'], torch), caller), 'Whole qualification preserves caller streams')
        except BaseException as error:
            report.update(status='failed', complete=False, qualification_passed=False, cleanup_failure=dict(type=type(error).__name__, message=str(error)))
        end = resource.getrusage(resource.RUSAGE_SELF)
        report.update(seconds=time.perf_counter()-started, CPU_user_seconds=end.ru_utime-usage.ru_utime,
            CPU_system_seconds=end.ru_stime-usage.ru_stime,
            cumulative_process_RSS_peak_bytes=int(end.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)),
            RSS_peak_is_process_lifetime_highwater_not_incremental=True)
        write(folder / 'QUALIFICATION_REPORT.json', report)
        write(folder / 'COMPLETE.json', dict(status=report['status'], complete=report['complete'], qualification_passed=report['qualification_passed'],
            accuracy_fit=False, TEST_file_access=False))
    return report


if __name__ == '__main__':
    print(json.dumps(dict(inactive=True, root_callable='qualify', runtime_or_host_launch=False)))
