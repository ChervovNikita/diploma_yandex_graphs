"""Prospective common-builder adapter and ten-mask attribution interface.

No numerical imports, launch, generated admission or change to portable sources.
Root must first close/support original E, verify the portable runtime, and admit
this future acquisition. Export/calibration/scoring integration remains separate.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
import time
import types

SEEDS = (8101, 8219, 8363, 8501, 8663, 8819, 9001, 9199, 9403, 9623)
PORTABLE_MANIFEST = 'd52b185077d81692472a0bedc002df864c1c7a34f6de55b4e398126c2ce80781'
LIMITS = dict(wall_seconds=172800, cuda_bytes=48 * 1024 ** 3, rss_bytes=64 * 1024 ** 3)
T_975_DF9 = 2.2621571628540993
MARGIN_PP = .10


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            digest.update(part)
    return digest.hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def _timeout(signum, frame):
    raise TimeoutError('Finite common-builder acquisition cap')


def _tie(family, body):
    """Fresh CPU body only, before AdamW; preserve every non-builder-r/s object."""
    torch = family.torch
    before = {name: parameter for name, parameter in body.named_parameters()}
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in before.values()),
            'Tie freshly constructed CPU float32 builders before optimizer/placement')
    rng = torch.get_rng_state().clone()
    removed_ids, added_ids, changes = set(), set(), []

    class CommonBuilder(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.weight, self.bias = original.weight, original.bias
            self.r = torch.nn.Parameter(original.r[:1].detach().clone())
            self.s = torch.nn.Parameter(original.s[:1].detach().clone())

        def forward(self, x):
            # The original sequential FactorLinear r/W/s/b operation order.
            # This module is intentionally outside FactorLinear member_context:
            # its one trainable row is common while each live input stays private.
            y = torch.nn.functional.linear(x * self.r[0], self.weight) * self.s[0]
            return y + self.bias

    require(len(body.residual_modules) == 2, 'Original two native graph layers')
    for layer, residual in enumerate(body.residual_modules):
        conv = residual.module.conv
        require(conv.value_conditioning == 'endpoint', 'Exact endpoint action only')
        for name in ('value_source_builder', 'value_target_builder'):
            original = getattr(conv, name)
            require(isinstance(original, family.factors.FactorLinear)
                    and original.weight.shape == (16, 256) and original.bias is not None
                    and original.bias.shape == (16,) and original.r.shape == (4, 256)
                    and original.s.shape == (4, 16), 'Exact original four-route endpoint builder')
            require(torch.equal(original.r, torch.ones_like(original.r))
                    and torch.equal(original.s, torch.ones_like(original.s))
                    and torch.count_nonzero(original.weight).item() == 0
                    and torch.count_nonzero(original.bias).item() == 0,
                    'Fresh original all-one factors/zero W,b; never tie a trained checkpoint')
            common = CommonBuilder(original)
            require(common.weight is original.weight and common.bias is original.bias,
                    'Preserve original builder W/b objects and exact values')
            removed_ids.update((id(original.r), id(original.s)))
            added_ids.update((id(common.r), id(common.s)))
            setattr(conv, name, common)
            changes.append(dict(layer=layer, builder=name, route_inputs=4, factor_rows=1,
                                original_factor_parameters=1088, common_factor_parameters=272,
                                W_bias_parameter_identity_preserved=True))
    after = {name: parameter for name, parameter in body.named_parameters()}
    require({id(p) for p in after.values()} == (({id(p) for p in before.values()} - removed_ids) | added_ids),
            'Only the eight builder r/s tensors changed ownership/shape')
    require(sum(p.numel() for p in before.values()) - sum(p.numel() for p in after.values()) == 3264,
            'Exact 3264-parameter reduction, without dummy parameters')
    require(torch.equal(torch.get_rng_state(), rng), 'Tying consumes no RNG draw')
    for record in family._value_installations:
        record.update(builder_factor_rows=1, builder_private_parameters=0,
                      builder_common_factor_parameters=544, route_inputs=4)
    return dict(builders=changes, original_builder_factor_parameters=4352,
                common_builder_factor_parameters=1088, parameter_reduction=3264,
                native_parameter_objects_preserved=True, draws_added=0,
                endpoint_action_unchanged=True, live_private_route_states_retained=True)


def bind_common_builders(family):
    """Bind only this fresh Family instance; existing make owns optimizer/RNG."""
    require(not getattr(family, '_common_builder_bound', False) and family._fit_model is None,
            'One binding before any model/optimizer')
    original = family.native_body

    def native_body(self, seed, native_member, factor_member, factorized, members):
        require(self._operation == 'endpoint' and factorized and members == 4
                and native_member == factor_member == 0, 'Only fresh common-builder E4 is admitted')
        body, wrapper = original(seed, native_member, factor_member, factorized, members)
        require(wrapper is not None and wrapper.kind == 'baseline' and wrapper.members == 4,
                'Four native private-state routes without cross-route block')
        self._common_builder_last = _tie(self, body)
        return body, wrapper

    family.native_body = types.MethodType(native_body, family)
    family._common_builder_bound = True
    return family


def acquire_control(engine, common, cfg, output, original_closure_receipts, root_guard):
    """Ten fresh common-builder fits; no engine.acquire/Family.run or old refit.

    root_guard first performs the applicable host/GPU checks, then validates all
    exact source/plan/config/role/graph receipts, original whole confirmation and
    E support, actual portable check, and finite external process owner. It must
    return the explicitly requested future admission, never an inherited PASS.
    Root supplies an actual portable cfg with LOCAL_RUN_PERMITTED and protocol
    schema Physics_E_common_builder_attribution_v1. No cfg/admission is generated.
    """
    require(callable(root_guard), 'Future explicit root admission callback required')
    source, plan_path = Path(__file__).resolve(), Path(__file__).resolve().with_name('PLAN.json')
    proof = root_guard(copy.deepcopy(cfg), action='common_builder_E4_attribution_fit',
                       source_path=str(source), plan_path=str(plan_path), output=str(output),
                       original_closure_receipts=copy.deepcopy(original_closure_receipts))
    require(isinstance(proof, dict) and proof.get('admitted') is True
            and proof.get('action') == 'common_builder_E4_attribution_fit'
            and proof.get('whole_original_confirmation_complete') is True
            and proof.get('complete_E_support') is True
            and proof.get('actual_portable_runtime_check_complete') is True
            and proof.get('finite_external_owner_bound') is True
            and proof.get('source_sha256') == sha(source) and proof.get('plan_sha256') == sha(plan_path),
            'Future root-owned whole-confirmation/support/runtime/source/owner admission')
    package = Path(engine.__file__).resolve().parent
    require(Path(common.__file__).resolve().parent == package
            and sha(package / 'MANIFEST.json') == PORTABLE_MANIFEST,
            'Exact sealed portable engine/common API package')
    require(cfg['seeds'] == list(SEEDS) and cfg['family_limits'] == LIMITS
            and cfg['device'] == 'cuda:0' and cfg['profile']['dataset'] == 'CoauthorPhysics'
            and cfg['protocol'].get('schema') == 'Physics_E_common_builder_attribution_v1',
            'Fixed ten original masks, acquisition caps and explicit attribution protocol')
    require(set(original_closure_receipts) == {'cohort', 'calibration', 'test'},
            'Root binds the entire original fit/calibration/TEST closure, not a partial E bank')
    plan = json.loads(plan_path.read_text())
    require(plan['primary_paired_effect']['margin_pp'] == MARGIN_PP, 'Previously saved margin remains frozen')
    output = Path(output).resolve()
    require(not output.exists(), 'Fresh control output only; never alter an existing attempt')
    started, records, family = time.monotonic(), [], None
    previous_handler = signal.getsignal(signal.SIGALRM)
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'Run in a dedicated root-owned child, not an existing alarm owner')
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(LIMITS['wall_seconds'])
    try:
        family = bind_common_builders(engine.construct(copy.deepcopy(cfg), output, common.local_validate))
        for seed in SEEDS:
            family.activate_role_block(seed)
            family.current.update(arm='common_builder_E4', seed=seed)
            family._operation = 'endpoint'
            folder = output / ('common_builder_E4_seed' + str(seed))
            folder.mkdir()
            logits, fit = family.fit(folder / 'member0', seed, 0, 'baseline', True, 4,
                                     'own_only', False, 'shared')
            require(fit['value_parameter_roles']['private_builder_factor_parameters'] == 1088,
                    'Recorded common builder factors match actual parameter count')
            # The preserved owner labels all r/s as private. Correct only this
            # fresh control's metadata; shared rows carry zero private factors.
            fit['value_parameter_roles']['private_builder_factor_parameters'] = 0
            fit['value_parameter_roles']['common_builder_factor_parameters'] = 1088
            write(folder / 'member0' / 'RESULT.json', fit)
            original_rule_metrics = family.metrics(logits)
            probabilities = logits.softmax(-1)
            pool = probabilities.mean(0)
            y = family.valid_y.cpu()
            archive = folder / 'selected_VALID.npz'
            family.np.savez(archive, ids=family.valid_ids.cpu().numpy(), y=y.numpy(),
                           raw_logits=logits.numpy(), probability_mean=pool.numpy(),
                           member_errors=probabilities.argmax(-1).ne(y).numpy(),
                           pooled_errors=pool.argmax(-1).ne(y).numpy())
            record = dict(seed=seed, arm='common_builder_E4', members=4, fits=[fit],
                          optimizer_acquisition_bundles=1, native_bodies_fitted=1,
                          builder_tying=copy.deepcopy(family._common_builder_last),
                          selected_VALID_archive=receipt(archive), SELECT_metrics=original_rule_metrics,
                          CAL_CONFIRM_truth_access=False, original_E_score_recalculation=False)
            write(folder / 'RESULT.json', record)
            records.append(record)
        require(len(records) == 10 and family.completed_units == 10, 'All ten prescribed common-builder fits close')
        require(sha(source) == proof['source_sha256'] and sha(plan_path) == proof['plan_sha256'], 'Frozen source/plan remained unchanged')
        result = dict(complete=True, banks=10, optimizer_acquisition_bundles=10, native_body_fit_records=10,
                      fixed_seed_roster=list(SEEDS), fixed_arm_roster=['common_builder_E4'], records=records,
                      original_closure_receipts=original_closure_receipts, future_root_admission=proof,
                      source=receipt(source), plan=receipt(plan_path), seconds=time.monotonic() - started,
                      CAL_CONFIRM_truth_access=False, original_score_recalculation=False,
                      same_graph_attribution_not_unused_confirmation=True,
                      automatic_export_calibration_or_CONFIRM_release=False)
        write(output / 'COMPLETE_COMMON_BUILDER_FITS.json', result)
        return result
    except BaseException as error:
        if output.exists():
            write(output / 'COMMON_BUILDER_FAILURE.json', dict(complete=False, status='FAILED_RETAINED',
                  type=type(error).__name__, message=str(error), completed_banks=len(records),
                  retry_or_survivor_subset=False, automatic_CONFIRM_release=False))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def paired_effect(records):
    """Only completed scalar records; original metrics are supplied verbatim.

    Each exact row: seed,common_raw_accuracy_pct,E4_raw_accuracy_pct,
    common_calibrated_NLL,BASE_calibrated_NLL,I4_calibrated_NLL. Root must bind
    original sealed scalar custody and all ten new fit/export/CAL/CONFIRM closure
    before invoking. This function neither reads labels nor recalculates scores.
    """
    fields = {'seed', 'common_raw_accuracy_pct', 'E4_raw_accuracy_pct',
              'common_calibrated_NLL', 'BASE_calibrated_NLL', 'I4_calibrated_NLL'}
    require(len(records) == 10 and {r['seed'] for r in records} == set(SEEDS)
            and all(set(r) == fields for r in records), 'Complete exact ten paired scalar records; no survivor mean')
    index = {r['seed']: r for r in records}
    rows = [index[seed] for seed in SEEDS]
    require(all(math.isfinite(r[k]) for r in rows for k in fields - {'seed'}), 'Finite completed paired metrics')
    d = [r['common_raw_accuracy_pct'] - r['E4_raw_accuracy_pct'] for r in rows]
    average = statistics.mean(d)
    halfwidth = T_975_DF9 * statistics.stdev(d) / math.sqrt(10)
    interval = [average - halfwidth, average + halfwidth]
    protection = {}
    for reference in ('BASE', 'I4'):
        harm = [r['common_calibrated_NLL'] - r[reference + '_calibrated_NLL'] for r in rows]
        protection[reference] = dict(paired_NLL_harm=harm, equal_seed_mean_harm=statistics.mean(harm),
                                     mean_limit=.02, every_block_limit=.05,
                                     protected=statistics.mean(harm) <= .02 and max(harm) <= .05)
    noninferior = interval[0] >= -MARGIN_PP
    protected = all(r['protected'] for r in protection.values())
    return dict(complete_pairs=True, seeds=list(SEEDS), paired_accuracy_difference_pp=d,
                equal_seed_mean_difference_pp=average, paired_two_sided_95_CI_pp=interval, df=9,
                noninferiority_margin_pp=MARGIN_PP, accuracy_noninferiority=noninferior,
                confidence_protection=protection, protected=protected,
                material_private_builder_necessity_refuted_within_scope=noninferior and protected,
                interval_supports_E_accuracy_advantage=interval[1] < 0,
                interval_supports_E_advantage_exceeding_margin=interval[1] < -MARGIN_PP,
                interval_crosses_noninferiority_margin=interval[0] < -MARGIN_PP <= interval[1],
                original_scores_recalculated=False, same_graph_attribution_not_unused_confirmation=True,
                pure_tensor_tying_cause_or_new_primitive_claim=False, automatic_followup_admission=False)


if __name__ == '__main__':
    raise SystemExit('Inactive source: root supplies future admission/API objects; no standalone launch or PASS.')
