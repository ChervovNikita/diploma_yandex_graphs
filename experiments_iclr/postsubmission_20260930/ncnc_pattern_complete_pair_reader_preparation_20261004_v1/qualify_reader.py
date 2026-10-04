"""Arithmetic/scope qualification on explicit fabricated scalar metadata only."""
import copy
import json
import math
from pathlib import Path
import reader


def source_row(index, loss):
    zero_slots, one_slots = index + 2, 1
    strata = {}
    for name in reader.STRATA:
        count = 65536 if name == 'all' else 1 if name in ('has_synthetic_removal', 'no_synthetic_removal', 'has_common') else 65535
        strata[name] = dict(queries=count, joint_query_sum=count * loss,
                            factorial_query_sum=count * (loss + .2),
                            joint_minus_factorial_query_sum=count * -.2,
                            between_component_spread_query_sum=count * .01)
    return dict(queries=65536, active_queries=2, residual_slots=zero_slots + one_slots,
                synthetic_removed_observed_slots=one_slots, source_unobserved_slots=zero_slots,
                component_entropy_query_sum=.4, between_component_spread_query_sum=.02,
                active_responsibility_entropy_sum=.8, active_responsibility_max_sum=1.6,
                normalized_native_q_vs_stable_sigmoid_max_abs=0., strata=strata,
                bit_marginal={'0': dict(slots=zero_slots, nll_sum=zero_slots * index,
                                        brier_sum=zero_slots * .1, predicted_observation_probability_sum=zero_slots * .2),
                              '1': dict(slots=one_slots, nll_sum=.3, brier_sum=.4,
                                        predicted_observation_probability_sum=.5)})


def fixture():
    identity = dict(driver_manifest_sha256=reader.DRIVER_SHA, family_id=reader.FAMILY)
    arms, selections = {}, {}
    for arm, hits, loss in [('J', .55, .1), ('F', .56, .2)]:
        selections[arm] = dict(order=25, hits50=hits)
        valid = dict(served_hits50=hits, member_hits50=[.5] * 4,
                     positive_coincident_error_fractions=[.25] * 6, positive_oracle_union=.7,
                     member_balanced_BCE=[.8] * 4, served_balanced_BCE=.7,
                     positive_queries=60084, negative_queries=100000,
                     not_a_selector=True, graph='complete_TRAIN_only', serving_pool='mean_raw_logits',
                     fixed_bank_counterfactual_routes={r: dict(served_hits50=hits, served_balanced_BCE=.7) for r in reader.ROUTES})
        arms[arm] = dict(selected_epoch=25, VALID=valid,
                         mask_event_batches=[dict(native_member_main_loss=[1.] * 4,
                             positive=source_row(i, loss), negative=source_row(i, loss)) for i in range(1, 18)],
                         source_pattern_mask_wall_seconds=1., TRAIN_stream={'fabricated_only': True},
                         support_digests=['FABRICATED'] * 34, source_teacher={'fabricated_only': True},
                         mask_seed=2026100307, new_optimization_or_selection=False, auxiliary_scorer_backward=False)
    pair = dict(schema='ncnc-pattern-pair-results-v1', identity=identity, selections=selections,
                representation_diagnostics=arms, all_100_native_streams_RNG_and_actual_supports_match=True,
                test_file_opened=False, scope='single_seed_validation_selected_development_pilot_not_confirmatory',
                source_pattern_scope='TRAIN_observation_incidence_not_latent_link_truth',
                stronger_claim_requires_covariance_aware_competent_single=True,
                selected_served_VALID_Hits50_J_minus_F=.55 - .56)
    closure = dict(identity=identity, preclosure_bound_stage_accounting={'fabricated_only': True},
                   inclusive_accounting={'fabricated_only': True}, cost_scope='FABRICATED_ONLY')
    terminal = dict(fabricated_only=True)
    return pair, closure, terminal


def main():
    pair, closure, terminal = fixture()
    report = reader.assemble(pair, closure, terminal)
    numerator = sum((i + 2) * i for i in range(1, 18))
    denominator = sum(i + 2 for i in range(1, 18))
    value = report['arms']['J']['positive_mask_events']['bit_marginal']['0']['nll']
    assert math.isclose(value, numerator / denominator, rel_tol=1e-12)
    assert not math.isclose(value, 9., rel_tol=1e-12), 'Unequal bit supports cannot use an unweighted mean of batch means'
    assert report['arms']['J']['positive_mask_events']['totals']['queries'] == 1114112
    assert report['confidence_intervals_or_significance_claims'] is False and report['seed_count'] == 1
    assert report['primary_interpretation'].startswith('No quality support'), 'Better reconstruction cannot rescue lower ranking'
    reader.render(report)
    cases = [dict(case='asymmetric_slot_weighted_arithmetic_and_adverse_primary', status='PASS')]
    for name in ('incomplete_batches', 'wrong_selection', 'missing_route', 'wrong_delta',
                 'unmatched_stream', 'nonfinite_diagnostic', 'TEST_opened', 'wrong_mask_seed'):
        bad = copy.deepcopy(pair)
        if name == 'incomplete_batches': bad['representation_diagnostics']['J']['mask_event_batches'].pop()
        if name == 'wrong_selection': bad['representation_diagnostics']['J']['selected_epoch'] = 26
        if name == 'missing_route': del bad['representation_diagnostics']['F']['VALID']['fixed_bank_counterfactual_routes']['crossed_cyclic_2']
        if name == 'wrong_delta': bad['selected_served_VALID_Hits50_J_minus_F'] = 0.
        if name == 'unmatched_stream': bad['representation_diagnostics']['F']['TRAIN_stream'] = {'different': True}
        if name == 'nonfinite_diagnostic': bad['representation_diagnostics']['F']['VALID']['member_balanced_BCE'][0] = float('nan')
        if name == 'TEST_opened': bad['test_file_opened'] = True
        if name == 'wrong_mask_seed': bad['representation_diagnostics']['F']['mask_seed'] = 7
        try: reader.assemble(bad, closure, terminal)
        except RuntimeError: cases.append(dict(case=name, status='PASS', invalid_scope_rejected=True))
        else: raise RuntimeError('Invalid fabricated case was accepted: ' + name)
    positive = copy.deepcopy(pair)
    positive['selections']['J']['hits50'] = .57
    positive['representation_diagnostics']['J']['VALID']['served_hits50'] = .57
    positive['representation_diagnostics']['J']['VALID']['fixed_bank_counterfactual_routes']['own']['served_hits50'] = .57
    positive['selected_served_VALID_Hits50_J_minus_F'] = .57 - .56
    success = reader.assemble(positive, closure, terminal)
    assert 'heldout confirmation still required' in success['primary_interpretation']
    assert success['manuscript_acceptance_or_general_superiority_established'] is False
    cases.append(dict(case='positive_screen_does_not_promote_confirmation_or_acceptance', status='PASS'))
    out = Path(__file__).parent / 'LOCAL_SCALAR_QUALIFICATION.json'
    with out.open('x') as handle:
        json.dump(dict(schema='ncnc-complete-pair-reader-fabricated-scalar-qualification-v1',
                       status='PASS', cases=cases, case_count=len(cases), fabricated_inputs_only=True,
                       scientific_data_checkpoint_TEST_outcome_access=False, numerical_libraries_imported=False,
                       file_pin_and_production_main_path_executed=False,
                       scope='Pure scalar aggregation and scientific interpretation guards only; not a production source/runtime/closure admission'), handle, indent=2)
        handle.write('\n')
    print('FABRICATED_SCALAR_READER_QUALIFICATION_PASS cases=' + str(len(cases)))


if __name__ == '__main__':
    main()
