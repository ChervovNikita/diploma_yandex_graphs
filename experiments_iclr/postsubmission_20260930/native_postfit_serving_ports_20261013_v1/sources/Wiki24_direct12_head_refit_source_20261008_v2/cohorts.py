"""Native fixed cohorts and explicit correctness/rival diagnostics; no I/O."""
def masks(np, logits, probability, pooled, y):
    correct = logits.argmax(-1) == y[None]
    truth = np.take_along_axis(logits, y[None, :, None], axis=2)
    rivals = (logits > truth).all(axis=0)
    pool_correct = pooled.argmax(-1) == y
    return dict(full_population=np.ones(len(y), dtype=np.bool_), pooled_correct=pool_correct,
                pooled_wrong=~pool_correct, any_member_correct=correct.any(0),
                all_members_wrong=~correct.any(0), strict_common_wrong_rival=rivals.any(1),
                prior_common_rivals=rivals, member_correct=correct)


def transitions(np, native, fitted, native_logits, fitted_logits, y, cohort):
    bc, fc = native['pooled_correct'], fitted['pooled_correct']
    ba, fa = native['any_member_correct'], fitted['any_member_correct']
    rivals = native['prior_common_rivals']
    truth = np.take_along_axis(fitted_logits, y[None, :, None], axis=2)
    # One SAME candidate member must beat EVERY original common rival.
    route_beats_all = ((truth > fitted_logits) | ~rivals[None]).all(-1)
    eligible = native['all_members_wrong'] & native['strict_common_wrong_rival']
    order_reversal = eligible & route_beats_all.any(0)
    strict_repair = eligible & (route_beats_all & fitted['member_correct']).any(0)
    native_combined = native['all_members_wrong'] & native['strict_common_wrong_rival']
    fitted_combined = fitted['all_members_wrong'] & fitted['strict_common_wrong_rival']
    values = dict(pooled_repairs=~bc & fc, pooled_harms=bc & ~fc,
                  member_coverage_repairs=~ba & fa, member_coverage_harms=ba & ~fa,
                  common_rival_order_reversal=order_reversal, strict_common_rival_repair=strict_repair,
                  introduced_common_rival_error=~native_combined & fitted_combined)
    return dict(nodes=int(cohort.sum()), **{key:int(value[cohort].sum()) for key,value in values.items()})
