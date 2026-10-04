"""Stdlib-only proposed reference availability; no scoring or numerical entry."""
import re

LABELS = ('False_anchor', 'True1', 'True2')
POOLS = ('positive', 'negative')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def choose_authenticated_reference_labels(historical_digests, receipts):
    """Choose only byte-identical references among the three fixed call outputs.

    The future caller must first authenticate original selected states/journal,
    identical inputs, full snapshot restoration, and these actual scorer receipts.
    A label identifies its process-private full FP32 vector; this helper cannot
    reconstruct an array or infer error from a mismatching hash.
    """
    require(set(historical_digests) == set(POOLS), 'Both historical pool digests required')
    require(all(re.fullmatch('[0-9a-f]{64}', value) for value in historical_digests.values()), 'Invalid historical digest')
    require(set(receipts) == set(LABELS), 'Only three predeclared call slots allowed')
    for receipt in receipts.values():
        if receipt is None:
            continue
        require(receipt['positive_queries'] == 60084 and receipt['negative_queries'] == 100000
                and receipt['all_query_rows_complete'] is True and receipt['encoder_calls'] == 1
                and receipt['query_batches'] == [1, 1] and receipt['graph'] == 'complete_TRAIN_only'
                and receipt['serving_pool'] == 'mean_raw_logits', 'Incomplete original scorer receipt')
        require(set(receipt['score_digests']) == set(POOLS), 'Both actual pool digests required')
        require(all(re.fullmatch('[0-9a-f]{64}', value) for value in receipt['score_digests'].values()), 'Invalid actual digest')
    labels = {pool: next((label for label in LABELS
        if receipts[label] is not None and receipts[label]['score_digests'][pool] == historical_digests[pool]), None)
        for pool in POOLS}
    return dict(reference_labels=labels, historical_reference_available=all(labels.values()),
        unavailable_pools=[pool for pool in POOLS if labels[pool] is None],
        historical_error_inferred_from_mismatching_hash=False,
        status='AUTHENTICATED_HISTORICAL_REFERENCE_AVAILABLE' if all(labels.values())
               else 'HISTORICAL_RAW_VECTOR_REFERENCE_UNAVAILABLE')


def proposed_fp32_arithmetic_criterion():
    """Same numeric values as qualify.close; the application scope is new."""
    epsilon = 2.0 ** -23
    return dict(dtype='float32', rtol=128 * epsilon, atol=128 * epsilon,
        elementwise_rule='abs(candidate-reference) <= atol + rtol*abs(reference)',
        finite_same_shape_same_dtype_required=True,
        scope='Proposed numerical output audit against a byte-authenticated historical reference; requires explicit fresh protocol approval.',
        original_scope='Engineering arithmetic comparison; neither predictive success nor selected experiment threshold.',
        old_exact_raw_replay_contract_changed=False, selected_Hits50_tolerance=0)
