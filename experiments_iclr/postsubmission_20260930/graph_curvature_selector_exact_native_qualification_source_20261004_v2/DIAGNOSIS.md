# Squirrel native engineering failure: source-only triage

## Established from the saved engineering report

The authorized Squirrel17 engineering child failed with `ValueError: Exact tensor state differs`. Its traceback identifies runner v1's `check_five_returns`, at the loop comparing each returned model's `state_dict` tensor with the independently reconstructed expectation. That comparison precedes this function's optimizer, RNG, mode, mean-logit and storage-independence checks.

The report records passes for serialized full-state/RNG custody, frozen-factor native/K4 Adam correspondence, full-graph native/K1/K4 identity, final-head affinity and head AD. Selector construction and its allotted trials completed. Those passes do not qualify the later returned-state comparison or the live-factor probe, which was not reached. No predictive continuation ran.

The failed report does **not** name the arm or tensor, nor provide differing values or their size. Therefore the supplied evidence cannot establish whether the rejection came from the reconstructed initialized head, an unchanged non-head tensor, or a shape/dtype mismatch. It cannot classify the failure as an oracle error or a method-state defect.

## Source comparison

The selector returns separate deep copies of its prototype, installs the selected **pretrial** head slices, reconstructs the original frozen optimizer and deep-copies the original RNG. Its trial function trains disposable copies and discards their states. No direct prototype-state mutation was found along that source path.

The independent oracle rebuilds the common center, the required graph/permuted/random basis and the selected pair from the receipt's radius. Its pair assembly matches selector `initial_metrics`: basis vectors are cast to the native center dtype **before** multiplying by the radius; the four offsets are `a, -a, b, -b`; and they are added to the center. Thus a different cast/multiply order is not a concrete oracle defect here. Both paths bind the final head and use the same common-only source initializer. The oracle's fixed reconstruction seeds match this runner's fixed seed17 scope.

This source inspection identifies no justified reconstruction or methodological repair. In particular, deterministic CUDA execution does not make the missing comparison path recoverable from the report. Independent recomputation through the native body also leaves numerical-reconstruction drift as a possible explanation; it has not been observed or measured by this triage.

## Runner v2 changes

V2 is a **diagnostic correction only**. It preserves the oracle, native bodies, selector, constants, schedules, every scientific check and the strict equality predicates. Each recursive comparison now carries a path such as `returned_arms['common_only'].model['head.R']`. On rejection, the report records the expected/actual tensor kind, dtype, shape and device; the number of unequal elements; the first differing index and values; and the maximum absolute difference. The last quantity explains the rejection and supplies no tolerance.

The exception remains a failed qualification and propagates to the existing runner error handler. The handler adds `exact_state_difference` to the failed report. Existing successful return values are unchanged. No extra model forward, selector trial, donor, warm update or retry is introduced. V1 and its failed report remain preserved.

The static check confirms that `check_five_returns` is identical after removing path keywords, and `run_qualification` is identical after removing those keywords and the exception-detail save. All other pre-existing functions except the equality reporter have identical ASTs. Exact tensor admission remains same dtype, same shape and `torch.equal(left.cpu(), right.cpu())`; dictionary, sequence and scalar admission are unchanged.

## What a later path would resolve

A named non-head mismatch would direct inspection to native/prototype custody and possible trial residue. A common-only head mismatch would direct inspection to the independently rebuilt common center versus the returned center. A selected-pair head mismatch would additionally implicate basis reconstruction, saved selection/radius or returned-slice custody. None of those paths alone would prove its cause; actual input/state comparison would still be needed.

This packet performed no numerical import or execution, server access, array/checkpoint inspection or predictive-score reading. It authorizes no rerun. The methodological candidate remains unqualified until the failed check and subsequent checks actually pass under a separately owned execution.
