# Independent source review: BLOCKED

Candidate manifest: `723988a209c21d86ded917c0109e7a122fe9620df75c78f8cc397d29a3227d2c`.

The exact disabled source was independently inspected. All 17 payload hashes and byte counts match. No source was imported, compiled or executed; no numerical module, raw data, features, state, scores, arrays, runtime binary or server was accessed. Only this new review folder was written. Execution remains unauthorized.

## Blocking finding F01

`prefix_observer.py:33–41` records aliases using Python object identity. Distinct tensor objects are independently CPU-cloned and receive value, device, layout, stride and storage-offset metadata. The schema does not record whether two such tensor paths share backing storage. The `fully_inspected` flag at line 71 only checks the opaque-object list.

A source-level witness is two equal-valued cache tensors with identical shape/stride/offset: one unit can hold two distinct views of a common storage, while another holds two independent allocations. Both records have distinct Python objects and identical recorded metadata. They therefore compare equal even though the storage alias graph differs. This witness follows from the schema; no numerical experiment or claim about actual runtime aliases was made.

`first_batch_repeat.py:123–144` can consequently label such a pair `MATCHED_COMPARISON`. The CPU Adam-step alias checks cover selected step scalars only. They do not close parameter, buffer or cache storage alias coverage. This prevents the requested full exposed alias comparison and the corresponding complete-state eligibility claim.

Closure requires a separate immutable source successor that compares canonical backing-storage/view relations for exposed tensor paths, or declares unsupported alias cases uninspected and isolates those pairs. Preserve this candidate and review. No in-place repair or execution release is authorized by this review.

## Source checks that passed

The pinned native training body calls `loss.backward()` immediately before `optimizer.step()`. The candidate wrapper delegates that backward exactly once, captures gradients, then raises `PrefixComplete`, a `BaseException` caught outside the original body. Its sentinel interrupts the body before the step call. Both optimizers also carry a prehook that raises before any unexpected optimizer body executes. The fresh factories contain no step call. Final accounting requires four complete backward prefixes and zero optimizer calls. This source sequence supports the zero-update design.

The four prefixes are prospectively fixed as A1, A2, B1 and B2, with two independent complete saved-state restore inputs. Within each unit only original entry RNG is reset; the original body establishes train flags and zero gradients. The native negative sampler and permutation iterator delegate their original calls/order/first batch. There is no cache/model/Adam reload between repeats. Numeric discrepancies are recorded after all prefixes and do not shorten them; structural or resource failures may stop collection.

The state authority, fixed tolerance, execution profile and caps match the diagnostic predecessor. TRAIN/feature hashes remain unchanged while input admission narrows to those two files. The original compare function retains candidate on the left and reference on the right. Values, flags, RNG, all registered buffers including nonpersistent ones, inspectable cache fields and object-identity aliases are checked. Mismatched or explicitly opaque states are isolated before numerical comparison. F01 limits the completeness of that check.

The ordinary supervisor physical helpers match reviewed native-control-v2. An independent AST comparison of its entire main function matches after exactly nine stage/path/status literal substitutions. The held-child identity/session/group mechanism, WNOWAIT exit observation, exclusive attempt lock, five-second cleanup, recorded-signal immediate stop and final drain, physical receipt before collection, and rejection of incomplete/cap-breaching outcomes are retained. The finite event/sampled caps and explicit final write/exception tails are accepted within their declared scope.

The candidate preserves 97 source/evidence descriptors and explicit earlier failed-shared4/native-exact-result preservation flags. Earlier numeric artifacts were not reopened. Genuine unchanged predecessor review evidence was reused for native helpers and physical supervision, then independently corroborated for the current source. The author AST checker was not treated as sole proof.

This is a bounded source review. Actual state identity, alias behavior, resource use and numerical repeatability remain unobserved. Observation/copy/hash/CSR work can alter scheduling or allocation history. Diagnostic agreement or variation cannot establish kernel causation, qualify the earlier full epoch, admit science or create a donor state.
