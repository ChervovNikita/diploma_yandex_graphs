# Independent review of the all-COMMON-rivals diagnostic

8 October 2026. Static source/method review plus two artificial CPU checks. No scientific predictions, model inputs, checkpoints, partial scores or server access.

## Verdict

The all-rivals adapter correctly implements **one same candidate member strictly beating every old COMMON rival**, separately from **each rival being beaten by some possibly different member**. Empty rival sets are excluded. Actual member correctness, native pooled correctness, paired repairs/harms and new rival identities are separated. No substantive flaw was found in these core calculations.

It is a supplemental diagnostic, not a replacement for the sealed whole-population quality readout or frozen gates. Three material reporting risks need explicit handling. Preserve sealed sources and gate decisions; add interpretation notices or a separately sealed successor, as root has proposed.

## 1. Some sealed fields named “repairs” are candidate-correct counts

In [readout.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/context_positive_closed9_readout_source_20261008_v1/readout.py:70>), `acquired_and_served_repaired` uses acquisition AND candidate pool correctness. At line72, `served_repairs_on_baseline_common_competitor` uses eligibility AND candidate pool correctness. Neither explicitly requires the baseline pool to be wrong. The underlying paired repair/harm contract does require that condition, as does [adapter.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/context_positive_all_common_competitors_CPU_adapter_20261008_v1/adapter.py:54>).

In exact arithmetic, a class above truth in every member is also above truth in the mean-probability pool, so the omitted condition is redundant. Floating probability rounding and native tie resolution can defeat the strict implication. The adapter already records baseline common-rival/pool-correct coincidences rather than rejecting them.

An independent synthetic check demonstrates the distinction: all four float32 logit vectors give a rival the smallest positive subnormal advantage over truth. Softmax probabilities round to a tie and the truth index wins native argmax. A candidate strictly reverses that rival while its pool remains correct. Actual paired repairs are zero; both named sealed “repair” fields count one. This proves a possible semantics mismatch, not its occurrence in any scientific archive. No native torch replay or microscopic rejection threshold is inferred.

**Handling:** retain the sealed values with a notice that they measure candidate correctness/acquisition-and-correctness if baseline coincidences exist. Use explicit baseline-wrong AND candidate-correct masks for actual repair claims. Report baseline coincidences and kept-correct objects separately. A successor can supply those distinctions without altering sealed results or gates.

## 2. Newly common rival counts are restricted to the old obstruction cohort

The adapter computes all new rival identities, then reports them under `COMMON_obstruction_population` with an `eligible` intersection ([adapter.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/context_positive_all_common_competitors_CPU_adapter_20261008_v1/adapter.py:81>)). These are correct scoped counts. They omit new obstructions on objects that had no common rival in COMMON, including initially correct objects.

A second artificial case creates a common rival and a served harm on an initially correct object with no old obstruction. Full-population harms count one, while the scoped new-common-object count is zero. The diagnostic's scope is behaving as written; calling that count “all newly common errors” would be misleading.

**Handling:** distinguish full-population appeared/cleared/persisted obstruction objects, full new rival class-object pairs, and replacement rivals within the old cohort. Reuse the sealed full-population common-competitor flows and add full new-class counts in a separate successor. Decreasing old obstruction support cannot substitute for whole-population repairs minus harms, NLL or competence. Class-object pairs and obstruction objects have different denominators.

## 3. COMMON-fixed and PERMUTED-baseline rank diagnostics differ

The all-rivals adapter freezes COMMON rival sets for both COMMON→ROUTE and COMMON→PERMUTED. The sealed readout has ROUTE−COMMON and ROUTE−PERMUTED contrasts. Its nested `pair` obtains rivals from `before` ([readout.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/context_positive_closed9_readout_source_20261008_v1/readout.py:56>)); consequently, its ROUTE−PERMUTED rank diagnostic uses PERMUTED's smallest qualifying rival. Its cohort masks still come from COMMON. In that contrast, eligibility is the intersection of a COMMON-defined cohort and the PERMUTED baseline's current common-rival mask.

**Handling:** label the anchor condition, transition direction, cohort origin and rival-selection rule on every table. The new adapter's ROUTE and PERMUTED counts are comparable on one COMMON-fixed population. They are not the same estimand as sealed PERMUTED→ROUTE rank/repair counts, and cannot be subtracted into paired transition counts without joint masks. Reversing one selected smallest-index class is weaker than eliminating all original rivals and is not class-label-invariant evidence of specialization.

## Remaining scientific limits

- Beating all old rivals is insufficient for correctness when another class blocks truth. It is unnecessary for a pool repair: different wrong members can beat different rivals and their mean can favor truth. Even reversing every old rival somewhere is insufficient for a pool repair. The existing four artificial examples correctly cover these distinctions and strict-versus-argmax ties; they were inspected, not rerun through the sealed writing script.
- Clearing every common rival only requires that each rival cease to beat truth in at least one member, possibly through a tie. It does not imply a correct member or pool. A changed or cleared obstruction is a rank statement; retain actual correctness, repairs and harms alongside it. An all-member-wrong pool rescue is possible once no single rival dominates truth in every member.
- The COMMON-frozen error population is enriched for COMMON failures. Its improvement is a valid conditional description but can coexist with new errors elsewhere. The complete population, all nine cells, fixed seed pairing and COMMON-before-candidate mask construction prevent favorable candidate cohort selection; they do not make the error cohort representative or remove checkpoint-selection bias on the reused development population.
- Counters overlap. Same-member acquisition, pool rescue, new rivals and class-object pairs cannot be added as disjoint explanations. Mean/worst member competence and pooled NLL remain essential. Pool-minus-mean-member accuracy is an accounting difference, not a causal measure of useful diversity. Raw margin size is sensitive to confidence/scale.
- The frozen numerical `stage1_gate` checks accuracy, NLL and member competence with the correct percentage-point conversion. It does not evaluate the separate frozen desired-mechanism statement about common-competitor support. A passing numerical gate must not be described as proof that this mechanism passed or competence-preserving context steering was established. Report mechanism success/failure separately; change no gate.
- Three optimizer seeds on one selected development graph, overlapping cohorts and repeated seed-node/class-node counts provide descriptive evidence. They do not supply independent graph replication, node-level significance, topology causality, or unused confirmation. ROUTE versus PERMUTED and the later competent references retain their frozen roles.

## Scope and immutable delivery

Reviewed adapter/readout sources and READMEs, their source bindings and static-check declarations, the frozen protocol, the prediction-analysis proposal, and targeted reused numerical-contract source passages. Did not call collection, either adapter's `run`, a model, the sealed synthetic script's writing entry point, or scientific inference. Two one-object synthetic checks used bundled NumPy and pure diagnostic functions only. The first system-Python attempt lacked NumPy and failed before any write. Source hashes and review files are bound in this folder; no adapter, sealed source, cohort, frozen gate or protocol was modified.
