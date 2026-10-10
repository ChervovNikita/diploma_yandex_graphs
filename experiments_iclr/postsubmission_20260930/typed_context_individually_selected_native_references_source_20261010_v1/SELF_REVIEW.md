# Separate native-reference source self-review

## Scope and decision

Source preparation for the two reference requirements already frozen in V3. All capabilities and release templates remain disabled. No source module or numerical provider was imported, no model/data/checkpoint/raw array was opened, no forward/fit/server action ran, and no current candidate or initializer outcomes were used. Stdlib AST compilation does not execute the source.

The constructor is separate; V3's paired TRAIN objective, complete VALID evaluation, own selector, checkpoint state and exact restoration methods are inherited without changes. A static attribute check found every attribute needed by those methods. This is a source-contract finding; it does not establish numerical compatibility or performance.

## Fixed references

| Property | Ordinary native independent4 | Contextual native independent4 |
| --- | --- | --- |
| Native body | Fresh complete native SeHGNN for each body | Fresh complete native SeHGNN for each body |
| Native inputs | All25 feature and12 half-positive label channels | Identical feature/positive channels |
| Added context information | None consumed by the native body | Exact three masses and15 derived balances |
| Point factors | None | Exact six adapter sites |
| Context generators | None | Exact V3 local multiplicative factory, two grouped sites, width32 per body |
| H initialization | None | Exact corresponding `_seed(generator_base, body_index, site)` |
| Native initial seed | Role optimizer seed+1009*body index | Same corresponding native initial seed |
| Optimizer and selector | Own Adam and complete own VALID two-context BCE | Own Adam and complete own VALID two-context BCE |
| Assembly | Four own-selected states, raw-logit mean | Four own-selected states, raw-logit mean |

The M1 adapter configuration is deliberately used as a scientific per-body implementation in this separate reference extension. Its six-site operations are unchanged; no V3 single128 control is substituted. A separate root scientific release governs repeated updates. The source extends the adapter's original M1 witness role without modifying the sealed adapter.

The four native initialization digests must differ within each reference, and corresponding ordinary/contextual digests must match. These references also change initialization and checkpoint selection relative to V3's copied-initial-state, commonly selected untied control. A result against them alone cannot identify a pure causal effect of tying. The ordinary reference lacks the extra mass information; only the contextual reference is same-information.

## Training, selection and replay

Frozen roles/context halves are prepared before setting the distinct native initialization seeds. Every TRAIN target is supervised once using its complement context. Each body trains all native/head/factor/generator parameters, with unscaled own BCE and its own Adam. Its dropout stream uses the corresponding V3 member seed. Four private AMP scaler lanes per family persist across ordered roles; replay uses a fresh separate scaler.

The unchanged selector chooses strict minimum complete VALID BCE, keeps the earliest tie, permits200 epochs and stops when epoch-best_epoch>50. M1 serving averages both frozen contexts' raw logits before its selector. TRAIN diagnostics use the complement only. Assembly takes exactly the four selected body logits in body order0–3, computes a CPU FP32 raw-logit mean, then sigmoid>.5. There is no ensemble assembly search or fitted pool.

Fresh replay restores full native weights/buffers, Adam, scaler and master/member streams. It checks pool/member/per-context/probability drift<=.001 and zero prediction changes, plus the complete VALID two-context roster. Floating output equality uses V3's declared tolerances. Fit cleanup restores and exactly checks its caller/master stream; cleanup failures are retained as failed body records.

## Authority, coverage and costs

Default CLOSED guards precede deliberate reads/work. The released execution route checks the literal allocation hostname/path/sole GPU before source/release reads, then pins this source/protocol, unchanged V3, exact interpreter/environment/providers, existing full native and complete-six V3 qualification descriptors, complete development inputs and named frozen roles. Root source approval, data/runtime approval and the scientific-fit flag must match the action. A fresh output is required. Original source, qualifier and scores are untouched.

Optional reference-only qualification covers all eight one-update/replay body paths at role1 and both assemblies. Full scientific mode requires all24 body fits and six banks across fixed roles1/2/3. No successful subset, automatic retry or model/task shrink is implemented. The existing V3 qualification is consumed by hash-bound descriptor; no repeated V3 qualification is requested.

Per-body fit/selector/snapshot/replay costs use the existing event logger. Assembly and payload serialization have explicit cost events. Cohort wall/CPU/cumulative RSS includes common full graph preparation, and body records include fit/replay CUDA peaks and full model/Adam storage accounting. Nested events must not be summed. Qualification records explicitly state one update per body and do not claim full-fit costs. Selected states and committee tensor payloads remain server-side.

Root retains responsibility for the separate finite owner, external resource/log/output bounds, direct child cleanup and release activation. This packet contains no owner or CLI.

## Verification and limits

STATIC_CHECKS.json contains45 passing stdlib AST/compile/hash/JSON and source-contract checks. The actual V3 protocol hash was read from local bytes, settling the summary's ambiguity. All21 V3 sealed source payloads remained unchanged. No actual numerical qualification or scientific scores were inspected during this preparation.

Independent root source review and qualification of the new constructor/replay/assembly paths remain necessary before their numerical output can support an analysis. No competence, superiority, broad generalization, novelty, efficiency or manuscript verdict is established by this source packet. The historical current-family analysis is complete and was not reopened here.
