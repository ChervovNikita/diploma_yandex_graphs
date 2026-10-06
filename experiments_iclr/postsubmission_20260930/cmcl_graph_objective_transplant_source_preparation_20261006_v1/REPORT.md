# Disabled CMCL objective transplant on the existing native bank

**Source/mathematical feasibility and fixture plan only.** Reuses the saved authenticated CMCL close design; no primary method reread, synthetic/native execution, stage, actual role/label/checkpoint/prediction payload access, fit or held scoring occurred. H16utility and full native references retain GPU priority. No runtime wrapper or training loop is added.

## Fixed loss helper

`cmcl_loss.py` (4330 bytes SHA `4f04eb87381512c9f7506b6458917b83fd12228f3c7651c6291e902c99fe6ac9`) keeps M4/K3/beta.75. For each supplied node and member, c=-log P(y), d=KL(U5||P)=-log5-mean(log P), and score a=c-.75d. Select the three smallest scores with stable index ties0/1/2/3; stop discrete ownership. The loss is **sum-owner CE +.75 sum-nonowner KL**, summed across members and averaged across nodes. There is no M/K division, all-member CE anchor, added pool term or optional feature sharing. Total training loss is (meanS+meanR)/2. Serve the arithmetic mean of all four class-softmax probabilities.

For owner set O, beta*sum(d)+sum_O(c-beta*d) proves the ranking rule. On a fixed branch, logit derivatives are P-e_y for owners and beta*(P-U5) for nonowners. Current ranking uses present loss/confidence, not a measured private learning step. At exact ties the helper follows a predetermined branch; it claims no unique derivative or global smoothness.

The helper uses supplied logits/targets only, with stdlib math at top level and numerical imports inside deliberate helper calls. Public training/serving entry points remain disabled; underscored pure engineering entry points require explicit engineering_authorized=True. No source import or numerical call was made during preparation.

## Native context and reuse boundary

A future reviewed caller must use the exact existing full24492x300 FP32 native shared4 bank, fixed native edges, commonW400 last W-only/seed17 state and same permitted S/R labels. Existing source/context/input identity gates are recorded in SOURCE_BINDINGS. S/R losses must come from the same incoming full logit tensor; shared theta and private rows receive one simultaneous plain-SGD commit at existing eta_core.001/eta_private.01. The fixedH16 endpoint is prospective, with no acquisition, extra stage or selector here.

The reviewed ordinary objective helper only supports own/own_pool and has no CMCL owner/KL operation, so it is not a direct loss drop-in. Preserve its sealed bytes and reuse its exact native/context/serving contract and existing SGD equation later. This packet does not select batched versus value/cotangent replay, choose resource caps, build a native caller or add a generic supervisor. Normal native gradients/RNG/resources remain unqualified for this new objective.

## Independent hand-calculable fixture plan

SYNTHETIC_FIXTURE_PLAN supplies six unexecuted closed cases: uniform exact K3 ties and member-sum scaling; rational probabilities with closed owner/loss formulas; KL direction and analytic gradient; a predetermined confident-wrong example where CMCL ranking differs from CE-only ranking; unequal S/R sizes with equal aggregate role mass; and full arithmetic probability serving. References are fractions/log identities and analytic P-e_y/beta*(P-U), not a mirrored helper. A uniform nonowner's zero derivative detects an accidentally retained all-member CE anchor. Native data/models and fitted states are unnecessary for these future small fixtures.

## Provenance and limits

Authenticated author objective is cmcl_v0, commit57f41f4b166b8544c56580f9d32ee98b997e3f59/src/model.py113–131. Exact committed Apache2.0 LICENSE and copyright notice are retained. The helper uses exact log_softmax KL rather than the author's1e-10 probability clamp, so extreme-probability behavior differs. Optional masked feature exchange is off; this does not remove the pre-existing tied core. M4/K3, equal roles, commonW400, native graph and plain SGD/H16 are explicit transplant choices; image architectures/momentum/weight decay/schedules, best feature-sharing/stochastic-label results and exact2017 revision are not reproduced. K3 is published overlap, beta.75 an authenticated default/example, neither a graph optimum. Prior K1 history remains preserved and superseded as the sole recommended close comparator; no new K/graph-vote variant is supplied.

Fair comparison uses equal allowed W/S/R information, complete fixed probability serving, all outcomes and actual resource bills, with no A/VALID/TEST-dependent choices. Directly using R labels in CMCL changes supervision timing/objective relative to the finite-response learner; it is a close conventional-confidence training challenge, not a matched finite-remainder ablation. Same H16 is not equal compute or equal gradient scale. CMCL uses a member-summed loss while ordinary own/own_pool use their declared normalization; do not silently change either to make a favorable comparison.

A CMCL gain can arise from ordinary owner CE correction or nonowner confidence reduction. Complete pooled net repairs and member competence must distinguish correction from lower confidence/diversity alone. If CMCL reproduces a finite-live gain, response-specific attribution is unsupported by that contrast. If finite-live wins, matched live first-order utility and competent own/own_pool/SINGLE/ENS4 references are still needed; CMCL alone cannot prove the finite remainder, graph causation, sharing necessity, novelty or universal advantage.

AST/JSON/hash preparation is complete; independent mathematical/source review and an independently admitted tiny synthetic qualification remain pending. No numerical or fit authority is granted.
