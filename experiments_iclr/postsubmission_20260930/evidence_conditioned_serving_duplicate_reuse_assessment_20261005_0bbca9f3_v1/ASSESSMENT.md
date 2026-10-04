# Evidence-conditioned serving: duplicate and reuse assessment

## Decision

**Reject this as a new methodological direction.** The exact structural-likelihood gate is already derived in `conditional_pattern_serving_alignment_note_20261004_v1/NOTE.md`, dated 4 October. Its status is derived and untested, not an implemented or empirically failed served model. No new theorem, experiment family or numerical harness is justified by renaming it “evidence-conditioned prediction.” The needed observation protocol and predictive evidence were already identified as missing.

The useful remaining distinction is a future, attributed implementation question. It has no demonstrated advantage over ordinary independent4 or a capable structured single. Root's latest instruction narrows this deliverable to duplication/reuse and stops another proposal.

## Exact existing probability rule

The retained note defines base context x **excluding** realized structural evidence E, a uniform latent member m, normalized q_m(E|x), and p_m(x)=P(Y=1|m,x). It assumes Y and E are conditionally independent given m,x. Its normalized joint and served conditional are already:

`P(Y=y,E|x) = (1/4) sum_m q_m(E|x) p_m(x)^y (1-p_m(x))^(1-y)`;

`rho_m(E,x) = q_m(E|x)/sum_j q_j(E|x)`;

`P(Y=1|E,x) = sum_m rho_m(E,x) p_m(x)`.

Each component integrates/sums to one, hence the joint normalizes. Inference uses E and x, never true Y. A joint NLL would train `-log P(Y,E|x)`; conditional BCE would train the displayed conditional probability and need not fit absolute evidence likelihoods. They differ from today's independent member BCE plus reconstruction auxiliary. Neither objective or Bayes' rule is new.

For the count-conditioned implementation, q_m is the product of **whole-endpoint conditional subset laws**, mixed only after multiplying those two laws. The `member_nll` and `rho` source use the untempered sum of side NLLs. Per-slot normalization is a training weighting; exponentiating a divided NLL would change the posterior. V5's F instead mixes each bit independently and is not the whole-endpoint product comparator.

The served expression is a probability mixture. `sigmoid(mean raw logits)`, mean raw logits, and a gate-weighted raw-logit pool do not implement it. Balanced sampled query labels do not establish calibrated population future-link probabilities. No TEST calibration is permitted.

## Why the current setup cannot supply its evidence

On unmasked complete TRAIN, native residual candidates have no TRAIN-observed counterpart link: the conditional teacher subsets have k_L=k_R=0. Every member likelihood is one, so rho is uniform. Nonzero native soft completion scores do not change this fact. Applying the existing teacher law directly at serving provides no nontrivial structural gate. A uniform probability pool can still differ from the current mean-logit pool, but that is a readout change, not evidence-conditioned posterior utility.

The retained amendment already requires a new artificial visible mask/view and observable reconstruction subset. Such a view must remove the query and reverse edge, choose supports without target-label knowledge, separate base topology from held-back evidence, and pay every graph/likelihood pass. Feeding full topology into x and then multiplying a nondegenerate density for its deterministic extracted E cannot justify new-information or calibrated generative-posterior claims. Repeated counterpart edges must be one observation coordinate, and multiplying overlapping local likelihoods is not a normalized whole-graph density. These gaps explain why the note was not executable; they do not document an adverse gate experiment.

NCNC v5/current exact-CB source uses complete-TRAIN mean-logit serving, never builds an inference evidence partition, and never feeds structural responsibilities into target prediction. Target BCE still trains all members uniformly. Existing specializations are not automatically label-conditional experts. The gate therefore requires changes to the observation provider, target objective, selection and inference; it is not an inference toggle on the current protocol.

## Known prior and counterexamples

The retained NOTE already supplies the probability-mixture derivation, standard Brier/log-loss information identities, a helpful toy example, and the counterexample that identical p_m make even perfectly informative structural responsibilities useless. Identical q_m or extreme supports yield uniform responsibilities; wrong structural-to-target alignment can harm prediction. Full-context determination of E leaves no additional information. The shared-latent packet gives strict structural-fit improvement with an unchanged served query score.

Closest retained ancestry: GRAN supplies shared graph-pattern mixtures; Newman–Leicht/Amini supply neighbourhood responsibilities and degree-conditioned likelihoods; MMSB supplies graph latent-role compatibility. The label-conditioned scout reduces the J/product Bayes factor to soft responsibility compatibility. Subgraph diffusion already uses class-conditioned structural likelihood for LP; inference must evaluate hypotheses rather than read true Y. Link-MoE (index-v55 record79) uses structure-conditioned pair gates over complete experts, albeit a weighted raw-score/sigmoid pool and second-stage supervision. BGCN/LDS supply nonlinear predictive marginalization; Graphite/PIFM/KREPE/ARK supply strong structured alternatives. Generic reconstruction, posterior gating, latent compatibility and motif completion are already considered ingredients. The exact neural composition is not globally novelty-cleared by these limited scopes.

## Duplicate fit classification and independent4 reuse

The earlier six-fit continuation in `conditional_endpoint_pattern_scientific_assessment_20261005_1ddec4e8_v1/CONTINUATION.json` must **not** be dispatched as a fresh family. Its seed0 J_K and J_K_sep cells duplicate the existing nine-fit exact-CB protocol's seed0 `joint` and `separate`: same fresh F4 width64/init, lambda1, native masking, 100 epochs/1,700 updates, all-query reductions, private route and selection. Seven copied model/data/evaluation helpers are byte-identical to v5, and the public conditional-mixture function has the same AST as core-v3. Bucketed normalization is an implementation successor, not a new scientific arm. This new assessment supersedes that continuation recommendation without editing the old sealed packet or current queue.

The proposed four native fits at seeds {0,5,10,15} also repeat the completed ordinary independent4 seed0 bank. Reuse its completed complete-family reference when the inference contract is unchanged; no automatic refit and no donor states for the existing exact-CB queue.

Source eligibility is partial, not an outcome adoption: both protocols use the official time split, width64 native NCNC recipe, native full-record batch masks, 100 epochs, same Adam rates, strict complete shared-pool VALID Hits50 and raw-logit serving. Differences remain: ordinary training has deterministic algorithms False while v5/CB explicitly enables them; the independent selector has 100 synchronized banks plus one individually selected bank, versus 100 F4 candidates. It is a competent historical ordinary reference, not an identical-stream paired cell. Saved complete-family source/data/runtime/selected-state pins and complete-cohort closure must bind any later reuse; no receipt payload, state or score was opened here.

A future gate with target-excluded mask views/probability pooling is a **different inference contract**. Original saved predictions cannot be reused as matched gate controls. Existing checkpoints might support a separately bound fixed-bank diagnostic, but cannot establish fresh fitting/selection competence or calibration on that new context. The source scoring graph is complete TRAIN and does not implement universal queried-pair removal; absence of target-pair overlap is not assumed. The existing protocol also records consumed Collab TEST, which cannot be fresh confirmation.

## Stop condition

No new experiment is frozen or authorized. If this already considered idea were ever reopened for an explicit empirical reason, it would require matched uniform probability pooling, a same-evidence discriminative gate, ordinary independent4 and a capable full-context structured single under one paid budget and observation/selection contract. Better structural NLL or concentration would not replace gains over both required quality references. This is a retained evidentiary requirement, not a proposed control set or another training queue.

Only this new packet was written. No server, predictive outcomes, graph/checkpoint payloads, canonical edits, scientific execution or new agents were used. Incidental existing nonpredictive diagnostic quantities in the protocol were not used to reach this decision.
