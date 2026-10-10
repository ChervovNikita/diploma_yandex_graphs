# Private-row normalization under PubMed coupled Adam

**One fixed M-derived foundation control is justified.** This is a regularization/optimizer convention, not a new architecture or novelty claim. The algebra identifies a concrete mismatch; existing predictions do not establish that it caused harm. No numeric test, implementation, model/data/checkpoint read or job was performed for this assessment.

## Exact objective and Adam identity

Let shared coordinates be θ, private row m be φₘ, and factual mean TRAIN CE be F=(1/M)ΣₘLₘ. Write gₘ=∂Lₘ/∂φₘ. Qualified own4 uses native coupled Adam: its private input is **gₘ/M + λφₘ**, whereas native factorM1 uses **gₘ + λφₘ**. The inspected source sets native ε=1e−8, decay .001 except attention-name groups1e−8, original rates .005/.0005, and Adam β=(.9,.999).

Current coupled gradients belong to

    R₀ = F + ½Σⱼλˢⱼθⱼ² + ½Σₘⱼλᵖⱼφₘⱼ².

For a positive constant a and Adam input qₜ=a gₜ+dφₜ, factor out a. With zero initial history or transported states mₜ=a m̄ₜ, vₜ=a²v̄ₜ and identical bias-correction clocks,

    η m̂ₜ/(√v̂ₜ+ε) = η m̂̄ₜ/(√v̂̄ₜ+ε/a),
    where the unit-data input is gₜ+(d/a)φₜ.

Thus current own4 has **effective unit-data decay Mλ and ε=Mε_native**. Coupled decay changes the moment input and may change direction; this does not mean measured parameter shrinkage is four times larger.

| Private data / decay / ε | Unit-data equivalent decay / ε |
|---|---|
| g/M, λ, ε_native: current own4 | Mλ, Mε_native |
| g/M, λ/M, ε_native: decay only | λ, Mε_native |
| g/M, λ/M, ε_native/M: sole matched control | λ, ε_native |

The matched control uses the regularized objective

    R₁ = F + ½Σⱼλˢⱼθⱼ² + (1/2M)Σₘⱼλᵖⱼφₘⱼ².

Its private Adam update is exactly equivalent in real arithmetic to multiplying only private data gradients by M before native coupled Adam with native λ/ε. These are two implementations of one control; no second performance arm is needed. Shared mean factual gradients and shared native groups stay unchanged.

At M=4, private nonattention decay becomes.00025, attention decay2.5e−9 and private ε2.5e−9. All46 R/S bank tensors at23 maps change; original W, biases, norms, rates, initial unit rows and dropout streams do not. Coupled decay is toward zero factors, not unit factors. The epsilon-zero scale-cancellation limit requires defined nonzero denominators; without moment evidence, native epsilon cannot be declared negligible. AdamW's decoupled shrinkage is a different rule and is not substituted.

Equivalence requires matched realized gradients, clocks, zero/mapped histories, masks and missing-versus-zero-gradient semantics. It establishes equivalence between the two normalized implementations. It does not make shared own4 identical to M1/I4: their shared slow gradients, capacity and selectors differ. No old optimizer history may be retroactively relabeled as normalized training, and no finite-step competence or convergence guarantee follows.

## History and closest collision

The strongest local collision is already implemented native **private-B bias transport**: the covariance-initialization adapter scales copied bias first/second moments by1/K and1/K², epsilon and coupled decay by1/K, retaining shared native options. The same code explicitly gives new R/S factors empty state with unchanged native options. This is prior normalization knowledge, not a completed all23-R/S PubMed contrast.

The saved AdaTask assessment already states exact epsilon matching under gradient rescaling; it changes shared route moments rather than this private regularizer. The saved reciprocal-scalar assessment concerns coordinate/rate diversity and zero-decay WikiCS. Inspected WikiCS and SeHGNN Adam recipes use zero decay. Those outcomes cannot settle this nonzero-decay normalization. No completed exact current-factor comparison was found in the bounded source/history search; no global absence claim is made. Classical Adam scale behavior and coupled-versus-decoupled weight decay are the closest general optimizer ancestry; no new primary paper was read.

## Scientific decision

FactorM1 reproduces most apparent shared gain, and the completed factorized I4 acquires99/102/93 exact correct alternatives over its included M1 with nearly unchanged mean competence but many serving losses. This motivates checking a basic private-row learning convention before attributing shared error scarcity to architectural sharing. The strong own4 member accuracy also prevents claiming its existing decay is already proven harmful.

Retain exactly one own4 control with private λ/M and ε/M, compared to the fresh full18 own anchors, capable factorM1 and factorized I4. Preserve original full graph, factual CE, native rates/initialization/dropout, max2000/patience250 selection and serving. A complete null/adverse result closes usefulness within this scope; a gain changes a foundation baseline and still needs uncertainty and unused confirmation. Report pooled accuracy/NLL, every member/class risk, acquired/served/lost alternatives, exact repairs/new harms and actual costs. No coefficient grid, new ensemble claim or rescue of a failed auxiliary mechanism follows. Root has authorized source preparation separately; this note grants no run admission.
