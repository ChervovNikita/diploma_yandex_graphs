# Degree conditioning and the graph mixture objective

Fixing how many neighbours are selected lets the auxiliary learn **which neighbours** form the observed pattern. It does not by itself remove a preference for popular candidate nodes, establish new latent-community modelling, or guarantee different ensemble members.

## Primary prior inspected

Karrer and Newman, *Stochastic blockmodels and community structure in networks*, arXiv:1008.3926v1. Their degree-corrected stochastic block model gives independent Poisson edges mean `theta_i theta_j omega[g_i,g_j]`. The fitted parameters preserve expected node degrees and expected edge counts between groups. Degree heterogeneity, latent graph blocks and their joint likelihood are established prior. The read is scoped to the model and likelihood discussion; published performance and their optimization algorithm were not adopted.

Our auxiliary differs in scope. It models two local binary subset observations under fixed counts, with a shared mixture member for one link. It is removed from the served predictor. These distinctions identify an experimental construction; they do not prove its novelty or effectiveness.

## What count conditioning actually cancels

For a candidate set `C`, selected identities `S`, and fixed count `k`, the component law is

`q(S | k,C) = exp(sum_{w in S} eta_w) / sum_{T subset C, |T|=k} exp(sum_{w in T} eta_w)`.

Adding a common offset `c` to every candidate logit multiplies numerator and denominator by `exp(k*c)` and cancels. This is standard conditional exponential-family algebra. In particular, the law cannot learn a per-side scalar intercept from these labels.

Candidate-specific offsets do not cancel. With two candidates, count one, and `eta_w = log(degree(w)+1)`, the odds of selecting candidate1 rather than candidate2 are `(degree(1)+1)/(degree(2)+1)`. Count conditioning therefore does not make the auxiliary degree invariant. Neighbourhood sizes, candidate identities, masking and the served encoder can all retain degree effects.

Conditioning the complete Poisson graph on **every node's degree sequence** is a different operation. The `theta_i` factors then become constant across feasible graphs, whereas our local subset constraints do not fix candidate-node degrees. Do not conflate these two conditionings or describe the current objective as a new degree-corrected block model.

## How this affects the next experiment

The DDI census supplies many more queries with nonconstant local subset choices. It establishes opportunity, not learned association. Dense degree and triangle patterns may still explain a fitted auxiliary. Evidence of low auxiliary loss alone would therefore be insufficient.

Before DDI prediction scores are accessed, the diagnostic plan fixes uniform and visible-degree subset references, member responsibility concentration, and endpoint responsibility overlap. The primary prediction comparison remains joint versus separately mixed endpoints and a competent ordinary target objective. A method advantage must appear in the unchanged served ranker and survive independent confirmation; these diagnostics cannot replace that requirement.

If members converge to identical component laws, responsibilities are uniform and joint equals separate. A quality gain in that case supports an ordinary reconstruction effect, not member specialization. Existing failures and this interpretation must remain available regardless of the outcome.

No new theorem, degree invariance, community recovery, diversity guarantee, prediction gain or exhaustive prior-art gap is claimed.
