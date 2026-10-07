# MolHIV message factors as coupled private LayerNorm coefficients

The existing message factor is an established affine modulation primitive. With the actual default affine LayerNorm, multiplying its output by a member factor is exactly a restricted member-conditioned affine normalization at that layer. Possible value lies in how this primitive is trained and whether it produces useful complementary graph predictions. Neither a new normalization operation nor graph-conditioned initialization follows from the source.

## Exact local mapping

The [model source](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/portable_internal_be_public_interface_20261007_v2/core/models.py:122) constructs `nn.LayerNorm(d)` and then multiplies its output by `message_factors[layer, member]`. PyTorch defaults to trainable affine weight and bias, initialized to one and zero; statistics come from the current input in both TRAIN and serving modes ([official source, LayerNorm](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/normalization.py#L94)). For a fixed layer and member input:

```text
q_m,v = h_m,v + virtual_m,graph(v)
qhat_m,v = (q_m,v − mean_channels(q_m,v)) / sqrt(var_channels(q_m,v) + eps)
u_m,v = alpha_shared ⊙ qhat_m,v + beta_shared
x_m,v = c_m ⊙ u_m,v
      = (c_m ⊙ alpha_shared) ⊙ qhat_m,v + c_m ⊙ beta_shared
alpha_m = c_m ⊙ alpha_shared,   beta_m = c_m ⊙ beta_shared.
```

Each member computes its own input statistics; there is no member-specific running-statistic bank. Per channel, every member's `(alpha_m, beta_m)` lies on the same line through the origin. Consequently, gain and offset cannot vary independently across members. When the shared bias is nonzero, changing `c_m` changes both effective gain and effective offset. The coefficients are static across nodes and molecules, indexed by member, layer and channel.

This is an exact pointwise mapping at fixed parameters. It gives no optimizer equivalence to independently trained private LayerNorm coefficients: shared norm parameters receive own supervision, while I gives `c` the GNCL mixture plus `.05` alignment. Updating these products couples shared and private changes; native Adam moments depend on the original coordinates and histories. Unrestricted private affine normalization also permits coefficient pairs excluded by the proportionality constraint. Inspect effective `(alpha_m, beta_m)`, rather than `c` spread alone, because compensating shared rescaling can preserve the products.

At initialization, the source has `alpha_shared=1`, `beta_shared=0`, and **all message factors equal one**. The [initialization source](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/portable_internal_be_public_interface_20261007_v2/core/factors.py:54) randomizes only the stem input factor for `be_init`. Different upstream states and member stochastic streams can break symmetry; the message tensor is neither randomly diversified nor initialized from topology, graph labels, or graph-conditioned factor geometry.

## Modulation ancestry and the remaining graph effect

[FiLM](https://arxiv.org/abs/1709.07871), method and conditional-normalization discussion, already treats featurewise affine conditioning as a general primitive, including replacements of normalization affine coefficients. Here the conditioning index is the ensemble member, with coupled coefficients. [GNN-FiLM](https://arxiv.org/abs/1906.12192v5), §2.1, already modulates incoming graph messages using receiver/relation-conditioned affine coefficients and discusses nonlinearity before aggregation. The current factor has no receiver-conditioned generator. BatchEnsemble and TabM supply shared matrices, private factors and initialization ancestry; the saved Kim thesis already applies BatchEnsemble inside GNNs.

The relevant placement remains **before shared bond addition and ReLU**:

```text
neighbor message_j→v = ReLU(alpha_m ⊙ qhat_m,j + beta_m + shared_bond_jv).
```

Different effective coefficients can therefore change admission of atom–bond channels before summation. The same factor also changes the GINE self term. This is possible graph behavior within known affine modulation, with no guarantee of useful evidence, chemistry-specific specialization, or survival through later residuals and normalization.

## What an empirical result could establish

A predictor cannot distinguish this computation from the same effective affine coefficients folded into LayerNorm at inference: the local maps are identical. Experiments can establish **utility of the restricted parameterization and supervision allocation**, with attributed ancestry.

The frozen O/I/P/G comparison can establish an allocation advantage if I improves the declared pool beyond matched own and relevant GNCL policies while members remain capable. It cannot isolate this tensor because other internal MLP and virtual-node factors change supervision too. Only after fixed18 warrants the question, the separately prepared, disabled message-own control can test whether pool-risk supervision of this existing tensor helps while keeping its alignment unchanged. A positive result would attribute benefit to its supervised loss exposure; it would not establish a new primitive, independent gain/bias necessity, or placement necessity. Its extra reverse collection must remain charged.

For a graph-evidence claim, require net full-population positive–negative ranking repair and persistent differences in bond/neighbor margin responses beyond per-member global affine score rescaling. All members strictly reversing a pair implies the mean-logit pool reverses it too. More coefficient spread, more changed ReLU gates, or lower TRAIN BCE alone does not establish complementary correct decisions. A label-blind response panel can diagnose sensitivity; chemically modified inputs are not automatically label-preserving causal interventions. The [saved mechanism assessment](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/molhiv_internal_pool_risk_message_mechanism_assessment_20261007_v1/NOTE.md) retains these readout limits.

## Scoped graph initialization search

Five targeted public-index queries surfaced graph ensemble and topology-expert ancestry. A scoped primary read of [GNN-Ensemble](https://arxiv.org/abs/2303.11376v1), §§III–IV, confirms independent base GNNs diversified by random substructures and subfeatures, followed by probability aggregation. Reused [GraphMoRE](https://arxiv.org/abs/2412.11085v1) passages describe differentiated expert curvature initialization and topology-aware routing; reused [PreGS](https://arxiv.org/abs/2609.26310v1) passages describe supervised GAT-head transfer into graph experts. These concern existing graph diversification or graph-informed expert acquisition; they do not make the current all-one message-factor initialization graph-conditioned.

The queries also surfaced GEENI and topology-specific molecular experts at abstract/metadata scope. Exact fast-factor or private-normalization overlap is unresolved for these leads. The bounded search supplies no exhaustive absence or originality verdict. Sources and reading scopes are preserved separately. No scientific execution or active-family amendment is adopted.
