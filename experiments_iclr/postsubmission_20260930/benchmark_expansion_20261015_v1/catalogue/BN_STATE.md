# Route-specific BN state: a distinct candidate

## Hypothesis and source evidence

Native OGB GCN/GIN routes can have different activation distributions because
their boundary factors, input adapters and native dropout differ. The frozen
shared-buffer policy applies each route sequentially to the same native BN
running mean/variance/counter. That one EMA is then used by every route at eval.
It can preferentially reflect later route calls. Test whether route-matched
running state improves source pooled VALID quality and proper loss. This is
normalization-state ownership, not a new backbone or a novelty claim.

For constant native momentum a and route batch means mu_r, one logical update
with M sequential routes gives
`mu_shared_after = (1-a)^M mu_before + a sum_r (1-a)^(M-1-r) mu_r`.
Private state instead updates each route's own previous mean once. Counters
likewise advance M times in the original shared policy versus once per called
route/module in its private bank. This demonstrates a real state difference;
it does not predict which resulting VALID logits are better.

Read-only native-source inspection finds BN in OGB `conv.py`: GIN's internal MLP,
the node-layer BN list, and virtual-node MLPs. The mean-pooling preset does not
instantiate the optional attention-pool BN. BUDDY's source has feature/embedding/
structural BN branches, including conditional inactive branches that must retain
their source call behavior. The inspected current PolyFormer and NCN/NCNC source
files do not expose a BN target. No BN is added to make them compatible.

## Exact candidate versus the original policy

Keep one copy of every learned native parameter, including original shared BN
affine weight/bias. Preserve BN class, eps, momentum, tracking flag, training
batch, dtype and call order. For each tracked native BN module, store M running
means, M running variances and M `num_batches_tracked` counters, one per route.
Initialize each bank from that module's original native state; do not randomize
or fit new affine terms. Entering route m binds its buffers for the original
native BN call and commits the updated buffers back to m only. Eval route m
reads its own stored bank. There are no copied GNN bodies or private learned
normalization parameters.

An implementation can preserve registered native buffer identities by copying
the selected bank into the original tensors around the existing call, then
copying updates back. It must define the canonical resident-buffer state outside
route contexts and include every bank/counter in checkpointing. A source identity
guard must continue to check original parameter objects. This is a newly labelled
extension to buffer ownership; it is not falsely asserted to satisfy the frozen
14 October shared-buffer policy unchanged.

With ordinary TRAIN-mode BN using current minibatch statistics, tracking buffers
do not determine the instantaneous normalized output. Thus under matching
inputs/RNG/parameters and no intervening eval dependence, the two ownership
policies have the same mathematical TRAIN forward/gradient, while their running
state and eval outputs differ. This statement is not bitwise replay qualification,
an eval no-harm theorem or a claim about modes that normalize using running state.
Momentum=None, tracking-disabled modules, unused branches, virtual-node calls and
any source eval-in-training logic must be audited explicitly. Do not concatenate
routes into a new BN batch or replace source BN with LayerNorm.

## Minimal controls, measurements and implementation gates

- Original shared native buffers are the baseline, including their four route
  updates. Source-reference M1 and genuine I4 retain their native own BN.
- Private per-route buffers with the same affine parameters are the candidate.
  A mixed/tied-buffer eval control may use the saved banks, but its combination
  law must be specified in advance; averaging means and variances is not silently
  equivalent to the native shared EMA or a pooled variance.
- Prospectively paired same-weight eval checks can isolate buffer effects.
  Capture original shared and route-bank histories during new authorized work;
  do not infer private histories from one shared EMA or recalibrate on VALID.
- Record per-route/module counts and mean/variance differences, selected VALID
  member/pool AUC/BCE, ranking/threshold repairs and harms, and route-order
  sensitivity. Hidden/state distance alone cannot promote this method.
- Verify no parameter/body copies, native instantaneous TRAIN output/gradient
  semantics, exact state restoration, correct unused/singleton handling and
  optimizer identity. Charge bank-copy/storage and eval costs. Current frozen
  sessions/pilots remain untouched; no BN bridge is implemented by this audit.

## Ancestry and reading limits

[BatchEnsemble](https://arxiv.org/abs/2002.06715) establishes shared slow weights
and route-specific fast factors. Here only its abstract and located BN-related
training paragraphs were read; that does not establish an exact private-running-
buffer precedent. The inspected Uncertainty Baselines WRN BatchEnsemble source
snapshot declares ordinary Keras BatchNormalization, so it is not cited as an
implementation of our separate route-bank rule.

[AdaBN](https://arxiv.org/abs/1603.04779), abstract and §3.2/Algorithm1 prose, is a
clear statistics-only normalization ancestry: domain-specific normalization can
change outputs without adding learned model components. Its use of target-domain
statistics is **not** adopted here. Only native TRAIN route histories may update
the proposed banks; VALID/TEST are not recalibration populations. Neither primary
paper's empirical claims or a complete paper/code audit transfers to this graph
candidate. Correct TabM source `2410.24210v3` supplies selection/initialization
context, not proof that this normalization variant is new or beneficial.
