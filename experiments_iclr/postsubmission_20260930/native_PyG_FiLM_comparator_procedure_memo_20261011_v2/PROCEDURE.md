# Native PyG FiLM comparator for the existing WikiCS scaffold

A native FiLM comparator is the next high-value known conditional baseline if the intended claim extends beyond E versus its matched N control. N retains a common receiver-generated linear action before mean; it omits FiLM's edgewise activation and additive modulation. Comparing with the actual published conditional-message family is more informative for that claim than adding another new mechanism. A narrow claim about E-minus-N under the matched paired-map operator can still retain N as its appropriate control.

This proposal adds one FiLM family in three ownership variants. It is a PyG2.7.0 operator adaptation in the established WikiCS scaffold and training protocol. It supplies no execution admission or reported-score reproduction.

## Exact native convolution

Use a fresh `FiLMConv(128, 128, num_relations=1, nn=None, act=ReLU(), aggr='mean')` in each of the two graph modules. Supply a **distinct freshly constructed ReLU module to each convolution**. The default activation expression in the source is one Python default object; omitting it in both constructors can share that object and violate the existing route wrapper's no-module-alias contract.

For the one-relation dense COO path, the actual forward is

$$(\beta_i,\gamma_i)=g(h_i),\quad(\beta_i^s,\gamma_i^s)=g_s(h_i),$$
$$u_i=\operatorname{ReLU}(\gamma_i^s\odot W_s h_i+\beta_i^s)
+\frac1{D_i}\sum_{j\to i}\operatorname{ReLU}(\gamma_i\odot W h_j+\beta_i).$$

`g` is a128→256 PyG Linear with bias; `g_s` is128→256 without bias. `W` and `W_s` are128→128 without bias. The source splits beta first and gamma second. Do not remove the skip term, move ReLU after mean, constrain gamma to a positive gate, add E/N transport builders, or convert mean to sum.

## Scaffold and support

Retain the complete original WikiCS feature rows, node order, stored COO records and safe TRAIN/VALID roles:11701 nodes,300 raw features,10 classes,580 TRAIN and5274 VALID rows. Retain the common width128, depth2, residual pre-LayerNorm, `[h, message]` FFN input, original FFN GELU/dropout/residual, stem and final LayerNorm/head. Replace only the two SAGE convolutions with the specified native FiLM convolutions.

The FiLM docstring says self-loops are automatically added and represented by their own relation. The actual forward has no edge-index loop insertion: it supplies a separate learned skip term. Pass the original factual support, which already contains its self entries, without inserting or removing loops. With one relation, each stored diagonal record remains in the ordinary relation mean; the FiLM skip is an additional center path outside that mean. Preserve both. This is the exact PyG-on-factual-support policy; it differs from the paper's PPI edge-typing setup and must be disclosed.

## Initialization and route ownership

Construct a fresh original scaffold under the original native seed, then replace its two convolutions while keeping the already constructed stem, FFNs, normalizations and head objects and values. Retain the new FiLM convolutions' own constructor/reset initialization. The discarded SAGE tensors are not reused as FiLM weights. Do not zero FiLM generators or impose identity modulation to mimic E's initialization: E/N builder identity starts are not native FiLM starts.

Before creating an optimizer or moving devices, wrap every materialized affine map once through the established FactorLinear. Its convention is $s_m\odot W(r_m\odot x)+b$, with the bias outside $s_m$. All four native FiLM affine modules are materialized PyG Linear instances and support that operation. Keep native W/b once in the shared variant, with private r/s for each route; keep normalizations shared. Retain all-one fast factors except the original first-stem Rademacher start. Never call FiLM reset_parameters after factorization, since its direct Linear reset calls expect native Linear modules.

The current NativeModelAdapter constructor only validates original SAGEConv under SAGEModule and rejects FiLMConv after replacement. Its forward/head-capture concept is reusable, but a future explicit FiLM scaffold adapter and source contract are required. Capturing a SAGE adapter before swapping does not establish that FiLM was validated. The current E installer and fixed SAGE owner guards are likewise not valid FiLM admission. This memo supplies no adapter or consumer implementation.

## One complete comparator roster

Use the original seeds7301,7403,7507 on the same complete graph and original roles. Retain native member seed stride1000003, factor offset2000003, dropout offset3000007 and the existing persistent member RNG policy.

| Variant | Ownership and selection per seed | Banks across three seeds | Fits and selected states |
| --- | --- | --- | --- |
| Equipped FiLM M1 | One full FiLM body with one factor row; its own CE and selected own accuracy | 3 | 3 |
| Shared FiLM4 | One full shared body and shared native/generator W/b; four live private-factor routes; one coherent selected bank | 3 | 3 |
| Genuine FiLM I4 | Four separately initialized, separately optimized full bodies and generators, each with its own selected state; no fitted sharing | 3 | 12 |

The complete family has9 banks,18 actual fits/states and27 native trajectories per roster serving pass. The existing fixed probability-mean serving rule applies to the four-member banks; M1 uses its single predictive distribution. No logit averaging, route-exchange block, cross-route loss or dummy parameter matching is added.

Use the original float32 full-graph loss and update implementation: arithmetic mean of the four own CEs for shared4 and unscaled own CE for each single or independent body. Keep AdamW learning rate0.001, zero weight decay, dropout0.2, maximum1000 updates, patience300 and the earliest strict maximum of full VALID probability-mean accuracy. Genuine I4 selects its bodies independently by own accuracy before pooling. Restore the actual selected model, optimizer and route RNG states with strict model loading before serving. Preserve every fit failure and all costs; interpret only the complete roster. The source-only maximum is18000 optimizer updates across18 fits, without predicting runtime.

If confidence is later compared, use the same original frozen five-fold scalar-calibration/readout procedure for all new banks, retaining original completed comparisons as archived authority. Do not tune temperatures, hyperparameters or the roster after seeing comparison outcomes. This remains encountered WikiCS development with three optimizer repeats, not unused independent-graph confirmation.

## Capacity and physical operation differences

At width128, the source-derived convolution counts are:

| Operator at one layer | Shared W/b | Private r/s per member |
| --- | ---: | ---: |
| Original SAGE | 32896 | 512 |
| SAGE plus E or N builders | 41120 | 1056 |
| One-relation native FiLM | 98560 | 1280 |

For the complete unchanged WikiCS scaffold, FiLM has336522 shared parameters plus4406 per-member fast scalars; E/N has221642 plus3958 per member. Thus shared4 FiLM has354146 total parameters versus237474 for E/N; equipped M1 has340928 versus225600. Genuine I4 stores four full M1 models. Both FiLM and E/N have14 factorized affine maps, but their dimensions and capacity differ. These comparisons cannot isolate sender conditioning through exact parameter equality.

One FiLM convolution evaluates four dense node maps, costing $6NC^2=98304N$ dense MACs per route, then performs modulation, addition and ReLU on every stored edge before mean, plus its analogous node skip operation. E instead retains $2NC^2=32768N$ SAGE native dense MACs and adds two edge-context builders ($8192E$), paired 4×4 products ($64E$) and the 4×32 VALUE action ($512E$). N evaluates its builders/pair products at nodes, then applies its map at edges. These are source-shape forward counts excluding common scaffold work, biases, r/s multiplications, aggregation, storage and transfers. Every shared route still performs a full live graph forward; no timing, memory or speed advantage follows from these formulas.

## Hyperparameters and claim scope

The proposed depth, width, optimizer, dropout and selector are conventional controlled-study choices inherited intact from the WikiCS protocol. They are appropriate for this matched-scaffold comparison, with learning competence assessed from complete training and prediction evidence. They are not established native-FiLM-optimal WikiCS settings. The official paper uses task-specific search and different selected PPI settings, a TensorFlow implementation, distinct early-stopping budgets and typed self relations; it supplies no WikiCS defaults in the inspected published scope.

A useful result would compare E with a capable implementation of an established richer receiver-conditioned operator under these declared costs and budgets. A weak or failed learning run would not establish that full GNN-FiLM is an inadequate explanation. This comparator strengthens the known-baseline boundary; it does not rename FiLM as a new method or supply a claim about the best possible FiLM configuration.
