# Shared graph ensemble member competence gap scout

The next useful question is whether the private parameter budget can learn a strong native graph predictor at the current sharing boundary. Retain one attributed capacity diagnostic: place private parameters in one existing complete native propagation block and compare with the same added parameter budget in node-only private capacity. Reuse an existing exact matched comparison if it already covers this question. This scout establishes no new ensemble principle or graph-specific quality advantage and authorizes no implementation or compute.

## Closed diagnoses and saved ancestry

Root supplied three closed aggregate diagnoses: the IMDB private-source candidate changes no decisions, uses sources almost commonly and loses about 1.15 percentage points in the reported quality comparison; the molecular internal-credit candidate has weaker members and less pool lift and loses 2.687 AUROC points to genuine independent4; centered sheaf partially repairs member learning but provides almost no complementary ranking. These are distinct tasks and metrics, not pooled statistical evidence. They support examining member function capacity and its actual ensemble benefit before another diversity or credit term.

CURRENT_SUPPLEMENT pointed to the exact v37 supplement and index v72 lineage. Saved TreeNets/Deep Sub-Ensembles, Recon/GDPS/AdaLoRA, Node-MoE/CoGNN, GNCL/TabM/TabLoRA/DICE/FoRDE/BSNN, function-prior and additive-rank1 conclusions are reused with zero new primary credit. The saved private-propagation synthesis already proposes untying one native block; the capacity diagnostic below refines that existing question using the new closed diagnoses. It is not a newly invented architecture. The just-added function-prior and additive-rank1 options remain different inactive proposals; this scout does not reimplement or relabel them.

Two bounded title queries led to two method scopes not found among the saved method-read records. No third primary body was opened. The frequency-filter AdaGNN below is distinct from the 2021 boosting AdaGNN metadata returned by discovery and the saved unread 2026 IEEE boosting AdaGNN. Neither boosting body was read or credited.

## Two new primary method scopes

### AdaShare

[AdaShare](https://arxiv.org/html/1911.12423v2#S3), Ximeng Sun, Rameswar Panda, Rogerio Feris and Kate Saenko, exact arXiv v2: complete Section 3, HTML lines 314–429, including equations 1–5, training paragraphs and the method figure caption. Figure pixels, experiments, appendices and author code were not read. Full-paper credit is zero.

AdaShare gives each task a binary execute/skip policy for each residual block. A block used by several tasks is shared; a block used by one task is task-specific. During training, both forward and backward use soft Gumbel-Softmax decisions. The initial temperature is 5 and is annealed toward zero. Its loss is

\[
\sum_k\lambda_k L_k
+\lambda_{sp}\sum_{l,k}\log\alpha_{l,k}
+\lambda_{sh}\sum_{k_1,k_2,l}\frac{L-l}{L}
  |\alpha_{l,k_1}-\alpha_{l,k_2}|.
\]

Here α is the block execution probability. Network weights and policies are optimized alternately on separate TRAIN splits. All blocks are initially shared for warmup; policy learning gradually admits more blocks from the end of the network. After policy training, a discrete policy is sampled from the best policy and the resulting network is optimized on the full training set. This is a multi-stage learning and selection budget, not a negligible policy-only cost.

This source addresses different tasks with different losses. It is neither a four-member same-task graph ensemble nor a teacher-first compression method. The inspected objective encourages task accuracy but provides no hard own-member competence floor. Its sharing regularizer explicitly rewards similar lower-block policies. Copying this setup to four identical labels/losses does not supply evidence that different strong routes will emerge. TreeNets is the closest same-task shared-branch ancestor; saved Recon/GDPS are closer to adaptive sharing/placement, and Gumbel policy learning is itself attributed in AdaShare. No broad adaptive-sharing gap remains open merely because this exact identity was newly read.

### AdaGNN with frequency response filters

[AdaGNN](https://arxiv.org/html/2104.12840v3#S2), Yushun Dong, Kaize Ding, Brian Jalaian, Shuiwang Ji and Jundong Li, exact arXiv v3: complete Section 2.1–2.3, HTML lines 326–412, equations 1–3 and both method figure captions. Section 3 proofs, experiments, code and figure pixels were not read. Full-paper credit is zero.

For an undirected graph, AdaGNN uses a normalized Laplacian of the graph with self-loops and learns channel-diagonal matrices Φ. Its graph filter is

\[
H^{(k)}=H^{(k-1)}-\widetilde L H^{(k-1)}\Phi_k.
\]

The first layer also applies a dense feature transform and ReLU; intermediate layers omit those transforms. A final dense classifier is trained with labeled-node CE plus L1 regularization on Φ and L2 on Φ, the first transform and the classifier. In the isolated linear filter calculation, channel j has response

\[
p_j(\lambda)=\prod_k(1-\phi_{j,k}\lambda).
\]

Thus channel-specific parameters can change graph-frequency response; they do more than rescale a fixed propagated channel afterward. The source is a single node-classification GNN. It supplies no jointly trained ensemble, private-route allocation or individual-competence guarantee. Its undirected normalized operator and simple linear middle layers are not a direct port of typed SeHGNN propagation, molecular GINE or a nonlinear sheaf network. Saved Node-MoE/filter-selection, private propagation and the linear common-filter boundary are the closest collisions. Calling a graph filter adaptive is insufficient methodological novelty.

## Concrete limitation that changes the next action

The printed AdaShare objective does not mathematically enforce competent tasks. Fix finite network weights and finite training inputs, and suppose the all-skipped residual route yields finite logits. Set every α to ε in (0,1). The weighted task losses remain bounded; the policy-sharing term is zero; the sparsity term is LK log ε. For positive λ_sp,

\[
L_{total}(\varepsilon)\longrightarrow-\infty
\quad\text{as}\quad\varepsilon\downarrow0.
\]

This is a manual limit of the inspected formula, not a numerical test or a claim that the author implementation failed. Probability floors, clipping, constraints or finite optimization could alter the actual implementation; they were not inspected. Consequently, that formula cannot be adopted as a competence-preserving architecture-selection constraint without a separately specified bounded implementation and actual member-quality evidence. There is no reason here to add its policy search to the closed recipes.

A second useful boundary explains why strong members alone cannot certify diversity. Under a shared input X and strictly proper Bernoulli loss, if each member can represent η(X)=P(Y=1|X),

\[
\frac1M\sum_m E\,BCE(p_m,Y)
=E\,H(\eta)+\frac1M\sum_m E\,KL(Ber(\eta)\Vert Ber(p_m)).
\]

The population optimum has every p_m=η almost surely. Different route parameters can therefore support equally competent identical predictions and no pool lift. This is not an impossibility result for finite-data ensembling, constrained models or different information sets. It shows that duplicating AdaShare task losses or AdaGNN CE does not add a same-task complementarity guarantee. The centered-sheaf diagnosis is compatible with this failure mode; it does not prove that the population optimum was reached.

AdaGNN also has a limited recovery boundary. A linear filter tail driven only by H stays in span{H, L H, …, L^K H}. In the symmetric-L case, a missing eigenspace component of every input channel remains missing under every such polynomial filter and a linear channel readout. This statement excludes new raw/self inputs, nonlinearities, node-dependent masks and feature-dependent attention. Current closed results do not demonstrate that any particular frequency was erased. The useful operation must enter where the relevant evidence is still available, rather than assume a late filter can recover it.

## One minimal meaningful matched diagnostic

Retain exactly one hypothesis: **private full native propagation capacity may repair own predictor quality and useful complete-task errors beyond equal-sized private node-only capacity.** The mechanism is conditional: separate native neighbor transforms could retain relation/hop/neighbor evidence that a common transform suppresses, while an own-loss objective requires each path to remain useful. No graph-frequency erasure or geometry benefit is claimed to have been observed.

Freeze one existing native propagation block, with no block/rank/policy search. Compare five fixed conditions on prospectively paired seeds:

| Condition | Exact operation and purpose |
|---|---|
| Shared own-only M4 | Current native graph/depth and sharing boundary, ordinary declared own loss, no extra credit or repulsion |
| Private native block M4 | Replace that one shared block with four separately trainable complete copies, including its native parameters/buffers; earlier processing remains shared |
| Matched node-only private capacity M4 | Keep graph propagation shared; add the same extra parameter budget, 3 times the copied block size, in private nonlinear node-only capacity at the same boundary |
| Capable native single | Full native own recipe and same permitted information; practical single reference |
| Genuine independent4 | Four complete independently fitted native bodies, their own selections and full acquisition/serving costs |

All cells start from prospectively seeded untrained constructors, without fitted donor checkpoints. The private-block copies begin with the same fresh prototype block values but own separate parameter/buffer objects and declared member streams. The graph/depth, role information, feature widths where compatible, full TRAIN exposures, member loss reduction/decay, update horizon, selector, fixed deployed pooling and member-serving scope must be declared before fits. Node-only parameter matching and initialization must be exact and qualified, not assumed. This is a capacity diagnostic with known partial-sharing ancestry. Stronger sharing attribution additionally needs the same operation untied, and a capable joint four-path single where relevant; it cannot be inferred from these five cells alone.

Use actual mean and worst own-member quality plus complete deployed native quality against both single and genuine ensemble. Retain every seed, failed cell, full repairs/harms, common-error and rank/coverage diagnostics, and complete costs. Three paired seeds would be exploratory; unused confirmation remains separate. A better TRAIN surrogate or weaker-but-different member does not satisfy the hypothesis.

If node-only capacity matches private graph capacity, the graph-specific explanation is unsupported. If own quality improves without deployed gain, the complementary-prediction problem remains. If either competent reference matches or beats the bank, the intended quality advantage is unestablished. Do not respond by searching branch locations, priors, coefficients or pooling thresholds. Complete/reuse an existing exact comparison before admitting any new one.

## Provenance and scope

PRIMARY_RETRIEVALS and READ_SCOPES record the exact versioned URLs, raw HTML and method-excerpt hashes. REUSED_MEMORY_BINDINGS retains the pointer/lineage and saved conclusions consulted. CLOSED_DIAGNOSES preserves the root-supplied aggregate context without new outcome reads. New primary method scopes: 2. Full papers, author-code/proof audits, performance adoptions and scientific executions: 0. No server allocation, data, checkpoint, current partial outcome, numerical fixture, source modification or new framework was accessed or created. Root retains novelty judgment, implementation and compute admission.
