# Competence retention and graph view correction from primary methods

Two new bounded primary method reads support one prospective architecture hypothesis: retain competent base members unchanged and train independent side members with explicit ego and neighborhood information. This is a composition of established methods, not an established novel method. The existing failures do not identify a shared representation blind spot, and the primary papers do not guarantee correction of common errors or improved pooled accuracy.

## Evidence motivating the question

The Amazon reference audit establishes small recorded TRAIN CE for each native member and small mean own CE for ordinary banks, despite poor A accuracy and very large A NLL. That supports training fit without establishing held competence or its failure mechanism. The stopped G0 has lower accuracy than native/ordinary references and essentially the same reported correction counts as graph free. Complete39 has weak live members and small or inconsistent joint sharing advantages. None of this proves erased graph information. Query error correlation alone does not identify common offending negative edges.

The saved literature index v72 and earlier scoped reports were checked before primary reading. Neo-GNN and SEAL lack direct records in that index but have saved primary retrieval and conclusions in `link_covariance_idea_v1`. Their conclusions are reused, with zero new reading credit. H2GCN had baseline/citation exposure in saved packets but no direct primary method scope located in this check. Side-Tuning had no located prior direct method scope. Canonical indices and sealed prior conclusions are unchanged.

## Side Tuning and the exact preservation boundary

Zhang, Sax, Zamir, Guibas, and Malik, *Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks*, ECCV2020, arXiv1912.13503v4. Read PDF pages 1 to 6 and 14; visually checked page 5. This is a scoped method, interpretation, and merge/limitation read, not a full paper or author code read.

The method freezes base model B, learns side model S, and combines representations:

`R(x) = B(x) ⊕ S(x)`;

`R(x) = α B(x) + (1−α) S(x)` for the paper's alpha blending.

Alpha is learned in the main formulation. The paper also describes the curriculum `α(N)=k/(k+N)`. Its incremental learning objective is `L(x_t,y_t)=||D_t(α_t B(x_t)+(1−α_t)S_t(x_t))−y_t||`, with a separate side network and decoder for each task. If architectures differ, side initialization uses distillation. Freezing B retains its function; freezing earlier task branches and decoders retains those earlier functions. This does **not** say that the current combined predictor D_t(R) preserves B's classifications, nor that same-task ensemble members remain competent after their output changes.

The explicit prior overlap is a fixed pretrained representation, an additional trainable branch with input access, late fusion, and separate preserved task paths. Frozen anchors or an additive graph side branch cannot be claimed as a new learning principle. The paper's benchmark evidence covers vision, navigation, reinforcement learning, NLP, and transfer tasks, with only one base and one side network in its experiments. It does not study graph common-negative correction or four shared members.

A branch that only observes B(x) cannot distinguish two inputs that B maps to the same representation. Giving S original input access is the relevant mechanism. Even with a frozen complete base classifier, mixing base probabilities 0.6 for the true binary class with side probabilities 0.0 at equal weight makes the true probability 0.3: preserved base competence does not guarantee pool competence.

## H2GCN and the exact graph observations

Zhu, Yan, Zhao, Heimann, Akoglu, and Koutra, *Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs*, NeurIPS2020, arXiv2006.11468. Read PDF pages 1 to 9 and 20 to 23; visually checked Algorithm1 on page21. This is a bounded method, theorem-condition, main result, algorithm, complexity, and tuning-scope read. Proof appendices and the remaining experimental appendices were not read.

The three designs are separate ego and neighbor embeddings, explicitly distinct higher-order neighborhoods, and concatenated intermediate representations. For feature embedding and K propagation rounds:

`R^(0)=σ(X W_e)`;

`R^(k)=[P_1 R^(k−1) || P_2 R^(k−1)]`;

`R_final=[R^(0) || R^(1) || … || R^(K)]`;

`p_v=softmax(dropout(R_final)_v W_c)`.

Here P_i is symmetric degree normalization of the i-hop adjacency without ego loops. The final dimension is `(2^(K+1)−1)p`. Propagation rounds contain no learned nonlinear transformation; the initial embedding and final classifier are learned. H2GCN-1 uses K=1 and H2GCN-2 uses K=2. The training objective is labelled-node CE plus L2 on W_e and W_c. Concatenation retains the ego block explicitly rather than irreversibly averaging it into a neighborhood block.

Theorem1 assumes one-hot label features, equal class counts, constant training degree, and uniform class compatibility. Theorem2 assumes conditional independence of neighbor labels and uniform off-class probabilities. Its expected second-hop homophily statement is conditional on those assumptions, not a universal claim about Amazon. Neither theorem proves same-negative correction, ensemble competence, or generalization on the present split.

The paper's literal Algorithm1 constructs `Abar_1=I[A−I_n>0]` and `Abar_2=I[A²−A−I_n>0]`. The prose defines minimum distance exactly two and excludes the ego. Taken literally with ordinary count-valued matrix multiplication, that second expression is not equivalent to the prose: in loopless K4, adjacent nodes have two common neighbors and `(A²−A)_uv=1`, while the diagonal also remains positive. A faithful author implementation and its binarization order must be resolved before execution; author code was not read here. This is a paper notation/implementation ambiguity, not a claim that the author's implemented baseline is wrong. For an explicitly defined exact-distance test, first binarize two-step reachability and then remove direct neighbors and the diagonal.

The paper tunes methods on shared development splits. Its main real experiments use old heterophily benchmarks and mostly 10 splits with actual 48/32/20 per-class train/validation/test ratios; Cora-Full uses three 25/25/50 splits. Amazon Ratings is not evaluated. Those results motivate a structural comparator, not a transfer guarantee to modern large graphs or Polynormer. Exact two-hop preprocessing costs `O(|E| d_max)` and can greatly expand support; feature aggregation costs `O(2^K (|E|+|E_2|)p)`. A deterministic shared cache is not automatically cheap.

## Reused link prediction prior and remaining limits

The saved SEAL conclusion supplies target-labelled enclosing subgraphs; the saved Neo-GNN conclusion supplies explicit neighborhood overlap; BUDDY/ELPH supplies sketches and NCN/NCNC supplies common-neighbor structure and completion. Query-specific structure cannot be presented as a new graph observation. These priors remain strong controls if a link version is eventually proposed. This packet makes no new method read claim for them and does not infer a link ranking mechanism from the node classification H2GCN evidence.

The already read TabM supports useful weak members, not a standalone competence guarantee. TreeNet/shared branches, factorized ensembles, graph filter banks, BernNet/PolyFormer/HOPPER, hidden fusion, generic contrastive diversity, and graph expert routing remain competing priors. A raw-input or hop bypass is not a novel operator. Separate ego and neighbor blocks can be available in an existing expressive native model already; improvement cannot be presumed from a schematic information-loss argument.

## One prospective hypothesis

**H1: a frozen competent bank plus independently supervised H2GCN side members can correct more common opponent errors than the same bank plus feature-only side members, while improving pooled accuracy.** This tests extra graph observations while preserving the original base members as separate output paths. It does not assert preservation of the new side members or the combined pool. Sharing is limited to deterministic graph operators; side W_e/W_c parameters are private. No learned router, new diversity penalty, gradient meta rule, or loss grid is proposed.

Use four frozen native base members and four independently trained side members. H2GCN-1 uses K=1; feature-only controls use K=0. Each side member trains on ordinary own CE using the same role policy and optimizer schedule; there is no hard-error mining from A. Independently supervising S avoids relying solely on gradients of a frozen base that already nearly solves TRAIN CE. This is an adaptation of the Side-Tuning idea: an equal probability pool of base and side predictions, rather than its learned representation alpha, must be named as an experimental change. Use one fixed 50/50 bank mixture; no alpha search.

The minimal representative development experiment is one full graph, with three paired seed blocks and a separately authorized, previously uninspected development split. Amazon's existing A must not select the branch or its hyperparameters. All architectures, one regularization/optimizer recipe, role policy, horizon, and the development comparison must be fixed before those outcomes. Charge training, operator construction, memory, and additional serving work; no saving claim follows from deterministic sharing.

Required controls are frozen native SINGLE/independent4, an independent native8 control for the combined eight members, the competent ordinary own plus pool bank, standalone H2GCN-1, four H2GCN side models alone, and equal-capacity feature-only side models with the same freeze and pooling scheme. A parameter-matched concatenated ego/neighbor MLP control distinguishes extra learnable capacity from separated graph input. The H2GCN side-only bank is essential: a gain explained completely by a better standalone architecture is not an ensemble-specific contribution. Strong native/MLP/LINKX or other already admitted modern heterophily controls must remain visible; weak GCN is insufficient. The native8 control may cost more than the small side bank; disclose its cost rather than dropping the quality comparison.

Primary outcome is pooled accuracy, with member accuracy, NLL/Brier, and common strict opponent identities reported. On nodes where a shared non-target class strictly outranks the target in every base member, count repairs and whether a side member becomes correct; also count losses on previously correct nodes. Report paired seed-block differences without treating graph nodes as iid replications. If a link transfer is later considered, use aligned negative-slot margins and capable NCN/NCNC, BUDDY, and SEAL/Neo controls rather than node error correlation alone.

**Falsifiers:** no accuracy gain over the frozen native bank; a probability-loss gain accompanied by accuracy loss; no excess common-error repair over the feature-only side control; side heads with poor standalone competence; or improvement explained fully by the side-only H2GCN bank. A mixed seed pattern remains inconclusive. Success would establish only development utility for this known-method composition, not graph erasure, causal historical diagnosis, novelty, or confirmation.

No second hypothesis is retained. The nearly solved TRAIN CE and the source limits do not justify another optimizer or diversity-loss mechanism. This is a literature and prospective-test packet only: no scientific implementation, fit, prediction analysis, held scoring, or launch was performed.

## Read accounting and retrieval exceptions

Two new bounded primary method reads, zero full-paper reads, and zero new author-code reads. Mechanical extraction covered all PDF pages but does not count as reading. Neo-GNN and SEAL were redundantly retrieved before the older packet was found, but were not newly read; their duplicate retrievals have zero new method credit. A guessed Side-Tuning identifier resolved to a basketball paper; it was excluded after the first-page identity check. The corrected exact-title locator resolves Side-Tuning to arXiv1912.13503v4 and DOI10.1007/978-3-030-58580-8_41. The wrong paper supplies no hypothesis evidence. No canonical literature index was modified.
