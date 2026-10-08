# UniMP author-code reference and limits

8 October 2026. Source inspection only; no imports, installation, benchmark, numerical result adoption or code port.

The official OGB node leaderboard links the PGL team's UniMP entry and arXiv:2009.03509 to `PaddlePaddle/PGL/ogb_examples/nodeproppred/unimp`. This provides a primary implementation locator. Repository-search discovery also found an explicitly unofficial implementation, which was not adopted. The inspected PGL files are pinned to commit `6dbb47c4559352ea1b1e327ee0039c47095583af`; current repository source is not proof of exact 2020/2021 submission-version equivalence.

## Original arxiv operation inspected

`main_arxiv.py` shuffles the official TRAIN IDs in place each epoch, exposes the first `floor(label_rate * n_train)` labels and uses the complementary TRAIN IDs as CE targets. The default visible fraction is 0.625. At inference all official TRAIN labels are available. No inverse-inclusion value scaling is present in the inspected original masking/label-input path.

`Arxiv_label_embedding_model` gathers only label_idx labels, embeds them, adds them to node features, then normalizes the combined input. Three graph-transformer layers propagate the resulting states. Queries, keys and values all depend on these evolving combined states, with learned residual gates and nonlinear hidden layers. The final attention layer averages its heads. Its parameters are fitted end to end through masked-target CE. This is an existing alternative to treating label messages as a separate late residual.

`transformer_gat_pgl.py` implements multihead scaled query-key dot products, incoming-neighbor softmax, transformed message values, and head concatenation or averaging. `utils.py` makes the graph undirected with deduplicated edges, then adds/deduplicates self loops. The original arxiv CLI uses 3 layers, width128/head, 2 heads, dropout0.3, 2000 epochs, Adam lr0.001 and L2 regularization0.0005. The CLI's attn_dropout argument is not passed into the inspected original model call; the transformer helper default is zero. These are original arxiv settings, not a justified WikiCS recipe transfer.

## Stronger source interface

The OGB leaderboard also links a UniMP_v2 arxiv implementation with virtual nodes and attention-based APPNP smoothing. Its inspected driver/model interfaces use three virtual nodes, 3 layers, hidden width100, 3 hidden heads, label-visible fraction0.65, 1500 epochs, and a final attention-based ten-hop APPNP call with alpha0.2. Labels are normalized and rectified before feature addition. The model uses input dropout0.1, while internal feature dropout is the parser value0.3. The graph-transformer/APPNP helpers and virtual-node utility were not inspected: no complete v2 implementation audit or port is claimed. V2 is a stronger-reference obligation, not numerical evidence for this project.

## Difference from the current prototype

The retained GNNM corrector uses feature-only detached native H/logits, label-only message values, one nonself-neighbor attention hop, four factorized correction routes, and a separate own-CE correction update. It preserves the native feature-training policy rather than jointly mixing labels into the native feature states. Its linear-value first-moment scaling has a narrower validity domain than UniMP's nonlinear propagated-label operation. These differences define an attributed composition to evaluate; they do not establish methodological novelty, useful diversity or superiority.

Four attention routes also make a one-head baseline insufficient to establish ensemble benefit. The representative screen therefore requires a four-head single with a joint prediction readout, alongside four untied correctors on the same native trajectory. A full ordinary independent GNN4 and capable source-native label-aware models remain necessary after a positive screen.

## Reproduction boundaries

The original example pins paddlepaddle_gpu1.8.3.post107, torch1.5.1 and OGB1.2.1. Nothing was installed or executed. Its stock evaluator reads and prints TEST scores each epoch and tracks best TEST alongside best VALID. Although its returned score follows best VALID, this loop does not satisfy our frozen heldout protocol. Any execution here must separate allowed TRAIN/development tensors and perform no TEST scoring until the final locked confirmation. Loading a full label array in the stock example does not itself prove label-input leakage: the model gathers label_idx, which the driver restricts to TRAIN.

Original arxiv constants include feature/label-embedding width128 and40 classes. A WikiCS port needs declared dimension/graph/label-role changes and competent fitting; the supplied GNNM core is not a faithful UniMP substitute. Source equivalence, runtime competence, accepted-paper equivalence, fresh training and a global novelty search remain incomplete. Published README/leaderboard numbers were exposed incidentally but are not imported as experiment results.
