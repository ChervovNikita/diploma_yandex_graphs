# graphmae bounded method scope

Primary URL: https://arxiv.org/html/2205.10803v3

Blocks 35 through 75 inclusive in graphmae_blocks.json. Zero full-paper credit.

## Block 35

3. The GraphMAE Approach

## Block 36

In this section, we present the self-supervised masked graph autoencoder framework—GraphMAE—to learn graph representations without supervision based on graph neural networks (GNNs). We introduce the critical components that differ GraphMAE from previous attempts on designing graph autoencoders (GAEs).

## Block 37

3.1. The GAE Problem and GraphMAE

## Block 38

Briefly, an autoencoder usually comprises an encoder, code (hidden states), and a decoder. The encoder maps the input data to code, and the decoder maps the code to reconstruct the input under the supervision of a reconstruction criterion. For graph autoencoders, they can be formalized as follows.

## Block 39

Let $\mathcal{G}=(\mathcal{V},{\bm{A}},{\bm{X}})$ denote a graph, where $\mathcal{V}$ is the node set, $N=|\mathcal{V}|$ is the number of nodes, ${\bm{A}}\in\{0,1\}^{N\times N}$ is the adjacency matrix, and ${\bm{X}}\in\mathbb{R}^{N\times d}$ is the input node feature matrix. Further, given $f_{E}$ as the graph encoder, $f_{D}$ as the graph decoder, and ${\bm{H}}\in\mathbb{R}^{N\times d_{h}}$ denoting the code encoded by the encoder, the goal of general GAEs is to reconstruct the input as

## Block 40

(1) $\begin{split}{\bm{H}}=f_{E}({\bm{A}},{\bm{X}}),\ \mathcal{G}^{\prime}=f_{D}({\bm{A}},{\bm{H}}),\end{split}$

## Block 41

where $\mathcal{G}^{\prime}$ denotes the reconstructed graph, which could be either reconstructed features or structures or both.

## Block 42

Despite their versatile applications in NLP and CV, autoencoders’ progress in graphs, especially for classification tasks, is relatively insignificant. To bridge the gap, in this work, we aim to identify and rectify the deficiencies of existing GAE approaches, and subsequently present the GraphMAE—a masked graph autoencoder—to further the idea and design of GAEs and generative SSL in graphs.

## Block 43

GraphMAE. The overall architecture of GraphMAE is illustrated in Figure 2 . Its core idea lies in the reconstruction of masked node features. And we introduce a re-mask decoding strategy with GNNs, rather than the widely-used MLP in GAEs, as the decoder to empower GraphMAE. To have a robust reconstruction, we also propose to use a scaled cosine error as the criterion. Figure 1a summarizes the technical differences between GraphMAE and existing GAEs.

## Block 44

In detail, the backbones for $f_{E}$ and $f_{D}$ can be any type of GNNs, such as GCN ( Kipf and Welling, 2017 ) , GAT ( Velickovic et al., 2018 ) , or GIN ( Xu et al., 2019 ) . As our encoder $f_{E}$ processes the whole graph ${\bm{A}}$ with partially observed node features $\widetilde{{\bm{X}}}$ , resonating to the backbones in other generative SSL methods (e.g., BERT and MAE), GraphMAE prefers a more expressive GNN encoder on features for different tasks. For instance, GAT is more expressive in node classification, and GIN provides a better inductive bias for graph-level applications (See Tables 6 and 4 ).

## Block 45

3.2. The Design of GraphMAE

## Block 46

In this part, we explore how to design a generative self-supervised graph pre-training framework that can match and outperform state-of-the-art contrastive models. Specifically, we discuss different strategies via answering the following four questions:

## Block 47

Q1 : What to reconstruct in GAEs?

## Block 48

Q2 : How to train robust GAEs to avoid trivial solutions?

## Block 49

Q3 : How to arrange the decoder for GAEs?

## Block 50

Q4 : What error function to use for reconstruction?

## Block 51

These questions concern the designs of the reconstruction objective, robust learning, loss function, and model architecture in GAEs, the answers to which enable us to develop GraphMAE.

## Block 52

Q1: Feature reconstruction as the objective. Given a graph $\mathcal{G}=(\mathcal{V},{\bm{A}},{\bm{X}})$ , a GAE could target reconstructing either the structure ${\bm{A}}$ or the features ${\bm{X}}$ , or both of them. Most classical GAEs ( Kipf and Welling, 2016 ; Pan et al., 2018 ) focus on the tasks of link prediction and graph clustering, and thus usually choose to reconstruct ${\bm{A}}$ —a target commonly used in network embeddings ( Perozzi et al., 2014 ; Grover and Leskovec, 2016 ; Tang et al., 2015 ) . More recent GAEs ( Wang et al., 2017 ; Park et al., 2019 ; Salehi and Davulcu, 2020 ) tend to adopt a combined objective of reconstructing both features and structure, which unfortunately does not empower GAEs to produce as significant progress in node and graph classifications as autoencoders have done in NLP and CV.

## Block 53

A very recent study shows that simple MLPs distilled from trained GNN teachers can work comparably to advanced GNNs on node classification ( Zhang et al., 2022 ) , indicating the vital role of features in such tasks. Thus, to enable GraphMAE to achieve a good performance on classification, we adopt feature reconstruction as the training objective. Our empirical examination also shows that the explicit prediction of structural proximity has no contributions to the downstream classification tasks in GraphMAE (See Figure 1b ).

## Block 54

Q2: Masked feature reconstruction. When the code’s dimension size is larger than input’s, the vanilla autoencoders have risks to learn the notorious “identity function”—the trivial solution—that makes the learned code useless ( Vincent et al., 2008 ) . Relatively speaking, it is not a severe problem in CV since the image input is usually high-dimensional. However, in graphs, the node feature dimension size is typically quite small, making it a real challenge to train powerful feature-oriented GAEs. Unfortunately, existing GAEs that incorporate the reconstruction of features as their objective commonly ignore the threat ( Kipf and Welling, 2016 ; Pan et al., 2018 ; Park et al., 2019 ; Cui et al., 2020 ; Salehi and Davulcu, 2020 ) .

## Block 55

The denoising autoencoder ( Vincent et al., 2008 ) , which corrupts the input data on purpose, is a natural option to eliminate the trivial solution. Actually, the idea of employing masking as the corruption in masked autoencoders has found wide applications in CV ( He et al., 2021 ; Bao et al., 2021 ) and NLP ( Devlin et al., 2019 ) . Inspired by their success, we propose to adopt masked autoencoders as the backbone of GraphMAE.

## Block 56

Formally, we sample a subset of nodes $\widetilde{\mathcal{V}}\subset\mathcal{V}$ and mask each of their features with a mask token [MASK], i.e., a learnable vector ${\bm{x}}_{[M]}\in\mathbb{R}^{d}$ . Thus, the node feature $\widetilde{{\bm{x}}}_{i}$ for $v_{i}\in\mathcal{V}$ in the masked feature matrix $\widetilde{{\bm{X}}}$ can be defined as:

## Block 57

$\widetilde{{\bm{x}}}_{i}=\begin{cases}{\bm{x}}_{[M]}&v_{i}\in\widetilde{\mathcal{V}}\\ {\bm{x}}_{i}&v_{i}\notin\widetilde{\mathcal{V}}\end{cases}$

## Block 58

The objective of GraphMAE is to reconstruct the masked features of nodes in $\widetilde{\mathcal{V}}$ given the partially observed node signals $\widetilde{{\bm{X}}}$ and the input adjacency matrix ${\bm{A}}$ .

## Block 59

We apply a uniform random sampling strategy without replacement to obtain masked nodes. In GNNs, each node relies on its neighbor nodes to enhance/recover its features ( Gilmer et al., 2017 ) . Random sampling with a uniform distribution helps prevent a potential bias center, i.e., one’s neighbors are neither all masked nor all visible. Additionally, similar to MAE ( He et al., 2021 ) , a relatively large mask ratio (e.g., 50%) is necessary to reduce redundancy in the attributed graphs in most cases and thus form a challenging self-supervision to learn meaningful node representations.

## Block 60

The use of [MASK], on the other hand, potentially creates a mismatch between training and inference since the [MASK] token does not appear during inference ( Yang et al., 2019 ) . To mitigate the discrepancy, BERT proposes to not always replace “masked” words with the actual [MASK] token, but with a small probability (i.e., 15% or smaller) to leave it unchanged or to substitute it with another random token. Our experiments find that the “leave-unchanged” strategy actually harms GraphMAE’s learning, while the “random-substitution” method could help form more high-quality representations.

## Block 61

Q3: GNN decoder with re-mask decoding. The decoder $f_{D}$ maps the latent code ${\bm{H}}$ back to the input ${\bm{X}}$ , and its design would depend on the semantic level ( He et al., 2021 ) of target ${\bm{X}}$ . For example, in language, since targets are one-hot missing words with rich semantics, usually a trivial decoder such as MLP is sufficient ( Devlin et al., 2019 ) . But in vision, previous studies ( He et al., 2021 ) discover that a more advanced decoder (e.g., the Transformer model ( Vaswani et al., 2017 ) ) is necessary to recover pixel patches with low-level semantics.

## Block 62

In graphs, the decoder reconstructs relatively less informative multi-dimensional node features. Traditional GAEs employ either no neural decoders or a simple MLP for decoding with less expressiveness, causing the latent code ${\bm{H}}$ to be nearly identical to input features. However, it has no merit to learn such trivial latent representations because the goal is to embed input features with meaningful compressed knowledge. Therefore, GraphMAE resorts to a more expressive single-layer GNN as its decoder. The GNN decoder can recover the input features of one node based on a set of nodes instead of only the node itself, and it consequently helps the encoder learn high-level latent code.

## Block 63

To further encourage the encoder to learn compressed representations, we propose a re-mask decoding technique to process the latent code ${\bm{H}}$ for decoding. We replace ${\bm{H}}$ on masked node indices again with another mask token [DMASK], i.e., the decoder mask, with ${\bm{h}}_{[M]}\in\mathbb{R}^{d_{h}}$ . Specifically, the re-masked code $\widetilde{{\bm{h}}}_{i}$ in $\widetilde{{\bm{H}}}=\mathrm{REMASK}({\bm{H}})$ can be denoted as

## Block 64

$\widetilde{{\bm{h}}}_{i}=\begin{cases}{\bm{h}}_{[M]}&v_{i}\in\widetilde{\mathcal{V}}\\ {\bm{h}}_{i}&v_{i}\notin\widetilde{\mathcal{V}}\end{cases}$

## Block 65

With the GNN decoder, a masked node is forced to reconstruct its input feature from the neighboring unmasked latent representations. Similar to encoders, our empirical examination suggests that the GAT and GIN encoders are good options for node classification and graph classification, respectively. Note that the decoder is only used during the self-supervised training stage to perform the node feature reconstruction task. Therefore, the decoder architecture is independent of the encoder choice and can use any type of GNN.

## Block 66

Q4: Scaled cosine error as the criterion. The feature reconstruction criterion varies for masked autoencoders ( Devlin et al., 2019 ; He et al., 2021 ) in different domains. In NLP and CV, the de facto criterion is to predict discrete token indices derived from tokenizers using cross entropy error. An exception is the MAE work ( He et al., 2021 ) in CV, which directly predicts pixels in the masked patches using the mean square error (MSE); Nevertheless in fact, pixels are naturally normalized to 0–255, functioning similarly to tokenizers. But in graphs, it remains unexplored how to define a universal tokenizer.

## Block 67

In GraphMAE, we propose to directly reconstruct the raw features for each masked node, which can be challenging due to the multi-dimensional and continuous nature of node features. Existing GAEs with feature reconstruction have adopted MSE as their criterion ( Wang et al., 2017 ; Park et al., 2019 ; Jin et al., 2020 ) . But in preliminary experiments, we discover that the MSE loss can be minimized to nearly zero and may not be enough for feature reconstruction, which may (partly) explain why few existing GAEs use feature reconstruction as their only training objective. To this end, we found that MSE could suffer from the issues of sensitivity and low selectivity . Sensitivity means that MSE is sensitive to vector norms and dimensionality ( Friedman, 2004 ) . Extreme values in certain feature dimensions can also lead to MSE’s overfit on them. Low selectivity represents that MSE is not selective enough to focus on those harder ones among imbalanced easy-and-hard samples.

## Block 68

To handle its sensitivity, we leverage the cosine error as the criterion to reconstruct original node features, which gets rid of the impact of dimensionality and vector norms. The $l_{2}$ -normalization in the cosine error maps vectors to a unit hyper-sphere and substantially improves the training stability of representation learning. This benefit is also observed by some contrastive learning methods like BYOL ( Grill et al., 2020 ) .

## Block 69

To improve its selectivity, we further the cosine error by introducing the scaled cosine error (SCE) for GraphMAE. The intuition is that we can down-weight easy samples’ contribution in training by scaling the cosine error with a power of $\gamma\geq 1$ . For predictions with high confidence, their corresponding cosine errors are usually smaller than 1 and decay faster to zero when the scaling factor $\gamma>1$ . Formally speaking, given the original feature ${\bm{X}}$ and reconstructed output ${\bm{Z}}=f_{D}({\bm{A}},\widetilde{{\bm{H}}})$ , we define SCE for GraphMAE as

## Block 70

(2) $\mathcal{L}_{\textrm{SCE}}=\frac{1}{|\widetilde{\mathcal{V}}|}\sum_{v_{i}\in\widetilde{\mathcal{V}}}(1-\frac{{\bm{x}}^{T}_{i}{\bm{z}}_{i}}{\|{\bm{x}}_{i}\|\cdot\|{\bm{z}}_{i}\|})^{\gamma},~\gamma\geq 1,$

## Block 71

which is averaged over all masked nodes. The scaling factor $\gamma$ is a hyper-parameter adjustable over different datasets. This scaling technique could also be viewed as an adaptive sample reweighing, and the weight of each sample is adjusted with the reconstruction error. This error is also famous in the field of supervised object detection as the focal loss ( Lin et al., 2017 ) .

## Block 72

In summary, GraphMAE is a simple and scalable self-supervised graph learning framework. Figure 1 illustrates how each of its design choices directly impacts the performance of the self-supervised GraphMAE framework. By identifying the negative components and designing new strategies, GraphMAE unleashes the power of autoencoders for self-supervised graph pre-training.

## Block 73

3.3. Training and Inference

## Block 74

The overall training flow of GraphMAE is summarized by Figure 2 . First, given an input graph, we randomly select a certain proportion of nodes and replace their node features with the mask-token [MASK]. We feed the graph with partially observed features into the encoder to generate the encoded node representations. In decoding, we re-mask the selected nodes and replace their features with another token [DMASK]. Then the decoder is applied to the re-masked graph to reconstruct the original node features with the proposed scaled cosine error.

## Block 75

For downstream applications, the encoder is applied to the input graph without any masking in the inference stage. The generated node embeddings can be used for various graph learning tasks, such as node classification and graph classification. For graph-level tasks, we use a non-parameterized graph pooling (readout) function, e.g., MaxPooling and MeanPooling, to obtain the graph-level representation ${\bm{h}}^{g}=\mathrm{READOUT}(\{{\bm{h}}_{i},v_{i}\in\mathcal{G}_{g}\})$ . In addition, similar to ( Hu et al., 2020c ) , GraphMAE also enables robust transfer of pre-trained GNN models to various downstream tasks. In the experiments, we show that GraphMAE achieves competitive performance in both node-level and graph-level applications.
