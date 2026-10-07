# paretognn inspected method scope

Primary URL: https://arxiv.org/html/2210.02016v3

Zero full-paper or author-source credit.

## Block 15

2 Multi-Task Self-supervised Learning via ParetoGNN

## Block 16

In this section, we illustrate our proposed multi-task self-supervised learning framework for GNNs, namely ParetoGNN . As Figure 1 illustrates, ParetoGNN is trained with different SSL tasks simultaneously to enhance the task generalization. Specifically, given a graph $\mathcal{G}$ with $N$ nodes and their corresponding $D$ -dimensional input features $\mathbf{X}\in\mathbb{R}^{N\times D}$ , ParetoGNN learns a GNN encoder $f_{g}(\cdot;\bm{\theta}_{g}):\mathcal{G}\rightarrow\mathbb{R}^{N\times d}$ parameterized by $\bm{\theta}_{g}$ , that maps every node in $\mathcal{G}$ to a $d$ -dimensional vector (s.t. $d\ll N$ ). The resulting node representations should retain competitive performance across various downstream tasks without any update on $\bm{\theta}_{g}$ . With $K$ self-supervised tasks, we consider the loss function for $k$ -th SSL task as $\mathcal{L}_{k}(\mathcal{G};\mathcal{T}_{k},\bm{\theta}_{g},\bm{\theta}_{k}):\mathcal{G}\rightarrow\mathbb{R}^{+}$ , where $\mathcal{T}_{k}$ refers to the graph augmentation function required for $k$ -th task, and $\bm{\theta}_{k}$ refers to task-specific parameters for $k$ -th task (e.g., MLP projection head and/or GNN decoder). In ParetoGNN , all SSL tasks are dynamically reconciled by promoting Pareto optimality, where the norm of gradients w.r.t. the parameters of our GNN encoder $\bm{\theta}_{g}$ is minimized in the convex hull. Such gradients guarantee a descent direction to the Pareto optimality, which enhances task generalization while minimizing potential conflicts.

## Block 17

2.1 Multi-task Graph Self-supervised Learning

## Block 18

ParetoGNN is a general framework for multi-task self-supervised learning over graphs. We regard the full graph $\mathcal{G}$ as the data source; and for each task, ParetoGNN is self-supervised by sub-graphs sampled from $\mathcal{G}$ , followed by task-specific augmentations (i.e., $\mathcal{T}_{k}(\cdot)$ ). The rationale behind the exploration of sub-graphs is two-fold. Firstly, the process of graph sampling is naturally a type of augmentation ( Zeng et al., 2019 ) by enlarging the diversity of the training data. Secondly, modeling over sub-graphs is more memory efficient, which is significant especially under the multi-task scenario. In this work, we design five simple pretext tasks spanning three high-level philosophies, including generative reconstruction, whitening decorrelation, and mutual information maximization. However we note that ParetoGNN is not limited to the current learning objectives and the incorporation of other philosophies is a straightforward extension. Three high-level philosophies and their corresponding five pretext tasks are illustrated as follows:

## Block 19

Generative reconstruction . Recent studies ( Zhang et al., 2021d ; Hou et al., 2022 ) have demonstrated that node features contain rich information, which highly correlates to the graph topology. To encode node features into the representations derived by ParetoGNN , we mask the features of a random batch of nodes, forward the masked graph through the GNN encoder, and reconstruct the masked node features given the node representations of their local sub-graphs ( Hou et al., 2022 ) . Furthermore, we conduct the similar reconstruction process for links between the connected nodes to retain the pair-wise topological knowledge ( Zhang & Chen, 2018 ) . Feature and topology reconstruction are denoted as FeatRec and TopoRec respectively.

## Block 20

Whitening decorrelation . SSL based on whitening decorrelation has gained tremendous attention, owing its capability of learning representative embeddings without prohibitively expensive negative pairs or offline encoders ( Ermolov et al., 2021 ; Zbontar et al., 2021 ) . We adapt the same philosophy to graph SSL by independently augmenting the same sub-graph into two views, and then minimize the distance between the same nodes in the two views while enforcing the feature-wise covariance of all nodes equal to the identity matrix. We denote this pretext as RepDecor .

## Block 21

Mutual Information Maximization . Maximizing the mutual information between two corrupted views of the same target has been proved to learn the intrinsic patterns, as demonstrated by deep infomax-based methods ( Bachman et al., 2019 ; Velickovic et al., 2019 ) and contrastive learning methods ( Hassani & Khasahmadi, 2020 ; Zhu et al., 2020 ) . We maximize the local-global mutual information by minimizing the distance between the graph-level representation of the intact sub-graph and its node representations, while maximizing the distance between the former and the corrupted node representations. Besides, we also maximize the local sub-graph mutual information by maximizing similarity of representations of two views of the sub-graph entailed by the same anchor nodes, while minimizing the similarities of the representations of the sub-graphs entailed by different anchor nodes. The pretext tasks based on node-graph mutual information and node-subgraph mutual information are denoted as MI-NG and MI-NSG , respectively.

## Block 22

Technical details and objective formulation of these five tasks are provided in Appendix B . As described above, pretext tasks under different philosophies capture distinct dimensions of the same graph. Empirically, we observe that simply combining all pretext SSL tasks with weighted summation can sometimes already lead to better task generalization over various downstream tasks and datasets. Though promising, according to our empirical studies, such a multi-task self-supervised GNN falls short on some downstream tasks, if compared with the best-performing experts on these tasks. This phenomenon indicates that with the weighted summation there exist potential conflicts between different SSL tasks, which is also empirically shown by previous works from other domains such as Computer Vision ( Sener & Koltun, 2018 ; Chen et al., 2018 ) .

## Block 23

2.2 Multi-task Graph SSL promoting Pareto Optimality

## Block 24

To mitigate the aforementioned problem and simultaneously optimize multiple SSL tasks, we can derive the following empirical loss minimization formulation as:

## Block 25

$\min_{\begin{subarray}{c}\bm{\theta}_{g},\\ \bm{\theta}_{1},\ldots,\bm{\theta}_{K}\end{subarray}}\sum_{k=1}^{K}\alpha_{k}\cdot\mathcal{L}_{k}(\mathcal{G};\mathcal{T}_{k},\bm{\theta}_{g},\bm{\theta}_{k}),$ (1)

## Block 26

where $\alpha_{k}$ is the task weight for $k$ -th SSL task computed according to pre-defined heuristics. For instance, AutoSSL ( Jin et al., 2022 ) derives task weights that promote pseudo-homophily. Though such a formulation is intuitively reasonable for some graphs and tasks, heterophilous graphs, which are not negligible in the real world, directly contradict the homophily assumption. Moreover, not all downstream tasks benefit from such homophily assumption, which we later validate in the experiments. Hence, it is non-trivial to come up with a unified heuristic that suits all graphs and downstream tasks. In addition, weighted summation of multiple SSL objectives might cause undesirable behaviors, such as performance instabilities entailed by conflicting objectives or different gradient scales ( Chen et al., 2018 ; Kendall et al., 2018 ) .

## Block 27

Therefore, we take an alternative approach and formulate this problem as multi-objective optimization with a vector-valued loss $\bm{\mathcal{L}}$ , as the following:

## Block 28

$\min_{\begin{subarray}{c}\bm{\theta}_{g},\\ \bm{\theta}_{1},\ldots,\bm{\theta}_{K}\end{subarray}}\bm{\mathcal{L}}(\mathcal{G},\bm{\theta}_{g},\bm{\theta}_{1},\ldots,\bm{\theta}_{K})=\min_{\begin{subarray}{c}\bm{\theta}_{g},\\ \bm{\theta}_{1},\ldots,\bm{\theta}_{K}\end{subarray}}\big(\mathcal{L}_{1}(\mathcal{G};\mathcal{T}_{1},\bm{\theta}_{g},\bm{\theta}_{1}),\ldots,\mathcal{L}_{K}(\mathcal{G};\mathcal{T}_{K},\bm{\theta}_{g},\bm{\theta}_{K})\big).$ (2)

## Block 29

The focus of multi-objective optimization is approaching Pareto optimality ( Désidéri, 2012 ) , which in the multi-task SSL setting can be defined as follows:

## Block 30

Definition 1 (Pareto Optimality) .

## Block 31

A set of solution ( $\bm{\theta}_{g}^{\star}$ , $\bm{\theta}_{1}^{\star}$ , …, $\bm{\theta}_{K}^{\star}$ ) is Pareto optimal if and only if there does not exists a set of solution that dominates ( $\bm{\theta}_{g}^{\star}$ , $\bm{\theta}_{1}^{\star}$ , …, $\bm{\theta}_{K}^{\star}$ ). ( $\bm{\theta}_{g}^{\star}$ , $\bm{\theta}_{1}^{\star}$ , …, $\bm{\theta}_{K}^{\star}$ ) dominates ( $\bm{\hat{\theta}}_{g}$ , $\bm{\hat{\theta}}_{1}$ , …, $\bm{\hat{\theta}}_{K}$ ) if for every SSL task $k$ , $\mathcal{L}_{k}(\mathcal{G};\mathcal{T}_{k},\bm{\hat{\theta}}_{g},\bm{\hat{\theta}}_{k})\geq\mathcal{L}_{k}(\mathcal{G};\mathcal{T}_{k},\bm{\theta}_{g}^{\star},\bm{\theta}_{k}^{\star})$ and $\bm{\mathcal{L}}(\mathcal{G},\bm{\hat{\theta}}_{g},\bm{\hat{\theta}}_{1},\ldots,\bm{\hat{\theta}}_{K})\neq\bm{\mathcal{L}}(\mathcal{G},\bm{\theta}_{g}^{\star},\bm{\theta}_{1}^{\star},\ldots,\bm{\theta}_{K}^{\star})$ .

## Block 32

In other words, if a self-supervised GNN is Pareto optimal, it is impossible to further optimize any SSL task without sacrificing the performance of at least one other SSL task. Finding the Pareto optimal model is not sensible if there exist a set of parameters that can easily fit all SSL tasks (i.e., no matter how different SSL tasks are reconciled, such a model approaches Pareto optimality where every SSL task is perfectly fitted). However, this is rarely the case, because solving all difficult pretext tasks is non-trivial for only one set of parameters. By promoting the Pareto optimlaity, ParetoGNN is enforced to learn intrinsic patterns applicable to a number of pretext tasks, which further enhances the task generalization across various downstream tasks.

## Block 33

2.3 Pareto Optimality by Multiple Gradient Descent Algorithm

## Block 34

To obtain the Pareto optimal parameters, we explore the Multiple Gradient Descent Algorithm (MGDA) ( Désidéri, 2012 ) and adapt it to the multi-task SSL setting. Specifically, MGDA leverages the saddle-point test and theoretically proves that a solution (i.e., the combined gradient descent direction or task weight assignments in our case) that satisfies the saddle-point test gives a descent direction that improves all tasks and eventually approaches the Pareto optimality. We further elaborate descriptions of the saddle-point test for the share parameters $\bm{\theta}_{g}$ and task-specific parameters $\bm{\theta}_{k}$ in Appendix G . In our multi-task SSL scenario, the optimization problem can be formulated as:

## Block 35

$\min_{\alpha_{1},\ldots,\alpha_{K}}\Bigg\lvert\Bigg\lvert\sum_{k=1}^{K}\alpha_{k}\cdot\nabla_{\bm{\theta}_{g}}\mathcal{L}_{k}(\mathcal{G};\mathcal{T}_{k},\bm{\theta}_{g},\bm{\theta}_{k})\Bigg\rvert\Bigg\lvert_{F},\;\;\;\;\text{s.t.}\;\;\sum_{k=1}^{K}\alpha_{k}=1\;\;\;\;\text{and}\;\;\;\;\forall{k}\;\;\alpha_{k}\geq 0,$ (3)

## Block 36

where $\nabla_{\bm{\theta}_{g}}\mathcal{L}_{k}(\mathcal{G};\mathcal{T}_{k},\bm{\theta}_{g},\bm{\theta}_{k})\in\mathbb{R}^{1\times\lvert\bm{\theta}_{g}\rvert}$ refers to the gradients of parameters for the GNN encoder w.r.t. the $k$ -th SSL task. ParetoGNN can be trained using Equation 1 with the task weights derived by the above optimization. As shown in Figure 1 , optimizing the above objective is essentially finding descent direction with the minimum norm within the convex hull defined by the gradient direction of each SSL task. Hence, the solution to Equation 3 is straight-forward when $K=2$ (i.e., only two gradient descent directions involved). If the norm of one gradient is smaller than their inner product, the solution is simply the gradient with the smaller norm (i.e., ( $\alpha_{1}=0$ , $\alpha_{2}=1$ ) or vice versa). Otherwise, $\alpha_{1}$ can be calculated by deriving the descent direction perpendicular to the convex line with only one step, formulated as:

## Block 37

$\alpha_{1}=\frac{\nabla_{\bm{\theta}_{g}}\mathcal{L}_{2}(\mathcal{G};\mathcal{T}_{2},\bm{\theta}_{g},\bm{\theta}_{2})\cdot\big(\nabla_{\bm{\theta}_{g}}\mathcal{L}_{2}(\mathcal{G};\mathcal{T}_{2},\bm{\theta}_{g},\bm{\theta}_{2})-\nabla_{\bm{\theta}_{g}}\mathcal{L}_{1}(\mathcal{G};\mathcal{T}_{1},\bm{\theta}_{g},\bm{\theta}_{1})\big)^{\intercal}}{\big\lvert\big\lvert\nabla_{\bm{\theta}_{g}}\mathcal{L}_{2}(\mathcal{G};\mathcal{T}_{2},\bm{\theta}_{g},\bm{\theta}_{2})-\nabla_{\bm{\theta}_{g}}\mathcal{L}_{1}(\mathcal{G};\mathcal{T}_{1},\bm{\theta}_{g},\bm{\theta}_{1})\big\rvert\big\rvert_{F}}.$ (4)

## Block 38

When $K>2$ , we minimize the quadratic form $\bm{\alpha}(\nabla_{\bm{\theta}_{g}}\bm{\mathcal{L}})(\nabla_{\bm{\theta}_{g}}\bm{\mathcal{L}})^{\intercal}\bm{\alpha}^{\intercal}$ , where $\nabla_{\bm{\theta}_{g}}\bm{\mathcal{L}}\in\mathbb{R}^{K\times|\bm{\theta}_{g}|}$ refers to the vertically concatenated gradients w.r.t. $\bm{\theta}_{g}$ for all SSL tasks, and $\bm{\alpha}\in\mathbb{R}^{1\times K}$ is the vector for task weight assignments such that $\lvert\lvert\bm{\alpha}\rvert\rvert=1$ . Inspired by Frank-Wolfe algorithm ( Jaggi, 2013 ) , we iteratively solve such a quadratic problem as a special case of Equation 4 . Specifically, we first initialize every element in $\bm{\alpha}$ as $1/K$ , and we increment the weight of the task (denoted as $t$ ) whose descent direction correlates least with the current combined descent direction (i.e., $\sum_{k=1}^{K}\alpha_{k}\cdot\nabla_{\bm{\theta}_{g}}\mathcal{L}_{k}(\mathcal{G};\mathcal{T}_{k},\bm{\theta}_{g},\bm{\theta}_{k})$ ). The step size $\eta$ of this increment can be calculated by utilizing the idea of Equation 4 , where we replace $\nabla_{\bm{\theta}_{g}}\mathcal{L}_{2}(\mathcal{G};\mathcal{T}_{2},\bm{\theta}_{g},\bm{\theta}_{2})$ with $\sum_{k=1}^{K}\alpha_{k}\cdot\nabla_{\bm{\theta}_{g}}\mathcal{L}_{k}(\mathcal{G};\mathcal{T}_{k},\bm{\theta}_{g},\bm{\theta}_{k})$ and replace $\nabla_{\bm{\theta}_{g}}\mathcal{L}_{1}(\mathcal{G};\mathcal{T}_{1},\bm{\theta}_{g},\bm{\theta}_{1})$ with $\nabla_{\bm{\theta}_{g}}\mathcal{L}_{t}(\mathcal{G};\mathcal{T}_{t},\bm{\theta}_{g},\bm{\theta}_{t})$ . One iteration of solving this quadratic problem is formulated as:

## Block 39

$\bm{\alpha}:=(1-\eta)\cdot\bm{\alpha}+\eta\cdot\mathbf{e}_{t},\;\;\;\text{s.t. }\;\;\eta=\frac{\hat{\nabla}_{\bm{\theta}_{g}}\cdot\big(\hat{\nabla}_{\bm{\theta}_{g}}-\nabla_{\bm{\theta}_{g}}\mathcal{L}_{t}(\mathcal{G};\mathcal{T}_{t},\bm{\theta}_{g},\bm{\theta}_{t})\big)^{\intercal}}{\big\lvert\big\lvert\hat{\nabla}_{\bm{\theta}_{g}}-\nabla_{\bm{\theta}_{g}}\mathcal{L}_{t}(\mathcal{G};\mathcal{T}_{t},\bm{\theta}_{g},\bm{\theta}_{t})\big\rvert\big\rvert_{F}},$ (5)

## Block 40

where $t=\argmin_{r}\sum_{i=1}^{K}\alpha_{i}\cdot\nabla_{\bm{\theta}_{g}}\mathcal{L}_{i}(\mathcal{G};\mathcal{T}_{i},\bm{\theta}_{g},\bm{\theta}_{i})\cdot\nabla_{\bm{\theta}_{g}}\mathcal{L}_{r}(\mathcal{G};\mathcal{T}_{r},\bm{\theta}_{g},\bm{\theta}_{r})^{\intercal}$ and $\hat{\nabla}_{\bm{\theta}_{g}}=\sum_{k=1}^{K}\alpha_{k}\cdot\nabla_{\bm{\theta}_{g}}\mathcal{L}_{k}(\mathcal{G};\mathcal{T}_{k},\bm{\theta}_{g},\bm{\theta}_{k}).$ $\mathbf{e}_{t}$ in the above solution refers to an one-hot vector with $t$ -th element equal to 1. The optimization described in Equation 5 iterates until $\eta$ is smaller than a small constant $\xi$ or the number of iterations reaches the pre-defined number $\gamma$ . Furthermore, the above task reconciliation of ParetoGNN satisfies the following theorem.

## Block 41

Theorem 1 .

## Block 42

Assuming that $\bm{\alpha}(\nabla_{\bm{\theta}_{g}}\bm{\mathcal{L}})(\nabla_{\bm{\theta}_{g}}\bm{\mathcal{L}})^{\intercal}\bm{\alpha}^{\intercal}$ is $\beta$ -smooth. $\bm{\alpha}$ converges to the optimal point at a rate of $\mathcal{O}(1/\gamma)$ , and $\bm{\alpha}$ is at most $4\beta/(\gamma+1)$ away from the optimal solution.

## Block 43

The proof of Theorem 1 is provided in Appendix A . According to this theorem and our empirical observation, the optimization process in Equation 5 would terminate and deliver well-approximated results with $\gamma$ set to an affordable value (e.g., $\gamma=100$ ).

## Block 95

Appendix B Description of the Pretext Tasks

## Block 96

In this section, we demonstrate the design of our proposed five pretext tasks, including two based on generative reconstruction (i.e., FeatRec and TopoRec ), one based on whitening decorrelation (i.e., RepDecor ) and two based on mutual information maximization (i.e., MI-NG and MI-NSG ).

## Block 97

We first explain the operation of graph convolution as proposed by Kipf & Welling (2016a) . The key mechanism of graph convolution is layer-wise message passing where a node iteratively extracts information from its first-order neighbors and information from milti-hop neighbors can be captured through stacked convolution layers. Specifically, at $l$ -th layer, this process is formulated as follows:

## Block 98

$\mathbf{H}^{l+1}=\sigma(\mathbf{A}\cdot\mathbf{H}^{l}\cdot\mathbf{W}^{l}),$ (13)

## Block 99

where $\mathbf{H}^{0}=\mathbf{X}$ , $\mathbf{A}\in\{0,1\}^{N\times N}$ is the adjacency matrix of the input graph, $\sigma(\cdot)$ refers to the non-linear activation function, $\mathbf{W}^{l}\in\mathbb{R}^{d^{l}\times d^{l+1}}$ refers to the learnable parameters of $l$ -th layer, and $d^{l}$ and $d^{l+1}$ are the hidden dimensions at these two consecutive layers respectively. The graph encoder $f_{g}(\cdot;\bm{\theta}_{g}):\mathcal{G}\rightarrow\mathbb{R}^{N\times d}$ of ParetoGNN is constructed by stacked graph convolution layers as:

## Block 100

$f_{g}(\mathcal{G};\bm{\theta}_{g})=\mathbf{H}^{L}=\sigma(\mathbf{A}\cdot\mathbf{H}^{L-1}\cdot\mathbf{W}^{L-1}),$ (14)

## Block 101

where $L$ stands for the number of layers in the encoder of ParetoGNN and $\bm{\theta}_{g}=\{\mathbf{W}^{l}\}_{l=0}^{L-1}$ .

## Block 102

As briefly described in Section 2.1 , we regard the full graph $\mathcal{G}$ as the data source; and for each task, ParetoGNN is self-supervised by sub-graphs sampled from $\mathcal{G}$ , followed by task-specific augmentations (i.e., $\mathcal{T}_{t}(\cdot)$ ). The graph sampling strategy is fairly straightforward. For the pretext task $t$ , we select the sub-graph constituted by nodes within $k_{t}$ hops of $N_{t}$ randomly selected seed nodes, where $k_{t}$ and $N_{t}$ are two task-specific hyper-parameters. The graph augmentation operations we explore include feature masking, edge dropping, and node dropping. For the simplicity of denotation, we unify the operation of graph sampling and graph augmentation together as $\mathcal{T}_{t}(\cdot)$ . Task-specific hyper-parameters for graph augmentation and sub-graph sampling are covered in Appendix D .

## Block 103

B.1 Generative Reconstruction

## Block 104

Feat ure rec onstruction , denoted as FeatRec , utilizes the high-level idea from Zhang et al. (2021d) , proving that topological information can be referred purely from the node features. Hence to utilize such inductive bias, following the implementation of GraphMAE ( Hou et al., 2022 ) , we mask the node features and forward the masked graphs through $f_{g}(\cdot;\bm{\theta}_{g})$ . Then we re-mask the previous masked nodes and feed the resulted graph to a convolution-based decoder, formulated as:

## Block 105

$\hat{\mathbf{X}}^{\prime}=\mathbf{A}^{\prime}\cdot f_{g}(\mathcal{G}^{\prime};\bm{\theta}_{g})\odot\mathbf{M}\cdot\mathbf{W}^{\text{Dec}},$ (15)

## Block 106

where $\odot$ refers to Hadamard product, $\mathbf{A}^{\prime}$ is the adjacency matrix of the sampled sub-graph $\mathcal{G}^{\prime}\sim\mathcal{T}_{\texttt{FeatRec}}(\mathcal{G})$ , $\mathbf{M}\in\{0,1\}^{N^{\prime}\times d}$ is the feature mask matrix whose rows equal to $\mathbf{1}$ if their corresponding nodes are targeted for reconstruction, and $\mathbf{W}^{\text{Dec}}\in\mathbb{R}^{d\times D}$ is the parameter matrix for the feature decoder. The objective for FeatRec is formulated as:

## Block 107

$\mathcal{L}_{\texttt{FeatRec}}=\frac{\lvert\lvert\hat{\mathbf{X}}^{\prime}\odot\hat{\mathbf{M}}-\mathbf{X}^{\prime}\odot\hat{\mathbf{M}}\rvert\rvert_{F}}{\lvert\lvert\mathbf{X}^{\prime}\odot\hat{\mathbf{M}}\rvert\rvert_{F}},$ (16)

## Block 108

where $\hat{\mathbf{M}}\in\{0,1\}^{N^{\prime}\times D}$ is the mask matrix defined similarly to $\mathbf{M}$ with different dimension, and $\mathbf{X}^{\prime}\in\mathbb{R}^{N^{\prime}\times D}$ is the feature matrix of the sampled sub-graph $\mathcal{G}^{\prime}\sim\mathcal{T}_{\texttt{FeatRec}}(\mathcal{G})$ .

## Block 109

Topo logical rec onstruction , denoted as TopoRec , aims at capturing the pair-wise relationships between the connected nodes. Given a sampled sub-graph $\mathcal{G}^{\prime}\sim\mathcal{T}_{\texttt{TopoRec}}(\mathcal{G})$ , we randomly select $B$ pairs of nodes $V^{+}=\{(i,j)|\mathbf{A}^{\prime}_{i,j}=1\}$ and another $B$ pairs of nodes $V^{-}=\{(i,j)|\mathbf{A}^{\prime}_{i,j}=0\}$ . The connection between two node $i$ and $j$ is measured by a logit calculated as:

## Block 110

$P_{\texttt{TopoRec}}(i,j)=\sigma\big((f_{g}(\mathcal{G}^{\prime};\bm{\theta}_{g})[i]\odot f_{g}(\mathcal{G}^{\prime};\bm{\theta}_{g})[j])\cdot\mathbf{W}^{\text{Topo}}\big),$ (17)

## Block 111

where $[\cdot]$ refers to the indexing operation, and $\mathbf{W}^{\text{Topo}}\in\mathbb{R}^{d\times 1}$ is the parameter vector. The objective of TopoRec is maximizing the $P_{\texttt{TopoRec}}$ for nodes in $V^{+}$ and minimizing for nodes in $V^{-}$ , formulated as a binary cross entropy loss as:

## Block 112

$\mathcal{L}_{\texttt{TopoRec}}=-\frac{1}{2B}\sum_{(i,j)\in V^{+}}\log(P_{\texttt{TopoRec}}(i,j))+\sum_{(i,j)\in V^{-}}\log(1-P_{\texttt{TopoRec}}(i,j)).$ (18)

## Block 113

B.2 Whitening Decorrelation

## Block 114

Rep resentation decor relation , denoted as RepDecor , encourages the similarities between the representations of the same nodes in two independently augmented sub-graphs. During this process, to the prevent the node representations from collapsing into a trivial solution, the covariance between representations matrices of two sub-graphs are enforced to be an identity matrix, such that the knowledge learned by each dimension in the hidden space is orthogonal to each other ( Ermolov et al., 2021 ; Zbontar et al., 2021 ; Zhang et al., 2021b ) . Given two sub-graphs $\mathcal{G}^{\prime}_{1},\mathcal{G}^{\prime}_{2}\sim\mathcal{T}_{\texttt{TopoRec}}(\mathcal{G})$ that are constituted by the same seed nodes but augmented differently, the objective of RepDecor is formulated as:

## Block 115

$\mathcal{L}_{\texttt{RepDecor}}=\Big\lvert\Big\lvert f_{g}(\mathcal{G}^{\prime}_{1};\bm{\theta}_{g})-f_{g}(\mathcal{G}^{\prime}_{2};\bm{\theta}_{g})\Big\rvert\Big\rvert_{F}+\alpha\cdot\Big\lvert\Big\lvert f_{g}(\mathcal{G}^{\prime}_{1};\bm{\theta}_{g})^{\intercal}\cdot f_{g}(\mathcal{G}^{\prime}_{2};\bm{\theta}_{g})^{\intercal}-\mathbf{I}^{d\times d}\Big\rvert\Big\rvert_{F},$ (19)

## Block 116

where the first term encourages the node similarity and the second term regularize the solution from collapsing, $\mathbf{I}^{d\times d}$ is the square identity matrix with dimension $d\times d$ , and $\alpha$ refers to a pre-defined balancing term (i.e., we use an $\alpha$ of 1e-3 across all datasets).

## Block 117

B.3 Mutual Information Maximization

## Block 118

M utual i nformation between n odes and the whole g raph , denoted as MI-NG , enables the graph encoder to learn coarse graph-level knowledge. Specifically, given a sampled sub-graph $\mathcal{G}^{\prime}\sim\mathcal{T}_{\texttt{MI-NG}}(\mathcal{G})$ , we first corrupt $\mathcal{G}^{\prime}$ into $\mathcal{G}^{\prime\prime}$ by feature shuffling ( Velickovic et al., 2019 ) . Then we extract the hidden graph-level representation of $\mathcal{G}^{\prime}$ by the graph mean pooling ( Xu et al., 2018a ) , and enforce the representations of nodes in $\mathcal{G}^{\prime}$ similar to the pooled representation while nodes in $\mathcal{G}^{\prime\prime}$ far away from the pooled representation. This pretext task allows the graph encoder to capture the perturbation brought by the topological change (i.e., feature shuffling). The objective of MI-NG is formulated as a binary cross entropy as:

## Block 119

$\mathcal{L}_{\texttt{MI-NG}}=-\frac{1}{2N^{\prime}}\sum_{i=1}^{N^{\prime}}\log\big(P_{\texttt{MI-NG}}(f_{g}(\mathcal{G}^{\prime};\bm{\theta}_{g})[i],\mathbf{h}_{g})\big)+\log\big(1-P_{\texttt{MI-NG}}(f_{g}(\mathcal{G}^{\prime\prime};\bm{\theta}_{g})[i],\mathbf{h}_{g})\big),$

## Block 120

$\text{s.t.\;\;}\mathbf{h}_{g}=\textsc{Pool}\big(f_{g}(\mathcal{G}^{\prime};\bm{\theta}_{g})\big),\text{\;and\;\;}P_{\texttt{MI-NG}}(\mathbf{h},\mathbf{h}_{g}\big)=(\mathbf{h}||\mathbf{h}_{g})\cdot\mathbf{W}^{\texttt{MI-NG}},$ (20)

## Block 121

where $\textsc{Pool}(\cdot)$ refers to the graph mean pooling function ( Xu et al., 2018a ) , $||$ is the horizontal concatenation operation, and $\mathbf{W}^{\texttt{MI-NG}}\in\mathbb{R}^{2d\times 1}$ is the parameter vector.

## Block 122

M utual i nformation between n odes and their s ub- g raphs , denoted as MI-NSG , enables the graph encoder to learn fine-grained graph-level knowledge. Unlike MI-NG that enforces mutual information between node representations and the graph-level representation, MI-NSG maximizes the mutual information between the independently augmented sub-graphs entailed by the same anchor nodes, which learns fine-grained knowledge compared with MI-NG . Specifically, given two sub-graphs $\mathcal{G}^{\prime}_{1},\mathcal{G}^{\prime}_{2}\sim\mathcal{T}_{\texttt{TopoRec}}(\mathcal{G})$ that are constituted by the same seed nodes but augmented differently, the objective of MI-NSG is formulated as a variant of InfoNCE ( Chen et al., 2020 ) :

## Block 123

$\mathcal{L}_{\texttt{MI-NSG}}=-\frac{1}{N^{\prime}}\sum_{i=1}^{N^{\prime}}\log\Big(\frac{\textsc{Exp}\Big(\textsc{Sim}\big(f_{g}(\mathcal{G}^{\prime}_{1};\bm{\theta}_{g})[i],f_{g}(\mathcal{G}^{\prime}_{2};\bm{\theta}_{g})[i]\big)/\tau\Big)}{\sum_{j=1}^{N^{\prime}}\textsc{Exp}\Big(\textsc{Sim}\big(f_{g}(\mathcal{G}^{\prime}_{1};\bm{\theta}_{g})[i],f_{g}(\mathcal{G}^{\prime}_{1};\bm{\theta}_{g})[j]\big)/\tau\Big)+\textsc{Exp}\Big(\textsc{Sim}\big(f_{g}(\mathcal{G}^{\prime}_{1};\bm{\theta}_{g})[i],f_{g}(\mathcal{G}^{\prime}_{2};\bm{\theta}_{g})[j]\big)/\tau\Big)}\Big),$ (21)

## Block 124

where $\textsc{Sim}(\mathbf{h}_{1},\mathbf{h}_{2})=\frac{\mathbf{h}_{1}\cdot\mathbf{h}_{2}^{\intercal}}{\lvert\lvert\mathbf{h}_{1}\rvert\rvert_{F}\cdot\lvert\lvert\mathbf{h}_{2}\rvert\rvert_{F}}$ is the similarity metric, $\textsc{Exp}(\cdot)$ stands for the exponential function, and $\tau$ is the temperature hyper-parameter used to control the sharpness of the similarity distribution (i.e., we explore a $\tau$ of 0.1 across all datasets).
