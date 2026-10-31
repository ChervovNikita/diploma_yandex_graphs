# Exact primary and canonical source passages

Date:2026-10-02. Source-only text custody; no model/source import or execution. UTF-8 offsets retain original LF/CRLF. PyG2.7.0 Git commit 76ff9c2ce18c8cebf52122b57e2aeadce9793d10; selected runtime Torch2.1.2+cu118 per parent.

## C1: Canonical original branch and four serial member forwards

`sources/coordinate_models.py`, LF lines153–211; Unicode offset6282,length3134; slice SHA256 `74a272aedb4768083619439f9e291324862dc714a1a2cf8db5f7c904d60109e4`.

~~~~text
            for p in (block.R, block.S, block.B)
        }
        if arm == "original":
            return
        generator = torch.Generator(device="cpu").manual_seed(permutation_seed)
        for layer, residual in enumerate(base.residual_modules):
            module = residual.module
            conv = module.conv
            if (conv.aggr != "mean" or conv.project or conv.normalize
                    or not conv.root_weight):
                raise ValueError("Only the original default homogeneous SAGE contract is supported")
            sites = (
                (conv, "lin_l", f"residual_modules.{layer}.module.conv.lin_l"),
                (conv, "lin_r", f"residual_modules.{layer}.module.conv.lin_r"),
                (module.feed_forward_module, "linear_1",
                 f"residual_modules.{layer}.module.feed_forward_module.linear_1"),
                (module.feed_forward_module, "linear_2",
                 f"residual_modules.{layer}.module.feed_forward_module.linear_2"),
            )
            for owner, name, path in sites:
                original_map = getattr(owner, name)
                if isinstance(original_map, ContextLinear):
                    raise RuntimeError("Attempted to wrap a hidden map twice")
                context = ContextLinear(original_map, members, arm, generator)
                setattr(owner, name, context)
                self.hidden_context_paths.append("base." + path)
                self._private_parameter_ids.update(id(p) for p in context.private_parameters())
        # BEBlock.W is deliberately untouched: no input/class-axis permutation.

    def forward_member(self, graph, x: torch.Tensor, member: int) -> torch.Tensor:
        _member_check(member, self.members)
        x = self.base.input_be_block(x, tabm_seed=member)
        x = self.base.act(self.base.dropout(x))
        for residual in self.base.residual_modules:
            if self.arm == "original":
                x = residual(graph, x)
                continue
            normalized = residual.normalization(x)
            module = residual.module
            conv = module.conv
            # Exactly the default SAGEConv order: mean neighbors, lin_l,
            # lin_r(root), addition. ContextLinear needs an explicit member.
            neighbors = conv.propagate(
                graph.edge_index, x=(normalized, normalized), size=None,
            )
            message = conv.lin_l(neighbors, member) + conv.lin_r(normalized, member)
            ff = module.feed_forward_module
            branch = ff.linear_1(torch.cat([normalized, message], dim=1), member)
            branch = ff.act(ff.dropout_1(branch))
            branch = ff.dropout_2(ff.linear_2(branch, member))
            x = x + branch
        x = self.base.output_normalization(x)
        # Unlike the original class, retain the C axis even when C=1.
        return self.base.output_be_block(x, tabm_seed=member)

    def forward(self, graph, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.forward_member(graph, x, m) for m in range(self.members)])

    def storage_report(self) -> dict[str, Any]:

~~~~

## C2: Full untied trajectories and genuinely shared-trunk heads

`sources/coordinate_models.py`, LF lines217–264; Unicode offset9548,length2011; slice SHA256 `8af028cdc9b7729ee79af781752b68cb7f9d4ef3208f04309e8082fc7a79539a`.

~~~~text
        super().__init__()
        self.models = nn.ModuleList(models)
        self.arm = arm
        self.members = len(models)
        self._private_parameter_ids = (
            {id(p) for p in self.parameters()} if arm == "untied" else set()
        )

    def forward_member(self, graph, x: torch.Tensor, member: int) -> torch.Tensor:
        _member_check(member, self.members)
        base = self.models[member]
        return base.output_linear(_common_trunk(base, graph, x))

    def forward(self, graph, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.forward_member(graph, x, m) for m in range(self.members)])

    def storage_report(self) -> dict[str, Any]:
        return storage_report(self)


class SharedTrunkHeads(nn.Module):
    """One complete shared SAGE trunk with M capable h->h->C GELU heads."""

    def __init__(self, base, hidden_dim: int, output_dim: int, members: int, dropout: float):
        super().__init__()
        self.base = base
        # Remove the unused original readout and its registered parameters.
        self.base.output_linear = nn.Identity()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))
            for _ in range(members)
        ])
        self.arm = "heads"
        self.members = members
        self._private_parameter_ids = {id(p) for p in self.heads.parameters()}

    def forward_member(self, graph, x: torch.Tensor, member: int) -> torch.Tensor:
        _member_check(member, self.members)
        return self.heads[member](_common_trunk(self.base, graph, x))

    def forward(self, graph, x: torch.Tensor) -> torch.Tensor:
        # Credit this arm's real reuse: one trunk, followed by M heads.
        h = _common_trunk(self.base, graph, x)
        return torch.stack([head(h) for head in self.heads])

    def storage_report(self) -> dict[str, Any]:
        return storage_report(self)

~~~~

## F1: Feature-only residual norm, SAGE concat and FFN activation order

`sources/frozen_models.py`, LF lines8–68; Unicode offset226,length2329; slice SHA256 `76cae68dea146a4b66bb22b3749a81fc1e33554aea00370ee63d29cf133f563f`.

~~~~text
class ResidualModuleWrapper(nn.Module):
    def __init__(self, module, normalization, dim, **kwargs):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module(dim=dim, **kwargs)

    def forward(self, graph, x):
        x_res = self.normalization(x)
        x_res = self.module(graph, x_res)
        return x + x_res


class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x


class GCNModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
        super().__init__()
        self.conv = GCNConv(dim, dim, add_self_loops=False)
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        x = self.conv(x, graph.edge_index)
        x = self.feed_forward_module(graph, x)
        return x


class SAGEModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
        super().__init__()
        self.conv = SAGEConv(dim, dim)
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        message = self.conv(x, graph.edge_index)
        x = torch.cat([x, message], dim=1)
        x = self.feed_forward_module(graph, x)
        return x


~~~~

## F2: Private boundary BE and native TABM forward

`sources/frozen_models.py`, LF lines214–284; Unicode offset8201,length2776; slice SHA256 `2d4fa309f160ed1afe47d115e15275934d9d70baa0996f0ce2125986709e54dd`.

~~~~text
class BEBlock(torch.nn.Module):
    def __init__(self, in_shape, out_shape, is_first, device, tabm_inits):
        super().__init__()

        self.R = nn.Parameter(torch.empty(tabm_inits, in_shape, device=device))
        self.S = nn.Parameter(torch.empty(tabm_inits, out_shape, device=device))
        self.B = nn.Parameter(torch.empty(tabm_inits, out_shape, device=device))
        self.W = nn.Linear(in_shape, out_shape, bias=False)

        self.tabm_inits = tabm_inits
        self.is_first = is_first
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        if self.is_first:
            self.R.bernoulli_(0.5).mul_(2).sub_(1)
        else:
            self.R.fill_(1.0)
        self.S.fill_(1.0)
        self.B.zero_()

    def forward(self, x, tabm_seed):
        x = x * self.R[tabm_seed]
        x = self.W(x)
        x = x * self.S[tabm_seed]
        x = x + self.B[tabm_seed]
        return x


class TABMModel(nn.Module):
    def __init__(self, model_name, num_layers, input_dim, hidden_dim, output_dim, hidden_dim_multiplier, num_heads,
                 normalization, dropout, tabm_inits, device='cpu'):

        super().__init__()

        self.model_name = model_name
        normalization = NORMALIZATION[normalization]

        self.input_be_block = BEBlock(in_shape=input_dim, out_shape=hidden_dim, is_first=True, device=device, tabm_inits=tabm_inits)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            for module in MODULES[model_name]:
                residual_module = ResidualModuleWrapper(module=module,
                                                        normalization=normalization,
                                                        dim=hidden_dim,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        num_heads=num_heads,
                                                        dropout=dropout)

                self.residual_modules.append(residual_module)

        self.output_normalization = normalization(hidden_dim)
        self.output_be_block = BEBlock(in_shape=hidden_dim, out_shape=output_dim, is_first=False, device=device, tabm_inits=tabm_inits)

    def forward(self, graph, x, tabm_seed):
        x = self.input_be_block(x, tabm_seed=tabm_seed)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x = self.output_normalization(x)
        x = self.output_be_block(x, tabm_seed=tabm_seed).squeeze(1)

        return x

~~~~

## B1: Existing factor/feature vectorization and repeated test inputs

`sources/batchensemble.txt`, LF lines427–540; Unicode offset16246,length5411; slice SHA256 `27ef40b2d4c938e3a4ed3420153fca1eb5ec5581c1ae22403b0dc11dd756c27f`.

~~~~text
In this section, we introduce how to ensemble neural networks in an efficient way. Let WW be the weights in a neural network layer. Denote the input dimension as mm and the output dimension as nn, i.e. W∈ℝm×nW\in\mathbb{R}^{m\times n}. For ensemble, assuming the ensemble size is MM and each ensemble member has weight matrix W¯i\overline{W}_{i}. Each ensemble member owns a tuple of trainable vectors rir_{i} and sis_{i} which share the same dimension as input and output (mm and nn) respectively, where ii ranges from 11 to MM. Our algorithm generates a family of ensemble weights W¯i\overline{W}_{i} by the following:



Figure 2: An illustration on how to generate the ensemble weights for two ensemble members.








W¯i=W∘Fi, where ​Fi=ri​si⊤,\overline{W}_{i}=W\circ F_{i},\text{ where }F_{i}=r_{i}s_{i}^{\top},

(1)






For each training example in the mini-batch, it receives an ensemble weight W¯i\overline{W}_{i} by element-wise multiplying WW, which we refer to as “slow weights”, with a rank-one matrix FiF_{i}, which we refer to as “fast weights.” The subscript ii represents the selection of ensemble member. Since WW is shared across ensemble members, we term it as "shared weight" in the following paper. Figure 2 visualizes BatchEnsemble. Rather than modulating the weight matrices, one can also modulate the neural networks’ intermediate features, which achieves promising performance in visual reasoning tasks (Perez et al., 2017).





Vectorization: We show how to make the above ensemble weight generation mechanism parallelizable within a device, i.e., where one computes a forward pass with respect to multiple ensemble members in parallel. This is achieved by manipulating the matrix computations for a mini-batch (Wen et al., 2018). Let xx denote the activations of the incoming neurons in a neural network layer. The next layer’s activations are given by:






yn\displaystyle y_{n}
=ϕ⁡(W¯i⊤​xn)\displaystyle=\phi\left(\overline{W}_{i}^{\top}x_{n}\right)

(2)





=ϕ⁡((W∘ri​si⊤)⊤​xn)\displaystyle=\phi\left(\left(W\circ r_{i}s_{i}^{\top}\right)^{\top}x_{n}\right)

(3)





=ϕ⁡((W⊤​(xn∘ri))∘si),\displaystyle=\phi\left(\left(W^{\top}(x_{n}\circ r_{i})\right)\circ s_{i}\right),

(4)



where ϕ\phi denotes the activation function and the subscript nn represents the index in the mini-batch. The output represents next layer’s activations from the it​hi^{th} ensemble member. To vectorize these computations, we define matrices RR and SS whose rows consist of the vectors rir_{i} and sis_{i} for all examples in the mini-batch. The above equation is vectorized as:






Y=ϕ⁡(((X∘R)​W)∘S).Y=\phi\left(\left((X\circ R)W\right)\circ S\right).

(5)



where XX is the mini-batch input. By computing Eqn. 5, we can obtain the next layer’s activations for each ensemble member in a mini-batch friendly way. This allows us to take full advantage of parallel accelerators to implement the ensemble efficiently. To match the input and the ensemble weight, we can divide the input mini-batch into MM sub-batches and each sub-batch receives ensemble weight W¯i\overline{W}_{i}, i={1,…,M}i=\{1,\dots,M\}.





Ensembling During Testing: In our experiments, we take the average of predictions of each ensemble member. Suppose the test batch size is BB and there are MM ensemble members. To achieve an efficient implementation, one repeats the input mini-batch MM times, which leads to an effective batch size B⋅MB\cdot M. This enables all ensemble members to compute the output of the same BB input data points in a single forward pass. It eliminates the need to calculate the output of each ensemble member sequentially and therefore reduces the ensemble’s computational cost.






3.2 Computational Cost





The only extra computation in BatchEnsemble over a single neural network is the Hadamard product, which is cheap compared to matrix multiplication. Thus, BatchEnsemble incurs almost no additional computational overhead (Figure 1).22
              2
              
              
              
            In Figure 1, note the computational overhead of BatchEnsemble at the ensemble size 1 indicates the additional cost of Hadamard products.
 One limitation of BatchEnsemble is that if we keep the mini-batch size the same as single model training, each ensemble member gets only a portion of input data. In practice, the above issue can be remedied by increasing the batch size so that each ensemble member receives the same amount of data as ordinary single model training. Since BatchEnsemble is parallelizable within a device, increasing the batch size incurs almost no computational overhead in both training and testing stages on the hardware that can fully utilize large batch size. Moreover, when increasing the batch size reaches its diminishing return regime, BatchEnsemble can still take advantage from even larger batch size by increasing the ensemble size.





The only memory overhead in BatchEnsemble is the set of vectors, {r1,…,rm}\{r_{1},\dots,r_{m}\} and {s1,…,sm}\{s_{1},\dots,s_{m}\}, which are cheap to store compared to the weight matrices. By eliminating the need to store full weight matrices of each ensemble member, BatchEnsemble has almost no additional memory cost. For example, BatchEnsemble of ResNet-32 of size 4 incurs 10%10\% more parameters while naive ensemble incurs 3X more.






3.3 BatchEnsemble as an Approach to lifelong learning





~~~~

## B2: Hardware-dependent batch utilization and storage scope

`sources/batchensemble.txt`, LF lines540–568; Unicode offset21656,length2185; slice SHA256 `6d855f2d8b7d93a0036b855144b737fc06280ba99fc5a28d741bd2e36ec8eebb`.

~~~~text


The significant memory cost of ensemble methods limits its application to many real world learning scenarios such as multi-task learning and lifelong learning, where one might apply an independent copy of the model for each task. This is not the case with BatchEnsemble. Specifically, consider a total of TT tasks arriving in sequential order. Denote Dt=(xi,yi,t)D_{t}=(x_{i},y_{i},t) as the training data in task tt where t∈{1,2,…,T}t\in\{1,2,\dots,T\} and ii is the index of the data point. Similarly, denote the test data set as 𝒯t=(xi,yi,t)\mathcal{T}_{t}=(x_{i},y_{i},t). At test time, we compute the average performance on 𝒯t\mathcal{T}_{t} across all tasks seen so far as the evaluation metric. To extend BatchEnsemble to lifelong learning, we compute the neural network prediction in task tt with weight W¯t=W∘(rt​st⊤)\overline{W}_{t}=W\circ(r_{t}s_{t}^{\top}) in task tt. In other words, each ensemble member is in charge of one lifelong learning task. For the training protocol, we train the shared weight WW and two fast weights r1,s1r_{1},s_{1} on the first task,






minW,s1,r1⁡L1​(W,s1,r1,D1),\displaystyle\min_{W,s_{1},r_{1}}L_{1}(W,s_{1},r_{1};D_{1}),

(6)



where L1L_{1} is the objective function in the first task such as cross-entropy in image classification. On a subsequent task tt, we only train the relevant fast weights rt,str_{t},s_{t}.






minst,rt⁡Lt​(st,rt,Dt).\displaystyle\min_{s_{t},r_{t}}L_{t}(s_{t},r_{t};D_{t}).

(7)



BatchEnsemble shares similar advantages as progressive neural networks (PNN): it entirely prevents catastrophic forgetting as the model for previously seen tasks remains the same. This removes the need of storing any data from previous task. In addition, BatchEnsemble has significantly less memory consumption than PNN as only fast weights are trained to adapt to a new task. Therefore, BatchEnsemble can easily scale to up to 100 tasks as we showed in Section 4.1 on split ImageNet. Another benefit of BatchEnsemble is that if future tasks arrive in parallel rather than sequential order, one can train on all the tasks at once (see Section 3.1). We are not aware of any other lifelong learning methods can achieve this.

~~~~

## P1: Default selected mean/root SAGE and native linear order

`sources/sage_conv.py`, LF lines66–150; Unicode offset2843,length2706; slice SHA256 `c85ee6dcadf4de8d37002a4409af61212f203e07d8981c636d8ff7801784dfc4`.

~~~~text
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support lazy initialization with "
                                 f"`project=True`")
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):

~~~~

## P2: PyG node_dim=-2 default and feature decomposition

`sources/message_passing.py`, LF lines70–117; Unicode offset2534,length2448; slice SHA256 `02fc88cb5d284ac02037c1ab244211488528bda5355c9e65bb34fea282d5d913`.

~~~~text
            resolved. (default: :obj:`None`)
        flow (str, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)
        decomposed_layers (int, optional): The number of feature decomposition
            layers, as introduced in the `"Optimizing Memory Efficiency of
            Graph Neural Networks on Edge Computing Platforms"
            <https://arxiv.org/abs/2104.03058>`_ paper.
            Feature decomposition reduces the peak memory usage by slicing
            the feature dimensions into separated feature decomposition layers
            during GNN aggregation.
            This method can accelerate GNN execution on CPU-based platforms
            (*e.g.*, 2-3x speedup on the
            :class:`~torch_geometric.datasets.Reddit` dataset) for common GNN
            models such as :class:`~torch_geometric.nn.models.GCN`,
            :class:`~torch_geometric.nn.models.GraphSAGE`,
            :class:`~torch_geometric.nn.models.GIN`, etc.
            However, this method is not applicable to all GNN operators
            available, in particular for operators in which message computation
            can not easily be decomposed, *e.g.* in attention-based GNNs.
            The selection of the optimal value of :obj:`decomposed_layers`
            depends both on the specific graph dataset and available hardware
            resources.
            A value of :obj:`2` is suitable in most cases.
            Although the peak memory usage is directly associated with the
            granularity of feature decomposition, the same is not necessarily
            true for execution speedups. (default: :obj:`1`)
    """

    special_args: Set[str] = {
        'edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'size',
        'size_i', 'size_j', 'ptr', 'index', 'dim_size'
    }

    # Supports `message_and_aggregate` via `EdgeIndex`.
    # TODO Remove once migration is finished.
    SUPPORTS_FUSED_EDGE_INDEX: Final[bool] = False

    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = 'sum',
        *,
        aggr_kwargs: Optional[Dict[str, Any]] = None,
        flow: str = "source_to_target",
        node_dim: int = -2,
        decomposed_layers: int = 1,

~~~~

## P3: Node-size inference and lifting along node_dim

`sources/message_passing.py`, LF lines249–329; Unicode offset11005,length3475; slice SHA256 `ec07570f43769c5b1c11b3ebd8777901c4e9894f2b828064818b5019f6534d15`.

~~~~text
    def _set_size(
        self,
        size: List[Optional[int]],
        dim: int,
        src: Tensor,
    ) -> None:
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                f'Encountered tensor with size {src.size(self.node_dim)} in '
                f'dimension {self.node_dim}, but expected size {the_size}.')

    def _index_select(self, src: Tensor, index) -> Tensor:
        if torch.jit.is_scripting() or is_compiling():
            return src.index_select(self.node_dim, index)
        else:
            return self._index_select_safe(src, index)

    def _index_select_safe(self, src: Tensor, index: Tensor) -> Tensor:
        try:
            return src.index_select(self.node_dim, index)
        except (IndexError, RuntimeError) as e:
            if index.numel() > 0 and index.min() < 0:
                raise IndexError(
                    f"Found negative indices in 'edge_index' (got "
                    f"{index.min().item()}). Please ensure that all "
                    f"indices in 'edge_index' point to valid indices "
                    f"in the interval [0, {src.size(self.node_dim)}) in "
                    f"your node feature matrix and try again.") from e

            if (index.numel() > 0 and index.max() >= src.size(self.node_dim)):
                raise IndexError(
                    f"Found indices in 'edge_index' that are larger "
                    f"than {src.size(self.node_dim) - 1} (got "
                    f"{index.max().item()}). Please ensure that all "
                    f"indices in 'edge_index' point to valid indices "
                    f"in the interval [0, {src.size(self.node_dim)}) in "
                    f"your node feature matrix and try again.") from e

            raise e

    def _lift(
        self,
        src: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        dim: int,
    ) -> Tensor:
        if not torch.jit.is_scripting() and is_torch_sparse_tensor(edge_index):
            assert dim == 0 or dim == 1
            if edge_index.layout == torch.sparse_coo:
                index = edge_index._indices()[1 - dim]
            elif edge_index.layout == torch.sparse_csr:
                if dim == 0:
                    index = edge_index.col_indices()
                else:
                    index = ptr2index(edge_index.crow_indices())
            elif edge_index.layout == torch.sparse_csc:
                if dim == 0:
                    index = ptr2index(edge_index.ccol_indices())
                else:
                    index = edge_index.row_indices()
            else:
                raise ValueError(f"Unsupported sparse tensor layout "
                                 f"(got '{edge_index.layout}')")
            return src.index_select(self.node_dim, index)

        elif isinstance(edge_index, Tensor):
            if torch.jit.is_scripting():  # Try/catch blocks are not supported.
                index = edge_index[dim]
                return src.index_select(self.node_dim, index)
            return self._index_select(src, edge_index[dim])

        elif isinstance(edge_index, SparseTensor):
            row, col, _ = edge_index.coo()
            if dim == 0:
                return src.index_select(self.node_dim, col)
            elif dim == 1:
                return src.index_select(self.node_dim, row)


~~~~

## P4: COO separate propagation and feature aggregation

`sources/message_passing.py`, LF lines459–577; Unicode offset20053,length5060; slice SHA256 `08ede9914673c1dab126531776721a5ec47fc27e5718e693f1d37d70cf44837c`.

~~~~text
        decomposed_layers = 1 if self.explain else self.decomposed_layers

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        mutable_size = self._check_input(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        fuse = False
        if self.fuse and not self.explain:
            if is_sparse(edge_index):
                fuse = True
            elif (not torch.jit.is_scripting()
                  and isinstance(edge_index, EdgeIndex)):
                if (self.SUPPORTS_FUSED_EDGE_INDEX
                        and edge_index.is_sorted_by_col):
                    fuse = True

        if fuse:
            coll_dict = self._collect(self._fused_user_args, edge_index,
                                      mutable_size, kwargs)

            msg_aggr_kwargs = self.inspector.collect_param_data(
                'message_and_aggregate', coll_dict)
            for hook in self._message_and_aggregate_forward_pre_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs))
                if res is not None:
                    edge_index, msg_aggr_kwargs = res
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
            for hook in self._message_and_aggregate_forward_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs), out)
                if res is not None:
                    out = res

            update_kwargs = self.inspector.collect_param_data(
                'update', coll_dict)
            out = self.update(out, **update_kwargs)

        else:  # Otherwise, run both functions in separation.
            if decomposed_layers > 1:
                user_args = self._user_args
                decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
                decomp_kwargs = {
                    a: kwargs[a].chunk(decomposed_layers, -1)
                    for a in decomp_args
                }
                decomp_out = []

            for i in range(decomposed_layers):
                if decomposed_layers > 1:
                    for arg in decomp_args:
                        kwargs[arg] = decomp_kwargs[arg][i]

                coll_dict = self._collect(self._user_args, edge_index,
                                          mutable_size, kwargs)

                msg_kwargs = self.inspector.collect_param_data(
                    'message', coll_dict)
                for hook in self._message_forward_pre_hooks.values():
                    res = hook(self, (msg_kwargs, ))
                    if res is not None:
                        msg_kwargs = res[0] if isinstance(res, tuple) else res
                out = self.message(**msg_kwargs)
                for hook in self._message_forward_hooks.values():
                    res = hook(self, (msg_kwargs, ), out)
                    if res is not None:
                        out = res

                if self.explain:
                    explain_msg_kwargs = self.inspector.collect_param_data(
                        'explain_message', coll_dict)
                    out = self.explain_message(out, **explain_msg_kwargs)

                aggr_kwargs = self.inspector.collect_param_data(
                    'aggregate', coll_dict)
                for hook in self._aggregate_forward_pre_hooks.values():
                    res = hook(self, (aggr_kwargs, ))
                    if res is not None:
                        aggr_kwargs = res[0] if isinstance(res, tuple) else res

                out = self.aggregate(out, **aggr_kwargs)

                for hook in self._aggregate_forward_hooks.values():
                    res = hook(self, (aggr_kwargs, ), out)
                    if res is not None:
                        out = res

                update_kwargs = self.inspector.collect_param_data(
                    'update', coll_dict)
                out = self.update(out, **update_kwargs)

                if decomposed_layers > 1:
                    decomp_out.append(out)

            if decomposed_layers > 1:
                out = torch.cat(decomp_out, dim=-1)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, mutable_size, kwargs), out)
            if res is not None:
                out = res

        return out

    def message(self, x_j: Tensor) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        return x_j

    def aggregate(

~~~~

## P5: Aggregation forwards node_dim

`sources/message_passing.py`, LF lines577–603; Unicode offset25094,length1167; slice SHA256 `e1c44b2fade2b4244503d53ca0142406c1894b65f4fbf92ea6fcc5ab40f84c9c`.

~~~~text
    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\bigoplus_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to the underlying
        :class:`~torch_geometric.nn.aggr.Aggregation` module to reduce messages
        as specified in :meth:`__init__` by the :obj:`aggr` argument.
        """
        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size,
                                dim=self.node_dim)

    @abstractmethod
    def message_and_aggregate(self, edge_index: Adj) -> Tensor:
        r"""Fuses computations of :func:`message` and :func:`aggregate` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and

~~~~

## P6: MeanAggregation parameter-free selected operation

`sources/aggr_basic.py`, LF lines24–36; Unicode offset692,length521; slice SHA256 `23e805b43761d995765658822514757decc9caa9dd214af831f6baca67b790a8`.

~~~~text

class MeanAggregation(Aggregation):
    r"""An aggregation operator that averages features across a set of
    elements.

    .. math::
        \mathrm{mean}(\mathcal{X}) = \frac{1}{|\mathcal{X}|}
        \sum_{\mathbf{x}_i \in \mathcal{X}} \mathbf{x}_i.
    """
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='mean')

~~~~

## P7: Aggregation reduce/scatter dimension contract

`sources/aggr_base.py`, LF lines173–189; Unicode offset6718,length618; slice SHA256 `921a01942edd8582f55df5788fdb824544064904b1d9bd5b2c0a5331fed8c8b4`.

~~~~text
    def reduce(self, x: Tensor, index: Optional[Tensor] = None,
               ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
               dim: int = -2, reduce: str = 'sum') -> Tensor:

        if ptr is not None:
            if index is None or self._deterministic:
                ptr = expand_left(ptr, dim, dims=x.dim())
                return segment(x, ptr, reduce=reduce)

        if index is None:
            raise RuntimeError("Aggregation requires 'index' to be specified")

        return scatter(x, index, dim, dim_size, reduce)

    def to_dense_batch(
        self,
        x: Tensor,

~~~~

## P8: Mean degree counts, broadcast and empty-degree clamp

`sources/scatter.py`, LF lines55–90; Unicode offset2243,length1553; slice SHA256 `53de2d8399ac3220c034dc229e786db0d2db4b6b41eae5ad0b9fe53b8c53e327`.

~~~~text
    # `torch.scatter_reduce` has a faster forward implementation for
    # "min"/"max" reductions since it does not compute additional arg
    # indices, but is therefore way slower in its backward implementation.
    # More insights can be found in `test/utils/test_scatter.py`.

    size = src.size()[:dim] + (dim_size, ) + src.size()[dim + 1:]

    # For "any" reduction, we use regular `scatter_`:
    if reduce == 'any':
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_(dim, index, src)

    # For "sum" and "mean" reduction, we make use of `scatter_add_`:
    if reduce == 'sum' or reduce == 'add':
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_add_(dim, index, src)

    if reduce == 'mean':
        count = src.new_zeros(dim_size)
        count.scatter_add_(0, index, src.new_ones(src.size(dim)))
        count = count.clamp(min=1)

        index = broadcast(index, src, dim)
        out = src.new_zeros(size).scatter_add_(dim, index, src)

        return out / broadcast(count, out, dim)

    # For "min" and "max" reduction, we prefer `scatter_reduce_` on CPU or
    # in case the input does not require gradients:
    if reduce in ['min', 'max', 'amin', 'amax']:
        if (not torch_geometric.typing.WITH_TORCH_SCATTER or is_compiling()
                or is_in_onnx_export() or not src.is_cuda
                or not src.requires_grad):

            if (src.is_cuda and src.requires_grad and not is_compiling()
                    and not is_in_onnx_export()):

~~~~

## P9: Index/count broadcasting preserves other axes

`sources/scatter.py`, LF lines141–145; Unicode offset6082,length217; slice SHA256 `dc5078d7719e6cb51c2659f8c9e8f6618e469dad17382ca4796f94c9c62ff3d9`.

~~~~text
def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
    dim = ref.dim() + dim if dim < 0 else dim
    size = ((1, ) * dim) + (-1, ) + ((1, ) * (ref.dim() - dim - 1))
    return src.view(size).expand_as(ref)


~~~~

## P10: PyG dense map delegates F.linear

`sources/linear.py`, LF lines121–127; Unicode offset4266,length198; slice SHA256 `f052ea877fb1ed78cd83ea2c72e7d788f50e4d64054f411d894dd7240da12e17`.

~~~~text
    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input features.
        """
        return F.linear(x, self.weight, self.bias)

~~~~

## T1: Torch2.1.2 LayerNorm last-feature statistics and affine map

`sources/torch_layernorm.py`, LF lines87–198; Unicode offset2545,length5094; slice SHA256 `4bdf2cf9493921436351ec3f95bfc25ed5d0268487a86e3063853dd089570a08`.

~~~~text
class LayerNorm(Module):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`elementwise_affine` is ``True``). Default: ``True``.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> # NLP Example
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = nn.LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = torch.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = nn.LayerNorm([C, H, W])
        >>> output = layer_norm(input)

    .. image:: ../_static/img/nn/layer_norm.jpg
        :scale: 50 %

    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            if bias:
                self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)


~~~~

## T2: Torch2.1.2 Linear supports all leading dimensions

`sources/torch_linear.py`, LF lines48–114; Unicode offset1062,length2946; slice SHA256 `49196a0aa2b9d77de25a4214022e23a358e0bccaa231d5805da3565ca77b1886`.

~~~~text
class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

~~~~
