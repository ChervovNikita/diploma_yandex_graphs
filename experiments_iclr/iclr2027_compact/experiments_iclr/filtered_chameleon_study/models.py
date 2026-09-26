import torch
from torch import nn
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv, GATv2Conv, TransformerConv, TAGConv
from torch_geometric.nn import GraphNorm
from torch_geometric.utils import add_self_loops


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


class GATModule(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim_multiplier, dropout, **kwargs):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')
        head_dim = dim // num_heads
        self.conv = GATConv(dim, head_dim, heads=num_heads, dropout=dropout, concat=True, add_self_loops=False)
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        x = self.conv(x, graph.edge_index)
        x = self.feed_forward_module(graph, x)
        return x


class GATSepModule(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim_multiplier, dropout, **kwargs):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')
        head_dim = dim // num_heads
        self.conv = GATv2Conv(dim, head_dim, heads=num_heads, dropout=dropout, concat=True, add_self_loops=False)
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        message = self.conv(x, graph.edge_index)
        x = torch.cat([x, message], dim=1)
        x = self.feed_forward_module(graph, x)
        return x


class TransformerAttentionModule(nn.Module):
    def __init__(self, dim, num_heads, dropout, **kwargs):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')
        self.conv = TransformerConv(dim, dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
        self.output_linear = nn.Linear(in_features=dim, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        x = self.conv(x, graph.edge_index)
        x = self.output_linear(x)
        x = self.dropout(x)
        return x


class TransformerAttentionSepModule(nn.Module):
    def __init__(self, dim, num_heads, dropout, **kwargs):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')
        self.conv = TransformerConv(dim, dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
        self.output_linear = nn.Linear(in_features=dim * 2, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        message = self.conv(x, graph.edge_index)
        x = torch.cat([x, message], dim=1)
        x = self.output_linear(x)
        x = self.dropout(x)
        return x


class TAGModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
        super().__init__()
        self.conv = TAGConv(dim, dim)
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        x = self.conv(x, graph.edge_index)
        x = self.feed_forward_module(graph, x)
        return x


MODULES = {
    'ResNet': [FeedForwardModule],
    'GCN': [GCNModule],
    'SAGE': [SAGEModule],
    'GAT': [GATModule],
    'GAT-sep': [GATSepModule],
    'GT': [TransformerAttentionModule, FeedForwardModule],
    'GT-sep': [TransformerAttentionSepModule, FeedForwardModule],
    'TAG': [TAGModule],
}

NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d,
    'GraphNorm': GraphNorm,
}


class Model(nn.Module):
    def __init__(self, model_name, num_layers, input_dim, hidden_dim, output_dim, hidden_dim_multiplier, num_heads,
                 normalization, dropout):

        super().__init__()

        self.model_name = model_name
        normalization = NORMALIZATION[normalization]

        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
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
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, graph, x):
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


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
