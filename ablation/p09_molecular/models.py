"""Reuse the pinned OGB backbone and the repository's original BEBlock."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from ogb.utils.features import get_atom_feature_dims

from external_sources.ogb_molecular.gnn import GNN
from models import BEBlock
from .normalization import install_private_batchnorm, select_batchnorm_route


class MolecularGNNM(GNN):
    """Four routes through one official molecular message-passing backbone.

    Concatenated one-hot atom features turn the sum of AtomEncoder tables into
    an equivalent linear input boundary. Native tables and native graph-head
    weights are retained. Private R/S/B factors follow the original BEBlock.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.members = 4
        self.atom_feature_dims = get_atom_feature_dims()
        native_atom = self.gnn_node.atom_encoder
        native_head = self.graph_pred_linear
        self.input_be_block = BEBlock(sum(self.atom_feature_dims), self.emb_dim,
                                      True, 'cpu', self.members)
        self.output_be_block = BEBlock(self.emb_dim, self.num_tasks,
                                       False, 'cpu', self.members)
        with torch.no_grad():
            tables = [embedding.weight.T for embedding in native_atom.atom_embedding_list]
            self.input_be_block.W.weight.copy_(torch.cat(tables, dim=1))
            self.output_be_block.W.weight.copy_(native_head.weight)
            self.output_be_block.B.copy_(native_head.bias.expand(self.members, -1))
        self.gnn_node.atom_encoder = torch.nn.Identity()
        del self.graph_pred_linear

    def forward(self, batched_data, member=0, return_prehead=False):
        select_batchnorm_route(self.gnn_node, member)
        categorical = [F.one_hot(batched_data.x[:, column], size)
                       for column, size in enumerate(self.atom_feature_dims)]
        x = torch.cat(categorical, dim=-1).to(self.input_be_block.W.weight.dtype)
        encoded = self.input_be_block(x, tabm_seed=member)
        graph = SimpleNamespace(x=encoded, edge_index=batched_data.edge_index,
                                edge_attr=batched_data.edge_attr, batch=batched_data.batch)
        h_node = self.gnn_node(graph)
        h_graph = self.pool(h_node, batched_data.batch)
        logits = self.output_be_block(h_graph, tabm_seed=member)
        return (logits, h_graph) if return_prehead else logits


def build_model(config, num_tasks):
    model_class = MolecularGNNM if config['method'].startswith('gnnm') else GNN
    model = model_class(num_tasks=num_tasks, num_layer=config['num_layers'],
                       emb_dim=config['hidden_dim'], gnn_type=config['model'],
                       virtual_node=False, residual=False, drop_ratio=config['dropout'],
                       JK='last', graph_pooling='mean')
    if 'private_bn' in config['method']:
        install_private_batchnorm(model.gnn_node)
    return model


def route_logits(model, batch, members):
    if members == 1:
        return model(batch).unsqueeze(0)
    return torch.stack([model(batch, member=member) for member in range(members)])
