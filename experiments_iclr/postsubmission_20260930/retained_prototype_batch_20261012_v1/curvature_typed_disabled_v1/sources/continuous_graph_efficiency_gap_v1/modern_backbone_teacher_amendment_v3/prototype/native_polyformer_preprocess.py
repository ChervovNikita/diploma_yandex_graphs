"""Attributed exact native monomial preprocessing bodies, no dataset imports."""
import numpy as np
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_scipy_sparse_matrix

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def mono_base(K, x, edge_index, edge_attr):
    edge_index, norm = gcn_norm(edge_index, edge_attr, num_nodes=x.size(0), dtype=x.dtype)
    adj = to_scipy_sparse_matrix(edge_index, norm, x.size(0))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    device = x.device
    adj = adj.to(device)

    list_mat = []
    list_mat.append(x)
    tmp_mat = x
    for _ in range(K):
        tmp_mat = torch.spmm(adj, tmp_mat)
        list_mat.append(tmp_mat)
    return list_mat

