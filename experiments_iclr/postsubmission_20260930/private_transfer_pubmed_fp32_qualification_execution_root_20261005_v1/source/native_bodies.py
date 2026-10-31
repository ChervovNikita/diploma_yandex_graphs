"""Pinned native bodies and explicit available-only projections; no launch until release."""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling, to_undirected
from torch_geometric.data import Data
from typing import Iterable
from gnn_model import SAGE
from scoring import mlp_score
from baseline_models.NCN.model import GCN, IncompleteCN1Predictor
from baseline_models.NCN.util import PermIterator
from types import FunctionType
NODES = 19717
device = torch.device("cuda:0")

def reference_sage_train(model, score_func, train_pos, x, optimizer, batch_size):
    model.train()
    score_func.train()
    total_loss = total_examples = 0
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        num_nodes = x.size(0)
        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
        train_edge_mask = train_pos[mask].transpose(1, 0)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1, 0]]), dim=1)
        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
        h = model(x, adj)
        edge = train_pos[perm].t()
        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long, device=h.device)
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples

@torch.no_grad()
def reference_sage_test_edge(score_func, input_data, h, batch_size, negative_data=None):
    pos_preds = []
    neg_preds = []
    if negative_data is not None:
        for perm in DataLoader(range(input_data.size(0)), batch_size):
            pos_edges = input_data[perm].t()
            neg_edges = torch.permute(negative_data[perm], (2, 0, 1))
            pos_scores = score_func(h[pos_edges[0]], h[pos_edges[1]]).cpu()
            neg_scores = score_func(h[neg_edges[0]], h[neg_edges[1]]).cpu()
            pos_preds += [pos_scores]
            neg_preds += [neg_scores]
        neg_preds = torch.cat(neg_preds, dim=0)
    else:
        neg_preds = None
        for perm in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            pos_preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
    pos_preds = torch.cat(pos_preds, dim=0)
    return (pos_preds, neg_preds)

def reference_ncnc_train(model, predictor, data, split_edge, optimizer, batch_size, maskinput: bool=True, cnprobs: Iterable[float]=[], alpha: float=None):

    def penalty(posout, negout):
        scale = torch.ones_like(posout[[0]]).requires_grad_()
        loss = -F.logsigmoid(posout * scale).mean() - F.logsigmoid(-negout * scale).mean()
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(torch.square(grad))
    if alpha is not None:
        predictor.setalpha(alpha)
    model.train()
    predictor.train()
    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()
    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    for perm in PermIterator(adjmask.device, adjmask.shape[0], batch_size):
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei, sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        h = model(data.x, adj)
        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        pos_losss = -F.logsigmoid(pos_outs).mean()
        edge = negedge[:, perm]
        neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_losss + pos_losss
        loss.backward()
        optimizer.step()
        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss

def candidate_sage_train(model, score_func, train_pos, x, optimizer, batch_size):
    model.train()
    score_func.train()
    total_loss = total_examples = 0
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        num_nodes = x.size(0)
        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
        train_edge_mask = train_pos[mask].transpose(1, 0)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1, 0]]), dim=1)
        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
        h = model(x, adj)
        edge = train_pos[perm].t()
        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long, device=h.device)
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples

@torch.no_grad()
def candidate_sage_test_edge(score_func, input_data, h, batch_size, negative_data=None):
    pos_preds = []
    neg_preds = []
    if negative_data is not None:
        for perm in DataLoader(range(input_data.size(0)), batch_size):
            pos_edges = input_data[perm].t()
            neg_edges = torch.permute(negative_data[perm], (2, 0, 1))
            pos_scores = score_func(h[pos_edges[0]], h[pos_edges[1]]).cpu()
            neg_scores = score_func(h[neg_edges[0]], h[neg_edges[1]]).cpu()
            pos_preds += [pos_scores]
            neg_preds += [neg_scores]
        neg_preds = torch.cat(neg_preds, dim=0)
    else:
        neg_preds = None
        for perm in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            pos_preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
    pos_preds = torch.cat(pos_preds, dim=0)
    return (pos_preds, neg_preds)

def candidate_ncnc_train(model, predictor, data, split_edge, optimizer, batch_size, maskinput: bool=True, cnprobs: Iterable[float]=[], alpha: float=None):

    def penalty(posout, negout):
        scale = torch.ones_like(posout[[0]]).requires_grad_()
        loss = -F.logsigmoid(posout * scale).mean() - F.logsigmoid(-negout * scale).mean()
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(torch.square(grad))
    if alpha is not None:
        predictor.setalpha(alpha)
    model.train()
    predictor.train()
    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()
    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    for perm in PermIterator(adjmask.device, adjmask.shape[0], batch_size):
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei, sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        h = model(data.x, adj)
        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        pos_losss = -F.logsigmoid(pos_outs).mean()
        edge = negedge[:, perm]
        neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_losss + pos_losss
        loss.backward()
        optimizer.step()
        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss

def seed_native(seed):
    import random
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def candidate_factory(model_name, seed, x, train):
    train_device = None
    if model_name == 'SAGE':
        seed_native(seed)
        edge = torch.cat((train.t(), train.t()[[1, 0]]), dim=1)
        adj = SparseTensor.from_edge_index(edge, torch.ones(edge.size(1)), [NODES, NODES])
        model = SAGE(x.size(1), 256, 256, 2, 0.1, 2, 1, NODES, False).to(device)
        predictor = mlp_score(256, 256, 1, 3, 0.1).to(device)
        seed_native(seed)
        model.reset_parameters()
        predictor.reset_parameters()
        x = x.to(device)
        train_device = train.to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.001, weight_decay=0)
        data = {'adj': adj}
    else:
        edge = to_undirected(train.t())
        adj = SparseTensor.from_edge_index(edge, sparse_sizes=(NODES, NODES)).to_symmetric().coalesce()
        data = Data(x=x, edge_index=edge, adj_t=adj, num_nodes=NODES, max_x=-1).to(device)
        split = {'train': {'edge': train}}
        seed_native(seed)
        model = GCN(x.size(1), 256, 256, 1, 0.1, True, False, -1, 'puregcn', True, 0.0, xdropout=0.3, taildropout=0.0, noinputlin=False).to(device)
        predictor = IncompleteCN1Predictor(256, 256, 1, 1, 0.1, 0.0, True, cndeg=-1, use_xlin=True, tailact=True, twolayerlin=False, beta=1.0, alpha=0.3, scale=5.3, offset=0.5, trainresdeg=-1, testresdeg=-1, pt=0.5, learnablept=False, depth=1, splitsize=-1).to(device)
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.001}, {'params': predictor.parameters(), 'lr': 0.001}], weight_decay=0)
    return (model, predictor, optimizer, x, data, train_device)

def reference_factory(model_name, seed, x, train):
    # Resolved native pubmed.sh/default policy; no native loader is imported.
    if model_name == "SAGE":
        seed_native(999)
        torch.randperm(train.size(0))  # discarded diagnostic row-selection draw
        edge = torch.cat((train.t(), train.t()[[1, 0]]), dim=1)
        adj = SparseTensor.from_edge_index(edge, torch.ones(edge.size(1)), [NODES, NODES])
        model = SAGE(x.size(1), 256, 256, 2, 0.1, 2, 1, NODES, False).to(device)
        predictor = mlp_score(256, 256, 1, 3, 0.1).to(device)
        seed_native(seed)
        model.reset_parameters()
        predictor.reset_parameters()
        x = x.to(device)
        train_device = train.to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.001, weight_decay=0)
        data = {"adj": adj}
    else:
        edge = to_undirected(train.t())
        adj = SparseTensor.from_edge_index(edge, sparse_sizes=(NODES, NODES)).to_symmetric().coalesce()
        data = Data(x=x, edge_index=edge, adj_t=adj, num_nodes=NODES, max_x=-1).to(device)
        seed_native(seed)
        model = GCN(data.num_features, 256, 256, 1, 0.1, True, False, -1, "puregcn", True, 0.0, xdropout=0.3, taildropout=0.0, noinputlin=False).to(device)
        predictor = IncompleteCN1Predictor(256, 256, 1, 1, 0.1, 0.0, True, cndeg=-1, use_xlin=True, tailact=True, twolayerlin=False, beta=1.0, depth=1, splitsize=-1, scale=5.3, offset=0.5, trainresdeg=-1, testresdeg=-1, pt=0.5, learnablept=False, alpha=0.3).to(device)
        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": 0.001}, {"params": predictor.parameters(), "lr": 0.001}], weight_decay=0)
        train_device = None
    return model, predictor, optimizer, x, data, train_device


@torch.no_grad()
def candidate_validate():
    model.eval()
    predictor.eval()
    if model_name == 'SAGE':
        h = model(x, data['adj'].to(device))
        iter(DataLoader((), batch_size=1024))
        (pos, neg) = sage_test_edge(predictor, valid, h, 1024, negative_data=valid_neg)
        iter(DataLoader((), batch_size=1024))
        (pos, neg) = (pos.flatten(), neg.squeeze(-1))
    else:
        h = model(data.x, data.adj_t)
        positive = valid.to(device)
        negatives = valid_neg.to(device)
        (pos, neg) = ([], [])
        for perm in PermIterator(device, positive.shape[0], 512, False):
            pos.append(predictor(h, data.adj_t, positive[perm].t()).squeeze().cpu())
            targets = torch.permute(negatives[perm], (2, 0, 1)).view(2, -1)
            neg.append(predictor(h, data.adj_t, targets).squeeze().cpu())
        pos = torch.cat(pos).flatten()
        neg = torch.cat(neg).view(-1, 500)
    return (pos, neg)

@torch.no_grad()
def reference_sage_validate():
    model.eval()
    predictor.eval()
    h = model(x, data["adj"].to(device))
    iter(DataLoader(range(2216), batch_size=1024))
    pos, neg = reference_sage_test_edge(predictor, valid, h, 1024, negative_data=valid_neg)
    iter(DataLoader(range(2051), batch_size=1024))
    return pos.flatten(), neg.squeeze(-1)


@torch.no_grad()
def reference_ncnc_validate():
    model.eval()
    predictor.eval()
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    adj = data.adj_t
    h = model(data.x, adj)
    neg_num = neg_valid_edge.size(1)
    pos_preds = []
    neg_preds = []
    for perm in PermIterator(pos_valid_edge.device, pos_valid_edge.shape[0], batch_size, False):
        pos_preds += [predictor(h, adj, pos_valid_edge[perm].t()).squeeze().cpu()]
        neg_edges = torch.permute(neg_valid_edge[perm], (2, 0, 1))
        neg_edges = neg_edges.view(2, -1)
        neg_preds += [predictor(h, adj, neg_edges).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_preds, dim=0)
    neg_valid_pred = torch.cat(neg_preds, dim=0)
    neg_valid_pred = neg_valid_pred.view(-1, neg_num)
    pos_valid_pred = torch.flatten(pos_valid_pred)
    return (pos_valid_pred, neg_valid_pred)

def bound_validation(which, model_name, scope):
    scope["sage_test_edge"] = candidate_sage_test_edge
    scope["batch_size"] = 512
    fn = candidate_validate if which == "candidate" else (reference_sage_validate if model_name == "SAGE" else reference_ncnc_validate)
    body = fn.__wrapped__
    bound = FunctionType(body.__code__, scope, body.__name__, body.__defaults__, body.__closure__)
    return torch.no_grad()(bound)

