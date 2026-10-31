"""Exact required SeHGNN helper bodies; no dataset constructor or heldout scorer."""
import gc
import random
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, remove_diag
import numpy as np
from sklearn.metrics import f1_score


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def evaluator(gt, pred):
    gt = gt.cpu().squeeze()
    pred = pred.cpu().squeeze()
    return f1_score(gt, pred, average='micro'), f1_score(gt, pred, average='macro')

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def hg_propagate_feat_dgl(g, tgt_type, num_hops, max_length, extra_metapath, echo=False):
    for hop in range(1, max_length):
        reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)
            # if hop == args.num_hops and dtype != tgt_type: continue
            for k in list(g.nodes[stype].data.keys()):
                if len(k) == hop:
                    current_dst_name = f'{dtype}{k}'
                    if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) \
                      or (hop > num_hops and k not in reserve_heads):
                        continue
                    if echo: print(k, etype, current_dst_name)
                    g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name), etype=etype)

        # remove no-use items
        for ntype in g.ntypes:
            if ntype == tgt_type: continue
            removes = []
            for k in g.nodes[ntype].data.keys():
                if len(k) <= hop:
                    removes.append(k)
            for k in removes:
                g.nodes[ntype].data.pop(k)
            if echo and len(removes): print('remove', removes)
        gc.collect()

        if echo: print(f'-- hop={hop} ---')
        for ntype in g.ntypes:
            for k, v in g.nodes[ntype].data.items():
                print(f'{ntype} {k} {v.shape}', v[:,-1].max(), v[:,-1].mean())
        if echo: print(f'------\n')
    return g

def hg_propagate_sparse_pyg(adjs, tgt_types, num_hops, max_length, extra_metapath, prop_feats=False, echo=False, prop_device='cpu'):
    store_device = 'cpu'
    if type(tgt_types) is not list:
        tgt_types = [tgt_types]

    label_feats = {k: v.clone() for k, v in adjs.items() if prop_feats or k[-1] in tgt_types} # metapath should start with target type in label propagation
    adjs_g = {k: v.to(prop_device) for k, v in adjs.items()}

    for hop in range(2, max_length):
        reserve_heads = [ele[-(hop+1):] for ele in extra_metapath if len(ele) > hop]
        new_adjs = {}
        for rtype_r, adj_r in label_feats.items():
            metapath_types = list(rtype_r)
            if len(metapath_types) == hop:
                dtype_r, stype_r = metapath_types[0], metapath_types[-1]
                for rtype_l, adj_l in adjs_g.items():
                    dtype_l, stype_l = rtype_l
                    if stype_l == dtype_r:
                        name = f'{dtype_l}{rtype_r}'
                        if (hop == num_hops and dtype_l not in tgt_types and name not in reserve_heads) \
                          or (hop > num_hops and name not in reserve_heads):
                            continue
                        if name not in new_adjs:
                            if echo: print('Generating ...', name)
                            if prop_device == 'cpu':
                                new_adjs[name] = adj_l.matmul(adj_r)
                            else:
                                with torch.no_grad():
                                    new_adjs[name] = adj_l.matmul(adj_r.to(prop_device)).to(store_device)
                        else:
                            if echo: print(f'Warning: {name} already exists')
        label_feats.update(new_adjs)

        removes = []
        for k in label_feats.keys():
            metapath_types = list(k)
            if metapath_types[0] in tgt_types: continue  # metapath should end with target type in label propagation
            if len(metapath_types) <= hop:
                removes.append(k)
        for k in removes:
            label_feats.pop(k)
        if echo and len(removes): print('remove', removes)
        del new_adjs
        gc.collect()

    if prop_device != 'cpu':
        del adjs_g
        torch.cuda.empty_cache()

    return label_feats

def train(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator, mask=None, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []

    for batch in train_loader:
        # batch = batch.to(device)
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
        batch_y = labels_cuda[batch]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(batch, batch_feats, batch_labels_feats, batch_mask)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att = model(batch, batch_feats, batch_labels_feats, batch_mask)
            loss_train = loss_fcn(output_att, batch_y)
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc
