"""Independent Torch ports of released HGB GAT/Simple-HGN and DBLP SeHGNN.

Source qualifications, commit bindings and deviations are in PROVENANCE.json.
No DGL/PyG or heldout label input is needed by these model implementations.
"""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class HomogeneousGraph:
    nodes: int
    src: torch.Tensor
    dst: torch.Tensor
    etype: torch.Tensor
    num_etypes: int

    def to(self, device):
        return HomogeneousGraph(self.nodes,self.src.to(device),self.dst.to(device),
                                self.etype.to(device),self.num_etypes)


def incoming_softmax(scores, dst, nodes):
    """Per destination, per head normalization; never normalize across heads."""
    index = dst[:,None].expand_as(scores)
    maximum = scores.new_full((nodes,scores.shape[1]),-float('inf'))
    # Max is a numerical stabilizer, not part of the model's derivatives.
    maximum.scatter_reduce_(0,index,scores.detach(),reduce='amax',include_self=True)
    weights = (scores-maximum[dst]).exp()
    denominator = scores.new_zeros((nodes,scores.shape[1]))
    denominator.scatter_add_(0,index,weights)
    return weights/denominator[dst]


class AttentionConv(nn.Module):
    def __init__(self, cin, cout, heads, dropout=.5, slope=.05,
                 residual=False, activation=None, edge_dim=None,
                 num_etypes=None, alpha=.05):
        super().__init__()
        self._out_feats,self._num_heads = cout,heads
        self.fc = nn.Linear(cin,heads*cout,bias=False)
        self.attn_l = nn.Parameter(torch.empty(1,heads,cout))
        self.attn_r = nn.Parameter(torch.empty(1,heads,cout))
        self.feat_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(slope)
        self.res_fc = (nn.Linear(cin,heads*cout,bias=False) if cin!=cout else nn.Identity()) if residual else None
        self.activation,self.alpha,self.edge_dim = activation,alpha,edge_dim
        if edge_dim is not None:
            self.edge_emb = nn.Embedding(num_etypes,edge_dim)
            self.fc_e = nn.Linear(edge_dim,edge_dim*heads,bias=False)
            self.attn_e = nn.Parameter(torch.empty(1,heads,edge_dim))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight,gain=gain)
        nn.init.xavier_normal_(self.attn_l,gain=gain)
        nn.init.xavier_normal_(self.attn_r,gain=gain)
        if isinstance(self.res_fc,nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight,gain=gain)
        if self.edge_dim is not None:
            nn.init.xavier_normal_(self.attn_e,gain=gain)
            nn.init.xavier_normal_(self.fc_e.weight,gain=gain)

    def forward(self, graph, features, prior_attention=None):
        h = self.feat_drop(features)
        projected = self.fc(h).view(-1,self._num_heads,self._out_feats)
        left = (projected*self.attn_l).sum(-1)
        right = (projected*self.attn_r).sum(-1)
        scores = left[graph.src]+right[graph.dst]
        if self.edge_dim is not None:
            edge = self.fc_e(self.edge_emb(graph.etype)).view(-1,self._num_heads,self.edge_dim)
            scores = scores+(edge*self.attn_e).sum(-1)
        attention = self.attn_drop(incoming_softmax(self.leaky_relu(scores),graph.dst,graph.nodes))
        if prior_attention is not None:
            if self.edge_dim is None:
                raise ValueError('Native GAT has no attention residual')
            attention = (1-self.alpha)*attention+self.alpha*prior_attention
        output = projected.new_zeros((graph.nodes,self._num_heads,self._out_feats))
        # Limit the temporary edge-message tensor. Algebra matches native sum.
        for start in range(0,len(graph.src),32768):
            end = start+32768
            output.index_add_(0,graph.dst[start:end],
                projected[graph.src[start:end]]*attention[start:end,:,None])
        if self.res_fc is not None:
            output = output+self.res_fc(h).view(len(h),-1,self._out_feats)
        if self.activation is not None:
            output = self.activation(output)
        return output,attention.detach()


class HGBGAT(nn.Module):
    """Two hidden convolutions plus one class convolution, native DBLP defaults."""
    def __init__(self, input_dims, classes=4, simple=False, hidden=64, heads=8,
                 edge_dim=64, num_etypes=13, dropout=.5):
        super().__init__()
        self.simple,self.num_layers = simple,2
        self.fc_list = nn.ModuleList([nn.Linear(d,hidden,bias=True) for d in input_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight,gain=1.414)
        dims = [(hidden,hidden,heads,False,F.elu),
                (hidden*heads,hidden,heads,simple,F.elu),
                (hidden*heads,classes,1,simple,None)]
        self.gat_layers = nn.ModuleList([AttentionConv(cin,cout,h,dropout=dropout,
                residual=residual,activation=activation,
                edge_dim=edge_dim if simple else None,num_etypes=num_etypes)
                for cin,cout,h,residual,activation in dims])

    def forward(self, graph, features=None):
        if features is None:
            # Native DBLP identity features: exact I @ W^T + bias without dense I.
            h = torch.cat([fc.weight.T+fc.bias for fc in self.fc_list],0)
        else:
            def project(fc,x):
                return fc(x) if x.layout==torch.strided else torch.sparse.mm(x,fc.weight.T)+fc.bias
            h = torch.cat([project(fc,x) for fc,x in zip(self.fc_list,features)],0)
        previous = None
        for layer in self.gat_layers[:-1]:
            h,previous = layer(graph,h,previous if self.simple else None)
            h = h.flatten(1)
        # Author output deliberately resets residual attention (8 heads ->1).
        logits,_ = self.gat_layers[-1](graph,h,None)
        logits = logits.mean(1)
        if self.simple:
            logits = logits/logits.norm(dim=1,keepdim=True).clamp_min(1e-12)
        return logits


class LinearPerMetapath(nn.Module):
    def __init__(self, cin, cout, channels):
        super().__init__()
        self.W = nn.Parameter(torch.empty(channels,cin,cout))
        self.bias = nn.Parameter(torch.empty(channels,cout))
        self.reset_parameters()

    def reset_parameters(self):
        cin,cout = self.W.shape[-2:]
        limit = math.sqrt(3.)*nn.init.calculate_gain('relu')*math.sqrt(2./(cin+cout))
        nn.init.uniform_(self.W,-limit,limit); nn.init.zeros_(self.bias)

    def forward(self,x):
        return torch.einsum('bcm,cmn->bcn',x,self.W)+self.bias[None]


class Transformer(nn.Module):
    def __init__(self, hidden, heads=1, att_drop=0.):
        super().__init__()
        if hidden%(heads*4):
            raise ValueError('Native Transformer width must divide heads*4')
        self.num_heads = heads
        self.query = nn.Linear(hidden,hidden//4)
        self.key = nn.Linear(hidden,hidden//4)
        self.value = nn.Linear(hidden,hidden)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.att_drop = nn.Dropout(att_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.query,self.key,self.value):
            layer.reset_parameters()
        nn.init.zeros_(self.gamma)

    def forward(self,x):
        batch,channels,width = x.shape; heads = self.num_heads
        q = self.query(x).view(batch,channels,heads,-1).permute(0,2,1,3)
        k = self.key(x).view(batch,channels,heads,-1).permute(0,2,3,1)
        v = self.value(x).view(batch,channels,heads,-1).permute(0,2,1,3)
        attention = self.att_drop(F.softmax(q@k/math.sqrt(q.shape[-1]),dim=-1))
        fused = self.gamma*(attention@v)
        return x+fused.permute(0,2,1,3).reshape(batch,channels,width)


class SeHGNN(nn.Module):
    """DBLP branch of the released architecture; native parameter names retained."""
    def __init__(self, data_size, label_keys, classes=4, embed=512, hidden=512,
                 n_fp_layers=2, n_task_layers=3, dropout=.5, input_drop=.5,
                 residual=True, att_drop=0.):
        super().__init__()
        self.feat_keys,self.label_feat_keys = sorted(data_size),sorted(label_keys)
        self.tgt_type,self.residual = 'A',residual
        channels = len(self.feat_keys)+len(self.label_feat_keys)
        self.input_drop = nn.Dropout(input_drop)
        self.embeding = nn.ParameterDict({k:nn.Parameter(torch.empty(v,embed)) for k,v in data_size.items()})
        self.labels_embeding = nn.ParameterDict({k:nn.Parameter(torch.empty(classes,embed)) for k in self.label_feat_keys})
        projection = []
        for i in range(n_fp_layers):
            projection += [LinearPerMetapath(embed if i==0 else hidden,hidden,channels),
                           nn.LayerNorm([channels,hidden]),nn.PReLU(),nn.Dropout(dropout)]
        self.feature_projection = nn.Sequential(*projection)
        self.semantic_fusion = Transformer(hidden,att_drop=att_drop)
        self.fc_after_concat = nn.Linear(channels*hidden,hidden)
        if residual:
            self.res_fc = nn.Linear(embed,hidden)
        task = [nn.PReLU(),nn.Dropout(dropout)]
        for _ in range(n_task_layers-1):
            task += [nn.Linear(hidden,hidden),nn.BatchNorm1d(hidden,affine=False),nn.PReLU(),nn.Dropout(dropout)]
        task += [nn.Linear(hidden,classes),nn.BatchNorm1d(classes,affine=False,track_running_stats=False)]
        self.task_mlp = nn.Sequential(*task)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self._modules.values():
            if isinstance(module,nn.ParameterDict):
                for value in module.values():
                    nn.init.uniform_(value,-.5,.5)
            elif isinstance(module,nn.Sequential):
                for layer in module:
                    if hasattr(layer,'reset_parameters'):
                        layer.reset_parameters()
            elif hasattr(module,'reset_parameters'):
                module.reset_parameters()

    def forward(self, batch, feature_dict, label_dict):
        features = {k:self.input_drop(x@self.embeding[k]) for k,x in feature_dict.items()}
        labels = {k:self.input_drop(x@self.labels_embeding[k]) for k,x in label_dict.items()}
        x = torch.stack([features[k] for k in self.feat_keys]+[labels[k] for k in self.label_feat_keys],1)
        x = self.semantic_fusion(self.feature_projection(x)).transpose(1,2)
        x = self.fc_after_concat(x.reshape(len(x),-1))
        if self.residual:
            x = x+self.res_fc(features['A'])
        return self.task_mlp(x)
