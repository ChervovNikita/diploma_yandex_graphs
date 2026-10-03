"""Independent HGEN GCN adapter for the settled code-literal finite profile.

This is a prospective paired-HGB adaptation, not a native reproduction claim.
Pinned author bytes remain private in the earlier source-qualification packet.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


PROFILES = ('code_literal',)
VIEW_NAMES = ('APA','APTPA','APCPA')


def require(condition,message):
    if not condition:
        raise ValueError(message)


@dataclass
class NormalizedView:
    name: str
    adjacency: torch.Tensor

    def to(self,device,dtype=None):
        return NormalizedView(self.name,self.adjacency.to(device=device,dtype=dtype or self.adjacency.dtype))


class GCNConv(nn.Module):
    """PyG GCNConv defaults for fixed unweighted metapath support, bias=True.

    Normalized views already have exactly one loop per node and coefficients
    d_src^-1/2 d_dst^-1/2. Aggregation is sparse matrix multiplication rather than
    allocating an E-by-hidden edge message tensor. Edge weights are not learned.
    """
    def __init__(self,cin,cout):
        super().__init__()
        self.lin = nn.Linear(cin,cout,bias=False)
        self.bias = nn.Parameter(torch.empty(cout))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.bias)

    def forward(self,x,view):
        return torch.sparse.mm(view.adjacency,self.lin(x))+self.bias


class GCN_embed(nn.Module):
    def __init__(self,in_dim,hidden,dropout,layers):
        super().__init__()
        self.enc = nn.Linear(in_dim,hidden)
        convolutions = [GCNConv(hidden,hidden) for _ in range(layers)]
        # Preserve the author's unused decoder parameters and parameter names.
        self.dec = nn.Linear(hidden,hidden)
        self.layers = nn.ModuleList(convolutions)
        self.dropout = dropout

    def forward(self,x,view):
        x = F.dropout(x,self.dropout,training=self.training)
        x = F.leaky_relu(self.enc(x),.1)
        for layer in self.layers:
            x = F.leaky_relu(layer(x,view),.1)
        return x


class AttentionH(nn.Module):
    def __init__(self,hidden,attention_dim,members,profile):
        super().__init__()
        require(profile in PROFILES,'Explicit supported HGEN profile required')
        require(members>1,'Author min-max fusion is undefined with one member')
        self.num_gcn,self.profile = members,profile
        # The author never uses self.att; retain it for honest capacity/mapping.
        self.attention_list = nn.ModuleList()
        self.att = nn.Linear(hidden,attention_dim)
        for _ in range(members):
            self.attention_list.append(nn.Linear(hidden,attention_dim))
        self.agg = nn.Linear(members*attention_dim,members)

    def forward(self,embeddings):
        scores = self.agg(torch.cat([layer(value) for layer,value in zip(self.attention_list,embeddings)],dim=1))
        scores = scores-scores.mean(dim=1,keepdim=True)
        minimum = scores.min(dim=1,keepdim=True).values
        maximum = scores.max(dim=1,keepdim=True).values
        span = maximum-minimum
        # No invented epsilon or fallback predictor. On the source's finite,
        # positive-range domain this is exactly its min-max function. A zero
        # range is surfaced as a failed case instead of silently producing NaN.
        require(bool(torch.isfinite(scores).all()) and not bool((span==0).any()),
                'HGEN min-max attention has a nonfinite or zero range; no smoothing recipe is qualified')
        attention = (scores-minimum)/span
        # Preserve the code's asymmetric residual (paper Eq6 differs).
        fused = attention[:,0,None]*embeddings[0]
        for i in range(1,self.num_gcn):
            fused += (attention[:,i,None]+1./self.num_gcn)*embeddings[i]
        return fused,attention


class HGEN(nn.Module):
    def __init__(self,in_dim,profile='code_literal',hidden=64,attention_dim=8,members=3,
                 dropout=.1,layers=2,classes=4):
        super().__init__()
        require(profile in PROFILES,'Only the code-literal finite profile is settled')
        require(classes==4,'Paired HGB-DBLP classifier schema must come from TRAIN4')
        self.profile,self.num_path,self.num_gcn = profile,3,members
        self.gcn_list = nn.ModuleList([nn.ModuleList([GCN_embed(in_dim,hidden,dropout,layers)
            for _ in range(members)]) for _ in range(3)])
        self.attention_list = nn.ModuleList([AttentionH(hidden,attention_dim,members,profile) for _ in range(3)])
        self.dec_list = nn.ModuleList([nn.Linear(hidden,classes) for _ in range(3)])

    def forward(self,features,views,return_attention=False):
        require(tuple(v.name for v in views)==VIEW_NAMES,'Native author metapath order APA/APTPA/APCPA required')
        # Preserve all first-pass dropout draws from the author. GCN has no
        # mutable running statistics; unused first-pass outputs have no loss
        # path. no_grad removes only unnecessary retained autograd graphs.
        with torch.no_grad():
            for learners,view in zip(self.gcn_list,views):
                for learner in learners:
                    learner(features,view)
        fused,attention = [],[]
        for learners,view,fusion in zip(self.gcn_list,views,self.attention_list):
            value,weights = fusion([learner(features,view) for learner in learners])
            fused.append(value); attention.append(weights)
        pool = torch.stack([value.mean(0) for value in fused])
        gram = pool@pool.T
        logits = torch.stack([decoder(value) for decoder,value in zip(self.dec_list,fused)]).sum(0)
        result = (F.log_softmax(logits,dim=1),gram,torch.stack(fused).sum(0))
        return result+(attention,) if return_attention else result


def objective(log_probabilities,gram,train_ids,train_labels,profile='code_literal',lambda_cov=0.):
    require(profile in PROFILES and isinstance(lambda_cov,(int,float)) and not isinstance(lambda_cov,bool)
        and math.isfinite(lambda_cov) and lambda_cov==0.,
        'Only documented code-default lambda_cov0 is settled; a regularized profile requires separate source/protocol qualification')
    nll = F.nll_loss(log_probabilities[train_ids],train_labels)
    penalty = torch.norm(gram,p=1)
    penalty = penalty**2
    return nll+lambda_cov*penalty,nll,penalty
