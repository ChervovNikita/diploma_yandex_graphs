"""Attributed exact author PolyFormer body; seeding prefix removed."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from .native_polyformer import PolyFormerBlock

class PolyFormer(nn.Module): 
    def __init__(self, dataset, args):
        super(PolyFormer,self).__init__()
        self.dropout = args.dropout
        self.nlayers = args.nlayer
        self.dataset = args.dataset
        
        self.attn = nn.ModuleList([PolyFormerBlock(dataset, args) for _ in range(self.nlayers)])
        self.K = args.K + 1
        self.base = args.base

        self.lin1 = Linear(args.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.hidden)
        self.lin3 = Linear(args.hidden, args.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        input_mat = data.list_mat
        input_mat = torch.stack(input_mat, dim = 1) # [N,k,d]
        input_mat = self.lin1(input_mat) # just for common dataset
       
        for block in self.attn:
            input_mat = block(input_mat)
            
        x = torch.sum(input_mat, dim = 1) # [N,d]
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        return x

