"""Four private-hop target-only routes on the verified shared HL-GNN powers."""
import torch
from torch.utils.data import DataLoader
from native.model import BaseModel, create_input_layer, create_predictor_layer
from native.utils import get_pos_neg_edges
from member_filter import SharedPowerHLGNN


class F4PrivateHopTargetModel(BaseModel):
    MEMBERS = 4

    def __init__(self, lr, dropout, grad_clip_norm, gnn_num_layers, mlp_num_layers,
                 emb_hidden_channels, gnn_hidden_channels, mlp_hidden_channels,
                 num_nodes, num_node_feats, gnn_encoder_name, predictor_name,
                 loss_func, optimizer_name, device, use_node_feats,
                 train_node_emb, pretrain_emb, alpha, init):
        if (gnn_encoder_name != 'HLGNN' or predictor_name != 'MLP' or optimizer_name != 'Adam'
                or loss_func != 'WeightedHingeAUC' or use_node_feats or not train_node_emb
                or pretrain_emb is not None or init != 'KI' or alpha != 0.5
                or gnn_num_layers != 15 or mlp_num_layers != 2
                or (emb_hidden_channels,gnn_hidden_channels,mlp_hidden_channels) != (512,512,512)):
            raise ValueError('The fixed native DDI recipe with fresh learned embeddings is required.')
        self.loss_func_name = loss_func
        self.num_nodes, self.num_node_feats = num_nodes, num_node_feats
        self.use_node_feats, self.train_node_emb = use_node_feats, train_node_emb
        self.clip_norm, self.device = grad_clip_norm, device
        self.input_channels, self.emb = create_input_layer(
            num_nodes=num_nodes,num_node_feats=num_node_feats,hidden_channels=emb_hidden_channels,
            use_node_feats=use_node_feats,train_node_emb=train_node_emb,pretrain_emb=pretrain_emb)
        self.emb = self.emb.to(device)
        self.encoder = SharedPowerHLGNN(
            in_channels=self.input_channels,hidden_channels=gnn_hidden_channels,K=gnn_num_layers,
            dropout=dropout,alpha=alpha,members=self.MEMBERS,private_alpha=True).to(device)
        self.predictor = torch.nn.ModuleList([
            create_predictor_layer(hidden_channels=mlp_hidden_channels,num_layers=mlp_num_layers,
                                   dropout=dropout,predictor_name=predictor_name)
            for _ in range(self.MEMBERS)]).to(device)
        self.para_list = (list(self.encoder.parameters()) + list(self.predictor.parameters())
                          + list(self.emb.parameters()))
        self.optimizer = torch.optim.Adam(self.para_list,lr=lr)

    def param_init(self):
        # The shared lin1 retains its fresh constructor initialization, as native KI does.
        self.encoder.reset_parameters()
        for predictor in self.predictor:
            predictor.reset_parameters()
        torch.nn.init.xavier_uniform_(self.emb.weight)

    def train(self, data, split_edge, batch_size, neg_sampler_name, num_neg):
        if set(split_edge) != {'train','valid'} or set(split_edge['train']) != {'edge'}:
            raise ValueError('This DDI target-only control requires native missing TRAIN weights.')
        self.encoder.train()
        self.predictor.train()
        # Exactly one native negative draw per epoch, shared by all four routes.
        pos_train_edge, neg_train_edge = get_pos_neg_edges(
            'train',split_edge,edge_index=data.edge_index,num_nodes=self.num_nodes,
            neg_sampler_name=neg_sampler_name,num_neg=num_neg)
        pos_train_edge, neg_train_edge = pos_train_edge.to(self.device), neg_train_edge.to(self.device)
        total_loss = total_examples = 0
        for perm in DataLoader(range(pos_train_edge.size(0)),batch_size,shuffle=True):
            self.optimizer.zero_grad()
            # No persistent training cache: embeddings and common dropout change each call.
            h = self.encoder(self.create_input_feat(data),data.adj_t,data.edge_weight,storage='aggregates')
            pos_edge = pos_train_edge[perm].t()
            neg_edge = torch.reshape(neg_train_edge[perm],(-1,2)).t()
            route_losses = []
            for member,predictor in enumerate(self.predictor):
                pos_out = predictor(h[member,pos_edge[0]],h[member,pos_edge[1]])
                neg_out = predictor(h[member,neg_edge[0]],h[member,neg_edge[1]])
                # Inherited native dispatch falls back to exact auc_loss when margin is None.
                route_losses.append(self.calculate_loss(pos_out,neg_out,num_neg,margin=None))
            loss = torch.stack(route_losses).mean()
            loss.backward()
            if self.clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(),self.clip_norm)
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(),self.clip_norm)
            # Native embeddings remain in Adam and are not included in native clipping groups.
            self.optimizer.step()
            num_examples = pos_edge.size(1)
            total_loss += loss.item()*num_examples
            total_examples += num_examples
        return total_loss/total_examples

    @torch.no_grad()
    def batch_predict(self, h, edges, batch_size):
        predictions = []
        for perm in DataLoader(range(edges.size(0)),batch_size):
            edge = edges[perm].t()
            routes = [predictor(h[member,edge[0]],h[member,edge[1]]).reshape(-1)
                      for member,predictor in enumerate(self.predictor)]
            predictions.append(torch.stack(routes,dim=0).mean(dim=0).cpu())
        return torch.cat(predictions,dim=0)

    @torch.no_grad()
    def validate(self, data, split_edge, batch_size, evaluator):
        self.encoder.eval()
        self.predictor.eval()
        h = self.encoder(self.create_input_feat(data),data.adj_t,data.edge_weight,storage='aggregates')
        # Retain native mean unseen-node representation independently for each route.
        h = torch.cat([h,h.mean(dim=1,keepdim=True)],dim=1)
        pos_valid_edge, neg_valid_edge = get_pos_neg_edges('valid',split_edge)
        pos_valid_pred = self.batch_predict(h,pos_valid_edge.to(self.device),batch_size)
        neg_valid_pred = self.batch_predict(h,neg_valid_edge.to(self.device),batch_size)
        results = {}
        for K in [20,50,100]:
            evaluator.K = K
            results[f'Hits@{K}'] = evaluator.eval({
                'y_pred_pos':pos_valid_pred,'y_pred_neg':neg_valid_pred})[f'hits@{K}']
        return results
