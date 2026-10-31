"""Source-derived Torch port of pinned GNNBenchmark meanSAGE.

No numerical imports at module load. This is not bitwise TensorFlow replication.
Root admits the provider/runtime and prospective ten-mask protocol.
"""
SEEDS = (8101, 8219, 8363, 8501, 8663, 8819, 9001, 9199, 9403, 9623)
POINT = {'dataset': 'CoauthorPhysics', 'num_layers': 2, 'hidden_size': 32,
         'aggregator': 'mean', 'dropout_prob': .4, 'learning_rate': .001,
         'tf_slim_weight_decay': .1, 'effective_first_layer_square_penalty': .05,
         'normalize_features': False, 'max_updates': 100000, 'patience_updates': 50,
         'selector': 'latest_tied_minimum_SELECT_total_CE_plus_L2',
         'optimizer': 'TF1_Adam_bias_powers_epsilon_before_bias_correction',
         'beta1': .9, 'beta2': .999, 'epsilon': 1e-8,
         'support_policy': 'shared_prepared530417_with34493_self_entries_no_added_loops'}


def model_class(torch, F):
    class MeanLayer(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.aggregate_weight = torch.nn.Parameter(torch.empty(input_dim, output_dim))
            self.skip_weight = torch.nn.Parameter(torch.empty(input_dim, output_dim))
            self.bias = torch.nn.Parameter(torch.zeros(output_dim))
            torch.nn.init.xavier_uniform_(self.aggregate_weight)
            torch.nn.init.xavier_uniform_(self.skip_weight)

        def forward(self, features, adjacency, degrees):
            if self.training:
                if features.is_sparse:
                    # Pinned util.py drops only stored values and retains indices,
                    # including values changed to zero; agg and skip share this X.
                    values = F.dropout(features.values(), p=.4, training=True)
                    features = torch.sparse_coo_tensor(features.indices(), values, features.shape,
                                                       device=features.device, is_coalesced=True)
                else:
                    features = F.dropout(features, p=.4, training=True)
            multiply = torch.sparse.mm if features.is_sparse else torch.matmul
            transformed = multiply(features, self.aggregate_weight)
            aggregated = torch.sparse.mm(adjacency, transformed) / degrees
            output = aggregated + multiply(features, self.skip_weight) + self.bias
            # Author clip(norm,1,inf), including logits before softmax/CE.
            return output / torch.linalg.vector_norm(output, ord=2, dim=1, keepdim=True).clamp_min(1.)

    class MeanSAGE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList((MeanLayer(8415, 32), MeanLayer(32, 5)))

        def forward(self, features, adjacency, degrees):
            hidden = F.relu(self.layers[0](features, adjacency, degrees))
            return self.layers[1](hidden, adjacency, degrees)

        def regularizer(self):
            first = self.layers[0]
            return .05 * (first.aggregate_weight.square().sum() + first.skip_weight.square().sum())

    return MeanSAGE


def optimizer_class(torch):
    class TF1Adam(torch.optim.Optimizer):
        """TF1 Adam formula, with float32 beta powers saved as optimizer state.

        lr_t=lr*sqrt(1-beta2_power)/(1-beta1_power), p-=lr_t*m/(sqrt(v)+eps).
        Torch kernels/RNG/reduction order remain a port, not TF bit identity.
        """
        def __init__(self, parameters):
            parameters = list(parameters)
            super().__init__(parameters, {'lr': .001, 'beta1': .9, 'beta2': .999, 'eps': 1e-8})
            self.state['tf1_global'] = {'updates': 0,
                'beta1_power': parameters[0].new_tensor(.9),
                'beta2_power': parameters[0].new_tensor(.999)}

        @torch.no_grad()
        def step(self, closure=None):
            if closure is not None: raise ValueError('Fixed own-CE step has no closure')
            global_state = self.state['tf1_global']
            for group in self.param_groups:
                b1, b2 = group['beta1'], group['beta2']
                for parameter in group['params']:
                    gradient = parameter.grad
                    if gradient is None or gradient.is_sparse:
                        raise ValueError('Every registered published matrix/bias has a dense gradient')
                    state = self.state[parameter]
                    if not state:
                        state['exp_avg'] = torch.zeros_like(parameter)
                        state['exp_avg_sq'] = torch.zeros_like(parameter)
                    m, v = state['exp_avg'], state['exp_avg_sq']
                    m.mul_(b1).add_(gradient, alpha=1 - b1)
                    v.mul_(b2).addcmul_(gradient, gradient, value=1 - b2)
                    beta1_power = global_state['beta1_power'].to(parameter.device)
                    beta2_power = global_state['beta2_power'].to(parameter.device)
                    lr_t = group['lr'] * (1 - beta2_power).sqrt() / (1 - beta1_power)
                    parameter.add_(-lr_t * m / (v.sqrt() + group['eps']))
            global_state['updates'] += 1
            global_state['beta1_power'].mul_(.9)
            global_state['beta2_power'].mul_(.999)

    return TF1Adam


def inspect_model(model):
    parameters = dict(model.named_parameters())
    expected = {'layers.0.aggregate_weight': (8415, 32), 'layers.0.skip_weight': (8415, 32),
                'layers.0.bias': (32,), 'layers.1.aggregate_weight': (32, 5),
                'layers.1.skip_weight': (32, 5), 'layers.1.bias': (5,)}
    if {k: tuple(v.shape) for k, v in parameters.items()} != expected or list(model.named_buffers()):
        raise ValueError('Exact published meanSAGE two-layer matrix/bias structure required')
    count = sum(p.numel() for p in parameters.values())
    if count != 538917: raise ValueError('Symbolic published Physics parameter count differs')
    return {'parameters': count, 'inference_parameter_bytes': sum(p.numel() * p.element_size() for p in parameters.values()),
            'native_bodies': 1, 'members': 1, 'buffer_count': 0,
            'regularized_parameters': ['layers.0.aggregate_weight', 'layers.0.skip_weight'],
            'row_normalization': 'z/max(L2norm(z),1) before activation, also last-layer logits',
            'sparse_input_dropout': 'stored values only, preserved indices, same dropped X for aggregate and skip'}
