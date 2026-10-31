"""CPU check of lambda=1 and the matched-control GNNM update."""
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from lambda_objective import train_step_lambda
from projector_controls import compute_loss, make_model, model_output
from run_common import set_seed


def step_control(model, optimizer, data, m):
    model.train()
    optimizer.zero_grad()
    for member in range(m):
        loss = compute_loss(model_output(model, 'gnnm', data, member), data, False) / m
        loss.backward()
    optimizer.step()


def main():
    data = Data(
        x=torch.arange(40, dtype=torch.float32).reshape(8, 5) / 40,
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                 [1, 2, 3, 4, 5, 6, 7, 0]]),
        y=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
        train_mask=torch.ones(8, dtype=torch.bool),
    )
    args = SimpleNamespace(model='SAGE', num_layers=2, hidden_dim=16, m=4)
    set_seed(7)
    control = make_model(args, 'gnnm', 5, 3, torch.device('cpu'))
    control_rng = torch.get_rng_state().clone()
    set_seed(7)
    lam_one = make_model(args, 'gnnm', 5, 3, torch.device('cpu'))
    lam_rng = torch.get_rng_state().clone()
    assert torch.equal(control_rng, lam_rng)
    assert all(torch.equal(a, b) for a, b in zip(control.state_dict().values(),
                                                  lam_one.state_dict().values()))
    control_opt = torch.optim.AdamW(control.parameters(), lr=3e-5, weight_decay=0)
    lam_opt = torch.optim.AdamW(lam_one.parameters(), lr=3e-5, weight_decay=0)
    torch.set_rng_state(control_rng)
    for _ in range(3):
        step_control(control, control_opt, data, args.m)
    after_control_rng = torch.get_rng_state().clone()
    torch.set_rng_state(lam_rng)
    for _ in range(3):
        train_step_lambda(lam_one, lam_opt, data, args.m, 1.0)
    after_lambda_rng = torch.get_rng_state().clone()
    max_param_difference = max((a - b).abs().max().item()
                               for a, b in zip(control.state_dict().values(),
                                               lam_one.state_dict().values()))
    assert torch.equal(after_control_rng, after_lambda_rng)
    assert max_param_difference == 0.0
    print('initial_parameters_equal', True)
    print('initial_rng_equal', True)
    print('rng_equal_after_three_steps', torch.equal(after_control_rng, after_lambda_rng))
    print('max_parameter_difference_after_three_steps', max_param_difference)


if __name__ == '__main__':
    main()
