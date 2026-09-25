"""Small shape and gradient smoke check for the projector controls."""
from types import SimpleNamespace
import torch
from torch_geometric.data import Data
from projector_controls import VARIANTS, apply_common_backbone_init, make_model, model_output
from run_common import set_seed


def main():
    data = Data(
        x=torch.randn(8, 5),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                 [1, 2, 3, 4, 5, 6, 7, 0]]),
        y=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
    )
    for name in ('SAGE', 'GAT-sep'):
        args = SimpleNamespace(model=name, num_layers=2, hidden_dim=16, m=4)
        for variant in VARIANTS:
            model = make_model(args, variant, 5, 3, torch.device('cpu'))
            m = 1 if variant in ('base', 'gnnm_m1') else 4
            out = torch.stack([model_output(model, variant, data, i)
                               for i in range(m)])
            assert out.shape == (m, 8, 3), (name, variant, out.shape)
            loss = torch.nn.functional.cross_entropy(out.mean(0), data.y)
            loss.backward()
            assert all(p.grad is not None for p in model.parameters() if p.requires_grad), (name, variant)
            if variant == 'output_only':
                with torch.no_grad():
                    before = model.output_be.R.detach().clone()
                    after_gradient = model.output_be.R.grad.detach().clone()
                    assert not torch.allclose(after_gradient[0], after_gradient[1]), name
                    assert torch.allclose(before[0], before[1]), name
            print(name, variant, out.shape,
                  sum(p.numel() for p in model.parameters() if p.requires_grad))
    args = SimpleNamespace(model='SAGE', num_layers=2, hidden_dim=16, m=4)
    aligned = []
    for variant in ('gnnm', 'independent_projectors'):
        set_seed(0)
        model = make_model(args, variant, 5, 3, torch.device('cpu'))
        apply_common_backbone_init(model, args, 5, 3, 0, torch.device('cpu'))
        aligned.append({k: v.clone() for k, v in model.residual_modules.state_dict().items()})
    assert all(torch.equal(aligned[0][k], aligned[1][k]) for k in aligned[0])
    print('Common backbone initialization matches exactly')

    # Untying the propagation stack must preserve every initial member
    # function and the training RNG state of the original GNNM run.
    set_seed(123)
    tied = make_model(args, 'gnnm', 5, 3, torch.device('cpu'))
    tied_rng = torch.get_rng_state().clone()
    set_seed(123)
    untied = make_model(args, 'untied_backbone', 5, 3, torch.device('cpu'))
    assert torch.equal(tied_rng, torch.get_rng_state())
    tied.eval()
    untied.eval()
    with torch.no_grad():
        for member in range(4):
            for key, tensor in tied.residual_modules.state_dict().items():
                assert torch.equal(
                    tensor, untied.residual_modules_by_member[member].state_dict()[key]
                ), (member, key)
            actual = model_output(untied, 'untied_backbone', data, member)
            expected = model_output(tied, 'gnnm', data, member)
            assert torch.equal(actual, expected), member
    print('Untied backbones preserve all initial logits and training RNG exactly')

    set_seed(123)
    frozen = make_model(args, 'freeze_output_factors', 5, 3,
                        torch.device('cpu'))
    assert torch.equal(tied_rng, torch.get_rng_state())
    assert all(torch.equal(tied.state_dict()[key], tensor)
               for key, tensor in frozen.state_dict().items())
    assert frozen.output_be_block.W.weight.requires_grad
    for factor in (frozen.output_be_block.R, frozen.output_be_block.S,
                   frozen.output_be_block.B):
        assert not factor.requires_grad
    assert torch.all(frozen.output_be_block.R == 1)
    assert torch.all(frozen.output_be_block.S == 1)
    assert torch.all(frozen.output_be_block.B == 0)
    frozen.eval()
    with torch.no_grad():
        for member in range(4):
            actual = model_output(frozen, 'freeze_output_factors', data,
                                  member)
            expected = model_output(tied, 'gnnm', data, member)
            assert torch.equal(actual, expected), member
    print('Frozen output factors preserve all initial logits and training RNG exactly')


if __name__ == '__main__':
    main()
