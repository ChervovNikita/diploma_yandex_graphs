"""Meaningful prospective numerical checks. Written and parsed; NOT EXECUTED.

Import and call check_factorization() later in the already qualified runtime.
check_native_reduction(native_hlgnn_class) additionally needs the pinned class.
No dataset, checkpoint, optimizer, training launch, or external artifact is used.
"""

import torch
from torch_sparse import SparseTensor
from member_filter import SharedPowerHLGNN, direct_reference


def irregular_graph():
    # Weighted undirected path/star plus isolated node: symmetric normalized P
    # does not preserve the all-ones vector, so ignoring bias propagation fails.
    row = torch.tensor([0, 1, 1, 2, 1, 3, 3, 4], dtype=torch.long)
    col = torch.tensor([1, 0, 2, 1, 3, 1, 4, 3], dtype=torch.long)
    weight = torch.tensor([0.7, 0.7, 1.2, 1.2, 0.4, 0.4, 1.8, 1.8], dtype=torch.float64)
    return SparseTensor(row=row, col=col, value=weight, sparse_sizes=(6, 6))


def probe(output):
    index = torch.arange(output.numel(), dtype=output.dtype).reshape(output.shape)
    coefficients = torch.cos(index * 0.37) + 0.3 * torch.sin(index * 0.11)
    return (torch.tanh(output) * coefficients + 0.07 * output.square()).sum()


def output_and_gradients(model, x, graph, mask, method):
    parameters = list(model.named_parameters())
    dropped_x = x * mask  # Common fixed dropout realization; preserves d/dX.
    if method == "direct":
        output = direct_reference(model, x, graph, dropped_x=dropped_x)
    else:
        z, P = model.context(x, graph, dropped_x=dropped_x)
        output = model.factored(z, P, storage=method)
    gradients = torch.autograd.grad(probe(output), [x] + [p for _, p in parameters])
    return output, {"input": gradients[0], **{n: g for (n, _), g in zip(parameters, gradients[1:])}}


def check_factorization():
    graph = irregular_graph()
    checks = []
    sample = SharedPowerHLGNN(5, 4).double()
    _, P = sample.context(torch.zeros(6, 5, dtype=torch.float64), graph)
    from torch_sparse import matmul
    assert not torch.allclose(matmul(P, torch.ones(6, 1, dtype=torch.float64)),
                              torch.ones(6, 1, dtype=torch.float64))
    # K=0 tests affine reduction; nonzero K tests actual sharing and bias handling.
    for K in (0, 4):
        for members in (1, 3):
            for private_alpha in (False, True):
                torch.manual_seed(9401)
                model = SharedPowerHLGNN(5, 4, K=K, members=members,
                                        private_alpha=private_alpha).double().train()
                with torch.no_grad():
                    model.lin1.bias.copy_(torch.tensor([0.4, -0.7, 0.3, 0.8], dtype=torch.float64))
                    if members > 1:
                        model.r.copy_(torch.randn_like(model.r) * 0.2 + 1)
                        model.s.copy_(torch.randn_like(model.s) * 0.2 + 1)
                        model.bias_delta.copy_(torch.randn_like(model.bias_delta) * 0.15)
                    # Nonzero signed arbitrary coefficients: tests free learning,
                    # not merely identical KI initial filters.
                    model.temp.copy_(torch.randn_like(model.temp) * 0.2 + 0.1)
                base_x = torch.randn(6, 5, dtype=torch.float64)
                mask = (torch.rand_like(base_x) > 0.3).to(base_x.dtype) / 0.7
                direct, gradients = output_and_gradients(model, base_x.clone().requires_grad_(),
                                                        graph, mask, "direct")
                for storage in ("powers", "aggregates"):
                    factored, other = output_and_gradients(model, base_x.clone().requires_grad_(),
                                                          graph, mask, storage)
                    torch.testing.assert_close(factored, direct, rtol=1e-9, atol=1e-10)
                    for name in gradients:
                        torch.testing.assert_close(other[name], gradients[name], rtol=1e-8, atol=1e-10)
                    checks.append((K, members, private_alpha, storage, tuple(gradients)))
    return checks


def check_native_reduction(native_hlgnn_class):
    """Pass HLGNN from the pinned author layer.py; no checkpoint is loaded."""
    torch.manual_seed(9402)
    native = native_hlgnn_class(5, 4, 4, 15, 0.3, 0.5, "KI").double().train()
    adapter = SharedPowerHLGNN(5, 4).double().train()
    # In-memory tied parameters, including nonzero native bias and learned alpha.
    adapter.load_state_dict(native.state_dict(), strict=True)
    graph = irregular_graph()
    x_native = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
    x_adapter = x_native.detach().clone().requires_grad_()
    # Identical RNG context replays the native input dropout placement.
    torch.manual_seed(9403)
    output_native = native(x_native, graph, None)
    torch.manual_seed(9403)
    output_adapter = adapter(x_adapter, graph, None)
    torch.testing.assert_close(output_adapter, output_native, rtol=1e-9, atol=1e-10)
    native_parameters = dict(native.named_parameters())
    adapter_parameters = dict(adapter.named_parameters())
    assert tuple(native_parameters) == tuple(adapter_parameters)
    names = tuple(native_parameters)
    gradients_native = torch.autograd.grad(probe(output_native), [x_native] + [native_parameters[n] for n in names])
    gradients_adapter = torch.autograd.grad(probe(output_adapter), [x_adapter] + [adapter_parameters[n] for n in names])
    for a, b in zip(gradients_adapter, gradients_native):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)
    return {"members": 1, "K": 15, "dropout": 0.3, "init": "KI", "alpha": 0.5, "parameter_names": names}
