"""Resolve two source-review limits with independent arithmetic oracles.

Synthetic CPU engineering only. No dataset, selector or scientific fit.
The sealed prototype remains unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert hashlib.sha256((args.packet / 'MANIFEST.json').read_bytes()).hexdigest() == 'ac04bc7f6f6f37b86c50d38436aee111e987195c5f992816c4a5ef9d175887ae'
    sys.path.insert(0, str(args.packet))
    import torch
    from torch.nn import functional as F
    from native_reference import verify_seal
    from prototype import FactorLinear, CompletionTwin, Recipe, training_batch_loss
    from qualify import fixture, patterned
    verify_seal()
    assert not torch.cuda.is_available()
    torch.manual_seed(31003)
    torch.set_num_threads(1)
    dtype = torch.float64
    tolerance = 128 * torch.finfo(dtype).eps
    def close(x, y):
        torch.testing.assert_close(x, y, rtol=tolerance, atol=tolerance)

    # Explicit derivatives of a weighted linear response, with distinct
    # inputs, upstream derivatives and factors for every member.
    layer = FactorLinear(5, 3, 4).to(dtype)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.copy_(torch.randn_like(parameter))
    inputs = torch.randn(4, 7, 5, dtype=dtype)
    upstream = torch.randn(4, 7, 3, dtype=dtype)
    outputs = torch.stack([layer.forward_member(inputs[m], m) for m in range(4)])
    (outputs * upstream).sum().backward()
    analytic_weight = torch.zeros_like(layer.weight)
    analytic_bias = upstream.sum((0, 1))
    analytic_r = torch.zeros_like(layer.r)
    analytic_s = torch.zeros_like(layer.s)
    with torch.no_grad():
        for m in range(4):
            scaled_input = inputs[m] * layer.r[m]
            scaled_derivative = upstream[m] * layer.s[m]
            analytic_weight += scaled_derivative.T @ scaled_input
            analytic_r[m] = ((scaled_derivative @ layer.weight) * inputs[m]).sum(0)
            analytic_s[m] = ((scaled_input @ layer.weight.T) * upstream[m]).sum(0)
    for actual, expected in [(layer.weight.grad, analytic_weight),
                             (layer.bias.grad, analytic_bias),
                             (layer.r.grad, analytic_r),
                             (layer.s.grad, analytic_s)]:
        close(actual, expected)

    # The independent reference is twice balanced BCE on captured positive
    # and negative logits, rather than the prototype's two log-sigmoid means.
    model = CompletionTwin(Recipe(features=4, hidden=4)).double().train()
    patterned(model, diverse=True)
    features, pairs, _ = fixture()
    captured = []
    encoder_calls = []
    hooks = [model.decoder.register_forward_hook(lambda module, inputs, output: captured.append(output)),
             model.encoder.register_forward_hook(lambda *unused: encoder_calls.append(1))]
    try:
        actual_loss = training_batch_loss(model, features, pairs, torch.tensor([0, 3]),
                                          torch.tensor([[0, 4], [1, 6]]), 'private')
    finally:
        for hook in hooks:
            hook.remove()
    assert len(captured) == 2 and len(encoder_calls) == 1
    positive, negative = captured
    assert positive.shape == negative.shape == (2, 4)
    combined = torch.cat((positive, negative), 0)
    labels = torch.cat((torch.ones_like(positive), torch.zeros_like(negative)), 0)
    expected_loss = 2 * F.binary_cross_entropy_with_logits(combined, labels, reduction='mean')
    close(actual_loss, expected_loss)
    parameters = tuple(model.parameters())
    actual_gradients = torch.autograd.grad(actual_loss, parameters, retain_graph=True, allow_unused=True)
    expected_gradients = torch.autograd.grad(expected_loss, parameters, allow_unused=True)
    checked = 0
    for actual, expected in zip(actual_gradients, expected_gradients):
        if expected is None:
            assert actual is None
        else:
            close(actual, expected)
            checked += 1
    verify_seal()
    result = dict(status='SUPPLEMENTAL_ENGINEERING_ORACLES_PASSED',
                  private_factor_analytic_gradient_oracle=True,
                  shared_weight_and_bias_analytic_gradient_oracle=True,
                  twice_balanced_BCE_loss_and_parameter_gradient_oracle=True,
                  loss_parameter_gradients_checked=checked,
                  prototype_manifest_sha256=hashlib.sha256((args.packet / 'MANIFEST.json').read_bytes()).hexdigest(),
                  executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  torch_version=torch.__version__, dtype=str(dtype),
                  native_float64_encoder_parity_claim=False,
                  CUDA_used=False, benchmark_data_accessed=False, optimizer_updates=0,
                  scientific_predictive_result=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
