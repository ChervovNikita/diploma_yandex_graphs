"""Frozen CPU tensor checks only: no Session/model/dataset/checkpoint or fit.

Root runs this once on its existing Torch environment. Source import/help is
stdlib-only; numerical providers import after explicit invocation. Values in
this fixture have no scientific performance interpretation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PUBLIC_MANIFEST_SHA = '190940ca9f8141ac45f739d064cb1ef76aaf1965ba8c91adb544ad8e38fef724'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-interface', type=Path,
                        default=HERE.parent/'portable_internal_be_public_interface_20261007_v2')
    args = parser.parse_args()
    root = args.public_interface.resolve()
    assert sha(root/'MANIFEST.json') == PUBLIC_MANIFEST_SHA
    for row in json.loads((root/'MANIFEST.json').read_text())['files']:
        assert (root/row['path']).stat().st_size == row['bytes']
        assert sha(root/row['path']) == row['sha256']

    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    started = time.monotonic()
    losses = load('_fixture_candidate_losses', HERE/'losses.py')
    original = load('_fixture_original_objectives', root/'core/objectives.py')
    driver = load('_fixture_source_only_driver', HERE/'train.py')
    assert driver.LOSS_MODULE_SHA == sha(HERE/'losses.py')
    assert driver.RECOMPUTE_SHA == sha(HERE/'recompute.py')
    labels = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    # Fixed literal tensors avoid any random draw or dependence on scientific data.
    a = (torch.arange(40, dtype=torch.float64).reshape(2, 5, 4)/17 - 1.1).requires_grad_()
    b = (torch.sin(torch.arange(40, dtype=torch.float64)+.3).reshape(2, 5, 4)).requires_grad_()
    rng = torch.get_rng_state().clone()

    def gradients(value, first=a, second=b):
        return torch.autograd.grad(value, (first, second), retain_graph=True)

    def compare(value, reference, exact=False):
        assert torch.equal(value, reference) if exact else torch.allclose(value, reference, atol=1e-11, rtol=1e-11)
        for actual, expected in zip(gradients(value), gradients(reference)):
            assert torch.equal(actual, expected) if exact else torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)

    cross = losses.class_full_cross_view(a, b, labels)
    baseline = original.alignment_loss(a, b, labels, 'wikics', .2, None)
    compare(cross, baseline, exact=True)

    # Independent literal Eq2 reference: loop over anchor/candidate identities.
    z = torch.cat((F.normalize(a, dim=-1), F.normalize(b, dim=-1)), dim=1)
    y = labels.repeat(2)
    reference_rows, inside_rows = [], []
    for member in range(len(z)):
        for anchor in range(len(y)):
            candidates = [j for j in range(len(y)) if j != anchor]
            positives = [j for j in candidates if y[j] == y[anchor]]
            candidate_scores = torch.stack([z[member, anchor].dot(z[member, j])/.2 for j in candidates])
            positive_scores = torch.stack([z[member, anchor].dot(z[member, j])/.2 for j in positives])
            denominator = torch.logsumexp(candidate_scores, 0)
            reference_rows.append(torch.stack([denominator - p for p in positive_scores]).mean())
            inside_rows.append(denominator - torch.logsumexp(positive_scores, 0)
                               + torch.log(torch.tensor(float(len(positives)), dtype=a.dtype)))
    canonical = losses.canonical_supcon_eq2(a, b, labels)
    reference = torch.stack(reference_rows).mean()
    compare(canonical, reference)
    inside = torch.stack(inside_rows).mean()
    assert abs(float(canonical-inside)) > 1e-5
    assert any(not torch.allclose(x, q, atol=1e-8, rtol=1e-8)
               for x, q in zip(gradients(canonical), gradients(inside)))

    # Independent derivative formula includes both anchor and candidate roles.
    raw = torch.matmul(z, z.transpose(1, 2))/.2
    diagonal = torch.eye(len(y), dtype=torch.bool)
    positive = (y[:, None] == y[None, :]) & ~diagonal
    class_counts = torch.tensor([2, 2, 1, 2, 2], dtype=torch.long).repeat(2)
    assert torch.equal(positive.sum(-1), 2*class_counts-1)
    assert torch.equal((~diagonal).sum(-1), torch.full((10,), 9))
    R = raw.masked_fill(diagonal, -float('inf')).softmax(-1)
    U = positive.to(raw.dtype)/positive.sum(-1, keepdim=True)
    pair_derivative = (R-U)/(len(z)*len(y))
    grad_z = torch.matmul(pair_derivative+pair_derivative.transpose(1, 2), z)/.2
    for h, projected, expected in ((a, z[:, :5], grad_z[:, :5]),
                                    (b, z[:, 5:], grad_z[:, 5:])):
        jacobian_applied = (expected-projected*(projected*expected).sum(-1, keepdim=True))/h.norm(dim=-1, keepdim=True)
        actual = torch.autograd.grad(canonical, h, retain_graph=True)[0]
        assert torch.allclose(actual, jacobian_applied, atol=1e-10, rtol=1e-10)

    # Alter only member1: member0 has no cross-member auxiliary path.
    changed_a, changed_b = a.detach().clone(), b.detach().clone()
    changed_a[1] += .37; changed_b[1] -= .19
    changed_a.requires_grad_(); changed_b.requires_grad_()
    for operation in (losses.class_full_cross_view, losses.canonical_supcon_eq2):
        first = gradients(operation(a, b, labels))
        second = torch.autograd.grad(operation(changed_a, changed_b, labels), (changed_a, changed_b))
        for x, q in zip(first, second):
            assert torch.equal(x[0], q[0])

    # Zero and below-epsilon rows must be finite: no zero times masked -infinity.
    tiny_a, tiny_b = a.detach().clone(), b.detach().clone()
    tiny_a[:, 0] = 0.; tiny_b[:, 1] = 1e-20
    tiny_a.requires_grad_(); tiny_b.requires_grad_()
    for operation in (losses.class_full_cross_view, losses.canonical_supcon_eq2):
        value = operation(tiny_a, tiny_b, labels)
        assert torch.isfinite(value)
        assert all(torch.isfinite(g).all() for g in torch.autograd.grad(value, (tiny_a, tiny_b)))
    # A singleton class still has the opposite-view counterpart as its positive.
    assert int(positive.sum(-1)[2]) == 1

    # Both conditions retain complete580-node mean-member two-view own CE.
    full_labels = torch.arange(580, dtype=torch.long) % 10
    logits_a = (torch.cos(torch.arange(4*580*10, dtype=a.dtype)).reshape(4, 580, 10)).requires_grad_()
    logits_b = (torch.sin(torch.arange(4*580*10, dtype=a.dtype)).reshape(4, 580, 10)).requires_grad_()
    own = .5*(original.own_supervision(logits_a, full_labels, 'wikics')
              + original.own_supervision(logits_b, full_labels, 'wikics'))
    for auxiliary in (cross, canonical):
        total = own.mean()+.05*auxiliary
        for logits, actual in zip((logits_a, logits_b),
                                  torch.autograd.grad(total, (logits_a, logits_b), retain_graph=True)):
            expected = (logits.softmax(-1)-F.one_hot(full_labels, 10))/(2*4*580)
            assert torch.allclose(actual, expected, atol=1e-15, rtol=1e-12)
        for actual, expected in zip(gradients(total), gradients(auxiliary)):
            assert torch.allclose(actual, .05*expected, atol=1e-12, rtol=1e-11)
    # Preserve the actual original linspace operator; no replacement panel is fed.
    panel = torch.linspace(0, 579, steps=512, device='cpu').long()
    assert panel.shape == (512,) and len(panel.unique()) == 512
    assert int(panel[0]) == 0 and int(panel[-1]) == 579
    assert torch.equal(torch.get_rng_state(), rng)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(schema='WikiCS-SupCon-CPU-semantic-fixture-v1', complete=True,
        torch=str(torch.__version__), python=sys.executable, losses_sha256=sha(HERE/'losses.py'),
        wrapper_sha256=sha(HERE/'train.py'), replay_sha256=sha(HERE/'recompute.py'),
        public_manifest_sha256=PUBLIC_MANIFEST_SHA,
        cross_view_original_value_and_gradient_exact=True,
        canonical_explicit_Eq2_value_and_gradient=True, Eq3_value_and_gradient_distinct=True,
        canonical_analytic_both_role_gradient=True, no_cross_member_gradient=True,
        singleton_positive_and_exact_self_exclusion=True, zero_floor_finite=True,
        complete580_two_view_own_CE_and_coefficient_scaling=True,
        CPU_panel_operator_preserved=True, CPU_panel_raw_sha256=hashlib.sha256(panel.numpy().tobytes()).hexdigest(),
        RNG_unchanged=True, CUDA_initialized=False, fixtures_only=True,
        scientific_models_constructed=0, data_or_checkpoints_loaded=False,
        training_updates=0, current_outcomes_opened=False, seconds=time.monotonic()-started), sort_keys=True))


if __name__ == '__main__':
    main()
