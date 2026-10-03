"""Data-free analytic reparameterization witness in the qualified NCNC decoder.

This is a mathematical/implementation diagnostic, not a predictive pilot.
"""
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
SOURCE = PHASE / 'graph_ncNC_member_completion_qualification_preparation_20261003_v2'
OUT = PHASE / 'source_model_gauge_probe_20261003_v1'


def main():
    os.chdir(REPO)
    assert subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip() == str(REPO)
    uuids = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    assert set(uuids) == {'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998',
                          'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    sys.path.insert(0, str(SOURCE))
    import native_reference
    import prototype
    from graph_ops import Graph
    assert native_reference.verify_seal() == 'a99b0e3b0e8ec597b03e2c390ac176597b5b6422d365fc20624ab98a113d42f9'
    torch.manual_seed(20261003)
    base = prototype.CompletionTwin().eval()
    transformed = copy.deepcopy(base)
    width = base.recipe.hidden
    assert width == 64 and base.recipe.member_count == 4
    # Positive diagonal scalings; disjoint emphasis, fixed before execution.
    c = torch.full((4, width), 1.0 / 32.0)
    for member in range(4):
        c[member, member * 16:(member + 1) * 16] = 32.0
    with torch.no_grad():
        transformed.decoder.lin.ops['1'].weight.mul_(c)
        transformed.decoder.lin.ops['1'].bias.mul_(c)
        transformed.decoder.lin.ops['8'].r.div_(c)
    captures = {}

    def run(model, x, graph, queries, mode):
        final = model.decoder.lin.ops['8']
        original = final.forward_member
        captured = {}

        def intercepted(a, member):
            captured[member] = a.detach().clone()
            return original(a, member)

        final.forward_member = intercepted
        try:
            with torch.no_grad():
                z = model(x, graph, queries, mode)
        finally:
            del final.forward_member
        assert set(captured) == set(range(4))
        assert all(a.shape == (len(queries), width) for a in captured.values())
        h = torch.stack([captured[i] for i in range(4)])
        assert bool(torch.isfinite(z).all()) and bool(torch.isfinite(h).all())
        return z, h

    x = torch.randn(9, 128)
    pairs = torch.tensor([[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4],
                          [3, 5], [4, 6], [5, 6], [6, 7], [7, 8], [5, 8]], dtype=torch.long)
    queries = torch.tensor([[0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [6, 8]], dtype=torch.long)
    rows = []
    probability_maps = {}
    for mode in ['private', 'pooled_after_clamp']:
        before, after = [], []
        for graph_name, edges in [('native', pairs), ('drop_first_two', pairs[2:]),
                                  ('drop_last_two', pairs[:-2])]:
            graph = Graph.from_pairs(edges, 9)
            z, h = run(base, x, graph, queries, mode)
            zg, hg = run(transformed, x, graph, queries, mode)
            assert torch.allclose(hg, h * c[:, None, :], rtol=1e-5, atol=1e-6)
            assert torch.equal(z, zg), 'Power-of-two gauge changed served/member logits'
            def cosine_energy(a):
                a = a.flatten(1)
                assert bool((a.norm(dim=1) > 0).all())
                return float(torch.stack([F.cosine_similarity(a[i:i+1], a[j:j+1]).square()
                                          for i in range(4) for j in range(i + 1, 4)]).mean())
            rows.append(dict(mode=mode, graph=graph_name, query_count=len(queries),
                             raw_member_logits_bitwise_identical=True,
                             maximum_logit_difference=float((z - zg).abs().max()),
                             mean_squared_hidden_cosine_before=cosine_energy(h),
                             mean_squared_hidden_cosine_after=cosine_energy(hg)))
            before.append(torch.sigmoid(z))
            after.append(torch.sigmoid(zg))
        response = torch.cat([before[0] - before[1], before[0] - before[2]])
        response_g = torch.cat([after[0] - after[1], after[0] - after[2]])
        assert torch.equal(response, response_g)
        probability_maps[mode] = dict(finite_probability_responses_bitwise_identical=True,
                                      maximum_response_difference=float((response - response_g).abs().max()))
    result = dict(schema='ncnc-positive-diagonal-hidden-gauge-witness-v1',
                  UTC=datetime.now(timezone.utc).isoformat(),
                  source_manifest_sha256=native_reference.verify_seal(), torch_version=torch.__version__,
                  engineering_seed=20261003, dtype='float32', device='CPU',
                  changed_parameters=['decoder.lin.ops.1.weight', 'decoder.lin.ops.1.bias',
                                      'decoder.lin.ops.8.r'],
                  fixed_scales=[1.0 / 32.0, 32.0], comparisons=rows, responses=probability_maps,
                  evidence_role='analytic implementation witness, not representative predictive research',
                  dataset_checkpoint_or_label_access=False, GPU_compute=False,
                  predictive_superiority_or_new_primitive_claim=False,
                  original_paper_scores_changed=False)
    OUT.mkdir(exist_ok=False)
    with (OUT / 'WITNESS.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
