"""Independent strict-byte successor of the preserved data-free gauge probe.

The fixed seed, fixture, model, transformation, routing and hidden tolerance
are unchanged. Numerical equality and storage-byte identity are distinct fields.
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
ORIGINAL = PHASE / 'source_model_gauge_probe_20261003_v1'
OUT = PHASE / 'source_model_gauge_byte_recheck_20261003_v1'
SOURCE_SEAL = 'a99b0e3b0e8ec597b03e2c390ac176597b5b6422d365fc20624ab98a113d42f9'
WITNESS_SHA = '9a6a738fa48816300b574cba57851d31831004867b1a6f9acf24346e79255bef'
INTERPRETER_SHA = '14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'


def main():
    os.chdir(REPO)
    assert subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip() == str(REPO)
    uuids = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    assert set(uuids) == {'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998',
                          'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
    assert hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest() == INTERPRETER_SHA
    witness_bytes = (ORIGINAL / 'WITNESS.json').read_bytes()
    assert hashlib.sha256(witness_bytes).hexdigest() == WITNESS_SHA
    original_witness = json.loads(witness_bytes)
    assert OUT.is_dir() and set(p.name for p in OUT.iterdir()) == {'recheck.py'}
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    assert torch.__version__ == original_witness['torch_version'] == '2.7.1'
    assert torch.get_default_dtype() == torch.float32
    assert str(torch.get_default_device()) == 'cpu'
    assert not torch.cuda.is_initialized()
    sys.path.insert(0, str(SOURCE))
    import native_reference
    import prototype
    from graph_ops import Graph
    assert native_reference.verify_seal() == SOURCE_SEAL

    def uint8_view(a):
        assert a.layout == torch.strided
        return a.detach().cpu().contiguous().reshape(-1).view(torch.uint8)

    def digest(a):
        return hashlib.sha256(bytes(uint8_view(a).tolist())).hexdigest()

    def comparison(a, b):
        same_dtype = a.dtype == b.dtype
        same_shape = a.shape == b.shape
        numeric = bool(torch.equal(a, b))
        # Metadata gates precede the actual byte comparison. Float torch.equal
        # is recorded separately; only uint8 torch.equal establishes bytes.
        byte_equal = False
        different_bytes = None
        if same_dtype and same_shape:
            ba, bb = uint8_view(a), uint8_view(b)
            byte_equal = bool(torch.equal(ba, bb))
            different_bytes = int((ba != bb).sum())
        return dict(dtype_before=str(a.dtype), dtype_after=str(b.dtype),
                    shape_before=list(a.shape), shape_after=list(b.shape),
                    dtype_identical=same_dtype, shape_identical=same_shape,
                    numerical_equal=numeric, contiguous_uint8_bytes_identical=byte_equal,
                    strict_byte_identity=same_dtype and same_shape and byte_equal,
                    differing_byte_count=different_bytes,
                    before_sha256=digest(a), after_sha256=digest(b),
                    before_bytes=a.numel() * a.element_size(),
                    after_bytes=b.numel() * b.element_size())

    # These controls use no random draws and precede the original manual seed.
    positive_zero = torch.tensor([0.0], dtype=torch.float32)
    negative_zero = torch.tensor([-0.0], dtype=torch.float32)
    controls = dict(signed_zero=comparison(positive_zero, negative_zero),
                    dtype=comparison(positive_zero, positive_zero.double()),
                    shape=comparison(positive_zero, positive_zero.reshape(())))
    noncontiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    controls['noncontiguous'] = comparison(noncontiguous, noncontiguous.contiguous())
    assert controls['signed_zero']['numerical_equal']
    assert not controls['signed_zero']['strict_byte_identity']
    assert not controls['dtype']['strict_byte_identity']
    assert not controls['shape']['strict_byte_identity']
    assert controls['noncontiguous']['strict_byte_identity']

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
    original_rows = {(r['mode'], r['graph']): r for r in original_witness['comparisons']}
    for mode in ['private', 'pooled_after_clamp']:
        before, after = [], []
        for graph_name, edges in [('native', pairs), ('drop_first_two', pairs[2:]),
                                  ('drop_last_two', pairs[:-2])]:
            graph = Graph.from_pairs(edges, 9)
            z, h = run(base, x, graph, queries, mode)
            zg, hg = run(transformed, x, graph, queries, mode)
            assert z.shape == zg.shape == (len(queries), 4)
            hidden_scaling = bool(torch.allclose(hg, h * c[:, None, :], rtol=1e-5, atol=1e-6))
            assert hidden_scaling, 'Original hidden-scaling tolerance failed'

            def cosine_energy(a):
                a = a.flatten(1)
                assert bool((a.norm(dim=1) > 0).all())
                return float(torch.stack([F.cosine_similarity(a[i:i+1], a[j:j+1]).square()
                                          for i in range(4) for j in range(i + 1, 4)]).mean())

            cb, ca = cosine_energy(h), cosine_energy(hg)
            old = original_rows[(mode, graph_name)]
            rows.append(dict(mode=mode, graph=graph_name, query_count=len(queries),
                             logits=comparison(z, zg),
                             members=[dict(member=m, **comparison(z[:, m], zg[:, m])) for m in range(4)],
                             served_mean=comparison(prototype.CompletionTwin.serve(z),
                                                    prototype.CompletionTwin.serve(zg)),
                             maximum_logit_difference=float((z - zg).abs().max()),
                             hidden_scaling_allclose=hidden_scaling,
                             hidden_scaling_tolerance=dict(rtol=1e-5, atol=1e-6),
                             mean_squared_hidden_cosine_before=cb,
                             mean_squared_hidden_cosine_after=ca,
                             original_cosines_reproduced_exactly=(cb == old['mean_squared_hidden_cosine_before']
                                                                 and ca == old['mean_squared_hidden_cosine_after'])))
            before.append(torch.sigmoid(z))
            after.append(torch.sigmoid(zg))
        response = torch.cat([before[0] - before[1], before[0] - before[2]])
        response_g = torch.cat([after[0] - after[1], after[0] - after[2]])
        assert response.shape == response_g.shape == (12, 4)
        graph_responses = []
        for index, graph_name in [(1, 'drop_first_two'), (2, 'drop_last_two')]:
            r, rg = before[0] - before[index], after[0] - after[index]
            graph_responses.append(dict(graph=graph_name, responses=comparison(r, rg),
                                        members=[dict(member=m, **comparison(r[:, m], rg[:, m]))
                                                 for m in range(4)]))
        probability_maps[mode] = dict(concatenated=comparison(response, response_g),
                                     members=[dict(member=m, **comparison(response[:, m], response_g[:, m]))
                                              for m in range(4)], graph_responses=graph_responses,
                                     maximum_response_difference=float((response - response_g).abs().max()))
    claimed_comparisons = [row['logits'] for row in rows]
    claimed_comparisons += [probability_maps[m]['concatenated'] for m in probability_maps]
    all_numeric = all(r['numerical_equal'] for r in claimed_comparisons)
    all_bytes = all(r['strict_byte_identity'] for r in claimed_comparisons)
    assert all(t.device.type == 'cpu' for t in (x, pairs, queries, c))
    assert all(p.device.type == 'cpu' for model in (base, transformed) for p in model.parameters())
    assert not torch.cuda.is_initialized()
    assert native_reference.verify_seal() == SOURCE_SEAL
    assert hashlib.sha256((ORIGINAL / 'WITNESS.json').read_bytes()).hexdigest() == WITNESS_SHA
    import torch._C
    runtime_files = [Path(sys.executable), Path(torch.__file__), Path(torch._C.__file__)]
    result = dict(schema='ncnc-positive-diagonal-gauge-strict-byte-recheck-v1',
                  UTC=datetime.now(timezone.utc).isoformat(),
                  status=('BYTE_IDENTITY_CONFIRMED_ON_FIXED_FIXTURE' if all_bytes else
                          'NUMERICAL_IDENTITY_ONLY_ON_FIXED_FIXTURE' if all_numeric else
                          'NUMERICAL_INVARIANCE_NOT_REPRODUCED_ON_FIXED_FIXTURE'),
                  original_witness_sha256=WITNESS_SHA, source_manifest_sha256=SOURCE_SEAL,
                  executed_recheck_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  original_probe_checker='torch.equal: numerical equality, insufficient for strict bytes',
                  strict_checker='dtype and shape, then detach().cpu().contiguous().reshape(-1).view(torch.uint8)',
                  strict_checker_tolerance=None, engineering_seed=20261003, dtype='float32', device='CPU',
                  fixture_tensors=[dict(name=n, dtype=str(a.dtype), shape=list(a.shape), sha256=digest(a))
                                   for n, a in [('x', x), ('pairs', pairs), ('queries', queries), ('scales', c)]],
                  fixture_base_state=[dict(name=n, dtype=str(a.dtype), shape=list(a.shape), sha256=digest(a))
                                      for n, a in base.state_dict().items()],
                  changed_parameters=original_witness['changed_parameters'], fixed_scales=[1.0 / 32.0, 32.0],
                  comparisons=rows, responses=probability_maps, comparator_controls=controls,
                  all_original_claimed_tensors_numerically_equal=all_numeric,
                  all_original_claimed_tensors_strict_byte_identical=all_bytes,
                  claimed_logit_tensor_count=6, claimed_response_tensor_count=2,
                  explicit_member_logit_comparison_count=24, explicit_member_graph_response_count=16,
                  runtime=dict(torch_version=torch.__version__, executable=sys.executable,
                               files=[dict(path=str(p), bytes=p.stat().st_size,
                                           sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in runtime_files],
                               num_threads=torch.get_num_threads(), num_interop_threads=torch.get_num_interop_threads(),
                               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                               CUDA_initialized=torch.cuda.is_initialized(), normal_execution=True,
                               isolation_wrapper=False, base_environment_changed=False),
                  evidence_role='fixed data-free floating-point witness; analytic reparameterization argument separate',
                  universal_floating_point_byte_invariance_claim=False,
                  dataset_checkpoint_or_label_access=False, GPU_compute=False,
                  predictive_superiority_or_new_primitive_claim=False,
                  original_paper_scores_changed=False, original_witness_preserved=True)
    with (OUT / 'RECHECK.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], member_logit_comparisons=24,
                         member_graph_response_comparisons=16, all_numerical=all_numeric,
                         all_strict_bytes=all_bytes, controls_passed=True,
                         original_cosines_reproduced=all(r['original_cosines_reproduced_exactly'] for r in rows),
                         CUDA_initialized=torch.cuda.is_initialized())))


if __name__ == '__main__':
    main()
