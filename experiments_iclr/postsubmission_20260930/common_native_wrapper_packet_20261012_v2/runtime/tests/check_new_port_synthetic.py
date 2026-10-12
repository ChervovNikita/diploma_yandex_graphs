"""Focused CPU-only new-port QA using freshly made synthetic tensors.

This is not a scientific fit/pilot or a rerun of a closed audit. It exercises
the native source interface, changed provider dimensions, persistent route
RNG, exchange cross-route gradients and literal source train step behavior.
No provider loader, payload, outcome, checkpoint file, or CUDA path is used.
"""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

RUNTIME = Path(__file__).resolve().parents[1]
PACKET = RUNTIME.parent
ROOT = PACKET.parent
sys.path.insert(0, str(RUNTIME))


def main():
    missing = [name for name in ('torch', 'torch_geometric', 'numpy') if importlib.util.find_spec(name) is None]
    if missing:
        result = {'schema': 'common-wrapper-new-port-synthetic-QA-v1', 'status': 'not_run_missing_dependencies',
            'missing_dependencies': missing, 'python': sys.executable, 'scope': 'CPU synthetic engineering only',
            'scientific_fits': 0, 'real_tensor_reads': 0, 'CUDA_operations': 0, 'old_closed_audits_rerun': False}
        (PACKET / 'VALIDATION_SYNTHETIC.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        return
    import torch
    import torch_geometric
    from adapters.common_native_wrapper import ARMS, BACKBONES, bound, construct_source, load_module
    from engine import inspect_independence, inspect_ownership
    torch.set_num_threads(1)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('Source requires float32')
    bindings = json.loads((PACKET / 'SOURCE_BINDINGS.json').read_text())
    factors = load_module(ROOT, bindings['factors'])
    routes = load_module(ROOT, bindings['routes'])
    # Bias must remain outside fast scaling, and W/b must be the original objects.
    dense = torch.nn.Linear(5, 4)
    original_weight, original_bias = dense.weight, dense.bias
    factored = factors.FactorLinear(dense, 4)
    assert factored.weight is original_weight and factored.bias is original_bias
    with torch.no_grad():
        factored.r[2].copy_(torch.arange(1., 6.))
        factored.s[2].copy_(torch.arange(2., 6.))
        factored.bias.copy_(torch.arange(1., 5.))
    sample = torch.linspace(-1, 1, 15).reshape(3, 5)
    with factors.member_context(factored, 2):
        expected = (sample * factored.r[2]) @ factored.weight.T * factored.s[2] + factored.bias
        torch.testing.assert_close(factored(sample), expected, rtol=0, atol=0)
    assert factored.member == 0
    raw = torch.linspace(-1, 1, 3 * 4 * 128).reshape(3, 4, 128)
    block_receipts = []
    for kind in ('separable', 'exchange'):
        block = routes.make_route_block(torch, 128, 4, kind, 16, 123457)
        assert sum(p.numel() for p in block.parameters()) == 8192
        assert torch.equal(block(raw), raw)
        # Exercise nonzero learned residual weights without fitting any model.
        with torch.no_grad():
            output_map = block.C if kind == 'separable' else block.U
            output_map.copy_(torch.linspace(-.01, .01, output_map.numel()).reshape_as(output_map))
        live = raw.clone().requires_grad_()
        loss = block(live)[0, 0].square().sum()
        gradient = torch.autograd.grad(loss, live)[0]
        peer = gradient[0, 1:].abs().sum().item()
        assert (peer > 0) if kind == 'exchange' else (peer == 0)
        block_receipts.append({'kind': kind, 'parameters': 8192, 'identity_at_zero_output_map': True,
            'off_route_prefix_gradient': peer > 0})
    edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
                          [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7]])
    edges = torch.cat([edges, torch.arange(9).repeat(2, 1)], dim=1)
    native_cases = []
    # Two actual provider shapes, nine freshly synthesized node rows each.
    for backbone in BACKBONES:
        for features, classes in ((300, 10), (745, 8)):
            x = torch.linspace(-.5, .5, 9 * features).reshape(9, features)
            graph = SimpleNamespace(edge_index=edges)
            batch = {'x': x, 'edge_index': edges, 'graph': graph}
            mask = torch.tensor([True, True, True, False, False, False, False, False, False])
            y = torch.arange(9) % classes
            data = SimpleNamespace(x=x, edge_index=edges, y=y, train_mask=mask)
            for arm in ARMS:
                source, model, optimizer, streams, primitives, rendered = construct_source(torch, ROOT, bindings,
                    features=features, classes=classes, seed=88101, body=0, arm=arm, backbone=backbone, dropout=.2, learning_rate=.001, device='cpu')
                source.batch, source.data, source.train_ids = batch, data, torch.arange(3)
                adapter = SimpleNamespace(model=model, optimizer=optimizer)
                inspect_ownership(adapter)
                members = 4 if arm in ('shared4_unchanged', 'separable_equal_size', 'exchange', 'original_gnnm') else 1
                body = model.body if hasattr(model, 'body') else model
                if arm not in ('ordinary_M1', 'original_gnnm'):
                    for module in body.modules():
                        if isinstance(module, factors.FactorLinear):
                            assert torch.equal(module.s, torch.ones_like(module.s))
                            if module is body.input_linear:
                                for row in range(members):
                                    generator = torch.Generator().manual_seed(source.seeds(88101, row)[1])
                                    signs = torch.randint(0, 2, (1, features), generator=generator) * 2 - 1
                                    assert torch.equal(module.r[row], signs[0].float())
                            else:
                                assert torch.equal(module.r, torch.ones_like(module.r))
                model.eval()
                if arm in ('shared4_unchanged', 'separable_equal_size', 'exchange'):
                    eval_streams = copy.deepcopy(streams)
                    actual, _ = model(batch, route_scope=lambda row: source.scope(streams, row))
                    direct = []
                    for row in range(4):
                        with factors.member_context(body, row):
                            direct.append(body(graph, x))
                    # Complete native forwards must agree with split prefix/tail
                    # when the newly added residual is exactly zero.
                    torch.testing.assert_close(actual, torch.stack(direct), rtol=0, atol=0)
                    assert all(torch.equal(a['cpu'], b['cpu']) for a, b in zip(eval_streams, streams))
                # One new-port engineering source update on a tiny synthetic graph.
                before_model, before_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
                before_streams = copy.deepcopy(streams)
                def train_once():
                    if arm in ('shared4_unchanged', 'separable_equal_size', 'exchange'):
                        return primitives['packed_train_step'](source, model, optimizer, streams, 4)
                    with source.scope(streams, 0):
                        return (primitives['train_step_tabm'](model, data, optimizer, False) if arm == 'original_gnnm'
                                else primitives['train_step'](model, optimizer, data, False))
                loss = train_once()
                after_model, after_optimizer, after_streams = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), copy.deepcopy(streams)
                assert math.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert any(not torch.equal(before_model[name], after_model[name]) for name in before_model)
                # Exact restoration/replay must resume every packed prefix/tail
                # RNG trajectory as well as the selected optimizer moments.
                model.load_state_dict(before_model)
                optimizer.load_state_dict(before_optimizer)
                streams[:] = copy.deepcopy(before_streams)
                replay = train_once()
                assert loss == replay
                assert all(torch.equal(value, model.state_dict()[name]) for name, value in after_model.items())
                assert all(torch.equal(a['cpu'], b['cpu']) for a, b in zip(after_streams, streams))
                assert all(torch.equal(after_optimizer['state'][key]['exp_avg'], optimizer.state_dict()['state'][key]['exp_avg']) for key in after_optimizer['state'])
                native_cases.append({'backbone': backbone, 'features': features, 'classes': classes, 'arm': arm,
                    'source_initial_factors_checked': arm not in ('ordinary_M1', 'original_gnnm'),
                    'ownership_passed': True, 'synthetic_update_replay_exact': True})
            # Genuine I4 is four separately initialized complete bodies/optimizers.
            for arm in ('ordinary_M1', 'factorized_allmap_M1'):
                learners = []
                for body_id in range(4):
                    _, model, optimizer, _, _, _ = construct_source(torch, ROOT, bindings,
                        features=features, classes=classes, seed=88201, body=body_id, arm=arm, backbone=backbone,
                        dropout=.2, learning_rate=.001, device='cpu')
                    learners.append(SimpleNamespace(model=model, optimizer=optimizer))
                assert inspect_independence(learners)['independent_adapters'] == 4
    # Rejection is exercised on fresh native objects before any optimizer.
    native = load_module(ROOT, bindings['native'])
    for defect in ('cached', 'buffer', 'already_factorized'):
        body = native.Model('GCN', 2, 11, 128, 3, 1, 8, 'LayerNorm', .2)
        if defect == 'cached':
            body.residual_modules[0].module.conv.cached = True
        if defect == 'buffer':
            body.register_buffer('extra_static', torch.zeros(()))
        if defect == 'already_factorized':
            factors.install_factors(body, 4)
        try:
            routes.wrap_shared(torch, factors, body, routes.NativeModelAdapter(torch, body), members=4, kind='exchange', rank=16)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe source body accepted: ' + defect)
    assert not torch.cuda.is_initialized()
    result = {'schema': 'common-wrapper-new-port-synthetic-QA-v1', 'status': 'passed', 'scope': 'CPU tiny synthetic new-port engineering only',
        'torch_version': torch.__version__, 'pyg_version': torch_geometric.__version__, 'python': sys.executable,
        'native_cases': native_cases, 'route_blocks': block_receipts, 'synthetic_train_primitive_calls': 72,
        'I4_independence_cases': 12, 'unsafe_body_rejections': ['cached', 'buffer', 'already_factorized'],
        'real_tensor_reads': 0, 'scientific_fits': 0, 'scientific_scores': 0, 'CUDA_operations': 0,
        'old_closed_audits_rerun': False, 'source_recipe_or_accuracy_qualification': False}
    (PACKET / 'VALIDATION_SYNTHETIC.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'passed', 'native_cases': len(native_cases), 'scientific_fits': 0, 'real_tensor_reads': 0}))


if __name__ == '__main__':
    main()
