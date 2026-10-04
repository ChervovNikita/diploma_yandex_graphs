"""Small CPU tensor checks; no dataset or native training operation is called."""
import copy
import random
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def equal(left, right):
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right, (left, right)


def near(left, right):
    tol = 5e-6 if left.dtype == torch.float32 else 1e-10
    torch.testing.assert_close(left, right, atol=tol, rtol=tol)


def model_snapshot(model):
    return dict(state=copy.deepcopy(model.state_dict()),
                gradients={name: None if p.grad is None else p.grad.detach().clone()
                           for name, p in model.named_parameters(remove_duplicate=False)},
                modes={name: module.training for name, module in model.named_modules()},
                requires_grad={name: p.requires_grad for name, p in model.named_parameters()})


class ToyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gain = nn.Parameter(torch.linspace(0.8, 1.2, width))
        self.gain_alias = self.gain
        self.register_buffer("shift", torch.linspace(-0.01, 0.01, width))

    def forward(self, x):
        return (x * self.gain + self.shift).tanh()


class ToyNativeCore(nn.Module):
    """Tiny PolyFormer-shaped core with genuine shared parameters and buffers."""
    def __init__(self):
        super().__init__()
        self.K, self.dropout = 2, 0.25
        self.lin1, self.lin2, self.lin3 = nn.Linear(4, 256), nn.Identity(), nn.Linear(256, 3)
        self.attn = nn.ModuleList([ToyBlock(256)])
        with torch.no_grad():
            self.lin1.weight.mul_(0.2)
            self.lin1.bias.mul_(0.2)
            self.lin3.weight.mul_(0.3)
            self.lin3.bias.mul_(0.3)

    def forward(self, data):
        x = self.lin1(torch.stack(data.list_mat, dim=1))
        for block in self.attn:
            x = block(x)
        x = F.dropout(x.sum(1), self.dropout, self.training)
        x = F.relu(self.lin2(x))
        return self.lin3(F.dropout(x, self.dropout, self.training))


class ToyTeacher(nn.Module):
    def __init__(self, spec=None):
        super().__init__()
        self.specification = copy.deepcopy(spec or dict(backbone="polyformer_mono",
            seed=17, config=0, family="single_author"))
        self.members, self.global_stage = 1, False
        self.models = nn.ModuleList([ToyNativeCore()])

    def set_global_stage(self, enabled):
        assert not enabled
        self.global_stage = False

    def forward(self, graph):
        data = SimpleNamespace(list_mat=graph.teacher_input.unbind(1))
        return self.models[0](data)[None]


def fixture(ctx, dtype=torch.float64, connected_train=True):
    torch.manual_seed(204)
    teacher = ToyTeacher().to(dtype=dtype).eval()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=0.002, betas=(0.7, 0.93),
                                 eps=1e-6, weight_decay=0.03, amsgrad=True)
    # Analytic nonzero history, never obtained by fitting or a warm update.
    for parameter in teacher.parameters():
        m = torch.linspace(-2e-4, 3e-4, parameter.numel(), dtype=dtype).view_as(parameter)
        v = torch.linspace(0.002, 0.003, parameter.numel(), dtype=dtype).view_as(parameter)
        optimizer.state[parameter] = dict(step=torch.tensor(7.0), exp_avg=m.clone(),
                                         exp_avg_sq=v.clone(), max_exp_avg_sq=1.2 * v)
        parameter.grad = torch.full_like(parameter, 1e-4)
    generator = torch.Generator().manual_seed(301)
    graph = SimpleNamespace(teacher_backbone="polyformer_mono", classes=3,
        teacher_input=torch.randn((10, 2, 4), generator=generator, dtype=dtype))
    train = SimpleNamespace(nodes=torch.tensor([0, 2, 4, 7, 9]), labels=torch.tensor([0, 1, 2, 0, 2]))
    before = model_snapshot(teacher)
    native_frozen = ctx.integration.named_optimizer_snapshot(teacher, optimizer)
    k1 = ctx.integration.clone_boundary(teacher, ctx.boundary, 1).eval()
    k4 = ctx.integration.clone_boundary(teacher, ctx.boundary, 4)
    ctx.integration.identity_logits_audit(teacher, k1, k4, graph)
    raw_optimizer, transport = ctx.integration.transport_optimizer(teacher, native_frozen, k4)
    frozen = ctx.integration.named_optimizer_snapshot(k4, raw_optimizer)
    equal(before, model_snapshot(teacher))
    edges = [(i, i + 1) for i in range(9)] + [(0, 5), (1, 7), (3, 8)]
    if connected_train:
        # Declared TRAIN chain activates the S term; original independent set stays available.
        edges += [(0, 2), (2, 4), (4, 7), (7, 9)]
    canonical = torch.tensor(sorted(edges), dtype=torch.int64).T.contiguous()
    S = ctx.initializer.symmetric_normalized_adjacency(10, canonical, dtype, torch.device("cpu"))
    return SimpleNamespace(teacher=teacher, native_optimizer=optimizer, native_frozen=native_frozen,
        k1=k1, prototype=k4, frozen=frozen, transport=transport, graph=graph, train=train,
        S=S, nodes=torch.arange(10), args=(graph.teacher_input,))


def rng_roundtrip(ctx):
    random.seed(125)
    np.random.seed(126)
    torch.manual_seed(127)
    snapshot = ctx.integration.rng_snapshot()
    def draw():
        return dict(python=[random.random() for _ in range(4)], numpy=np.random.rand(4).tolist(),
                    cpu=torch.rand(4), cuda=[torch.rand(4, device="cuda:%d" % i).cpu()
                        for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [])
    first = draw()
    ctx.integration.rng_restore(snapshot)
    equal(first, draw())
    ctx.integration.rng_restore(snapshot)
    equal(snapshot, ctx.integration.rng_snapshot())
    return dict(cuda_rng_devices=len(snapshot["torch_cuda"]))


def optimizer_and_checkpoint_restore(ctx):
    f = fixture(ctx)
    restored = ctx.integration.restore_named_optimizer(f.prototype, f.frozen)
    equal(f.frozen, ctx.integration.named_optimizer_snapshot(f.prototype, restored))
    for row in f.transport["mappings"]:
        old = f.native_frozen["state"][row["native"]]
        new = f.frozen["state"][row["raw_aliases"][0]]
        equal(old["step"], new["step"])
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if row["bias_replication"]:
                divisor = 4 if key == "exp_avg" else 16
                equal(old[key].expand_as(new[key]) / divisor, new[key])
            else:
                equal(old[key], new[key])
    for name, state in f.frozen["state"].items():
        if name.endswith((".R", ".S")):
            assert state == {}, name
    bias_groups = [g for g in f.frozen["groups"] if all(n.endswith(".B") for n in g["names"])]
    assert bias_groups and all(g["options"]["eps"] == 1e-6 / 4 and
        g["options"]["weight_decay"] == 0.03 / 4 for g in bias_groups)
    saved = ctx.integration.native_checkpoint(f.teacher, f.native_optimizer, {"synthetic_fixture": True})
    with tempfile.TemporaryDirectory(prefix="state_", dir=str(ctx.fixtures)) as directory:
        path = Path(directory) / "synthetic_state.pt"
        torch.save(saved, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        equal(saved, loaded)
        api = SimpleNamespace(TeacherFamily=lambda spec: ToyTeacher(spec).to(dtype=next(f.teacher.parameters()).dtype))
        teacher, optimizer = ctx.integration.restore_native(api, loaded, torch.device("cpu"))
        equal(f.teacher.state_dict(), teacher.state_dict())
        equal(saved["optimizer"], ctx.integration.named_optimizer_snapshot(teacher, optimizer))
        equal(saved["rng"], ctx.integration.rng_snapshot())
        assert teacher.training == saved["model_training"] and not teacher.global_stage
    return dict(nonzero_native_history=True, aliases_preserved=True, factors_start_empty=True)


def analytic_adam(optimizer):
    """Independent coupled-decay Adam/AMSGrad equation, including empty history."""
    expected = {}
    for group in optimizer.param_groups:
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                expected[id(p)] = p.detach().clone()
                continue
            state = optimizer.state.get(p, {})
            step = float(state.get("step", 0)) + 1
            effective = p.grad.detach() + group["weight_decay"] * p.detach()
            m = beta1 * state.get("exp_avg", torch.zeros_like(p)) + (1 - beta1) * effective
            v = beta2 * state.get("exp_avg_sq", torch.zeros_like(p)) + (1 - beta2) * effective.square()
            if group["amsgrad"]:
                v = torch.maximum(state.get("max_exp_avg_sq", torch.zeros_like(p)), v)
            denominator = v.sqrt() / ((1 - beta2 ** step) ** 0.5) + group["eps"]
            expected[id(p)] = p.detach() - group["lr"] * m / (1 - beta1 ** step) / denominator
    return expected


def adam_trial(ctx, dtype):
    f = fixture(ctx, dtype)
    prototype_before, frozen_before = model_snapshot(f.prototype), copy.deepcopy(f.frozen)
    rng = ctx.integration.rng_snapshot()
    center = torch.ones(256, dtype=dtype)
    offsets = torch.zeros((4, 256), dtype=dtype)
    offsets[0, 0], offsets[1, 0], offsets[2, 1], offsets[3, 1] = 0.02, -0.02, 0.02, -0.02
    slices = center[None] + offsets
    manual = copy.deepcopy(f.prototype)
    optimizer = ctx.integration.restore_named_optimizer(manual, f.frozen)
    ctx.selector.install_head(manual, slices, "polyformer_mono")
    manual.eval()
    optimizer.zero_grad(set_to_none=True)
    logits = manual(*f.args)
    loss = torch.stack([F.cross_entropy(member[f.train.nodes], f.train.labels) for member in logits]).mean()
    loss.backward()
    for name, p in manual.named_parameters():
        assert p.grad is not None and bool((p.grad != 0).any()), name
    # Independent normalization check: one member's private gradient is four times its mean-loss row.
    single = copy.deepcopy(manual)
    single.zero_grad(set_to_none=True)
    F.cross_entropy(single(*f.args)[0, f.train.nodes], f.train.labels).backward()
    near(single.head.R.grad[0] / 4, manual.head.R.grad[0])
    expected = analytic_adam(optimizer)
    optimizer.step()
    for p in manual.parameters():
        near(p.detach(), expected[id(p)])
    expected_value = float(F.cross_entropy(manual(*f.args).mean(0)[f.train.nodes], f.train.labels))
    captured = []
    def capture(model, frozen):
        opt = ctx.integration.restore_named_optimizer(model, frozen)
        captured.append((model, opt))
        return opt
    adapter = SimpleNamespace(restore_named_optimizer=capture, rng_restore=ctx.integration.rng_restore)
    receipts = [ctx.selector.one_adam_trial(f.prototype, f.frozen, slices, "polyformer_mono",
        f.args, f.train.nodes, f.train.labels, rng, adapter) for _ in range(2)]
    assert all(r["status"] == "eligible" and r["updates"] == 1 for r in receipts), receipts
    assert all(abs(r["post_trial_pooled_train_ce"] - expected_value) <= (5e-6 if dtype == torch.float32 else 1e-10)
               for r in receipts)
    assert captured[0][0] is not captured[1][0]
    for trial, trial_optimizer in captured:
        for a, b in zip(manual.parameters(), trial.parameters()):
            near(a.detach(), b.detach())
            near(a.grad, b.grad)
        equal(dict(manual.named_buffers()), dict(trial.named_buffers()))
        # Same Adam step/options/state, with float comparison for the independent reduction order.
        a = ctx.integration.named_optimizer_snapshot(manual, optimizer)
        b = ctx.integration.named_optimizer_snapshot(trial, trial_optimizer)
        equal(a["groups"], b["groups"])
        equal(a["aliases"], b["aliases"])
        for name in a["state"]:
            for key in a["state"][name]:
                near(a["state"][name][key], b["state"][name][key])
    equal(prototype_before, model_snapshot(f.prototype))
    equal(frozen_before, f.frozen)
    equal(rng, ctx.integration.rng_snapshot())
    assert not any(torch.is_tensor(v) for receipt in receipts for v in receipt.values())
    return dict(dtype=str(dtype), independent_trials=2, analytic_coupled_adam_matches=True,
                mean_private_gradient_scale=0.25)


def failed_trial_restores(ctx):
    f = fixture(ctx)
    before, frozen = model_snapshot(f.prototype), copy.deepcopy(f.frozen)
    rng = ctx.integration.rng_snapshot()
    class BrokenOptimizer:
        def __init__(self, optimizer):
            self.inner, self.state = optimizer, optimizer.state
        def zero_grad(self, **kwargs):
            self.inner.zero_grad(**kwargs)
        def step(self):
            with torch.no_grad():
                self.inner.param_groups[0]["params"][0].add_(0.5)
            random.random(); np.random.rand(); torch.rand(3)
            raise RuntimeError("intentional synthetic partial step failure")
    adapter = SimpleNamespace(rng_restore=ctx.integration.rng_restore,
        restore_named_optimizer=lambda model, state: BrokenOptimizer(ctx.integration.restore_named_optimizer(model, state)))
    slices = torch.ones((4, 256), dtype=torch.float64)
    result = ctx.selector.one_adam_trial(f.prototype, f.frozen, slices, "polyformer_mono",
        f.args, f.train.nodes, f.train.labels, rng, adapter)
    assert result["status"] == "trial_failure" and result["partial_optimizer_step_possible"]
    equal(before, model_snapshot(f.prototype)); equal(frozen, f.frozen)
    equal(rng, ctx.integration.rng_snapshot())
    return dict(partial_trial_failure_discarded=True)


def rank_and_D(ctx, dtype):
    center = torch.zeros(8, dtype=dtype)
    A = torch.zeros((6, 3, 8), dtype=dtype)
    A[:, 0, 0] = 1
    for coordinate, amplitude in ((1, 1.0), (2, 0.5), (3, 1.7)):
        A[2 * (coordinate - 1), 0, coordinate] = amplitude
        A[2 * (coordinate - 1) + 1, 0, coordinate] = -amplitude
    def logits_fn(theta):
        return torch.einsum("ncd,d->nc", A, theta)
    rows, labels = torch.arange(6), torch.zeros(6, dtype=torch.int64)
    gradient = torch.func.grad(lambda theta: F.cross_entropy(logits_fn(theta), labels))(center).double()
    bank = torch.zeros((4, 8), dtype=torch.float64)
    bank[0, 1] = bank[1, 2] = bank[2, 3] = 1
    bank[3] = -bank[:3].sum(0)
    basis, receipt = ctx.selector.ordered_basis(bank, gradient, ctx.constants, ctx.initializer.ALGEBRA_TOLERANCE)
    assert receipt["rank"] == 3 and basis is not None
    for row in basis:
        assert row[torch.nonzero(row.abs() > ctx.constants.sign_atol)[0, 0]] > 0
    assert ctx.selector.ordered_basis(bank, torch.zeros_like(gradient), ctx.constants,
        ctx.initializer.ALGEBRA_TOLERANCE)[0] is None
    bank2 = bank.clone(); bank2[:, 3] = 0
    assert ctx.selector.ordered_basis(bank2, gradient, ctx.constants,
        ctx.initializer.ALGEBRA_TOLERANCE)[0] is None
    reference = ctx.selector.initial_metrics(logits_fn, center, basis, (0, 1), ctx.constants.radius, rows, labels)
    target = reference[1]["D"]
    assert target > ctx.constants.d_positive_min
    diagnostics = [ctx.selector.initial_metrics(logits_fn, center, basis, (0, 1), scale, rows, labels)[1]["D"]
                   for scale in (0.0, 0.125, 0.25, 0.5, 1.0)]
    assert all(a <= b + 1e-12 for a, b in zip(diagnostics, diagnostics[1:])), diagnostics
    matched = []
    for pair in ctx.selector.PAIRS:
        item, failure = ctx.selector.match_pair(logits_fn, center, basis, pair, target, rows, labels, ctx.constants)
        assert failure is None, failure
        assert abs(item[1]["D"] - target) <= ctx.constants.d_match_atol + ctx.constants.d_match_rtol * target
        assert ctx.selector.finite_guard(item, target, 10.0, rows, ctx.constants,
                                       ctx.initializer, center, gradient) is None
        assert ctx.selector.finite_guard(item, target, -1.0, rows, ctx.constants,
                                       ctx.initializer, center, gradient) == "original_TRAIN_safeguard_failure"
        matched.append(item[1]["radius"])
    cap_D = ctx.selector.initial_metrics(logits_fn, center, basis, (0, 1), ctx.constants.radius_cap, rows, labels)[1]["D"]
    assert ctx.selector.match_pair(logits_fn, center, basis, (0, 1), cap_D + 1, rows, labels, ctx.constants)[1]["status"] == "cap_or_nonfinite_failure"
    def nonfinite(_):
        return torch.full((6, 3), float("nan"), dtype=dtype)
    assert ctx.selector.match_pair(nonfinite, center, basis, (0, 1), target, rows, labels, ctx.constants)[1]["status"] == "cap_or_nonfinite_failure"
    null_basis = basis.clone(); null_basis[0] = 0; null_basis[0, 4] = 1
    null = ctx.selector.initial_metrics(logits_fn, center, null_basis, (0, 1), ctx.constants.radius, rows, labels)
    assert ctx.selector.finite_guard(null, null[1]["D"], 10, rows, ctx.constants,
                                   ctx.initializer, center, gradient) == "class_centered_response_null_or_duplicate"
    return dict(dtype=str(dtype), rank=receipt["rank"], positive_D=target, matched_radii=matched,
                monotonic_fixture_D=diagnostics, cap_and_nonfinite_failures_retained=True)


def control_identities_and_returned_state(ctx, dtype):
    f = fixture(ctx, dtype)
    f.prototype.train()
    before, frozen = model_snapshot(f.prototype), copy.deepcopy(f.frozen)
    theta, logits_fn, binding = ctx.selector.bind_head_only(f.k1, "polyformer_mono", f.args)
    assert binding["dimensions"] == 256
    common, _ = ctx.initializer.initialize_four_routes(logits_fn, theta, f.S, f.nodes,
        f.train.nodes, f.train.labels, tangent_mode="common_only")
    assert torch.equal(common, common[0].expand_as(common))
    center = common[0]
    bank, gradient, _ = ctx.selector.projected_bank(logits_fn, center, f.S, f.nodes,
        f.train.nodes, f.train.labels, ctx.initializer)
    graph_basis, rank = ctx.selector.ordered_basis(bank, gradient, ctx.constants, ctx.initializer.ALGEBRA_TOLERANCE)
    assert graph_basis is not None, rank
    permuted, permutation = ctx.initializer.permute_topology_nodes(f.S, 80017)
    near(torch.linalg.eigvalsh(f.S.to_dense()), torch.linalg.eigvalsh(permuted.to_dense()))
    assert permutation.unique().numel() == 10
    perm_bank, perm_gradient, _ = ctx.selector.projected_bank(logits_fn, center, permuted,
        f.nodes, f.train.nodes, f.train.labels, ctx.initializer)
    near(gradient, perm_gradient)
    bases = {"graph": graph_basis}
    bases["permuted"], _ = ctx.selector.ordered_basis(perm_bank, gradient, ctx.constants, ctx.initializer.ALGEBRA_TOLERANCE)
    generator = torch.Generator().manual_seed(70017)
    random_bank = torch.randn((4, 256), dtype=torch.float64, generator=generator)
    bases["random"], _ = ctx.selector.ordered_basis(random_bank, gradient, ctx.constants, ctx.initializer.ALGEBRA_TOLERANCE)
    assert bases["random"] is not None
    rng = ctx.integration.rng_snapshot()
    outputs, record = ctx.selector.select_initializations(f.prototype, f.frozen, logits_fn,
        theta, f.S, f.nodes, f.train.nodes, f.train.labels, "polyformer_mono", f.args,
        rng, 17, ctx.constants, ctx.initializer, ctx.integration)
    assert tuple(outputs) == ctx.selector.ARMS
    assert all(len(rows) == 3 for rows in record["candidates"].values())
    assert record["trial_maps_started"] <= 10 and record["trial_maps_completed"] <= 10
    equal(before, model_snapshot(f.prototype)); equal(frozen, f.frozen)
    equal(rng, ctx.integration.rng_snapshot())
    for arm, output in outputs.items():
        equal(f.frozen, ctx.integration.named_optimizer_snapshot(output["model"], output["optimizer"]))
        equal(rng, output["rng"])
        assert output["model"] is not f.prototype and output["model"].training
        selected = record["selection"][arm]["pair"]
        expected = common
        if selected is not None:
            span, index = selected
            pair = ctx.selector.PAIRS[index]
            scale = record["candidates"][span][index]["radius"]
            a, b = (bases[span][j].to(dtype) * scale for j in pair)
            expected = center[None] + torch.stack((a, -a, b, -b))
        equal(expected, output["model"].head.R.detach())
        for name, value in output["model"].state_dict().items():
            if name != "head.R":
                equal(before["state"][name], value)
        evaluation = copy.deepcopy(output["model"]).eval()(*f.args)
        torch.testing.assert_close(evaluation.mean(0), logits_fn(center),
            atol=ctx.constants.mean_logit_atol, rtol=ctx.constants.mean_logit_rtol)
    a, b = (outputs[name] for name in ctx.selector.ARMS[:2])
    assert a["rng"] is not b["rng"] and a["rng"]["torch_cpu"].data_ptr() != b["rng"]["torch_cpu"].data_ptr()
    assert a["model"].head.R.data_ptr() != b["model"].head.R.data_ptr()
    return dict(dtype=str(dtype), graph_rank=rank["rank"], candidate_slots=9,
        trial_maps_started=record["trial_maps_started"], selection=record["selection"],
        returned_pretrial_parameters_optimizer_and_rng=True, five_arm_return_state_assertions_completed=5)


def independent_train_null_regression(ctx, dtype):
    """Preserve the exact v1 independent-TRAIN topology as a valid null case."""
    f = fixture(ctx, dtype, connected_train=False)
    f.prototype.train()
    before, frozen = model_snapshot(f.prototype), copy.deepcopy(f.frozen)
    theta, logits_fn, _ = ctx.selector.bind_head_only(f.k1, "polyformer_mono", f.args)
    induced = f.S.to_dense()[f.train.nodes][:, f.train.nodes]
    assert torch.equal(induced, torch.zeros_like(induced))
    common, _ = ctx.initializer.initialize_four_routes(logits_fn, theta, f.S, f.nodes,
        f.train.nodes, f.train.labels, tangent_mode="common_only")
    bank, gradient, _ = ctx.selector.projected_bank(logits_fn, common[0], f.S, f.nodes,
        f.train.nodes, f.train.labels, ctx.initializer)
    basis, receipt = ctx.selector.ordered_basis(bank, gradient, ctx.constants, ctx.initializer.ALGEBRA_TOLERANCE)
    assert basis is None and receipt["rank"] <= 2, receipt
    rng = ctx.integration.rng_snapshot()
    outputs, record = ctx.selector.select_initializations(f.prototype, f.frozen, logits_fn,
        theta, f.S, f.nodes, f.train.nodes, f.train.labels, "polyformer_mono", f.args,
        rng, 17, ctx.constants, ctx.initializer, ctx.integration)
    assert tuple(outputs) == ctx.selector.ARMS
    assert record["trial_maps_started"] == record["trial_maps_completed"] == 1
    assert all(len(rows) == 3 and all(row["status"] == "rank_or_D0_failure" for row in rows)
               for rows in record["candidates"].values())
    for arm, output in outputs.items():
        assert record["selection"][arm]["pair"] is None
        equal(common, output["model"].head.R.detach())
        equal(f.frozen, ctx.integration.named_optimizer_snapshot(output["model"], output["optimizer"]))
        equal(rng, output["rng"])
        for name, value in output["model"].state_dict().items():
            if name != "head.R": equal(before["state"][name], value)
    equal(before, model_snapshot(f.prototype)); equal(frozen, f.frozen)
    equal(rng, ctx.integration.rng_snapshot())
    return dict(dtype=str(dtype), projected_rank=receipt["rank"], TRAIN_internal_edges=0,
        original_v1_topology_preserved=True, null_candidate_slots=9, trial_maps_started=1,
        five_arm_return_state_assertions_completed=5, all_five_arms_return_common=True)


def photo_head_identity(ctx):
    """Exercise the admitted 512-wide head closure without a native Photo body."""
    class PhotoToy(nn.Module):
        def __init__(self, members):
            super().__init__()
            self.members = members
            self.core = nn.Module(); self.core._global = True
            linear = nn.Linear(512, 3)
            self.global_head = ctx.boundary.BoundaryProjector(linear, members)
            self.register_buffer("fixed_hidden", torch.randn(4, 512) * 0.01)
        def forward(self, _):
            return torch.stack([self.global_head(self.fixed_hidden, i) for i in range(self.members)])
    model = PhotoToy(1).eval()
    before = model_snapshot(model)
    theta, logits_fn, binding = ctx.selector.bind_head_only(model, "polynormer_r", (None,))
    direction = torch.zeros_like(theta); direction[0] = ctx.constants.radius
    near((logits_fn(theta + direction) + logits_fn(theta - direction)) / 2, logits_fn(theta))
    assert binding["dimensions"] == 512 and binding["name"] == "global_head.R"
    equal(before, model_snapshot(model))
    return dict(final_head_dimensions=512, antithetic_affinity=True, native_photo_body_tested=False)


def run(ctx):
    assert hasattr(torch, "func"), "PyTorch 2.0+ with torch.func is required"
    original = ctx.integration.rng_snapshot()
    cases = [("rng_exact_roundtrip", lambda: rng_roundtrip(ctx)),
             ("named_adam_transport_and_checkpoint_restore", lambda: optimizer_and_checkpoint_restore(ctx)),
             ("partial_trial_failure_restores_custody", lambda: failed_trial_restores(ctx)),
             ("photo_512_head_identity", lambda: photo_head_identity(ctx))]
    for dtype in (torch.float32, torch.float64):
        cases.extend([(str(dtype) + "_one_step_adam", lambda dtype=dtype: adam_trial(ctx, dtype)),
                      (str(dtype) + "_rank_D_and_guards", lambda dtype=dtype: rank_and_D(ctx, dtype)),
                      (str(dtype) + "_control_and_returned_state", lambda dtype=dtype: control_identities_and_returned_state(ctx, dtype)),
                      (str(dtype) + "_independent_TRAIN_null_regression", lambda dtype=dtype: independent_train_null_regression(ctx, dtype))])
    results = []
    try:
        for name, operation in cases:
            ctx.integration.rng_restore(original)
            start = time.monotonic()
            try:
                detail = operation()
                results.append(dict(name=name, passed=True, seconds=time.monotonic() - start, detail=detail))
            except Exception as error:
                results.append(dict(name=name, passed=False, seconds=time.monotonic() - start,
                    error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
            finally:
                ctx.integration.rng_restore(original)
    finally:
        ctx.integration.rng_restore(original)
    return results, dict(torch_version=torch.__version__, numpy_version=np.__version__,
        numerical_model_device="cpu", CUDA_available=torch.cuda.is_available(),
        real_native_bodies_tested=False, native_warm_or_continuation_invoked=False,
        fixture_comparison_tolerances={"float32": 5e-6, "float64": 1e-10},
        frozen_recipe_tolerances_unchanged=True)
