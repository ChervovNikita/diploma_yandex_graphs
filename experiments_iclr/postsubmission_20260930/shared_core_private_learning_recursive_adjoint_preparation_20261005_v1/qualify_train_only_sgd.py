"""UNEXECUTED source: one discarded exact recursive-adjoint NCN SGD qualification.

All numerical/model imports are delayed until an explicitly reviewed job enters
main. No fit, epoch, VALID/TEST loader, checkpoint, approximate Hessian update, or final Adam recipe is provided.
A local constant-adjacency recursive adjoint preserves the saved native forward. Finite differences verify a
derivative only; they never supply an update. This preparation itself was not run.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import random
import resource
import socket
import sys
import time

from recursive_adjoint import make_constant_adjacency_spmm

NODES, FEATURES, WIDTH, MEMBERS, BATCH = 3327, 3703, 256, 4, 1024
INNER_ROUTE_STEP, SHARED_STEP = 0.001, 0.001
FD_STEPS = (2.0 ** -8, 2.0 ** -10)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    if job.get("source_review_approved") is not True or job.get("operator_qualification_only") is not True:
        raise ValueError("Root review of this discarded operator qualification is required")
    if job.get("fits_authorized") is not False or job.get("expected_hostname") != socket.gethostname():
        raise ValueError("Wrong scope/host; this source authorizes no fits")
    if job.get("external_360_second_hard_bound_confirmed") is not True:
        raise ValueError("A reviewed external hard bound is required")
    if str(args.output.resolve()) != job.get("output_directory") or args.output.exists():
        raise ValueError("Require the reviewed fresh diagnostic output directory")
    seed, factor_seed = job["seed"], job["factor_seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Require a prospectively fixed NumPy-compatible integer seed")
    if type(factor_seed) is not int or not 0 <= factor_seed < 2**63:
        raise ValueError("Require a prospectively fixed native factor seed")
    if sha(__file__) != job.get("qualification_source_sha256"):
        raise ValueError("Qualification source bytes differ")
    if sha(Path(__file__).with_name("recursive_adjoint.py")) != job.get("recursive_adjoint_sha256"):
        raise ValueError("Reviewed recursive-adjoint bytes differ")
    if Path(importlib.import_module("recursive_adjoint").__file__).resolve() != Path(__file__).with_name("recursive_adjoint.py").resolve():
        raise ValueError("Recursive-adjoint factory module shadowed")
    if job.get("constant_adjacency_recursive_adjoint_authorized") is not True:
        raise ValueError("This exact repair needs separate root source review")
    source = Path(job["native_source_directory"]).resolve(strict=True)
    manifest = source / "SOURCE_MANIFEST.json"
    if sha(manifest) != job["native_source_manifest_sha256"]:
        raise ValueError("Reviewed native manifest differs")
    for row in json.loads(manifest.read_text())["files"]:
        if sha(source / row["path"]) != row["sha256"]:
            raise ValueError("Native source changed: " + row["path"])
    partition_path = Path(__file__).with_name("PARAMETER_PARTITION.json")
    if sha(partition_path) != job["partition_sha256"]:
        raise ValueError("Reviewed parameter partition differs")
    partition_spec = json.loads(partition_path.read_text())
    inputs = job["train_only_inputs"]
    if set(inputs) != {"train_pos.txt", "gnn_feature"} or job.get("feature_authority_verified") is not True:
        raise ValueError("Only authenticated TRAIN/features may enter this qualification")
    paths = {}
    for role, record in inputs.items():
        path = Path(record["path"]).resolve(strict=True)
        if path.name != role or sha(path) != record["sha256"]:
            raise ValueError("Input role/bytes differ: " + role)
        paths[role] = path
    args.output.mkdir(exist_ok=False)
    receipt = {"scope": "discarded_recursive_adjoint_SGD_operator_only", "fits": 0,
               "persistent_training_updates": 0, "VALID_TEST_access": False,
               "accuracy_or_ranking_metrics": False, "stages": [],
               "source_sha256": sha(__file__), "job_sha256": sha(args.job),
               "native_manifest_sha256": sha(manifest), "partition_sha256": sha(partition_path)}

    def stage(name, ok, **details):
        entry = dict(name=name, passed=bool(ok), elapsed_seconds=time.monotonic()-start, **details)
        receipt["stages"].append(entry)
        print(json.dumps(entry), flush=True)
        if time.monotonic()-start > 300:
            raise TimeoutError("300-second discarded CPU qualification budget exceeded")

    native = saved_native_spmm = wrapped_spmm = failure = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import numpy as np
        import torch
        from torch import nn
        from torch.func import functional_call
        import torch_sparse
        import torch_geometric
        import torch_scatter
        from torch_sparse import SparseTensor
        from torch_geometric.utils import negative_sampling, to_undirected
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        versions = {"torch": torch.__version__, "numpy": np.__version__,
                    "torch_sparse": torch_sparse.__version__, "torch_geometric": torch_geometric.__version__,
                    "torch_scatter": torch_scatter.__version__}
        if versions != job["expected_runtime_versions"]:
            raise ValueError("Installed CPU runtime differs; no substitute backend")
        receipt["runtime"] = versions
        supplied = torch.load(paths["gnn_feature"], map_location="cpu", weights_only=True)
        x = supplied.get("entity_embedding") if isinstance(supplied, dict) else None
        if not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or x.layout != torch.strided:
            raise ValueError("Require exact native float32 dense features")
        if tuple(x.shape) != (NODES, FEATURES) or not bool(torch.isfinite(x).all()):
            raise ValueError("Full authenticated native feature shape differs")
        pairs, raw_rows, self_rows = [], 0, 0
        with paths["train_pos.txt"].open() as stream:
            for line in stream:
                fields = line.strip().split("\t")
                if len(fields) != 2:
                    raise ValueError("Native TRAIN requires two tab-separated endpoints")
                u, v = map(int, fields); raw_rows += 1
                if not (0 <= u < NODES and 0 <= v < NODES):
                    raise ValueError("TRAIN endpoint outside native population")
                if u == v:
                    self_rows += 1
                else:
                    pairs.append((u, v))
        if len(pairs) != 3870 or len({tuple(sorted(p)) for p in pairs}) != len(pairs):
            raise ValueError("Native nonself TRAIN count/uniqueness differs")
        expected_counts = inputs["train_pos.txt"]["counts"]
        if (raw_rows, self_rows, len(pairs)) != (expected_counts["raw_rows"], expected_counts["self_loops"], expected_counts["native_nonself_rows"]):
            raise ValueError("TRAIN acquisition count binding differs")
        train = torch.tensor(pairs, dtype=torch.long)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        sys.path.insert(0, str(source / "vendor"))
        native = importlib.import_module("baseline_models.NCN.model")
        util = importlib.import_module("baseline_models.NCN.util")
        if Path(native.__file__).resolve() != source / "vendor/baseline_models/NCN/model.py":
            raise ValueError("NCN donor module shadowed")
        if Path(util.__file__).resolve() != source / "vendor/baseline_models/NCN/util.py":
            raise ValueError("NCN utility module shadowed")
        spec = importlib.util.spec_from_file_location("discarded_qualification_heads", source / "heads.py")
        heads = importlib.util.module_from_spec(spec); spec.loader.exec_module(heads)
        saved_native_spmm = native.spmm_add
        if saved_native_spmm is not importlib.import_module("torch_sparse.matmul").spmm_add:
            raise ValueError("Donor sparse alias was already replaced")
        wrapped_spmm = make_constant_adjacency_spmm(torch, saved_native_spmm)
        encoder = native.GCN(FEATURES, WIDTH, WIDTH, 1, .3, True, False, -1,
                             "puregcn", True, 0., xdropout=.4, taildropout=0., noinputlin=False)
        predictor = heads.make_predictor(native, arm="private_frame_f4", members=MEMBERS,
                                         width=WIDTH, factor_seed=factor_seed)

        class Bundle(nn.Module):
            def __init__(self):
                super().__init__(); self.encoder, self.predictor = encoder, predictor
            def forward(self, features, support, positive, negative):
                h = self.encoder(features, support)
                return self.predictor(h, support, positive), self.predictor(h, support, negative)

        model = Bundle()
        params = {n: p.detach().clone().requires_grad_(True) for n, p in model.named_parameters()}
        buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        expected = {row["name"]: row for row in partition_spec["parameters"]}
        if set(params) != set(expected) or any(list(p.shape) != expected[n]["shape"] for n, p in params.items()):
            raise ValueError("Exact shared/private parameter name/shape map differs")
        expected_buffers = {row["name"]: row for row in partition_spec["buffers"]}
        if set(buffers) != set(expected_buffers) or any(list(b.shape) != expected_buffers[n]["shape"] for n, b in buffers.items()):
            raise ValueError("Exact native buffer name/shape map differs")
        shared = tuple(n for n in params if expected[n]["role"] == "shared")
        private = tuple(n for n in params if expected[n]["role"] == "private")
        if sum(params[n].numel() for n in shared) != 1409026 or sum(params[n].numel() for n in private) != 26632:
            raise ValueError("Native partition count differs")
        edge = to_undirected(train.t())
        negative = negative_sampling(edge, NODES)
        if negative.ndim != 2 or negative.shape[0] != 2 or negative.shape[1] < len(train):
            raise ValueError("Native negative sampler supplied too few indexed queries")
        iterator = iter(util.PermIterator(torch.device("cpu"), len(train), BATCH))
        inner_ids, outer_ids = next(iterator), next(iterator)
        if len(inner_ids) != BATCH or len(outer_ids) != BATCH or len(torch.unique(torch.cat((inner_ids, outer_ids)))) != 2*BATCH:
            raise ValueError("Require two full disjoint native TRAIN query batches")
        keep = torch.ones(len(train), dtype=torch.bool)
        keep[inner_ids] = False; keep[outer_ids] = False
        support = SparseTensor.from_edge_index(train[keep].t(), sparse_sizes=(NODES, NODES)).to_symmetric().coalesce()
        if support.sparse_sizes() != (NODES, NODES) or support.nnz() != 2*(len(train)-2*BATCH):
            raise ValueError("Union-masked full-node TRAIN support differs")
        batches = {"inner": (train[inner_ids].t(), negative[:, inner_ids]),
                   "outer": (train[outer_ids].t(), negative[:, outer_ids])}
        rng = torch.get_rng_state().clone()
        receipt["geometry"] = {"features": [NODES, FEATURES], "hidden": WIDTH, "members": MEMBERS,
                               "inner_positive_negative_rows": [BATCH, BATCH], "outer_positive_negative_rows": [BATCH, BATCH],
                               "TRAIN_nonself_rows": len(train), "support_nnz": support.nnz(),
                               "query_edge_disjoint": True, "endpoint_separation_claimed": False}
        receipt["steps"] = {"inner_per_route_SGD": INNER_ROUTE_STEP, "shared_SGD": SHARED_STEP,
                            "final_Adam_recipe": False, "all_operator_states_discarded": True}

        def tensor_sha(t):
            return hashlib.sha256(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        def sparse_content(matrix):
            row, col, value = matrix.coo()
            return {"shape": list(matrix.sparse_sizes()), "row": tensor_sha(row), "col": tensor_sha(col),
                    "value": None if value is None else tensor_sha(value)}
        pristine = {n: tensor_sha(t) for n, t in model.state_dict().items()}
        pristine_modes = {n: m.training for n, m in model.named_modules()}
        pristine_support, pristine_features = sparse_content(support), tensor_sha(x)

        def call(mapping, which, training=False):
            modes = {m: m.training for m in model.modules()}
            model.train(training)
            try:
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(rng)
                    pos, neg = functional_call(model, (mapping, {n: b.clone() for n, b in buffers.items()}),
                                               (x, support, *batches[which]), strict=True)
                if pos.shape != (BATCH, MEMBERS) or neg.shape != pos.shape:
                    raise ValueError("Full native query/member shape differs")
                if not bool(torch.isfinite(pos).all() and torch.isfinite(neg).all()):
                    raise FloatingPointError("Nonfinite native logits")
                return pos, neg
            finally:
                for module, mode in modes.items(): module.training = mode

        def own_loss(pos, neg):
            return -torch.nn.functional.logsigmoid(pos).mean()-torch.nn.functional.logsigmoid(-neg).mean()
        def outer_loss(pos, neg):
            return .5*own_loss(pos.mean(1), neg.mean(1))+.5*own_loss(pos, neg)
        def fresh(mapping):
            return {n: p.detach().clone().requires_grad_(True) for n, p in mapping.items()}
        def finite(values):
            return all(bool(torch.isfinite(v).all()) for v in values)
        def dot(values, direction):
            return sum((v.double()*direction[n].double()).sum() for n, v in zip(shared, values))

        # First-order and forward equivalence precede every higher-order gate.
        # Assignment affects only the donor module alias in this qualification.
        # The torch_sparse package and immutable donor source are untouched.
        parity_ok = True
        for which, training, loss_fn in (("inner", True, own_loss), ("outer", False, outer_loss)):
            native.spmm_add = saved_native_spmm
            original_params = fresh(params)
            original_logits = call(original_params, which, training)
            original_grad = torch.autograd.grad(loss_fn(*original_logits), list(original_params.values()), allow_unused=False)
            native.spmm_add = wrapped_spmm
            wrapped_params = fresh(params)
            wrapped_logits = call(wrapped_params, which, training)
            wrapped_grad = torch.autograd.grad(loss_fn(*wrapped_logits), list(wrapped_params.values()), allow_unused=False)
            forward_equal = all(torch.equal(a, b) for a, b in zip(original_logits, wrapped_logits))
            first_equal = finite((*original_grad, *wrapped_grad)) and all(torch.equal(a, b) for a, b in zip(original_grad, wrapped_grad))
            parity_ok &= forward_equal and first_equal
            stage("full_native_forward_first_gradient_parity:"+which, forward_equal and first_equal,
                  full_route_logits_bitwise_equal=forward_equal, all_parameter_first_gradients_bitwise_equal=first_equal,
                  parameters_checked=len(params), max_abs_first_gradient_error=max(float((a-b).abs().max()) for a, b in zip(original_grad, wrapped_grad)))
            del original_params, original_logits, original_grad, wrapped_params, wrapped_logits, wrapped_grad
        native.spmm_add = wrapped_spmm

        # Exact native constant-adjacency gate: for Q=||KZ||^2/2,
        # H_Q d = K^T K d. No dense or differentiable replacement supplies K.
        model.eval()
        with torch.no_grad(): encoded = encoder(x, support)
        for name, module in model.named_modules(): module.training = pristine_modes[name]
        sparse_ok, inner_cn_active = True, False
        matrices = [("encoder_spmm", support)]
        for which in ("inner", "outer"):
            for label, queries in zip(("positive", "negative"), batches[which]):
                cn = native.adjoverlap(support, support, queries, False, cnsampledeg=-1)
                matrices.append((which+"_cn_"+label, cn))
                if which == "inner" and cn.nnz(): inner_cn_active = True
        for name, matrix in matrices:
            if matrix.nnz() == 0:
                stage(name, False, status="path_not_exercised_empty_CN"); continue
            try:
                z = encoded.detach().clone().requires_grad_(True)
                direction = torch.ones_like(z)/z.numel()**.5
                y = native.spmm_add(matrix, z)
                first = torch.autograd.grad(.5*y.square().sum(), z, create_graph=True)[0]
                expected_hvp = saved_native_spmm(matrix.t(), saved_native_spmm(matrix, direction))
                if not first.requires_grad:
                    sparse_ok = False
                    stage(name, False, status="dense_input_adjoint_has_no_second_order_graph",
                          expected_hvp_nonzero=bool(torch.count_nonzero(expected_hvp)))
                    continue
                got = torch.autograd.grad((first*direction).sum(), z)[0]
                ok = finite((first, got, expected_hvp)) and bool(torch.count_nonzero(expected_hvp)) and torch.allclose(got, expected_hvp, rtol=3e-4, atol=3e-5)
                sparse_ok &= bool(ok)
                stage(name, ok, status="quadratic_native_adjoint_identity", max_abs_error=float((got-expected_hvp).abs().max()))
            except Exception as error:
                sparse_ok = False; stage(name, False, status="unsupported_double_backward", error=repr(error))
        sparse_ok &= inner_cn_active
        stage("inner_common_neighbor_path_active", inner_cn_active)

        # Full native-shaped operator diagnosis is still attempted after a gate
        # failure. It cannot authorize/commit a shared update after such failure.
        inner_pos, inner_neg = call(params, "inner", True)
        g_inner = torch.autograd.grad(own_loss(inner_pos, inner_neg), [params[n] for n in private],
                                      create_graph=True, retain_graph=True, allow_unused=False)
        alpha = MEMBERS*INNER_ROUTE_STEP  # native mean-member loss -> per-route private scale
        fast = dict(params)
        fast.update({n: params[n]-alpha*g for n, g in zip(private, g_inner)})
        out_pos, out_neg = call(fast, "outer", False)
        objective = outer_loss(out_pos, out_neg)
        q = torch.autograd.grad(objective, [fast[n] for n in private], retain_graph=True, allow_unused=False)
        # q is held constant only for algebraic/FD verification of the pullback.
        # The actual full gradient below differentiates the live fast map.
        q_fixed = tuple(value.detach() for value in q)
        h = meta = None
        try:
            h = torch.autograd.grad(g_inner, [params[n] for n in shared], grad_outputs=q_fixed,
                                    retain_graph=True, allow_unused=False)
            meta = torch.autograd.grad(objective, [params[n] for n in shared], allow_unused=False)
            independent_fast = dict(params)
            independent_fast.update({n: fast[n].detach().clone().requires_grad_(True) for n in private})
            direct = torch.autograd.grad(outer_loss(*call(independent_fast, "outer", False)),
                                        [params[n] for n in shared], allow_unused=False)
            algebraic_ok = finite((*h, *meta, *direct)) and all(torch.allclose(a, b-alpha*c, rtol=5e-4, atol=3e-5)
                                                          for a, b, c in zip(meta, direct, h))
            mixed_nonzero = any(bool(torch.count_nonzero(t)) for t in h)
            algebraic_ok &= mixed_nonzero
            stage("full_chain_rule_decomposition", algebraic_ok,
                  mixed_pullback_nonzero=mixed_nonzero)
        except Exception as error:
            algebraic_ok = False; stage("full_chain_rule_decomposition", False, error=repr(error))

        fd_ok = h is not None
        for name in ("encoder.xemb.1.bias", "predictor.xlin.ops.0.bias"):
            direction = {n: torch.zeros_like(params[n]) for n in shared}
            target = direction[name]
            target.copy_((2*(torch.arange(target.numel()).reshape(target.shape)%2)-1).to(target)/target.numel()**.5)
            estimates = []
            for epsilon in FD_STEPS:
                scalars = []
                for sign in (1., -1.):
                    trial = fresh(params)
                    for n in shared: trial[n] = (trial[n]+sign*epsilon*direction[n]).detach().requires_grad_(True)
                    gi = torch.autograd.grad(own_loss(*call(trial, "inner", True)),
                                             [trial[n] for n in private], allow_unused=False)
                    scalars.append(sum((g.double()*v.double()).sum() for g, v in zip(gi, q_fixed)))
                estimates.append(float((scalars[0]-scalars[1])/(2*epsilon)))
            if h is None:
                stage("full_mixed_FD:"+name, False, status="autograd_pullback_unavailable", finite_differences=estimates)
                continue
            analytic = float(dot(h, direction))
            scale = max(abs(analytic), abs(estimates[-1]))
            agrees = abs(analytic-estimates[-1]) <= 1e-5+.02*scale
            stable = abs(estimates[0]-estimates[1]) <= 2e-5+.05*scale
            fd_ok &= agrees and stable
            stage("full_mixed_FD:"+name, agrees and stable, native_autograd=analytic,
                  finite_differences=estimates, convergence_checked=True,
                  FD_used_for_update=False, failure_may_include_nonsmooth_crossing=True)

        qualified = parity_ok and sparse_ok and algebraic_ok and fd_ok
        if qualified:
            shared_plus = fresh(params)
            for n, gradient in zip(shared, meta): shared_plus[n] = (params[n]-SHARED_STEP*gradient).detach().requires_grad_(True)
            # Recompute at new theta from ORIGINAL private parameters. Exactly
            # one candidate private SGD update is committed to a discarded copy.
            recomputed = torch.autograd.grad(own_loss(*call(shared_plus, "inner", True)),
                                             [shared_plus[n] for n in private], allow_unused=False)
            committed = dict(shared_plus)
            committed.update({n: shared_plus[n]-alpha*g for n, g in zip(private, recomputed)})
            again = fresh(shared_plus)
            repeated = torch.autograd.grad(own_loss(*call(again, "inner", True)),
                                          [again[n] for n in private], allow_unused=False)
            deterministic = all(torch.equal(a, b) for a, b in zip(recomputed, repeated))
            functional_logits = call(committed, "outer", False)
            committed_copy = copy.deepcopy(model)
            committed_copy.load_state_dict({**{n: p.detach().clone() for n, p in committed.items()},
                                             **{n: b.detach().clone() for n, b in buffers.items()}}, strict=True)
            committed_copy.eval()
            # Serving parity uses the restored original native forward alias.
            native.spmm_add = saved_native_spmm
            with torch.no_grad(): actual_logits = committed_copy(x, support, *batches["outer"])
            native.spmm_add = wrapped_spmm
            serving_ok = all(torch.equal(a, b) for a, b in zip(functional_logits, actual_logits))
            stage("private_recomputation_exact_rng_replay", deterministic)
            stage("committed_copy_raw_logit_serving", serving_ok,
                  actual_mean_logit_parity=all(torch.equal(a.mean(1), b.mean(1)) for a, b in zip(functional_logits, actual_logits)),
                  inference_adaptation=False)
            qualified &= deterministic and serving_ok
        else:
            stage("committed_operator", False, status="not_admitted_after_derivative_failure_or_inconclusive_FD")
        unchanged = pristine == {n: tensor_sha(t) for n, t in model.state_dict().items()}
        stage("initial_fixture_weights_buffers_unchanged", unchanged)
        inputs_unchanged = pristine_support == sparse_content(support) and pristine_features == tensor_sha(x)
        stage("TRAIN_support_features_unchanged", inputs_unchanged)
        qualified &= unchanged and inputs_unchanged and all(p.grad is None for p in model.parameters())
        receipt.update(status="PASS_DISCARDED_RECURSIVE_ADJOINT_SGD_OPERATOR_ONLY" if qualified else "FAIL_OR_UNQUALIFIED_RECURSIVE_ADJOINT_DERIVATIVE",
                       inclusive_seconds=time.monotonic()-start, final_predictive_recipe_admitted=False)
    except Exception as error:
        failure = error
        receipt.update(status="FAIL", error=type(error).__name__+": "+str(error),
                       inclusive_seconds=time.monotonic()-start, final_predictive_recipe_admitted=False)
        receipt["peak_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    finally:
        if native is not None and saved_native_spmm is not None:
            native.spmm_add = saved_native_spmm
            receipt["local_native_alias_restored"] = native.spmm_add is saved_native_spmm
        receipt["constant_adjacency_recursive_adjoint_declared"] = True
        receipt["global_torch_sparse_patch"] = False
        receipt["wrapper_sha256"] = sha(Path(__file__).with_name("recursive_adjoint.py"))
    if failure is not None:
        (args.output/"FAILURE.json").write_text(json.dumps(receipt, indent=2)+"\n")
        raise failure
    receipt["peak_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    (args.output/"RESULT.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
