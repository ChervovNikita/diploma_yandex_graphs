"""UNEXECUTED full-network float64 analytic dense-constant-adjacency oracle.

One sequential recursive-native/dense comparison; no finite differences, serving
replacement, shared update, fit, checkpoint, metric, or FP32 admission. Numerical
imports and model construction occur only inside a separately reviewed main.
"""
from pathlib import Path
import argparse
import gc
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
RTOL, ATOL = 1e-7, 1e-10  # fixed before every numerical output


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
    if job.get("float64_dense_oracle_authorized") is not True or job.get("serving_replacement") is not False:
        raise ValueError("Separately reviewed float64 diagnostic scope is required")
    if job.get("FP32_FD_pass_admitted") is not False:
        raise ValueError("This oracle cannot retroactively pass FP32 FD")
    comparison = job["diagnostic_comparison"]
    if comparison.get("dtype") != "float64" or comparison.get("rtol") != RTOL or comparison.get("atol") != ATOL:
        raise ValueError("Prospectively fixed oracle precision/tolerances differ")
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
    receipt = {"scope": "float64_full_network_dense_oracle_only", "fits": 0,
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
        support = SparseTensor.from_edge_index(train[keep].t(), edge_attr=torch.ones(int(keep.sum()), dtype=torch.float64), sparse_sizes=(NODES, NODES)).to_symmetric().coalesce()
        if support.sparse_sizes() != (NODES, NODES) or support.nnz() != 2*(len(train)-2*BATCH):
            raise ValueError("Union-masked full-node TRAIN support differs")
        batches = {"inner": (train[inner_ids].t(), negative[:, inner_ids]),
                   "outer": (train[outer_ids].t(), negative[:, outer_ids])}
        rng = torch.get_rng_state().clone()
        receipt["geometry"] = {"features": [NODES, FEATURES], "hidden": WIDTH, "members": MEMBERS,
                               "inner_positive_negative_rows": [BATCH, BATCH], "outer_positive_negative_rows": [BATCH, BATCH],
                               "TRAIN_nonself_rows": len(train), "support_nnz": support.nnz(),
                               "query_edge_disjoint": True, "endpoint_separation_claimed": False}
        receipt["steps"] = {"virtual_inner_per_route_SGD": INNER_ROUTE_STEP, "shared_update_applied": False,
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

        # Initialize the exact native FP32 fixture, then cast its fixed values.
        # No new float64 random initialization or row/column remapping occurs.
        x_fp32 = x
        x = x.to(torch.float64)
        params = {n: p.detach().to(torch.float64) for n, p in params.items()}
        buffers = {n: b.to(torch.float64) for n, b in buffers.items()}
        receipt["precision"] = {"authenticated_input_dtype": "float32", "diagnostic_dense_dtype": "float64",
                                "parameter_point": "same_native_FP32_initialization_cast_to_float64",
                                "constant_support_values_normalization": "float64",
                                "FP32_FD_pass_admitted": False, "serving_replacement": False}

        def call(mapping, which, training=False):
            modes = {m: m.training for m in model.modules()}
            model.train(training)
            try:
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(rng)
                    pos, neg = functional_call(model, (mapping, {n: b.clone() for n, b in buffers.items()}),
                                               (x, support, *batches[which]), strict=True)
                if pos.dtype != torch.float64 or neg.dtype != torch.float64:
                    raise ValueError("Require float64 throughout both diagnostic networks")
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

        # Dense matrices are independently assembled from the same fixed COO
        # constants. CN utility defaults may be FP32; both aliases explicitly
        # cast their exact values to FP64 at this operator boundary.
        sparse_constants, dense_constants = {}, {}
        def constants(adjacency):
            row, col, value = adjacency.coo()
            if value is not None and value.requires_grad:
                raise ValueError("Oracle requires constant sparse values")
            values = torch.ones(len(row), dtype=torch.float64) if value is None else value.detach().to(torch.float64)
            coordinates = row*adjacency.sparse_sizes()[1]+col
            if len(torch.unique(coordinates)) != len(row):
                raise ValueError("Require the native unique/coalesced constant supports")
            key = (adjacency.sparse_sizes(), tensor_sha(row), tensor_sha(col), tensor_sha(values))
            return key, row, col, values

        def recursive_native64(adjacency, dense):
            if dense.dtype != torch.float64:
                raise ValueError("Native diagnostic input must be float64")
            key, row, col, values = constants(adjacency)
            if key not in sparse_constants:
                sparse_constants[key] = SparseTensor(row=row, col=col, value=values,
                                                     sparse_sizes=adjacency.sparse_sizes()).coalesce()
            return wrapped_spmm(sparse_constants[key], dense)

        def independent_dense64(adjacency, dense):
            if dense.dtype != torch.float64:
                raise ValueError("Dense oracle input must be float64")
            key, row, col, values = constants(adjacency)
            if key not in dense_constants:
                matrix = torch.zeros(adjacency.sparse_sizes(), dtype=torch.float64)
                matrix[row, col] = values
                dense_constants[key] = matrix
            return dense_constants[key] @ dense

        def detached(mapping):
            return {name: value.detach().clone() for name, value in mapping.items()}

        def compare(name, left, right):
            if set(left) != set(right) or any(left[n].shape != right[n].shape for n in left):
                raise ValueError("Complete oracle coordinate map differs: "+name)
            ok, max_abs, max_scaled, count = True, 0., 0., 0
            for n in left:
                a, b = left[n], right[n]
                if a.dtype != torch.float64 or b.dtype != torch.float64 or a.requires_grad or b.requires_grad:
                    raise ValueError("Only detached float64 diagnostic coordinates may be compared")
                if not finite((a, b)):
                    raise FloatingPointError("Nonfinite oracle coordinates: "+name)
                error = (a-b).abs()
                limit = ATOL+RTOL*torch.maximum(a.abs(), b.abs())
                ok &= bool((error <= limit).all())
                max_abs = max(max_abs, float(error.max()))
                max_scaled = max(max_scaled, float((error/limit).max()))
                count += a.numel()
            stage(name, ok, tensors=len(left), coordinates=count,
                  rtol=RTOL, atol=ATOL, max_abs_error=max_abs, max_error_over_fixed_limit=max_scaled)
            return ok

        def branch(label, operator, reference_cotangent=None, reference_outer_point=None):
            native.spmm_add = operator
            work = fresh(params)
            inner_logits = call(work, "inner", True)
            inner_loss = own_loss(*inner_logits)
            # Complete first-order parity is recorded without retaining an
            # unnecessary graph for all shared first-gradient coordinates.
            first_inner_values = torch.autograd.grad(inner_loss, list(work.values()), retain_graph=True, allow_unused=False)
            first_inner = detached(dict(zip(work, first_inner_values)))
            inner_private = torch.autograd.grad(inner_loss, [work[n] for n in private],
                                                create_graph=True, retain_graph=True, allow_unused=False)
            alpha = MEMBERS*INNER_ROUTE_STEP
            fast = dict(work)
            fast.update({n: work[n]-alpha*g for n, g in zip(private, inner_private)})
            outer_logits = call(fast, "outer", False)
            objective = outer_loss(*outer_logits)
            q = torch.autograd.grad(objective, [fast[n] for n in private], retain_graph=True, allow_unused=False)
            q_fixed = tuple(value.detach() for value in q)
            # Both mixed pullbacks use exactly the native branch's fixed q.
            # The complete live meta gradients use each independent branch's
            # own evaluation of the same private SGD map and outer objective.
            common_q = q_fixed if reference_cotangent is None else tuple(reference_cotangent[n] for n in private)
            mixed = torch.autograd.grad(inner_private, [work[n] for n in shared], grad_outputs=common_q,
                                        retain_graph=True, allow_unused=False)
            meta = torch.autograd.grad(objective, [work[n] for n in shared], allow_unused=False)
            candidate = detached(fast)
            # Outer forward/first-gradient parity is evaluated at an identical
            # point, the native branch's detached virtual private-step values.
            parity_point = candidate if reference_outer_point is None else reference_outer_point
            independent = fresh(parity_point)
            parity_logits = call(independent, "outer", False)
            first_outer_values = torch.autograd.grad(outer_loss(*parity_logits), list(independent.values()), allow_unused=False)
            output = {"inner_logits": detached(dict(zip(("positive", "negative"), inner_logits))),
                      "outer_live_logits": detached(dict(zip(("positive", "negative"), outer_logits))),
                      "outer_parity_logits": detached(dict(zip(("positive", "negative"), parity_logits))),
                      "first_inner": first_inner,
                      "first_outer": detached(dict(zip(independent, first_outer_values))),
                      "outer_point": candidate,
                      "fast_private": {n: candidate[n] for n in private},
                      "cotangent": detached(dict(zip(private, q))),
                      "mixed": detached(dict(zip(shared, mixed))),
                      "meta": detached(dict(zip(shared, meta)))}
            if not all(finite(mapping.values()) for mapping in output.values()):
                raise FloatingPointError("Nonfinite full-network branch: "+label)
            stage("completed_sequential_branch:"+label, True, retained_outputs_detached=True)
            return output

        native_result = branch("recursive_native64", recursive_native64)
        gc.collect()  # no native higher-order graph is retained for dense run
        dense_result = branch("independent_dense64", independent_dense64,
                              native_result["cotangent"], native_result["outer_point"])
        gc.collect()
        ok = True
        for field in ("inner_logits", "first_inner", "outer_parity_logits", "first_outer", "fast_private",
                      "outer_live_logits", "cotangent", "mixed", "meta"):
            ok &= compare("complete_float64_comparison:"+field, native_result[field], dense_result[field])
        if len(native_result["mixed"]) != len(shared) or sum(v.numel() for v in native_result["mixed"].values()) != 1409026:
            raise ValueError("Mixed oracle did not cover every shared coordinate")
        mixed_nonzero = any(bool(torch.count_nonzero(v)) for v in native_result["mixed"].values())
        stage("complete_shared_mixed_nonzero", mixed_nonzero, coordinates=1409026)
        unchanged = pristine == {n: tensor_sha(t) for n, t in model.state_dict().items()}
        input_unchanged = pristine_support == sparse_content(support) and pristine_features == tensor_sha(x_fp32)
        stage("original_FP32_fixture_and_authenticated_inputs_unchanged", unchanged and input_unchanged)
        ok &= mixed_nonzero and unchanged and input_unchanged and all(p.grad is None for p in model.parameters())
        receipt.update(status="PASS_FLOAT64_DENSE_ORACLE_EQUIVALENCE_ONLY" if ok else "FAIL_OR_UNQUALIFIED_FLOAT64_DENSE_ORACLE",
                       inclusive_seconds=time.monotonic()-start, final_predictive_recipe_admitted=False,
                       FP32_FD_pass_admitted=False, serving_replacement=False, shared_update_applied=False,
                       matrices_cached=len(dense_constants), dense_constant_bytes=sum(m.numel()*m.element_size() for m in dense_constants.values()),
                       all_shared_mixed_and_meta_coordinates_compared=1409026)
    except Exception as error:
        failure = error
        receipt.update(status="FAIL", error=type(error).__name__+": "+str(error),
                       inclusive_seconds=time.monotonic()-start, final_predictive_recipe_admitted=False,
                       FP32_FD_pass_admitted=False, serving_replacement=False)
    finally:
        if native is not None and saved_native_spmm is not None:
            native.spmm_add = saved_native_spmm
            receipt["local_native_alias_restored"] = native.spmm_add is saved_native_spmm
        receipt["global_torch_sparse_patch"] = False
        receipt["wrapper_sha256"] = sha(Path(__file__).with_name("recursive_adjoint.py"))
        receipt["peak_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    output_name = "FAILURE.json" if failure is not None else "RESULT.json"
    (args.output/output_name).write_text(json.dumps(receipt, indent=2)+"\n")
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
