"""Inactive fixed-context-key native SAGE source; root review/qualification required."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ARMS = ("ordinaryM1_key_mixture", "factorizedM1_key_mixture",
        "genuinefactorizedI4_key_mixture", "genuinefactorizedI4_key_assigned",
        "shared4_key_private", "shared4_key_mixture", "shared4_key_uniform",
        "shared4_projected_peer_private", "shared4_key_moment_private", "shared4_key_shuffled_private")
CONTEXT_MODES = {ARMS[0]: "key_mixture", ARMS[1]: "key_mixture", ARMS[2]: "key_mixture",
                 ARMS[3]: "key_assigned", ARMS[4]: "key_private", ARMS[5]: "key_mixture",
                 ARMS[6]: "key_uniform", ARMS[7]: "projected_peer_private",
                 ARMS[8]: "key_moment_private", ARMS[9]: "key_shuffled_private"}


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Family:
    def __init__(self, cfg, output, config_sha256):
        import numpy as np
        import torch
        import torch.nn.functional as F
        sys.path.insert(0, cfg["native_repo"])
        from models import Model
        from run_base import train_step
        from run_common import set_seed
        self.np, self.torch, self.F = np, torch, F
        self.Model, self.train_step, self.set_seed = Model, train_step, set_seed
        self.routes = load_source("sage_family_routes", cfg["common_routes"])
        self.factors = load_source("sage_family_factors", cfg["factors"])
        self.cfg, self.output = cfg, output
        self.current, self.completed_units = {"arm": None, "seed": None, "member": None, "update": 0}, 0
        helper_path = Path(__file__).with_name("fixed_context_key_objective.py")
        source_paths = [__file__, str(helper_path), cfg["common_routes"], cfg["factors"]] + [str(Path(cfg["native_repo"]) / name) for name in ("models.py", "run_base.py", "run_common.py")]
        self.source_hashes = {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_paths}
        self.config_sha256 = config_sha256
        write_json(output / "SOURCE_HASHES.json", {"config_sha256": config_sha256, "files": self.source_hashes})
        self.device = torch.device(cfg["device"])
        self.cuda = self.device.index if self.device.index is not None else 0
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        with np.load(cfg["train_npz"], allow_pickle=False) as t, np.load(cfg["valid_npz"], allow_pickle=False) as v:
            if set(t.files) != {"x", "edge_index", "ids", "y"} or set(v.files) != {"ids", "y"}:
                raise ValueError("Exact safe TRAIN/VALID keys required")
            arrays = [t[k].copy() for k in ("x", "edge_index", "ids", "y")] + [v[k].copy() for k in ("ids", "y")]
        x, edges, train_ids, train_y, valid_ids, valid_y = arrays
        if x.shape != (11701, 300) or edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("Full WikiCS graph/features required; graph must already be prepared")
        if x.dtype != np.float32 or any(a.dtype != np.int64 for a in arrays[1:]):
            raise ValueError("Preserve safe float32 features and int64 graph/roles")
        if train_ids.shape != (580,) or valid_ids.shape != (5274,) or train_y.shape != train_ids.shape or valid_y.shape != valid_ids.shape:
            raise ValueError("TRAIN580/VALID5274 required")
        if any(len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(x) for ids in (train_ids, valid_ids)) or np.intersect1d(train_ids, valid_ids).size:
            raise ValueError("Unique disjoint role IDs required")
        if any(y.min() < 0 or y.max() >= 10 for y in (train_y, valid_y)):
            raise ValueError("Ten-class role labels required")
        if any(np.bincount(train_y, minlength=10) < 2):
            raise ValueError("Context targets require at least two TRAIN examples in each of the ten classes")
        tensor = lambda a: torch.from_numpy(a).to(self.device)
        self.train_ids, self.valid_ids, self.valid_y = map(tensor, (train_ids, valid_ids, valid_y))
        labels = torch.zeros(len(x), dtype=torch.long, device=self.device)
        labels[self.train_ids] = tensor(train_y)
        mask = torch.zeros(len(x), dtype=torch.bool, device=self.device)
        mask[self.train_ids] = True
        self.data = SimpleNamespace(x=tensor(x), edge_index=tensor(edges), y=labels, train_mask=mask)
        self.batch = {"x": self.data.x, "edge_index": self.data.edge_index,
                      "graph": SimpleNamespace(edge_index=self.data.edge_index)}
        expected = {"seeds": [7301, 7403, 7507], "backbone": "SAGE", "depth": 2,
                    "hidden": 128, "dropout": .2, "learning_rate": .001, "weight_decay": 0,
                    "max_updates": 1000, "patience_updates": 300, "rank": 16,
                    "member_seed_stride": 1000003, "factor_seed_offset": 2000003,
                    "dropout_seed_offset": 3000007, "block_seed_offset": 4000037,
                    "context_coefficient": .05, "context_temperature": .2, "context_teacher_temperature": .2,
                    "query_seed_offset": 5000011, "context_shuffle_seed": 11711}
        if any(cfg.get(key, value) != value for key, value in expected.items()):
            raise ValueError("Exact prospectively fixed native SAGE recipe required; no grid")
        self.context_objective = load_source("sage_fixed_context_key_objective_v1", str(helper_path))
        self.context_contract = {"target_policy": self.context_objective.TARGET_POLICY,
            "coefficient": self.context_objective.COEFFICIENT,
            "student_temperature": self.context_objective.TEMPERATURE,
            "teacher_temperature": self.context_objective.TEACHER_TEMPERATURE,
            "objective": "unit_logZ_fixed_keys_minus_actual_Q_key_moment_dot_over_tau",
            "key_policy": self.context_objective.KEY_POLICY,
            "query_map": "one shared bare128to300_no_bias_training_only_per_bank; own_map_per_independent_body",
            "query_initialization": "separateCPU_float64_QR_sign_fixed_float32_semiorthogonal; same_per_seed_bytes_all_arms_bodies",
            "query_seed_offset": self.context_objective.QUERY_SEED_OFFSET,
            "target_mean": "actual_float32_Q_times_actual_float32_keys; not_normalized",
            "lognormalizer_coefficient": "literal1; inherited_cast_Q_rowmass_diagnostic_only",
            "moment_control": "(1_minus_query_dot_actual_target_key_mean)/student_temperature",
            "peer_control": "same_query_map_on_live_query_and_live_same_route_peer_keys; both_gradient_paths",
            "shuffle": "one_class_preserving_permutation_on_both_Q_axes; fixed_keys_unchanged",
            "shuffle_seed": self.context_objective.SHUFFLE_SEED,
            "mixture": "mean_of_four_complete_key_Q_denominator_objectives; never_average_keys",
            "assigned_independent_body": "body_m_uses_Q_m_with_explicit_context_index",
            "native_factual_forwards_per_route_update": 1}
        self.synchronize()
        prep_start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        target_arrays, preparation = self.context_objective.prepare_targets(np, x, edges, train_ids, train_y)
        bank_order = np.arange(580, dtype=np.int64)
        single_order = np.argsort(train_ids, kind="stable")
        # Native single train_step uses boolean-mask order, unlike bank ids order.
        self.context_targets = {
            "bank": self.context_objective.tensor_bank(torch, target_arrays, bank_order, self.device),
            "single": self.context_objective.tensor_bank(torch, target_arrays, single_order, self.device)}
        target_file = output / "CONTEXT_TARGETS.npz"
        np.savez(target_file, single_order=single_order, **target_arrays)
        preparation["archive_sha256"] = hashlib.sha256(target_file.read_bytes()).hexdigest()
        preparation["archive_bytes"] = target_file.stat().st_size
        preparation["ordered_TRAIN_ids_int64_bytes_sha256"] = hashlib.sha256(train_ids.tobytes(order="C")).hexdigest()
        preparation["ordered_TRAIN_y_int64_bytes_sha256"] = hashlib.sha256(train_y.tobytes(order="C")).hexdigest()
        preparation["single_order_int64_bytes_sha256"] = hashlib.sha256(single_order.tobytes(order="C")).hexdigest()
        preparation["array_hashes"] = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
            "C_bytes_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()} for key, value in target_arrays.items()}
        preparation["actual_loss_tensor_hashes"] = {
            order_name: {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                "C_bytes_sha256": hashlib.sha256(value.detach().cpu().numpy().tobytes(order="C")).hexdigest()}
                for key, value in tensors.items()} for order_name, tensors in self.context_targets.items()}
        preparation["torch_version"] = torch.__version__
        preparation["numpy_version"] = np.__version__
        preparation["target_loss_dtype"] = "float32; unit_logZ_coefficient; actual_Q_key_moments_not_normalized"
        preparation["actual_Q_rowmass_diagnostics"] = {
            order_name: {name: {"min": tensors[name].sum(-1).min().item(),
                "max": tensors[name].sum(-1).max().item(),
                "max_abs_error_from1": (tensors[name].sum(-1) - 1).abs().max().item()}
                for name in ("weights", "uniform_weights", "shuffled_weights")}
            for order_name, tensors in self.context_targets.items()}
        preparation["actual_key_moment_preparation"] = {
            order_name: {"bmm_calls": 3, "matrix_products": 12,
                "multiply_accumulates": 12 * 580 * 580 * 300,
                "output_coordinates": 12 * 580 * 300,
                "transferred_buffer_bytes": sum(v.numel() * v.element_size() for v in tensors.values()),
                "moment_normalization": False}
            for order_name, tensors in self.context_targets.items()}
        preparation["same_target_bank_shared_across_all_arms_and_seeds"] = True
        preparation["objective_contract"] = self.context_contract
        self.synchronize()
        preparation["inclusive_preparation_seconds"] = time.perf_counter() - prep_start
        preparation["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        preparation["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        preparation["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0
        self.context_preparation = preparation
        write_json(output / "CONTEXT_TARGETS.json", preparation)


    def progress(self, member, update):
        self.current.update(member=member, update=update)
        write_json(self.output / "PROGRESS.json", dict(self.current, completed_units=self.completed_units))

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def seeds(self, seed, member):
        native = seed + self.cfg.get("member_seed_stride", 1000003) * member
        return native, native + self.cfg.get("factor_seed_offset", 2000003), native + self.cfg.get("dropout_seed_offset", 3000007)

    def make(self, seed, member, kind, factorized, members):
        torch, c = self.torch, self.cfg
        native_seed, _, _ = self.seeds(seed, member)
        self.set_seed(native_seed)
        body = self.Model("SAGE", c.get("depth", 2), 300, c.get("hidden", 128), 10,
                          1, 8, "LayerNorm", c.get("dropout", .2))
        wrapper = self.routes.wrap_shared(torch, self.factors, body, self.routes.NativeModelAdapter(torch, body),
                    members=members, kind=kind, rank=c.get("rank", 16), block_seed=seed + c.get("block_seed_offset", 4000037)) if factorized else None
        if factorized:
            # Row-wise seeds match each factorized I4 body and the shared bank.
            with torch.no_grad():
                for row in range(members):
                    _, factor_seed, _ = self.seeds(seed, member + row)
                    gen = torch.Generator(device="cpu").manual_seed(factor_seed)
                    signs = torch.randint(0, 2, (1, 300), generator=gen) * 2 - 1
                    body.input_linear.r[row].copy_(signs[0].to(body.input_linear.r))
            # FactorLinear construction fixes every other r/s row to one.
        model = wrapper if members > 1 else body
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c.get("learning_rate", .001), weight_decay=0)
        if wrapper is not None:
            wrapper.check_optimizer_ownership(optimizer)
        streams = []
        for row in range(members):
            _, _, dropout_seed = self.seeds(seed, member + row)
            self.set_seed(dropout_seed)
            stream = {"cpu": torch.get_rng_state()}
            if self.device.type == "cuda":
                stream["cuda"] = torch.cuda.get_rng_state(self.device)
            streams.append(stream)
        return model, optimizer, streams

    def scope(self, streams, member):
        return self.routes.existing_member_rng_scope(self.torch, streams, member,
                                                     self.cuda if self.device.type == "cuda" else None)

    def logits(self, model, streams, bank, ids):
        if bank:
            batch = dict(self.batch, ids=ids)
            return model(batch, route_scope=lambda m: self.scope(streams, m))[0]
        return model(self.data, self.data.x)[ids].unsqueeze(0)

    def metrics(self, logits):
        torch, y = self.torch, self.valid_y.to(logits.device)
        probs = logits.softmax(-1)
        pool = probs.mean(0)
        correct = probs.argmax(-1).eq(y)
        pooled_correct = pool.argmax(-1).eq(y)
        log_probs = logits.to(torch.float64).log_softmax(-1)
        log_pool = torch.logsumexp(log_probs, dim=0) - math.log(len(logits))
        nll = -log_pool.gather(1, y[:, None]).mean()
        member_nll = [-log_probs[m].gather(1, y[:, None]).mean().item() for m in range(len(logits))]
        return {"accuracy": pooled_correct.float().mean().item(), "nll": nll.item(),
                "member_accuracy": correct.float().mean(1).tolist(), "member_nll": member_nll,
                "correct_alternative_count": correct.any(0).sum().item(),
                "coverage_lost_in_pooling": (correct.any(0) & ~pooled_correct).sum().item()}

    def training_outputs(self, model, streams, bank):
        if bank:
            return model(dict(self.batch, ids=self.train_ids),
                         route_scope=lambda m: self.scope(streams, m))
        adapter = self.routes.NativeModelAdapter(self.torch, model)
        with self.scope(streams, 0):
            logits, hidden = adapter.native(dict(self.batch, graph=self.data))
        # Preserve the original single train_step's boolean TRAIN-mask ordering.
        return (logits[self.data.train_mask].unsqueeze(0),
                hidden[self.data.train_mask].unsqueeze(0))

    def probability_pool_ce(self, logits, targets):
        log_probs = logits.log_softmax(-1)
        log_pool = self.torch.logsumexp(log_probs, dim=0) - math.log(len(logits))
        return -log_pool.gather(1, targets[:, None]).mean()

    def fit(self, folder, seed, member, kind, factorized, members, context_mode, context_index=None):
        torch = self.torch
        folder.mkdir()
        self.progress(member, 0)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.synchronize()
        start = time.perf_counter()
        model, optimizer, streams = self.make(seed, member, kind, factorized, members)
        query_start = time.perf_counter()
        query_map = self.context_objective.make_query_map(torch, self.F, seed, self.device)
        optimizer.add_param_group({"params": list(query_map.parameters()),
                                   "lr": self.cfg.get("learning_rate", .001), "weight_decay": 0})
        native_parameters, query_parameters = list(model.parameters()), list(query_map.parameters())
        expected_owners = {id(p) for p in native_parameters + query_parameters}
        actual_owners = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if len(actual_owners) != len(expected_owners) or set(actual_owners) != expected_owners:
            raise ValueError("Every native/query parameter must occur exactly once in optimizer")
        query_initial_sha256 = hashlib.sha256(query_map.weight.detach().cpu().numpy().tobytes(order="C")).hexdigest()
        self.synchronize()
        query_creation_seconds = time.perf_counter() - query_start
        bank, best, stale = members > 1, -1., 0
        if (context_mode not in ("key_private", "key_mixture", "key_uniform", "key_assigned",
                                 "projected_peer_private", "key_moment_private", "key_shuffled_private")
                or not bank and context_mode not in ("key_mixture", "key_assigned")):
            raise ValueError("Fixed private/shared or complete mixture/assigned independent objective required")
        if context_mode == "key_assigned":
            if bank or not isinstance(context_index, int) or isinstance(context_index, bool) or context_index != member or not 0 <= member < 4:
                raise ValueError("Assigned independent body m must use exactly context index m")
        elif context_index is not None:
            raise ValueError("Other fixed arms cannot receive a context index")
        target_order_name = "bank" if bank else "single"
        target_bank = self.context_targets[target_order_name]
        target_tensor_hashes = self.context_preparation["actual_loss_tensor_hashes"][target_order_name]
        context_counts = {}
        checkpoint = folder / "selected.pt"
        with (folder / "trace.jsonl").open("w") as trace:
            for step in range(1, self.cfg.get("max_updates", 1000) + 1):
                model.train()
                query_map.train()
                optimizer.zero_grad()
                logits, hidden = self.training_outputs(model, streams, bank)
                targets = self.data.y[self.train_ids] if bank else self.data.y[self.data.train_mask]
                own_ce = self.F.cross_entropy(logits.flatten(0, 1), targets.repeat(members))
                with torch.no_grad():
                    pool_ce = self.probability_pool_ce(logits, targets) if bank else None
                auxiliary, route_losses, pair_losses, paid = self.context_objective.context_auxiliary(
                    torch, self.F, hidden, target_bank, context_mode, context_index=context_index, query_map=query_map)
                for key, value in paid.items():
                    context_counts[key] = context_counts.get(key, 0) + value
                loss = own_ce + .05 * auxiliary
                loss.backward()
                optimizer.step()
                train_loss, train_own_ce = loss.item(), own_ce.item()
                train_pool_ce = pool_ce.item() if pool_ce is not None else None
                train_context = auxiliary.item()
                train_context_routes = route_losses.detach().tolist()
                train_context_pairs = pair_losses.detach().tolist()
                model.eval()
                query_map.eval()
                with torch.no_grad():
                    metrics = self.metrics(self.logits(model, streams, bank, self.valid_ids))
                improved = metrics["accuracy"] > best
                if improved:
                    best, stale, selected_step = metrics["accuracy"], 0, step
                    torch.save({"model": model.state_dict(), "query_map": query_map.state_dict(),
                                "optimizer": optimizer.state_dict(), "streams": streams,
                                "query_seed": seed + self.context_objective.QUERY_SEED_OFFSET,
                                "query_initial_sha256": query_initial_sha256,
                                "selected_query_sha256": hashlib.sha256(query_map.weight.detach().cpu().numpy().tobytes(order="C")).hexdigest(),
                                "step": step, "selection": metrics,
                                "context_mode": context_mode, "context_index": context_index,
                                "context_target_archive_sha256": self.context_preparation["archive_sha256"],
                                "TRAIN_H_order": "train_ids" if bank else "native_boolean_mask",
                                "context_objective_contract": self.context_contract,
                                "context_target_tensor_hashes": target_tensor_hashes,
                                "source_sha256": self.source_hashes,
                                "selection_update_training_scalars": {
                                    "scope": "pre_AdamW_training_scalar_at_update_whose_post_update_VALID_selects_checkpoint",
                                    "step": step, "context_index": context_index, "own_ce": train_own_ce,
                                    "auxiliary": train_context, "weighted_auxiliary": .05 * train_context,
                                    "objective": train_loss}}, checkpoint)
                else:
                    stale += 1
                trace.write(json.dumps({"step": step, "learning_objective": "own_CE_plus_fixed_context_key_or_declared_control",
                                        "context_mode": context_mode, "context_index": context_index,
                                        "train_objective": train_loss, "train_own_ce": train_own_ce,
                                        "train_probability_pool_ce_diagnostic": train_pool_ce,
                                        "train_context_auxiliary": train_context,
                                        "train_context_route_losses": train_context_routes,
                                        "train_context_pair_losses": train_context_pairs,
                                        "train_context_weighted": .05 * train_context,
                                        "valid": metrics,
                                        "strict_improvement": improved, "stale_updates": stale}, allow_nan=False) + "\n")
                trace.flush()
                self.progress(member, step)
                if stale >= self.cfg.get("patience_updates", 300):
                    break
        self.synchronize()
        acquisition = time.perf_counter() - start
        restoration_start = time.perf_counter()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored_target_tensor_hashes = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype),
                  "C_bytes_sha256": hashlib.sha256(value.detach().cpu().numpy().tobytes(order="C")).hexdigest()}
            for key, value in target_bank.items()}
        if (state["context_mode"] != context_mode or state["context_index"] != context_index
                or state["context_target_archive_sha256"] != self.context_preparation["archive_sha256"]
                or state["TRAIN_H_order"] != ("train_ids" if bank else "native_boolean_mask")
                or state["context_objective_contract"] != self.context_contract
                or state["context_target_tensor_hashes"] != target_tensor_hashes
                or state["context_target_tensor_hashes"] != restored_target_tensor_hashes
                or state["source_sha256"] != self.source_hashes):
            raise ValueError("Selected checkpoint scalar/target/source/order contract mismatch")
        if (state["selection_update_training_scalars"]["step"] != state["step"]
                or state["selection_update_training_scalars"]["context_index"] != context_index
                or not all(math.isfinite(state["selection_update_training_scalars"][key])
                           for key in ("own_ce", "auxiliary", "weighted_auxiliary", "objective"))):
            raise ValueError("Selected update scalar provenance missing or nonfinite")
        if (state["query_seed"] != seed + self.context_objective.QUERY_SEED_OFFSET
                or state["query_initial_sha256"] != query_initial_sha256
                or set(state["query_map"]) != {"weight"}
                or tuple(state["query_map"]["weight"].shape) != (300, 128)):
            raise ValueError("Selected query-map initialization/ownership contract mismatch")
        model.load_state_dict(state["model"], strict=True)
        query_map.load_state_dict(state["query_map"], strict=True)
        if hashlib.sha256(query_map.weight.detach().cpu().numpy().tobytes(order="C")).hexdigest() != state["selected_query_sha256"]:
            raise ValueError("Restored selected query-map bytes mismatch")
        optimizer.load_state_dict(state["optimizer"])
        streams = state["streams"]
        self.synchronize()
        restoration_seconds = time.perf_counter() - restoration_start
        model.eval()
        self.synchronize()
        serving_start = time.perf_counter()
        with torch.no_grad():
            selected_logits = self.logits(model, streams, bank, self.valid_ids)
            restored = self.metrics(selected_logits)
            selected_logits = selected_logits.cpu()
        self.synchronize()
        serving = time.perf_counter() - serving_start
        query_optimizer_bytes = sum(value.numel() * value.element_size()
            for values in optimizer.state.values() for value in values.values()
            if torch.is_tensor(value))
        query_specific_optimizer_bytes = sum(value.numel() * value.element_size()
            for parameter in query_map.parameters() for value in optimizer.state[parameter].values()
            if torch.is_tensor(value))
        costs = {"acquisition_seconds": acquisition, "restoration_seconds": restoration_seconds,
                 "query_map_creation_seconds_nested_in_acquisition": query_creation_seconds,
                 "selected_forward_and_metrics_seconds": serving,
                 "training_parameters": sum(p.numel() for p in native_parameters + query_parameters),
                 "auxiliary_training_parameters": sum(p.numel() for p in query_parameters),
                 "training_parameter_bytes": sum(p.numel() * p.element_size() for p in native_parameters + query_parameters),
                 "auxiliary_training_parameter_bytes": sum(p.numel() * p.element_size() for p in query_parameters),
                 "optimizer_tensor_state_bytes": query_optimizer_bytes,
                 "auxiliary_optimizer_tensor_state_bytes": query_specific_optimizer_bytes,
                 "auxiliary_serving_parameters": 0, "auxiliary_serving_forwards": 0,
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "inference_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                 "selected_checkpoint_bytes": checkpoint.stat().st_size,
                 "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                 "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
                 "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0}
        result = {"member": member, "learning_objective": "own_CE_plus_fixed_context_key_or_declared_control",
                  "context_mode": context_mode, "context_index": context_index, "context_coefficient": .05, "context_temperature": .2,
                  "context_teacher_temperature": .2,
                  "context_objective_contract": self.context_contract,
                  "query_map_seed": state["query_seed"],
                  "query_map_initial_sha256": state["query_initial_sha256"],
                  "selected_query_map_sha256": state["selected_query_sha256"],
                  "query_map_ownership": "one_auxiliary_map_per_optimizer_body; genuinely_shared_across_shared_routes",
                  "query_map_served": False,
                  "context_target_tensor_hashes": target_tensor_hashes,
                  "source_sha256": self.source_hashes,
                  "selected_update_training_scalars": state["selection_update_training_scalars"],
                  "context_target_archive_sha256": self.context_preparation["archive_sha256"],
                  "TRAIN_H_order": "train_ids" if bank else "native_boolean_mask",
                  "preclassifier_width": hidden.shape[-1],
                  "native_factor_dropout_seeds": [self.seeds(seed, member + r) for r in range(members)],
                  "selected_step": selected_step, "completed_updates": step, "selection": state["selection"], "restored": restored,
                  "selected_state": str(checkpoint), "costs": costs,
                  "operation_counts": dict(context_counts, updates=step, backwards=step, adam_steps=step,
                    train_probability_pool_loss_evaluations=step if bank else 0,
                    train_probability_pool_backward_evaluations=0,
                    train_fullgraph_native_trajectories=members * step,
                    valid_selection_fullgraph_native_trajectories=members * step,
                    selected_valid_fullgraph_native_trajectories=members,
                    native_graph_block_calls=self.cfg.get("depth", 2) * members * (2 * step + 1))}
        write_json(folder / "RESULT.json", result)
        self.completed_units += 1
        self.progress(member, step)
        return selected_logits, result

    def run(self):
        torch, np, results, errors = self.torch, self.np, [], {}
        seeds = self.cfg["seeds"]
        if seeds != [7301, 7403, 7507]:
            raise ValueError("The fixed paired seeds are [7301, 7403, 7507]")
        write_json(self.output / "CONFIG.json", self.cfg)
        for seed in seeds:
            for arm in ARMS:
                self.current.update(arm=arm, seed=seed)
                folder = self.output / f"{arm}_seed{seed}"
                folder.mkdir()
                genuine = arm in ("genuinefactorizedI4_key_mixture", "genuinefactorizedI4_key_assigned")
                bank = arm.startswith("shared4_")
                factorized = arm != "ordinaryM1_key_mixture"
                kind = "baseline"
                context_mode = CONTEXT_MODES[arm]
                fits = [self.fit(folder / f"member{m}", seed, m, kind, factorized, 4 if bank else 1,
                                 context_mode, context_index=m if arm == "genuinefactorizedI4_key_assigned" else None)
                        for m in range(4 if genuine else 1)]
                pool_start = time.perf_counter()
                logits = torch.cat([x[0] for x in fits])
                metrics = self.metrics(logits)
                probs, y = logits.softmax(-1), self.valid_y.cpu()
                pool = probs.mean(0)
                member_errors, pooled_errors = probs.argmax(-1).ne(y), pool.argmax(-1).ne(y)
                pool_seconds = time.perf_counter() - pool_start
                errors[seed, arm] = pooled_errors.numpy()
                np.savez(folder / "selected_VALID.npz", ids=self.valid_ids.cpu().numpy(), y=y.numpy(),
                         raw_logits=logits.numpy(), probability_mean=pool.numpy(), member_errors=member_errors.numpy(), pooled_errors=pooled_errors.numpy())
                units = [x[1] for x in fits]
                costs = {k: sum(u["costs"][k] for u in units) for k in units[0]["costs"] if "peak" not in k}
                costs.update({k: max(u["costs"][k] for u in units) for k in units[0]["costs"] if "peak" in k})
                costs["selected_pool_seconds"] = pool_seconds
                costs["selected_serving_readout_seconds"] = costs["selected_forward_and_metrics_seconds"] + pool_seconds
                result = {"arm": arm, "seed": seed, "serving": "probability_mean", "valid": metrics, "fits": units, "costs": costs}
                write_json(folder / "RESULT.json", result)
                results.append(result)
                print(json.dumps({"finished": arm, "seed": seed, "selected_steps": [u["selected_step"] for u in units]}), flush=True)
        # Cross-arm contrasts are deferred to root's joined complete reader after
        # all45banks/81records/225terminal scalar endpoints. Fresh fit closure is
        # not interpretation eligibility and cannot choose a continuation.
        comparisons = []
        counts = {key: sum(unit["operation_counts"][key] for r in results for unit in r["fits"]) for key in results[0]["fits"][0]["operation_counts"]}
        if len(results) != 30 or self.completed_units != 48:
            raise ValueError("All30 banks/all48 independent optimizer fits must close")
        write_json(self.output / "COMPLETE_FAMILY.json", {"complete": True, "completion_scope": "fresh ten-arm fixed-context-key acquisition only",
                   "full_comparative_family_complete": False,
                   "requires_root_admitted_own_only_anchors": {"banks": 15, "selected_fit_records": 33},
                   "groups": len(results), "fit_units": self.completed_units,
                   "expected_groups": 30, "expected_fit_units": 48,
                   "joined_expected_groups": 45, "joined_expected_selected_fit_records": 81,
                   "quality_interpretation_eligible": False,
                   "requires_total_terminal_scalar_endpoints": 225,
                   "cached_scalar_endpoints": 75, "new_scalar_endpoints": 150,
                   "requires_root_frozen_quality_attribution_confirmation_criteria_before_fit": True,
                   "context_objective_contract": self.context_contract, "TEST_access": False, "operation_counts": counts,
                   "config_sha256": self.config_sha256, "source_sha256": self.source_hashes,
                   "context_preparation_once": self.context_preparation,
                   "results": results, "comparisons": comparisons,
                   "cost_scope": "sequential acquisition; selected full-graph forward, metrics, CPU transfer and probability pool; RSS is process lifetime peak; overlap supplied by root"})


def main():
    raise SystemExit("Source-only fixed-context-key comparison: root reviews/qualifies/binds/freezes quality criteria and invokes Family through its existing owner; no automatic scientific admission")


if __name__ == "__main__":
    main()
