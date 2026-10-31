"""Finite native PyG FiLM family; derived from the existing acquisition/export owner.

Import is stdlib-only. Source preparation admits no execution; metadata admission
precedes numerical imports. All nine banks are freshly acquired; source preparation admits no execution.
"""
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
from contextlib import nullcontext

ARM_SPECS = (
    {"stem": "FiLM_M1", "operation": "film", "architecture": "single", "members": 1, "bodies": 1, "bundles": 1, "factorized": True},
    {"stem": "FiLM_shared4", "operation": "film", "architecture": "shared", "members": 4, "bodies": 1, "bundles": 1, "factorized": True},
    {"stem": "FiLM_I4", "operation": "film", "architecture": "single", "members": 4, "bodies": 4, "bundles": 4, "factorized": True},
)


def file_receipt(item):
    """Validate a root-supplied portable materialization, without model/data reads."""
    if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
        raise ValueError("Exact path/bytes/sha256 receipt required")
    path = Path(item["path"])
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        raise ValueError("Provenance file size differs")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        raise ValueError("Provenance file hash differs")
    return path


def validate_config(cfg, qualification_only=False):
    contracts = load_source("endpoint_value_contracts", Path(__file__).with_name("contracts.py"))
    return contracts.validate(cfg, action="runtime_readiness_check" if qualification_only else "fixed_family_fit",
                              executing_source=__file__)


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
    def __init__(self, cfg, output, config_sha256, seed_block=None, training_only=False):
        preparation_started = time.perf_counter()
        self.family_started = preparation_started
        self.training_only = training_only
        self.learning_lock = validate_config(cfg, training_only)  # Before numerical/model imports.
        self.seed_block = cfg["seeds"] if seed_block is None else seed_block
        if not self.seed_block or len(set(self.seed_block)) != len(self.seed_block) or any(s not in cfg["seeds"] for s in self.seed_block):
            raise ValueError("Shard only complete seed blocks from the fixed three-seed roster")
        if self.seed_block != [s for s in cfg["seeds"] if s in self.seed_block]:
            raise ValueError("Preserve declared seed ordering within each shard")
        import numpy as np
        import torch
        import torch.nn.functional as F
        sys.path.insert(0, cfg["native_repo"])
        from models import Model
        from run_common import set_seed
        self.np, self.torch, self.F = np, torch, F
        self.Model, self.set_seed = Model, set_seed
        import inspect
        from torch_geometric.nn.conv import FiLMConv, MessagePassing
        import torch_geometric
        if np.__version__ != "1.26.4" or torch.__version__ != "2.1.2+cu118" or torch_geometric.__version__ != "2.7.0":
            raise ValueError("Original NumPy/Torch/PyG runtime required")
        self.FiLMConv = FiLMConv
        for key, obj in (("models", Model), ("run_common", set_seed),
                         ("film_conv", FiLMConv), ("message_passing", MessagePassing)):
            if Path(inspect.getfile(obj)).resolve() != Path(cfg["source_receipts"][key]["path"]).resolve():
                raise ValueError("Imported native/PyG source differs from admitted path: " + key)
        self.routes = load_source("sage_family_routes", cfg["common_routes"])
        self.factors = load_source("sage_family_factors", cfg["factors"])
        self.film_adapter_source = load_source("native_film_adapter", Path(__file__).with_name("film_adapter.py"))
        self.contracts = load_source("endpoint_value_contracts_counts", Path(__file__).with_name("contracts.py"))
        self._operation = "film"
        self._fit_model = self._single_adapter = None
        self._film_installations = []
        self._film_adapter = None
        self.cfg, self.output = cfg, output
        self.current, self.completed_units = {"arm": None, "seed": None, "member": None, "update": 0}, 0
        self.source_hashes = dict(self.learning_lock["source_sha256"])
        self.config_sha256 = config_sha256
        write_json(output / "SOURCE_HASHES.json", {"config_sha256": config_sha256, "files": self.source_hashes})
        self.device = torch.device(cfg["device"])
        self.cuda = self.device.index if self.device.index is not None else 0
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        with np.load(cfg["train_npz"], allow_pickle=False) as t:
            if set(t.files) != {"x", "edge_index", "ids", "y"}:
                raise ValueError("Exact safe full-graph TRAIN keys required")
            arrays = [t[k].copy() for k in ("x", "edge_index", "ids", "y")]
        if training_only:
            # Never deserialize VALID in disposable full-TRAIN qualification.
            arrays += [np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)]
        else:
            with np.load(cfg["valid_npz"], allow_pickle=False) as v:
                if set(v.files) != {"ids", "y"}:
                    raise ValueError("Exact safe VALID keys required")
                arrays += [v[k].copy() for k in ("ids", "y")]
        x, edges, train_ids, train_y, valid_ids, valid_y = arrays
        if x.shape != (11701, 300) or edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("Full WikiCS graph/features required; graph must already be prepared")
        if x.dtype != np.float32 or not np.isfinite(x).all() or any(a.dtype != np.int64 for a in arrays[1:]):
            raise ValueError("Preserve safe float32 features and int64 graph/roles")
        if train_ids.shape != (580,) or valid_ids.shape != ((0,) if training_only else (5274,)) or train_y.shape != train_ids.shape or valid_y.shape != valid_ids.shape:
            raise ValueError("TRAIN580/VALID5274 required")
        if any(len(np.unique(ids)) != len(ids) or (len(ids) and (ids.min() < 0 or ids.max() >= len(x))) for ids in (train_ids, valid_ids)) or np.intersect1d(train_ids, valid_ids).size:
            raise ValueError("Unique disjoint role IDs required")
        if any(len(y) and (y.min() < 0 or y.max() >= 10) for y in (train_y, valid_y)):
            raise ValueError("Ten-class role labels required")
        if edges.size == 0 or edges.min() < 0 or edges.max() >= 11701:
            raise ValueError("Prepared complete factual edge support required")
        self.edge_metadata = {"nodes": 11701, "edges": int(edges.shape[1]),
                              "self_edges": int((edges[0] == edges[1]).sum())}
        write_json(output / "DATA_SHAPES.json", dict(self.edge_metadata, TRAIN_rows=580,
            VALID_rows=0 if training_only else 5274, TEST_access=False,
            qualification_never_deserializes_VALID=training_only))
        tensor = lambda a: torch.from_numpy(a).to(self.device)
        self.train_ids, self.valid_ids, self.valid_y = map(tensor, (train_ids, valid_ids, valid_y))
        labels = torch.zeros(len(x), dtype=torch.long, device=self.device)
        labels[self.train_ids] = tensor(train_y)
        mask = torch.zeros(len(x), dtype=torch.bool, device=self.device)
        mask[self.train_ids] = True
        self.data = SimpleNamespace(x=tensor(x), edge_index=tensor(edges), y=labels, train_mask=mask)
        self.batch = {"x": self.data.x, "edge_index": self.data.edge_index,
                      "graph": SimpleNamespace(edge_index=self.data.edge_index)}
        self.synchronize()
        self.preparation = {"seconds": time.perf_counter() - preparation_started,
            "scope": "metadata/source/role hashing, runtime imports, full safe graph/role loading and tensor placement before native acquisition",
            "root_qualification_is_separate_paid_work": True,
            "VALID_deserialized": not training_only, "TEST_access": False,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}
        write_json(output / "PREPARATION.json", self.preparation)

    def progress(self, member, update):
        if not self.training_only:
            limits = self.cfg["family_limits"]
            if time.perf_counter() - self.family_started > limits["wall_seconds"]:
                raise RuntimeError("Finite entire-family wall budget exceeded")
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            if rss > limits["rss_bytes"]:
                raise RuntimeError("Finite family lifetime RSS budget exceeded")
            if self.device.type == "cuda" and self.torch.cuda.max_memory_allocated(self.device) > limits["cuda_bytes"]:
                raise RuntimeError("Finite native full-graph CUDA budget exceeded")
        self.current.update(member=member, update=update)
        write_json(self.output / "PROGRESS.json", dict(self.current, completed_units=self.completed_units))

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def seeds(self, seed, member):
        native = seed + self.cfg.get("member_seed_stride", 1000003) * member
        return native, native + self.cfg.get("factor_seed_offset", 2000003), native + self.cfg.get("dropout_seed_offset", 3000007)

    def native_body(self, seed, native_member, factor_member, factorized, members):
        torch, c = self.torch, self.cfg
        if not factorized or members not in (1, 4):
            raise ValueError("Exact equipped FiLM M1/M4 native ownership")
        native_seed, _, _ = self.seeds(seed, native_member)
        self.set_seed(native_seed)
        body = self.Model("SAGE", c["depth"], 300, c["hidden"], 10,
                          1, 8, "LayerNorm", c["dropout"])
        adapter = self.film_adapter_source.install(torch, body, self.FiLMConv, self.edge_metadata)
        wrapper = self.routes.wrap_shared(torch, self.factors, body, adapter,
                    members=members, kind="baseline", rank=c["rank"],
                    block_seed=seed + c["block_seed_offset"])
        adapter.assert_factorized(self.factors, members)
        with torch.no_grad():
            for row in range(members):
                _, factor_seed, _ = self.seeds(seed, factor_member + row)
                gen = torch.Generator(device="cpu").manual_seed(factor_seed)
                signs = torch.randint(0, 2, (1, 300), generator=gen) * 2 - 1
                body.input_linear.r[row].copy_(signs[0].to(body.input_linear.r))
        wrapper.native_adapter = adapter
        self._film_adapter = adapter
        self._film_installations = adapter.installations
        if members == 1:
            self._single_adapter = adapter
        return body, wrapper

    def make(self, seed, member, kind, factorized, members, architecture="shared"):
        torch, c = self.torch, self.cfg
        if (kind != "baseline" or architecture not in ("shared", "single") or not factorized
                or architecture == "single" and members != 1
                or architecture == "shared" and (members != 4 or member != 0)):
            raise ValueError("Fixed shared4 or genuine one-body factorized native routes required")
        body, wrapper = self.native_body(seed, member, member, True, members)
        model = wrapper if members > 1 else body
        self._fit_model = model
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c.get("learning_rate", .001), weight_decay=0)
        if members > 1:
            model.check_optimizer_ownership(optimizer)
        streams = []
        for row in range(members):
            _, _, dropout_seed = self.seeds(seed, member + row)
            self.set_seed(dropout_seed)
            stream = {"cpu": torch.get_rng_state()}
            if self.device.type == "cuda":
                stream["cuda"] = torch.cuda.get_rng_state(self.device)
            streams.append(stream)
        return model, optimizer, streams

    def body_records(self, seed, member, members, architecture, factorized, checkpoint, selected_step):
        body_members = range(4) if architecture == "joint_untied" else (member,)
        records = []
        for body_member in body_members:
            native_member = 0 if architecture in ("shared", "joint_untied") else body_member
            native_seed = self.seeds(seed, native_member)[0]
            route_members = range(4) if architecture == "shared" else (body_member,)
            records.append({"body_index": body_member, "native_initialization_seed": native_seed,
                            "factorized": factorized, "native_parameter_scope": "body" if architecture == "shared" else f"bodies.{body_member}" if architecture == "joint_untied" else "native_body",
                            "route_factor_dropout_seeds": [{"route": row, "factor_seed": self.seeds(seed, row)[1] if factorized else None,
                                                            "dropout_seed": self.seeds(seed, row)[2]} for row in route_members],
                            "selected_step": selected_step, "selected_state": str(checkpoint),
                            "selection_scope": "coherent_bank" if members > 1 else "own_body"})
        return records

    def scope(self, streams, member):
        return self.routes.existing_member_rng_scope(self.torch, streams, member,
                                                     self.cuda if self.device.type == "cuda" else None)

    def logits(self, model, streams, bank, ids):
        if bank:
            batch = dict(self.batch, ids=ids)
            return model(batch, route_scope=lambda m: self.scope(streams, m))[0]
        with self.factors.member_context(model, 0):
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
        adapter = self._single_adapter
        if adapter is None or adapter.body is not model:
            raise ValueError("Use the original validated native adapter for this installed single")
        with self.scope(streams, 0), self.factors.member_context(model, 0):
            logits, hidden = adapter.native(dict(self.batch, graph=self.data))
        # Preserve the original single train_step's boolean TRAIN-mask ordering.
        return (logits[self.data.train_mask].unsqueeze(0),
                hidden[self.data.train_mask].unsqueeze(0))

    def within_route_supcon(self, hidden, targets):
        torch = self.torch
        if hidden.ndim != 3 or hidden.shape[1] != len(targets) or len(targets) != 580:
            raise ValueError("Live member-by-all580-TRAIN preclassifier H required")
        normalized = self.F.normalize(hidden, p=2, dim=-1, eps=1e-12)
        cosine = normalized @ normalized.transpose(-2, -1)
        scores = cosine / .2
        allowed = ~torch.eye(len(targets), dtype=torch.bool, device=hidden.device)
        positives = targets[:, None].eq(targets[None, :]) & allowed
        positive_counts = positives.sum(-1)
        if (positive_counts == 0).any().item():
            raise ValueError("Every TRAIN anchor must have a nonself same-label positive")
        log_denominator = torch.logsumexp(scores.masked_fill(~allowed[None], float("-inf")), dim=-1)
        mean_positive_score = scores.masked_fill(~positives[None], 0).sum(-1) / positive_counts[None]
        route_losses = (log_denominator - mean_positive_score).mean(-1)
        return route_losses.mean(), route_losses

    def probability_pool_ce(self, logits, targets):
        log_probs = logits.log_softmax(-1)
        log_pool = self.torch.logsumexp(log_probs, dim=0) - math.log(len(logits))
        return -log_pool.gather(1, targets[:, None]).mean()

    def fit(self, folder, seed, member, kind, factorized, members, objective="own_only", use_supcon=False, architecture="shared"):
        torch = self.torch
        if self.training_only or objective != "own_only" or use_supcon:
            raise ValueError("Only the fixed own-CE scientific fit is admitted here")
        folder.mkdir()
        self.progress(member, 0)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.synchronize()
        start = time.perf_counter()
        model, optimizer, streams = self.make(seed, member, kind, factorized, members, architecture)
        bank, best, stale = members > 1, -1., 0
        if objective not in ("own_only", "gncl_half") or (objective == "gncl_half" and not bank):
            raise ValueError("Pool GNCL applies only to jointly optimized four-route banks")
        checkpoint = folder / "selected.pt"
        with (folder / "trace.jsonl").open("w") as trace:
            for step in range(1, self.cfg.get("max_updates", 1000) + 1):
                model.train()
                optimizer.zero_grad()
                logits, hidden = self.training_outputs(model, streams, bank)
                targets = self.data.y[self.train_ids] if bank else self.data.y[self.data.train_mask]
                own_ce = self.F.cross_entropy(logits.flatten(0, 1), targets.repeat(members))
                pool_ce = None
                if objective == "gncl_half":
                    pool_ce = self.probability_pool_ce(logits, targets)
                    loss = .5 * own_ce + .5 * pool_ce
                else:
                    loss = own_ce
                    if bank:
                        with torch.no_grad():
                            pool_ce = self.probability_pool_ce(logits, targets)
                sc_loss, sc_routes = self.within_route_supcon(hidden, targets) if use_supcon else (None, None)
                if use_supcon:
                    loss = loss + .05 * sc_loss
                loss.backward()
                optimizer.step()
                train_loss, train_own_ce = loss.item(), own_ce.item()
                train_pool_ce = pool_ce.item() if pool_ce is not None else None
                train_supcon = sc_loss.item() if sc_loss is not None else None
                train_supcon_routes = sc_routes.detach().tolist() if sc_routes is not None else None
                model.eval()
                with torch.no_grad():
                    metrics = self.metrics(self.logits(model, streams, bank, self.valid_ids))
                improved = metrics["accuracy"] > best
                if improved:
                    best, stale, selected_step = metrics["accuracy"], 0, step
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "streams": streams,
                                "step": step, "selection": metrics}, checkpoint)
                else:
                    stale += 1
                trace.write(json.dumps({"step": step, "learning_objective": objective,
                                        "train_objective": train_loss, "train_own_ce": train_own_ce,
                                        "train_probability_pool_ce": train_pool_ce,
                                        "train_supcon": train_supcon, "train_supcon_route_losses": train_supcon_routes,
                                        "train_supcon_weighted": .05 * train_supcon if use_supcon else 0.,
                                        "valid": metrics,
                                        "strict_improvement": improved, "stale_updates": stale}, allow_nan=False) + "\n")
                trace.flush()
                self.progress(member, step)
                if stale >= self.cfg.get("patience_updates", 300):
                    break
        self.synchronize()
        acquisition = time.perf_counter() - start
        restore_start = time.perf_counter()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        streams = state["streams"]
        model.eval()
        self.synchronize()
        restore_seconds = time.perf_counter() - restore_start
        serving_start = time.perf_counter()
        with torch.no_grad():
            selected_logits = self.logits(model, streams, bank, self.valid_ids)
            restored = self.metrics(selected_logits)
            selected_logits = selected_logits.cpu()
        self.synchronize()
        serving = time.perf_counter() - serving_start
        costs = {"acquisition_seconds": acquisition, "selected_restore_seconds": restore_seconds,
                 "selected_forward_and_metrics_seconds": serving,
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "inference_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                 "selected_checkpoint_bytes": checkpoint.stat().st_size,
                 "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                 "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
                 "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0}
        body_records = self.body_records(seed, member, members, architecture, factorized, checkpoint, selected_step)
        body = model.body if bank else model
        film_counts = self._film_adapter.operation_counts(members, step)
        costs["optimizer_tensor_state_bytes"] = sum(value.numel() * value.element_size()
            for values in optimizer.state.values() for value in values.values() if torch.is_tensor(value))
        film_parameter_roles = self._film_adapter.parameter_roles(self.factors)
        result = {"member": member, "architecture": architecture, "factorized": factorized,
                  "optimizer_acquisition_bundles": 1, "native_bodies_fitted": len(body_records), "body_fit_records": body_records,
                  "selection_scope": "coherent_bank" if bank else "own_body",
                  "learning_objective": objective, "within_route_supcon": use_supcon,
                  "supcon_coefficient": .05 if use_supcon else 0., "supcon_temperature": .2 if use_supcon else None,
                  "preclassifier_width": hidden.shape[-1], "route_seed_metadata": [record["route_factor_dropout_seeds"] for record in body_records],
                  "selected_step": selected_step, "completed_updates": step, "selection": state["selection"], "restored": restored,
                  "selected_state": str(checkpoint), "costs": costs,
                  "operation_counts": {"updates": step, "backwards": step, "adam_steps": step,
                    "train_probability_pool_loss_evaluations": step if bank else 0,
                    "train_probability_pool_backward_evaluations": step if objective == "gncl_half" else 0,
                    "supcon_batched_gram_calls": step if use_supcon else 0,
                    "supcon_route_grams": members * step if use_supcon else 0,
                    "supcon_normalized_coordinates": members * 580 * hidden.shape[-1] * step if use_supcon else 0,
                    "supcon_cosine_gram_entries": members * 580 * 580 * step if use_supcon else 0,
                    "supcon_gram_multiply_accumulates": members * 580 * 580 * hidden.shape[-1] * step if use_supcon else 0,
                    "supcon_anchor_rows": members * 580 * step if use_supcon else 0,
                    "train_fullgraph_native_trajectories": members * step, "valid_selection_fullgraph_native_trajectories": members * step,
                    "selected_valid_fullgraph_native_trajectories": members,
                    "native_graph_block_calls": self.cfg.get("depth", 2) * members * (2 * step + 1)}}
        result.update(selected_state_receipt={"path": str(checkpoint.resolve()), "bytes": checkpoint.stat().st_size,
                      "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
                      operator_kind=self._operation, film_installations=self._film_installations,
                      film_parameter_roles=film_parameter_roles, film_operation_counts=film_counts,
                      factual_message_policy="native FiLM edgewise ReLU/mean and separate skip on unchanged stored support",
                      native_factorized_maps=(model.factorized_maps if bank else sum(isinstance(module, self.factors.FactorLinear) for module in model.modules())))
        result["operation_counts"].update({key: film_counts["total"][key] for key in self.contracts.FILM_COUNT_KEYS})
        write_json(folder / "RESULT.json", result)
        self._fit_model = self._single_adapter = self._film_adapter = None
        self.completed_units += 1
        self.progress(member, step)
        return selected_logits, result

    def run(self):
        torch, np, results, errors = self.torch, self.np, [], {}
        if self.training_only or self.seed_block != self.cfg["seeds"]:
            raise ValueError("Only the entire fixed three-seed scientific roster may run")
        seeds, stage = self.cfg["seeds"], "F"
        arms = [spec["stem"] for spec in ARM_SPECS]
        write_json(self.output / "CONFIG.json", self.cfg)
        write_json(self.output / "LEARNING_LOCK.json", self.learning_lock)
        for seed in self.seed_block:
            for spec, arm in zip(ARM_SPECS, arms):
                self.current.update(arm=arm, seed=seed)
                self._operation = spec["operation"]
                folder = self.output / f"{arm}_seed{seed}"
                folder.mkdir()
                genuine = spec["bundles"] == 4
                jointly_optimized = spec["architecture"] in ("shared", "joint_untied")
                objective, use_supcon = "own_only", False
                fits = [self.fit(folder / f"member{m}", seed, m, "baseline", spec["factorized"],
                                 1 if genuine else spec["members"], objective, use_supcon, spec["architecture"])
                        for m in range(spec["bundles"])]
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
                archive_path = folder / "selected_VALID.npz"
                archive_receipt = {"path": str(archive_path.resolve()), "bytes": archive_path.stat().st_size,
                                   "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}
                units = [x[1] for x in fits]
                costs = {k: sum(u["costs"][k] for u in units) for k in units[0]["costs"] if "peak" not in k}
                costs.update({k: max(u["costs"][k] for u in units) for k in units[0]["costs"] if "peak" in k})
                costs["selected_pool_seconds"] = pool_seconds
                costs["selected_serving_readout_seconds"] = costs["selected_forward_and_metrics_seconds"] + pool_seconds
                costs["selected_restore_forward_and_pool_seconds"] = costs["selected_restore_seconds"] + costs["selected_serving_readout_seconds"]
                costs["acquisition_restore_selected_readout_seconds"] = costs["acquisition_seconds"] + costs["selected_restore_forward_and_pool_seconds"]
                result = {"arm": arm, "seed": seed, "stage": stage, "K": self.cfg.get("K"),
                          "architecture": spec["architecture"], "serving": "probability_mean", "valid": metrics,
                          "optimizer_acquisition_bundles": len(units),
                          "native_bodies_fitted": sum(u["native_bodies_fitted"] for u in units),
                          "native_trajectories_per_forward": len(logits),
                          "selection_scope": "independent_own_body_then_pool" if genuine else "coherent_bank" if jointly_optimized else "own_body",
                          "fits": units, "costs": costs, "operator_kind": spec["operation"],
                          "selected_VALID_archive": archive_receipt}
                if result["optimizer_acquisition_bundles"] != spec["bundles"] or result["native_bodies_fitted"] != spec["bodies"] or len(logits) != spec["members"]:
                    raise ValueError("Declared bank/bundle/body/route count differs")
                write_json(folder / "RESULT.json", result)
                results.append(result)
                print(json.dumps({"finished": arm, "seed": seed, "selected_steps": [u["selected_step"] for u in units]}), flush=True)
        full_stage = self.seed_block == seeds
        comparisons = []  # Seed shards expose no incomplete-family repair contrasts.
        if full_stage:
            for result in results:
                seed, arm = result["seed"], result["arm"]
                candidate = errors[seed, arm]
                versus = {ref: {"repairs": int((errors[seed, ref] & ~candidate).sum()),
                               "harms": int((~errors[seed, ref] & candidate).sum())} for ref in arms}
                comparisons.append({"arm": arm, "seed": seed, "versus": versus})
        counts = {key: sum(unit["operation_counts"][key] for r in results for unit in r["fits"]) for key in results[0]["fits"][0]["operation_counts"]}
        write_json(self.output / "COMPLETE_FAMILY.json", {"complete": True, "completion_scope": "declared stage and seed block only",
                   "full_same_runtime_stage_complete": full_stage, "complete_F_implies_complete_K": False,
                   "stage": stage, "K": self.cfg.get("K"), "fixed_seed_roster": seeds, "completed_seed_block": self.seed_block,
                   "fixed_arm_roster": arms, "full_expected_groups": 9, "full_expected_optimizer_acquisition_bundles": 18,
                   "full_expected_native_body_fit_records": 18, "full_native_trajectories_per_roster_forward": 27,
                   "groups": len(results), "fit_units": self.completed_units,
                   "fit_units_meaning": "optimizer/acquisition bundles, not native bodies",
                   "optimizer_acquisition_bundles": self.completed_units,
                   "native_body_fit_records": sum(r["native_bodies_fitted"] for r in results),
                   "expected_groups": 3 * len(self.seed_block), "expected_fit_units": 6 * len(self.seed_block),
                   "expected_native_body_fit_records": 6 * len(self.seed_block),
                   "TEST_access": False, "study_kind": "native_PyG_FiLM_WikiCS", "baseline_policy": "fresh_required",
                   "protocol": self.contracts.PROTOCOL,
                   "safe_role_sha256": self.learning_lock["safe_role_sha256"],
                   "runtime_admission": self.learning_lock["root_runtime_admission"],
                   "preparation": self.preparation,
                   "entire_family_observed_seconds": time.perf_counter() - self.family_started,
                   "finite_family_limits": self.cfg["family_limits"],
                   "pending_call_hard_limit": "root external finite watchdog required",
                   "operation_counts": counts, "learning_lock": self.learning_lock,
                   "config_sha256": self.config_sha256, "source_sha256": self.source_hashes, "results": results, "comparisons": comparisons,
                   "cost_scope": "native sequential route acquisition; no packed execution claim; selected fullgraph forward/metrics/CPU transfer/pool; lifetime RSS peak; overlap supplied by root"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_bytes = Path(args.config).read_bytes()
    family = None
    try:
        family = Family(json.loads(config_bytes), output, hashlib.sha256(config_bytes).hexdigest())
        family.run()
    except BaseException as error:
        write_json(output / "FAILURE.json", {"status": "FAILED_RETAINED", "complete": False,
            "type": type(error).__name__, "message": str(error),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "current": None if family is None else family.current,
            "completed_optimizer_bundles": 0 if family is None else family.completed_units,
            "TEST_access": False, "retry_or_rescue": False})
        raise


if __name__ == "__main__":
    main()
