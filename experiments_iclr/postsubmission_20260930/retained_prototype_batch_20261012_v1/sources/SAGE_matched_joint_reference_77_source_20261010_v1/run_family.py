"""Portable fresh native SAGE references with a coherent full-width untied bank."""
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
    {"stem": "shared4", "architecture": "shared", "members": 4, "bodies": 1, "bundles": 1, "factorized": True},
    {"stem": "joint_untied4", "architecture": "joint_untied", "members": 4, "bodies": 4, "bundles": 1, "factorized": True},
    {"stem": "factorized_M1", "architecture": "single", "members": 1, "bodies": 1, "bundles": 1, "factorized": True},
    {"stem": "genuine_factorized_I4", "architecture": "single", "members": 4, "bodies": 4, "bundles": 4, "factorized": True},
    {"stem": "ordinary_M1", "architecture": "single", "members": 1, "bodies": 1, "bundles": 1, "factorized": False},
)
K_ARMS = {"A": "shared4_GNCL", "B": "shared4_own_SupCon", "AB": "shared4_GNCL_SupCon"}


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


def validate_config(cfg):
    for key in ("native_repo", "common_routes", "factors", "train_npz", "valid_npz", "device"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise ValueError("Root must supply portable configuration field: " + key)
    seeds = cfg.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or any(type(x) is not int or x < 0 for x in seeds) or len(set(seeds)) != 3:
        raise ValueError("One fixed roster of three distinct paired integer seeds required")
    if cfg.get("stage") not in ("F", "K"):
        raise ValueError("Stage must be exactly F or K")
    if cfg["stage"] == "F":
        if cfg.get("K") is not None or cfg.get("k_selection_receipt") is not None:
            raise ValueError("F stage keeps K unselected and reads no pilot quality")
        return {"stage": "F", "K": None, "pilot_provenance_read": False}
    rule = cfg.get("K")
    if not isinstance(rule, str) or rule not in K_ARMS:
        raise ValueError("K must be one exact A, B or AB string; lists/grids are forbidden")
    receipt_path = file_receipt(cfg.get("k_selection_receipt"))
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema_version") != 1 or receipt.get("K") != rule or receipt.get("selection_scope") != "complete native SAGE GNCL SupCon pilot":
        raise ValueError("Locked K receipt differs from declared complete-pilot choice")
    summary_path = file_receipt(receipt.get("complete_analysis_summary"))
    summary = json.loads(summary_path.read_text())
    expected = {"complete": True, "backbone": "SAGE", "new_banks": 18, "new_native_fits": 27,
                "original_reference_banks": 15, "original_selected_fits": 33, "joined_selected_fit_records": 60,
                "TEST_access": False, "all165_scalar_attempts_terminal_before_interpretation": True}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("Complete33-bank/60-fit/165-terminal-attempt pilot provenance required")
    report_item = receipt.get("complete_pilot_report")
    file_receipt(report_item)
    frozen_report = summary.get("full_report", {})
    if any(report_item[key] != frozen_report.get(key) for key in ("bytes", "sha256")):
        raise ValueError("Materialized full pilot report differs from complete summary")
    field = receipt.get("selection_policy_field")
    if field not in ("first_raw_accuracy_qualifier", "first_joint_accuracy_confidence_qualifier"):
        raise ValueError("Root must identify the complete pilot's frozen selection policy field")
    if summary.get("frozen_policy", {}).get(field) != K_ARMS[rule]:
        raise ValueError("K differs from the selected frozen complete-pilot qualifier")
    return {"stage": "K", "K": rule, "pilot_provenance_read": True,
            "selection_policy_field": field, "selection_receipt": cfg["k_selection_receipt"],
            "complete_analysis_summary": receipt["complete_analysis_summary"],
            "complete_pilot_report": report_item}


def wrap_joint_untied(torch, factors, routes, bodies):
    """Four private complete native bodies; one simultaneous scalar and selector.

    This is a sequential native execution of full-width untied routes. It is
    neither Packed-Ensembles nor a claim of reduced graph-channel work.
    """
    if len(bodies) != 4 or len({id(body) for body in bodies}) != 4:
        raise ValueError("Four fresh complete native bodies required")
    all_params = [p for body in bodies for p in body.parameters()]
    if len({id(p) for p in all_params}) != len(all_params):
        raise ValueError("Untied bodies must have disjoint parameters")
    storages = [p.untyped_storage().data_ptr() for p in all_params]
    if len(set(storages)) != len(storages):
        raise ValueError("Untied parameter storage aliases are forbidden")
    adapters = [routes.NativeModelAdapter(torch, body) for body in bodies]
    class JointUntied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bodies = torch.nn.ModuleList(bodies)
            self.members, self.kind = 4, "joint_untied"
        def check_optimizer_ownership(self, optimizer):
            expected = {id(p) for p in self.parameters() if p.requires_grad}
            actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
            if len(actual) != len(set(actual)) or set(actual) != expected:
                raise ValueError("One optimizer must own every private body parameter exactly once")
        def forward(self, batch, *, route_scope=None):
            if any(k in batch for k in ("y", "labels", "train_mask", "valid_mask", "test_mask")):
                raise ValueError("Model interface is label-free")
            if self.training and route_scope is None:
                raise ValueError("Caller-owned persistent member streams required")
            if batch["x"].ndim != 2 or batch["edge_index"].ndim != 2 or batch["edge_index"].shape[0] != 2:
                raise ValueError("One factual graph and feature matrix required")
            versions = (batch["x"]._version, batch["edge_index"]._version)
            rows = []
            for member, (body, adapter) in enumerate(zip(self.bodies, adapters)):
                adapter.check(batch)
                with (route_scope(member) if route_scope else nullcontext()), factors.member_context(body, 0):
                    rows.append(adapter.native(batch))
            if versions != (batch["x"]._version, batch["edge_index"]._version):
                raise ValueError("A body mutated the factual graph/input")
            ids = batch.get("ids", torch.arange(len(batch["x"]), device=batch["x"].device))
            return torch.stack([row[0][ids] for row in rows]), torch.stack([row[1][ids] for row in rows])
    return JointUntied()


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
    def __init__(self, cfg, output, config_sha256, seed_block=None):
        self.learning_lock = validate_config(cfg)  # Before numerical/model imports.
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
        from run_base import train_step
        from run_common import set_seed
        self.np, self.torch, self.F = np, torch, F
        self.Model, self.train_step, self.set_seed = Model, train_step, set_seed
        self.routes = load_source("sage_family_routes", cfg["common_routes"])
        self.factors = load_source("sage_family_factors", cfg["factors"])
        self.cfg, self.output = cfg, output
        self.current, self.completed_units = {"arm": None, "seed": None, "member": None, "update": 0}, 0
        source_paths = [__file__, cfg["common_routes"], cfg["factors"]] + [str(Path(cfg["native_repo"]) / name) for name in ("models.py", "run_base.py", "run_common.py")]
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
        if cfg["stage"] == "K" and cfg["K"] in ("B", "AB") and any(np.bincount(train_y, minlength=10) < 2):
            raise ValueError("SupCon requires at least two TRAIN examples in each of the ten classes")
        tensor = lambda a: torch.from_numpy(a).to(self.device)
        self.train_ids, self.valid_ids, self.valid_y = map(tensor, (train_ids, valid_ids, valid_y))
        labels = torch.zeros(len(x), dtype=torch.long, device=self.device)
        labels[self.train_ids] = tensor(train_y)
        mask = torch.zeros(len(x), dtype=torch.bool, device=self.device)
        mask[self.train_ids] = True
        self.data = SimpleNamespace(x=tensor(x), edge_index=tensor(edges), y=labels, train_mask=mask)
        self.batch = {"x": self.data.x, "edge_index": self.data.edge_index,
                      "graph": SimpleNamespace(edge_index=self.data.edge_index)}

    def progress(self, member, update):
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
        native_seed, _, _ = self.seeds(seed, native_member)
        self.set_seed(native_seed)
        body = self.Model("SAGE", c.get("depth", 2), 300, c.get("hidden", 128), 10,
                          1, 8, "LayerNorm", c.get("dropout", .2))
        wrapper = self.routes.wrap_shared(torch, self.factors, body, self.routes.NativeModelAdapter(torch, body),
                    members=members, kind="baseline", rank=c.get("rank", 16),
                    block_seed=seed + c.get("block_seed_offset", 4000037)) if factorized else None
        if factorized:
            with torch.no_grad():
                for row in range(members):
                    _, factor_seed, _ = self.seeds(seed, factor_member + row)
                    gen = torch.Generator(device="cpu").manual_seed(factor_seed)
                    signs = torch.randint(0, 2, (1, 300), generator=gen) * 2 - 1
                    body.input_linear.r[row].copy_(signs[0].to(body.input_linear.r))
        return body, wrapper

    def make(self, seed, member, kind, factorized, members, architecture="shared"):
        torch, c = self.torch, self.cfg
        if kind != "baseline" or architecture not in ("shared", "joint_untied", "single"):
            raise ValueError("Fixed native baseline architectures required")
        if architecture == "single" and members != 1:
            raise ValueError("Each single or genuine independent fit has exactly one native route")
        if architecture == "shared" and (not factorized or members != 4 or member != 0):
            raise ValueError("The shared bank has one four-route factorized native body")
        if architecture == "joint_untied":
            if not factorized or members != 4 or member != 0:
                raise ValueError("The matched joint untied bank has four factorized bodies")
            # Identical native W/b/norm constructor seed to shared member0;
            # distinct input-factor signs and dropout streams match route m.
            bodies = [self.native_body(seed, 0, row, True, 1)[0] for row in range(4)]
            model = wrap_joint_untied(torch, self.factors, self.routes, bodies)
        else:
            body, wrapper = self.native_body(seed, member, member, factorized, members)
            model = wrapper if members > 1 else body
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
        write_json(folder / "RESULT.json", result)
        self.completed_units += 1
        self.progress(member, step)
        return selected_logits, result

    def run(self):
        torch, np, results, errors = self.torch, self.np, [], {}
        seeds, stage = self.cfg["seeds"], self.cfg["stage"]
        arms = [spec["stem"] + "_" + stage for spec in ARM_SPECS]
        write_json(self.output / "CONFIG.json", self.cfg)
        write_json(self.output / "LEARNING_LOCK.json", self.learning_lock)
        for seed in self.seed_block:
            for spec, arm in zip(ARM_SPECS, arms):
                self.current.update(arm=arm, seed=seed)
                folder = self.output / f"{arm}_seed{seed}"
                folder.mkdir()
                genuine = spec["bundles"] == 4
                jointly_optimized = spec["architecture"] in ("shared", "joint_untied")
                objective = "gncl_half" if stage == "K" and self.cfg["K"] in ("A", "AB") and jointly_optimized else "own_only"
                use_supcon = stage == "K" and self.cfg["K"] in ("B", "AB")
                # Singles/I4 are unscaled own CE (A's M1 pool term is identical),
                # plus within-body SupCon only when B is present.
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
                          "fits": units, "costs": costs}
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
                   "fixed_arm_roster": arms, "full_expected_groups": 15, "full_expected_optimizer_acquisition_bundles": 24,
                   "full_expected_native_body_fit_records": 33, "full_native_trajectories_per_roster_forward": 42,
                   "groups": len(results), "fit_units": self.completed_units,
                   "fit_units_meaning": "optimizer/acquisition bundles, not native bodies",
                   "optimizer_acquisition_bundles": self.completed_units,
                   "native_body_fit_records": sum(r["native_bodies_fitted"] for r in results),
                   "expected_groups": 5 * len(self.seed_block), "expected_fit_units": 8 * len(self.seed_block),
                   "expected_native_body_fit_records": 11 * len(self.seed_block),
                   "TEST_access": False, "operation_counts": counts, "learning_lock": self.learning_lock,
                   "config_sha256": self.config_sha256, "source_sha256": self.source_hashes, "results": results, "comparisons": comparisons,
                   "cost_scope": "native sequential route acquisition; no packed execution claim; selected fullgraph forward/metrics/CPU transfer/pool; lifetime RSS peak; overlap supplied by root"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-block", nargs="+", type=int, help="Only complete seed blocks from the fixed config roster")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config_bytes = Path(args.config).read_bytes()
    Family(json.loads(config_bytes), output, hashlib.sha256(config_bytes).hexdigest(), args.seed_block).run()


if __name__ == "__main__":
    main()
