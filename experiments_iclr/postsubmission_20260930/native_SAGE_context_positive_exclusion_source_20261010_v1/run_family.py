"""Inactive native SAGE fixed-context exclusion; existing Family, root admission."""
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

ARMS = ("ordinary_M1_context_mixture", "factorized_M1_context_mixture",
        "genuine_factorized_I4_context_mixture", "shared4_context_private_masked",
        "shared4_context_pair_mixture_masked", "shared4_context_private_full",
        "shared4_context_shuffled_masked")
CONTEXT_MODES = {ARMS[0]: "mixture", ARMS[1]: "mixture", ARMS[2]: "mixture",
                 ARMS[3]: "private", ARMS[4]: "mixture", ARMS[5]: "private_full",
                 ARMS[6]: "shuffled_private"}


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
        helper_path = Path(__file__).with_name("context_objective.py")
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
                    "dropout_seed_offset": 3000007, "block_seed_offset": 4000037}
        if any(cfg.get(key, value) != value for key, value in expected.items()):
            raise ValueError("Exact prospectively fixed native SAGE recipe required; no grid")
        self.context_objective = load_source("sage_fixed_context_objective", str(helper_path))
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
        np.savez(target_file, train_ids=train_ids, train_y=train_y,
                 single_order=single_order, **target_arrays)
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
        preparation["target_loss_dtype"] = "float32; literal row mass multiplies lognormalizer"
        preparation["same_target_bank_shared_across_all_arms_and_seeds"] = True
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

    def fit(self, folder, seed, member, kind, factorized, members, context_mode):
        torch = self.torch
        folder.mkdir()
        self.progress(member, 0)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.synchronize()
        start = time.perf_counter()
        model, optimizer, streams = self.make(seed, member, kind, factorized, members)
        bank, best, stale = members > 1, -1., 0
        if context_mode not in ("private", "mixture", "private_full", "shuffled_private") or (not bank and context_mode != "mixture"):
            raise ValueError("Fixed private/shared or fully informed single/I4 objective required")
        target_bank = self.context_targets["bank" if bank else "single"]
        context_counts = {}
        checkpoint = folder / "selected.pt"
        with (folder / "trace.jsonl").open("w") as trace:
            for step in range(1, self.cfg.get("max_updates", 1000) + 1):
                model.train()
                optimizer.zero_grad()
                logits, hidden = self.training_outputs(model, streams, bank)
                targets = self.data.y[self.train_ids] if bank else self.data.y[self.data.train_mask]
                own_ce = self.F.cross_entropy(logits.flatten(0, 1), targets.repeat(members))
                with torch.no_grad():
                    pool_ce = self.probability_pool_ce(logits, targets) if bank else None
                auxiliary, route_losses, pair_losses, paid = self.context_objective.context_auxiliary(
                    torch, self.F, hidden, target_bank, context_mode)
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
                with torch.no_grad():
                    metrics = self.metrics(self.logits(model, streams, bank, self.valid_ids))
                improved = metrics["accuracy"] > best
                if improved:
                    best, stale, selected_step = metrics["accuracy"], 0, step
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "streams": streams,
                                "step": step, "selection": metrics,
                                "context_mode": context_mode,
                                "context_target_archive_sha256": self.context_preparation["archive_sha256"],
                                "TRAIN_H_order": "train_ids" if bank else "native_boolean_mask"}, checkpoint)
                else:
                    stale += 1
                trace.write(json.dumps({"step": step, "learning_objective": "own_CE_plus_fixed_context",
                                        "context_mode": context_mode,
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
        if (state["context_mode"] != context_mode
                or state["context_target_archive_sha256"] != self.context_preparation["archive_sha256"]
                or state["TRAIN_H_order"] != ("train_ids" if bank else "native_boolean_mask")):
            raise ValueError("Selected checkpoint target/support/order custody mismatch")
        model.load_state_dict(state["model"], strict=True)
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
        costs = {"acquisition_seconds": acquisition, "restoration_seconds": restoration_seconds,
                 "selected_forward_and_metrics_seconds": serving,
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "inference_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                 "selected_checkpoint_bytes": checkpoint.stat().st_size,
                 "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                 "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
                 "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0}
        result = {"member": member, "learning_objective": "own_CE_plus_fixed_context",
                  "context_mode": context_mode, "context_coefficient": .05, "context_temperature": .2,
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
                genuine = arm == "genuine_factorized_I4_context_mixture"
                bank = arm.startswith("shared4_")
                factorized = arm != "ordinary_M1_context_mixture"
                kind = "baseline"
                context_mode = CONTEXT_MODES[arm]
                fits = [self.fit(folder / f"member{m}", seed, m, kind, factorized, 4 if bank else 1,
                                 context_mode) for m in range(4 if genuine else 1)]
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
        comparisons = []  # Reached only after the whole seven-arm × three-seed roster.
        for result in results:
            seed, arm = result["seed"], result["arm"]
            candidate = errors[seed, arm]
            versus = {ref: {"repairs": int((errors[seed, ref] & ~candidate).sum()),
                           "harms": int((~errors[seed, ref] & candidate).sum())} for ref in ARMS}
            comparisons.append({"arm": arm, "seed": seed, "versus": versus})
        counts = {key: sum(unit["operation_counts"][key] for r in results for unit in r["fits"]) for key in results[0]["fits"][0]["operation_counts"]}
        if len(results) != 21 or self.completed_units != 30:
            raise ValueError("All21 banks/all30 independent optimizer fits must close")
        write_json(self.output / "COMPLETE_FAMILY.json", {"complete": True, "completion_scope": "fresh seven-arm acquisition only",
                   "full_comparative_family_complete": False,
                   "requires_root_admitted_own_only_anchors": {"banks": 15, "selected_fit_records": 33},
                   "groups": len(results), "fit_units": self.completed_units,
                   "expected_groups": 21, "expected_fit_units": 30, "TEST_access": False, "operation_counts": counts,
                   "config_sha256": self.config_sha256, "source_sha256": self.source_hashes,
                   "context_preparation_once": self.context_preparation,
                   "results": results, "comparisons": comparisons,
                   "cost_scope": "sequential acquisition; selected full-graph forward, metrics, CPU transfer and probability pool; RSS is process lifetime peak; overlap supplied by root"})


def main():
    raise SystemExit("Source-only successor: root qualifies/binds/freezes and invokes Family through its existing owner; no automatic scientific admission")


if __name__ == "__main__":
    main()
