"""Fixed private-bootstrap GAT/SAGE screen; copied native trainer, no owner."""
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

ARMS = ("shared4_coherent_bootstrap", "shared4_paired_graph_bootstrap", "shared4_rank1_lora_graph_bootstrap")
STARTS = dict(zip(ARMS, ("coherent", "paired_graph", "rank1_lora_graph")))
UNWEIGHTED_COUNTERPARTS = ("shared4_coherent", "shared4_paired_graph", "shared4_rank1_lora_graph")
ORIGINAL_REFERENCES = ("ordinary_M1", "ordinary_genuine_I4", "factorized_allmap_M1", "factorized_allmap_genuine_I4", "shared4_unchanged")
CORRECTIONS_SOURCE = Path(__file__).resolve().parents[1] / "common_wrapper_paired_graph_native_family_source_20261010_v1/corrections.py"
CORRECTIONS_SHA256 = "58dc7ea7f15f3c0effd286e995503abb2606dab42222522985d102eceff63cc7"


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
        if cfg["backbone"] not in ("GAT", "SAGE"):
            raise ValueError("Explicit native GAT/SAGE backbone required")
        if cfg["seeds"] != [7301, 7403, 7507] or cfg.get("bootstrap_seed_offset", 6000119) != 6000119:
            raise ValueError("Fixed paired seed order and bootstrap offset required")
        if hashlib.sha256(CORRECTIONS_SOURCE.read_bytes()).hexdigest() != CORRECTIONS_SHA256:
            raise ValueError("Immutable existing graph correction source required")
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
        self.corrections = load_source("paired_graph_family_maps", CORRECTIONS_SOURCE)
        self.bootstrap = load_source("private_bootstrap_family_gradients", Path(__file__).with_name("bootstrap.py"))
        self.cfg, self.output = cfg, output
        self.current, self.completed_units = {"arm": None, "seed": None, "member": None, "update": 0}, 0
        source_paths = [__file__, str(Path(__file__).with_name("bootstrap.py")), str(CORRECTIONS_SOURCE), cfg["common_routes"], cfg["factors"]] + [str(Path(cfg["native_repo"]) / name) for name in ("models.py", "run_base.py", "run_common.py")]
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
        self.train_ids_np, self.train_y_np = train_ids.copy(), train_y.copy()
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

    def make(self, seed, member, kind, factorized, members):
        torch, c = self.torch, self.cfg
        native_seed, _, _ = self.seeds(seed, member)
        self.set_seed(native_seed)
        body = self.Model(c["backbone"], c.get("depth", 2), 300, c.get("hidden", 128), 10,
                          1, 8, "LayerNorm", c.get("dropout", .2))
        wrapper = self.routes.wrap_shared(torch, self.factors, body, self.routes.NativeModelAdapter(torch, body),
                    members=members, kind="baseline", rank=c.get("rank", 16), block_seed=seed + c.get("block_seed_offset", 4000037)) if factorized else None
        if factorized and kind == "rademacher":
            # Row-wise seeds match each factorized I4 body and the shared bank.
            with torch.no_grad():
                for row in range(members):
                    _, factor_seed, _ = self.seeds(seed, member + row)
                    gen = torch.Generator(device="cpu").manual_seed(factor_seed)
                    signs = torch.randint(0, 2, (1, 300), generator=gen) * 2 - 1
                    body.input_linear.r[row].copy_(signs[0].to(body.input_linear.r))
            # FactorLinear construction fixes every other r/s row to one.
        correction_sites = self.corrections.install_corrections(torch, self.factors, body,
            backbone=c["backbone"], kind=kind, members=members, seed=seed, cfg=c) if kind in ("paired_graph", "rank1_lora_graph", "paired_local") else []
        model = wrapper if members > 1 else body
        model._correction_sites = correction_sites
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
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Nonfinite selected or evaluation logits")
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

    def prepare_bootstrap(self, seed):
        np, torch = self.np, self.torch
        folder = self.output / f"bootstrap_seed{seed}"
        folder.mkdir()
        self.synchronize()
        start = time.perf_counter()
        raw, weights, record = self.bootstrap.draw_weights(np, seed, self.train_ids_np, self.train_y_np)
        archive = folder / "WEIGHTS.npz"
        np.savez(archive, ids=self.train_ids_np, y=self.train_y_np, raw_exponential=raw, weights=weights)
        tensor = torch.from_numpy(weights.copy()).to(self.device)  # Float64, untrained, exact saved weights.
        if tensor.requires_grad or tensor.dtype != torch.float64 or not (torch.isfinite(tensor) & (tensor > 0)).all().item():
            raise FloatingPointError("Saved positive float64 weights must survive unchanged device transfer")
        self.synchronize()
        record.update(family_seed=seed, archive=str(archive), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                      raw_exponential_sha256=hashlib.sha256(raw.tobytes()).hexdigest(), weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
                      ordered_TRAIN_ids_sha256=hashlib.sha256(self.train_ids_np.tobytes()).hexdigest(),
                      ordered_TRAIN_labels_sha256=hashlib.sha256(self.train_y_np.tobytes()).hexdigest(),
                      shared_across_arms=list(ARMS), setup_seconds_to_archive_and_device_tensor=time.perf_counter() - start,
                      trained_parameters_added=0, TEST_access=False)
        write_json(folder / "WEIGHTS.json", record)
        return tensor, record

    def fit(self, folder, seed, member, kind, factorized, members, bootstrap_weights, bootstrap_record):
        torch = self.torch
        folder.mkdir()
        self.progress(member, 0)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.synchronize()
        start = time.perf_counter()
        model, optimizer, streams = self.make(seed, member, kind, factorized, members)
        if member != 0 or not factorized or members != 4 or kind not in STARTS.values():
            raise ValueError("Fixed shared-four bootstrap acquisition required")
        roles, shared, private = self.bootstrap.parameter_roles(torch, self.factors, model)
        write_json(folder / "PARAMETER_ROLES.json", roles)
        bank, best, stale = members > 1, -1., 0
        checkpoint = folder / "selected.pt"
        with (folder / "trace.jsonl").open("w") as trace:
            for step in range(1, self.cfg.get("max_updates", 1000) + 1):
                if bank:
                    model.train()
                    optimizer.zero_grad()
                    logits = self.logits(model, streams, True, self.train_ids)
                    train_loss, weighted_train_loss, unconnected, member_train_losses = self.bootstrap.bank_gradients(
                        torch, self.F, logits, self.data.y[self.train_ids], bootstrap_weights, shared, private)
                    optimizer.step()
                else:
                    raise ValueError("No single/independent/reacquired reference arm is admitted")
                if not math.isfinite(train_loss):
                    raise FloatingPointError("Nonfinite TRAIN loss")
                model.eval()
                with torch.no_grad():
                    metrics = self.metrics(self.logits(model, streams, bank, self.valid_ids))
                improved = metrics["accuracy"] > best
                if improved:
                    best, stale, selected_step = metrics["accuracy"], 0, step
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "streams": streams,
                                "step": step, "selection": metrics, "bootstrap_weights_sha256": bootstrap_record["weights_sha256"]}, checkpoint)
                else:
                    stale += 1
                trace.write(json.dumps({"step": step, "train_own_ce": train_loss, "valid": metrics,
                                        "train_private_weighted_ce": weighted_train_loss, "reverse_mode_calls": 2,
                                        "train_member_losses": member_train_losses,
                                        "unconnected_gradient_names": unconnected,
                                        "strict_improvement": improved, "stale_updates": stale}, allow_nan=False) + "\n")
                trace.flush()
                self.progress(member, step)
                if stale >= self.cfg.get("patience_updates", 300):
                    break
        self.synchronize()
        acquisition = time.perf_counter() - start
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["bootstrap_weights_sha256"] != bootstrap_record["weights_sha256"]:
            raise ValueError("Selected checkpoint must bind the unchanged fixed weight table")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        streams = state["streams"]
        model.eval()
        self.synchronize()
        serving_start = time.perf_counter()
        with torch.no_grad():
            selected_logits = self.logits(model, streams, bank, self.valid_ids)
            restored = self.metrics(selected_logits)
            selected_logits = selected_logits.cpu()
        self.synchronize()
        serving = time.perf_counter() - serving_start
        costs = {"acquisition_seconds": acquisition, "selected_forward_and_metrics_seconds": serving,
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "inference_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                 "selected_checkpoint_bytes": checkpoint.stat().st_size,
                 "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                 "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
                 "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0}
        extra_names = [name for name, p in model.named_parameters() if name.endswith((".u1", ".u2", ".lora_a", ".lora_b"))]
        extra_count = sum(p.numel() for name, p in model.named_parameters() if name in extra_names)
        correction_calls = len(model._correction_sites) * members * (2 * step + 1)
        result = {"start": kind, "correction_sites": model._correction_sites, "extra_private_names": extra_names,
                  "bootstrap": bootstrap_record, "parameter_roles": roles,
                  "last_train_losses": {"own_ce": train_loss, "private_weighted_ce": weighted_train_loss, "members": member_train_losses},
                  "extra_private_parameters": extra_count, "member": member, "native_factor_dropout_seeds": [self.seeds(seed, member + r) for r in range(members)],
                  "selected_step": selected_step, "completed_updates": step, "selection": state["selection"], "restored": restored,
                  "selected_state": str(checkpoint), "costs": costs,
                  "operation_counts": {"updates": step, "backwards": 2 * step, "reverse_mode_calls": 2 * step,
                    "shared_own_ce_reverse_mode_calls": step, "private_weighted_ce_reverse_mode_calls": step, "adam_steps": step,
                    "train_fullgraph_native_trajectories": members * step, "valid_selection_fullgraph_native_trajectories": members * step,
                    "selected_valid_fullgraph_native_trajectories": members,
                    "native_graph_block_calls": self.cfg.get("depth", 2) * members * (2 * step + 1),
                    "fullgraph_corrected_dense_map_calls": correction_calls,
                    "Householder_reflection_calls": 2 * correction_calls if kind.startswith("paired") else 0,
                    "rank1_additive_correction_calls": correction_calls if kind == "rank1_lora_graph" else 0}}
        write_json(folder / "RESULT.json", result)
        self.completed_units += 1
        self.progress(member, step)
        return selected_logits, result

    def run(self):
        torch, np, results, errors = self.torch, self.np, [], {}
        seeds = self.cfg["seeds"]
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError("Complete three-seed roster required")
        selected_arms = ARMS
        expected_groups = 3 * len(selected_arms)
        expected_units = 3 * sum(4 if "genuine_I4" in arm else 1 for arm in selected_arms)
        write_json(self.output / "CONFIG.json", self.cfg)
        bootstrap_records = []
        for seed in seeds:
            bootstrap_weights, bootstrap_record = self.prepare_bootstrap(seed)
            bootstrap_records.append(bootstrap_record)
            for arm in selected_arms:
                self.current.update(arm=arm, seed=seed)
                folder = self.output / f"{arm}_seed{seed}"
                folder.mkdir()
                kind = STARTS[arm]
                fits = [self.fit(folder / "member0", seed, 0, kind, True, 4, bootstrap_weights, bootstrap_record)]
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
        comparisons = []  # Reached only after every declared acquisition in all three seeds.
        for result in results:
            seed, arm = result["seed"], result["arm"]
            candidate = errors[seed, arm]
            versus = {ref: {"repairs": int((errors[seed, ref] & ~candidate).sum()),
                           "harms": int((~errors[seed, ref] & candidate).sum())} for ref in selected_arms}
            comparisons.append({"arm": arm, "seed": seed, "versus": versus})
        counts = {key: sum(unit["operation_counts"][key] for r in results for unit in r["fits"]) for key in results[0]["fits"][0]["operation_counts"]}
        if len(results) != expected_groups or self.completed_units != expected_units:
            raise RuntimeError("Declared full acquisition roster did not complete")
        write_json(self.output / "COMPLETE_FAMILY.json", {"complete": True, "groups": len(results), "fit_units": self.completed_units,
                   "expected_groups": expected_groups, "expected_fit_units": expected_units,
                   "backbone": self.cfg["backbone"], "declared_acquisition_arms": list(selected_arms),
                   "full_comparative_roster": list(ORIGINAL_REFERENCES + UNWEIGHTED_COUNTERPARTS + ARMS),
                   "full_comparative_family_complete": False, "reused_original_reference_arms": list(ORIGINAL_REFERENCES),
                   "reused_unweighted_counterparts": list(UNWEIGHTED_COUNTERPARTS), "bootstrap_weight_records": bootstrap_records,
                   "block_gradient_law": "shared=partial mean ownCE; private=partial mean(weight*ownCE); same forward; generally nonconservative",
                   "bootstrap_setup_seconds_sum": sum(record["setup_seconds_to_archive_and_device_tensor"] for record in bootstrap_records),
                   "TEST_access": False, "operation_counts": counts,
                   "config_sha256": self.config_sha256, "source_sha256": self.source_hashes, "results": results, "comparisons": comparisons,
                   "cost_scope": "sequential acquisition; selected full-graph forward, metrics, CPU transfer and probability pool; RSS is process lifetime peak; overlap supplied by root"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config_bytes = Path(args.config).read_bytes()
    Family(json.loads(config_bytes), output, hashlib.sha256(config_bytes).hexdigest()).run()


if __name__ == "__main__":
    main()
