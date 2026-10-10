"""Fixed retrieval extension of the existing native Family fit/selector lifecycle."""
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

ARMS = ("live_shared4", "detached_shared4", "ordinary_M1_live", "genuine_ordinary_I4_live")


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
        if cfg["backbone"] not in ("GCN", "GAT", "SAGE"):
            raise ValueError("Explicit native GCN/GAT/SAGE backbone required")
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
        self.memory = load_source("native_family_label_memory", cfg["label_memory"])
        self.process_counts = {name: 0 for name in ("fullgraph_native_member_trajectories",
            "native_graph_block_calls", "retrieval_member_paths", "similarity_MACs",
            "query_draws", "backwards", "optimizer_steps")}
        self.query_generator = None
        self.latest_native_logits = None
        self.factors = load_source("sage_family_factors", cfg["factors"])
        self.cfg, self.output = cfg, output
        self.current, self.completed_units = {"arm": None, "seed": None, "member": None, "update": 0}, 0
        source_paths = [__file__, cfg["common_routes"], cfg["factors"], cfg["label_memory"]] + [str(Path(cfg["native_repo"]) / name) for name in ("models.py", "run_base.py", "run_common.py")]
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
        write_json(self.output / "PROGRESS.json", dict(self.current, completed_units=self.completed_units,
            process_operation_counts=dict(self.process_counts)))

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

    def full_native(self, model, streams, bank):
        # Same current forward supplies both native logits and native hidden.
        members = model.members if bank else 1
        self.process_counts["fullgraph_native_member_trajectories"] += members
        self.process_counts["native_graph_block_calls"] += self.cfg.get("depth", 2) * members
        if bank:
            return model(self.batch, route_scope=lambda m: self.scope(streams, m))
        adapter = self.routes.NativeModelAdapter(self.torch, model)
        with self.scope(streams, 0):
            z, h = adapter.native(self.batch)
        return z.unsqueeze(0), h.unsqueeze(0)

    def memory_read(self, logits, hidden, queries, mode):
        support = len(self.train_ids) - len(queries) if mode == "fit" else len(self.train_ids)
        self.process_counts["retrieval_member_paths"] += len(logits)
        self.process_counts["similarity_MACs"] += len(logits) * len(queries) * support * hidden.shape[-1]
        return self.memory.label_memory(self.torch, logits, hidden, self.train_ids,
            self.data.y[self.train_ids], queries, mode=mode,
            stop_attention_gradient=self.current["arm"] == "detached_shared4")

    def logits(self, model, streams, bank, ids):
        logits, hidden = self.full_native(model, streams, bank)
        self.latest_native_logits = logits[:, ids, :]
        return self.memory_read(logits, hidden, ids, "serve")["mixed_log_probs"]

    def retrieval_train_step(self, model, optimizer, streams, bank):
        torch = self.torch
        model.train()
        optimizer.zero_grad()
        queries = self.memory.common_half_query(torch, self.train_ids,
            self.data.y[self.train_ids], 10, generator=self.query_generator)
        self.process_counts["query_draws"] += 1
        logits, hidden = self.full_native(model, streams, bank)
        result = self.memory_read(logits, hidden, queries, "fit")
        members = len(logits)
        native_ce = self.F.cross_entropy(logits[:, self.train_ids, :].flatten(0, 1),
                                         self.data.y[self.train_ids].repeat(members))
        retrieval_ce = self.F.nll_loss(result["retrieval_log_probs"].flatten(0, 1),
                                       self.data.y[queries].repeat(members))
        loss = native_ce + retrieval_ce
        loss.backward()
        self.process_counts["backwards"] += 1
        optimizer.step()
        self.process_counts["optimizer_steps"] += 1
        return {"total_ce": loss.item(), "native_ce": native_ce.item(),
                "retrieval_ce": retrieval_ce.item()}

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
        native = self.latest_native_logits.softmax(-1)
        native_correct = native.argmax(-1).eq(y)
        native_log = self.latest_native_logits.to(torch.float64).log_softmax(-1)
        native_pool_log = torch.logsumexp(native_log, dim=0) - math.log(len(native_log))
        return {"native_probability_pool_accuracy": native.mean(0).argmax(-1).eq(y).float().mean().item(),
                "native_probability_pool_nll": -native_pool_log.gather(1, y[:, None]).mean().item(),
                "native_member_accuracy": native_correct.float().mean(1).tolist(),
                "native_member_nll": [-native_log[m].gather(1, y[:, None]).mean().item() for m in range(len(native_log))],
                "accuracy": pooled_correct.float().mean().item(), "nll": nll.item(),
                "member_accuracy": correct.float().mean(1).tolist(), "member_nll": member_nll,
                "correct_alternative_count": correct.any(0).sum().item(),
                "coverage_lost_in_pooling": (correct.any(0) & ~pooled_correct).sum().item()}

    def fit(self, folder, seed, member, kind, factorized, members):
        torch = self.torch
        folder.mkdir()
        self.progress(member, 0)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.synchronize()
        start = time.perf_counter()
        counts_before = dict(self.process_counts)
        # Reset the separate common-query stream identically for every arm/member.
        self.query_generator = torch.Generator(device="cpu").manual_seed(seed + 5000011)
        model, optimizer, streams = self.make(seed, member, kind, factorized, members)
        bank, best, stale = members > 1, -1., 0
        checkpoint = folder / "selected.pt"
        with (folder / "trace.jsonl").open("w") as trace:
            for step in range(1, self.cfg.get("max_updates", 1000) + 1):
                train_losses = self.retrieval_train_step(model, optimizer, streams, bank)
                model.eval()
                with torch.no_grad():
                    metrics = self.metrics(self.logits(model, streams, bank, self.valid_ids))
                improved = metrics["accuracy"] > best
                if improved:
                    best, stale, selected_step = metrics["accuracy"], 0, step
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "streams": streams,
                                "step": step, "selection": metrics,
                                "query_rng_state": self.query_generator.get_state(),
                                "query_rng_seed": seed + 5000011}, checkpoint)
                else:
                    stale += 1
                trace.write(json.dumps({"step": step, "train_total_ce": train_losses["total_ce"],
                                        "train_native_ce": train_losses["native_ce"],
                                        "train_retrieval_ce": train_losses["retrieval_ce"], "valid": metrics,
                                        "strict_improvement": improved, "stale_updates": stale}, allow_nan=False) + "\n")
                trace.flush()
                self.progress(member, step)
                if stale >= self.cfg.get("patience_updates", 300):
                    break
        self.synchronize()
        acquisition = time.perf_counter() - start
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        streams = state["streams"]
        self.query_generator.set_state(state["query_rng_state"])
        model.eval()
        self.synchronize()
        serving_start = time.perf_counter()
        with torch.no_grad():
            selected_logits = self.logits(model, streams, bank, self.valid_ids)
            restored = self.metrics(selected_logits)
            selected_logits = selected_logits.cpu()
            selected_native_logits = self.latest_native_logits.cpu()
        self.synchronize()
        serving = time.perf_counter() - serving_start
        costs = {"acquisition_seconds": acquisition, "selected_forward_and_metrics_seconds": serving,
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "inference_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                 "selected_checkpoint_bytes": checkpoint.stat().st_size,
                 "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                 "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
                 "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0}
        result = {"member": member, "native_factor_dropout_seeds": [self.seeds(seed, member + r) for r in range(members)],
                  "selected_step": selected_step, "completed_updates": step, "selection": state["selection"], "restored": restored,
                  "selected_state": str(checkpoint), "costs": costs,
                  "operation_counts": {"updates": step, "backwards": step, "adam_steps": step,
                    "train_fullgraph_native_trajectories": members * step, "valid_selection_fullgraph_native_trajectories": members * step,
                    "selected_valid_fullgraph_native_trajectories": members,
                    "native_graph_block_calls": self.cfg.get("depth", 2) * members * (2 * step + 1)},
                  "actual_process_count_delta": {k: self.process_counts[k] - counts_before[k] for k in counts_before},
                  "query_rng_seed": seed + 5000011, "query_rng_restored_selected_step": state["step"],
                  "output_semantics": "selected_logits are mixed log probabilities; native logits retained separately"}
        write_json(folder / "RESULT.json", result)
        self.completed_units += 1
        self.progress(member, step)
        return selected_logits, result, selected_native_logits

    def run(self):
        torch, np, results, errors = self.torch, self.np, [], {}
        seeds = self.cfg["seeds"]
        if seeds != [7301, 7403, 7507]:
            raise ValueError("Fixed seeds7301/7403/7507 required")
        if self.cfg.get("hidden", 128) != 128 or self.cfg.get("max_updates", 1000) != 1000 or self.cfg.get("patience_updates", 300) != 300:
            raise ValueError("Retain width128/max1000/patience300")
        write_json(self.output / "CONFIG.json", self.cfg)
        for seed in seeds:
            for arm in ARMS:
                self.current.update(arm=arm, seed=seed)
                folder = self.output / f"{arm}_seed{seed}"
                folder.mkdir()
                genuine = arm == "genuine_ordinary_I4_live"
                bank = arm in ("live_shared4", "detached_shared4")
                factorized = bank
                kind = "baseline"
                fits = [self.fit(folder / f"member{m}", seed, m, kind, factorized, 4 if bank else 1) for m in range(4 if genuine else 1)]
                pool_start = time.perf_counter()
                logits = torch.cat([x[0] for x in fits])
                native_logits = torch.cat([x[2] for x in fits])
                self.latest_native_logits = native_logits
                metrics = self.metrics(logits)
                probs, y = logits.softmax(-1), self.valid_y.cpu()
                pool = probs.mean(0)
                member_errors, pooled_errors = probs.argmax(-1).ne(y), pool.argmax(-1).ne(y)
                pool_seconds = time.perf_counter() - pool_start
                errors[seed, arm] = pooled_errors.numpy()
                np.savez(folder / "selected_VALID.npz", ids=self.valid_ids.cpu().numpy(), y=y.numpy(),
                         mixed_log_probs=logits.numpy(), native_logits=native_logits.numpy(),
                         probability_mean=pool.numpy(), member_errors=member_errors.numpy(), pooled_errors=pooled_errors.numpy())
                units = [x[1] for x in fits]
                costs = {k: sum(u["costs"][k] for u in units) for k in units[0]["costs"] if "peak" not in k}
                costs.update({k: max(u["costs"][k] for u in units) for k in units[0]["costs"] if "peak" in k})
                costs["selected_pool_seconds"] = pool_seconds
                costs["selected_serving_readout_seconds"] = costs["selected_forward_and_metrics_seconds"] + pool_seconds
                result = {"arm": arm, "seed": seed, "serving": "probability_mean", "valid": metrics, "fits": units, "costs": costs}
                write_json(folder / "RESULT.json", result)
                results.append(result)
                print(json.dumps({"finished": arm, "seed": seed, "selected_steps": [u["selected_step"] for u in units]}), flush=True)
        comparisons = []  # Reached only after the fixed four-arm × three-seed roster.
        for result in results:
            seed, arm = result["seed"], result["arm"]
            candidate = errors[seed, arm]
            versus = {ref: {"repairs": int((errors[seed, ref] & ~candidate).sum()),
                           "harms": int((~errors[seed, ref] & candidate).sum())} for ref in ARMS[:6]}
            comparisons.append({"arm": arm, "seed": seed, "versus": versus})
        counts = {key: sum(unit["operation_counts"][key] for r in results for unit in r["fits"]) for key in results[0]["fits"][0]["operation_counts"]}
        write_json(self.output / "COMPLETE_FAMILY.json", {"complete": True, "groups": len(results), "fit_units": self.completed_units,
                   "expected_groups": 12, "expected_fit_units": 21, "TEST_access": False, "operation_counts": counts,
                   "actual_process_operation_counts": dict(self.process_counts),
                   "closed_native_references": "root binds/reuses closed unchangedshared4 and raw native M1/I4 archives; no old-baseline rerun",
                   "method_ancestry": "Matching Networks/NCA family supervised metric retrieval; TPN/UniMP label-context ancestry",
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
