"""Source-only fitted screen. No model import/draw occurs outside future main.

Execution requires a separate root scientific freeze, root admission, external
operation supervision and matching packet/dependency hashes. None exists here
by default. Source preparation is not permission to execute this file.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
from types import ModuleType, SimpleNamespace

BASE = Path(__file__).resolve().parent
RESEARCH = BASE.parents[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_file(name, path):
    if "__ar_instrumenter__" in globals():
        return __ar_instrumenter__.load_file(name, path)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Pinned module unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def usage():
    value = resource.getrusage(resource.RUSAGE_SELF)
    return {"user_cpu_seconds": value.ru_utime, "system_cpu_seconds": value.ru_stime,
            "maxrss_raw": value.ru_maxrss, "maxrss_unit": "bytes" if sys.platform == "darwin" else "KiB_on_Linux; native_raw_otherwise",
            "memory_scope": "Process cumulative high-water mark, including imports/tables/models; not reset per arm"}


class Recorder:
    def __init__(self, output, freeze, freeze_sha, admission):
        self.output, self.index = output, 0
        self.data = {"status": "started", "root_freeze_sha256": freeze_sha, "root_admission": admission,
            "train_instances": [], "development_instances": [], "oracle_files": [], "fits": [], "evaluations": [], "classical": [],
            "pre_fit_identity_gates": [{"arm": arm["name"], "model_seed": freeze["model_seeds"][0], "status": "pending"}
                for arm in freeze["arms_in_order"] if arm["fixed_factor_names"]],
            "draw_gate": "Development generation allowed only after every planned fit has its final checkpoint",
            "scope": "Prospective bounded adaptation screen; no original-score recalculation, HPO or checkpoint selection"}
        for fit in freeze["fit_order"]:
            self.data["fits"].append({**fit, "status": "pending", "completed_updates": 0})
            for seed in freeze["development_topology_seeds"]:
                for fugacity in freeze["fugacities_in_order"]:
                    self.data["evaluations"].append({**fit, "graph_seed": seed, "fugacity": fugacity, "status": "pending"})
        for seed in freeze["development_topology_seeds"]:
            for fugacity in freeze["fugacities_in_order"]:
                self.data["classical"].append({"graph_seed": seed, "fugacity": fugacity, "status": "pending"})
        self.save()

    def save(self):
        self.data["latest_usage"] = usage()
        if "__ar_diagnostic_meter__" in globals():
            self.data["diagnostic_charged_union_seconds"] = __ar_diagnostic_meter__.charged
        temp = self.output / "progress.json.tmp"
        temp.write_text(json.dumps(self.data, indent=2, allow_nan=False) + "\n")
        temp.replace(self.output / "progress.json")

    def event(self, row):
        with (self.output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    def measure(self, name, function):
        index, start, cpu = self.index, time.perf_counter_ns(), time.process_time_ns()
        self.index += 1
        self.event({"event": "started", "index": index, "name": name, "utc": datetime.now(timezone.utc).isoformat(), "usage": usage()})
        try:
            explicit = name.startswith(("final_actual_H_scalar_batched_parity/", "rational_oracle_equality/", "pre_fit_identity_tiny_gate/"))
            context = __ar_diagnostic_meter__.block("explicit/" + name) if explicit else nullcontext()
            with context:
                result = function()
            __ar_diagnostic_meter__.check_budget()
        except BaseException:
            self.event({"event": "failed_or_interrupted", "index": index, "name": name,
                "wall_seconds": (time.perf_counter_ns() - start) / 1e9, "process_cpu_seconds": (time.process_time_ns() - cpu) / 1e9, "usage": usage()})
            raise
        self.event({"event": "completed", "index": index, "name": name,
            "wall_seconds": (time.perf_counter_ns() - start) / 1e9, "process_cpu_seconds": (time.process_time_ns() - cpu) / 1e9, "usage": usage()})
        return result


def verify_contract(freeze_path, expected_sha, admission_path):
    require(sha(freeze_path) == expected_sha, "Root scientific freeze hash mismatch")
    freeze = json.loads(freeze_path.read_text())
    admission = json.loads(admission_path.read_text())
    require(freeze["status"] == "scientifically_frozen_by_root", "Prospective draft is not a root scientific freeze")
    require(admission["status"] == "admitted_by_root" and admission["scientific_freeze_sha256"] == expected_sha,
            "Matching root execution admission required")
    manifest = json.loads((BASE / "AUTHORING_MANIFEST.json").read_text())
    require(admission["authoring_manifest_sha256"] == sha(BASE / "AUTHORING_MANIFEST.json"), "Admission did not bind this packet")
    for row in manifest["files"]:
        path = BASE / row["path"]
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Authored bytes changed: " + row["path"])
    pins = json.loads((BASE / "SOURCE_PINS.json").read_text())
    for row in pins["files"]:
        path = (RESEARCH / row["research_relative_path"]).resolve()
        require(path.is_relative_to(RESEARCH.resolve()) and sha(path) == row["sha256"], "Pinned dependency changed")
    require(freeze["runtime"]["dtype"] == "float64" and freeze["runtime"]["device"] == "cpu", "Qualified CPU float64 runtime required")
    require(freeze["baseline"] == 0 and freeze["samples_per_update"] == 8 and freeze["sampling_design"] == "stratified_per_branch", "Qualified scalar score path required")
    require(freeze["evaluation"]["normalization_atol"] == 1e-10 and freeze["evaluation"]["parity_atol"] == 1e-10,
            "Frozen normalization/parity tolerances required")
    arms = freeze["arms_in_order"]
    names = [arm["name"] for arm in arms]
    require(len(names) == len(set(names)) == 8, "Complete root eight-arm matrix required")
    planned = [(row["arm"], row["model_seed"]) for row in freeze["fit_order"]]
    require(len(planned) == len(set(planned)) and set(planned) == {(name, seed) for name in names for seed in freeze["model_seeds"]},
            "Every frozen arm/model seed must remain in fit matrix")
    train, dev = freeze["training_topology_seeds"], freeze["development_topology_seeds"]
    require(len(train) == len(set(train)) == 8 and len(dev) == len(set(dev)) == 8 and not set(train) & set(dev), "Distinct complete topology split required")
    require(freeze["fugacities_in_order"] == [4, 1] and freeze["updates_per_fit"] == 64, "Root prospective regimes/update count changed")
    cycle = [(row["graph_seed"], row["fugacity"]) for row in freeze["training_cycle"]]
    require(len(cycle) == 16 and set(cycle) == {(seed, fugacity) for seed in train for fugacity in [4, 1]}, "Complete balanced training cycle required")
    return freeze, admission, pins


def runtime_imports():
    import torch
    import torch_geometric
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    package = ModuleType("new_learning_gap_independent_v1")
    package.__path__ = [str(BASE.parent)]
    sys.modules[package.__name__] = package
    paths = {"domain": ".ar_mixture_source_v1.domain", "encoder": ".ar_mixture_source_v1.source_encoder",
             "work": ".ar_mixture_source_v1.work_accounting", "private": ".ar_mixture_source_v1.ar_decoder",
             "objective": ".ar_mixture_source_v1.mixture_objective", "shared": ".shared_transition_heads_source_v1.shared_head_decoder",
             "shared_work": ".shared_transition_heads_source_v1.work_accounting", "rational": ".finite_cache_source_v1.hard_core_reference"}
    modules = {name: importlib.import_module(package.__name__ + path) for name, path in paths.items()}
    Model = load_file("fitted_pinned_source_model", RESEARCH / "source_snapshot/models.py").Model
    spatial = load_file("fitted_pinned_spatial15", RESEARCH / "protocols/ar_qualification_root_v1/spatial_family_source_v2.py")
    evaluator = load_file("fitted_pinned_batched_evaluator", BASE.parent / "cpu_workload_qualifier_source_v3/batched_evaluation.py")
    factory = load_file("fitted_system_factory", BASE / "system_factory.py")
    metrics = load_file("fitted_complete_law", BASE / "complete_law.py")
    classical = load_file("fitted_classical_controls", BASE / "classical_controls.py")
    summary = load_file("fitted_summary", BASE / "summarize.py")
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, "Qualified thread pin failed")
    return torch, torch_geometric.__version__, modules, Model, spatial, evaluator, factory, metrics, classical, summary


def graph_case(torch, modules, instance, fugacity):
    domain = modules["domain"].ScheduleDomain(15, instance.edges, instance.order, instance.instance_id, "spatial-v2-separate-order-stream")
    edges = [pair for u, v in instance.edges for pair in ((u, v), (v, u))]
    graph = SimpleNamespace(edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous())
    features = torch.full((15, 1), math.log(fugacity))
    log_lambda = torch.full((15,), math.log(fugacity))
    return graph, features, domain, log_lambda


def graph_audit(modules, instance):
    return modules["work"].EncoderAudit("fitted-pinned-gcn-2x16-none", modules["encoder"].AUTHORIZED_SOURCE_SHA256,
        "pinned-GCN/add_self_loops=False/Identity/dropout0", "None", 16, 2, 4 * len(instance.edges), 15 * 2832)


def encode_case(encoder, ledger, modules, instance, case):
    # Only work metadata changes with E; the source module and recipe do not.
    ledger.audit = graph_audit(modules, instance)
    encoder.audit = ledger.audit
    graph, features, domain, _ = case
    return encoder(graph, features, domain, graph_id=domain.graph_id)


def checkpoint_parameters(torch, encoder, decoder, path):
    state = {"encoder": {n: p.detach().clone() for n, p in encoder.core.state_dict().items()},
             "decoder": {n: p.detach().clone() for n, p in decoder.state_dict().items()}}
    torch.save(state, path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def pre_fit_identity_gates(torch, modules, evaluator, factory, metrics, freeze, recorder):
    audit = modules["work"].EncoderAudit("pre-fit-identity-proxy", modules["encoder"].AUTHORIZED_SOURCE_SHA256,
        "arithmetic-proxy-no-source-encoding", "None", 16, 0, 0, 0)
    for row in recorder.data["pre_fit_identity_gates"]:
        arm = next(arm for arm in freeze["arms_in_order"] if arm["name"] == row["arm"])
        row["status"] = "checking"
        recorder.save()
        def gate():
            template = factory.decoder_template(torch, arm, row["model_seed"], freeze["private_initialization"])
            decoder, ledger = factory.construct_decoder(torch, modules, arm, template, audit)
            _, inventory = factory.parameter_inventory(torch, torch.nn.Module(), decoder)
            require(inventory["stored_elements"] == arm["expected_decoder_stored_elements"] and
                    inventory["trainable_elements"] == arm["expected_decoder_trainable_elements"], "Tiny identity raw counts changed")
            factory.assert_fixed_identity(torch, decoder, arm)
            parity = metrics.tiny_identity_parity(torch=torch, modules=modules, evaluator=evaluator,
                decoder=decoder, arm=arm, evaluation=freeze["evaluation"])
            return {"parameters": inventory, "parity": parity, "scalar_source_counts": dict(ledger.counts)}
        row.update(recorder.measure("pre_fit_identity_tiny_gate/" + arm["name"], gate))
        row["status"] = "completed"
        recorder.save()


def fit_one(torch, modules, Model, factory, freeze, recorder, row, instances):
    arm = next(arm for arm in freeze["arms_in_order"] if arm["name"] == row["arm"])
    seed = row["model_seed"]
    label = arm["name"] + "/seed" + str(seed)
    row["status"] = "fitting"
    recorder.save()
    encoder_state = recorder.measure("initialize_encoder/" + label, lambda: factory.initial_encoder_state(torch, Model, seed, freeze["source_encoder"]))
    template = recorder.measure("initialize_decoder/" + label, lambda: factory.decoder_template(torch, arm, seed, freeze["private_initialization"]))
    system = recorder.measure("construct_fit/" + label, lambda: factory.construct_system(torch, modules, Model, arm, seed,
        encoder_state, template, graph_audit(modules, instances[freeze["training_topology_seeds"][0]]), freeze["source_encoder"]))
    encoder, decoder, ledger, parameters, inventory = system
    row["parameters"] = inventory
    row["private_initialization_table_sha256"] = {name: hashlib.sha256(json.dumps(template[name].tolist(), separators=(",", ":")).encode()).hexdigest()
        for name in ("b_x", "b_h", "b_o", "h0")}
    trainable = [parameter for _, parameter in parameters if parameter.requires_grad]
    opt = freeze["optimizer"]
    optimizer = recorder.measure("optimizer_construction/" + label, lambda: torch.optim.AdamW(trainable,
        lr=opt["learning_rate"], betas=tuple(opt["betas"]), eps=opt["epsilon"], weight_decay=opt["weight_decay"],
        amsgrad=opt["amsgrad"], maximize=False, foreach=False, capturable=False, differentiable=False, fused=False))
    mixture = modules["domain"].FixedMixture((1.0,) * arm["members"])
    baseline = modules["domain"].IndependentBaseline((0.0,) * arm["members"], "predeclared_constant", "root-frozen-zero-baseline", True)
    branches = [m for m in range(arm["members"]) for _ in range(freeze["samples_per_update"] // arm["members"])]
    require(len(branches) == freeze["samples_per_update"], "Complete stratified schedule batch required")
    for step in range(freeze["updates_per_fit"]):
        choice = freeze["training_cycle"][step % len(freeze["training_cycle"])]
        instance = instances[choice["graph_seed"]]
        fugacity = choice["fugacity"]
        case = graph_case(torch, modules, instance, fugacity)
        suffix = label + "/update" + str(step + 1)
        uniforms = recorder.measure("sampling_uniforms/" + suffix, lambda: torch.rand((8, 15),
            generator=factory.generator(torch, seed, "training_schedule_uniforms/update" + str(step + 1))).tolist())
        encoded = recorder.measure("source_encoding_train/" + suffix, lambda: encode_case(encoder, ledger, modules, instance, case))
        samples = recorder.measure("scalar_complete_generation/" + suffix, lambda: tuple(decoder.generate(encoded, branch=m, uniforms=uniforms[i])
            for i, m in enumerate(branches)))
        cross = recorder.measure("scalar_cross_member_scores/" + suffix, lambda: decoder.cross_member_log_densities(encoded, samples))
        result = recorder.measure("qualified_score_surrogate/" + suffix, lambda: modules["objective"].fixed_mixture_reverse_kl_surrogate(
            torch_api=torch, encoded=encoded, samples=samples, cross_log_q=cross, mixture=mixture, log_fugacities=case[3], baseline=baseline,
            sampling_design=freeze["sampling_design"], objective_kind=arm["objective"], ledger=ledger))
        optimizer.zero_grad(set_to_none=True)
        recorder.measure("completed_backward/" + suffix, result.surrogate.backward)
        require(all(p.grad is None or torch.isfinite(p.grad).all().item() for _, p in parameters), "Nonfinite source gradient")
        recorder.measure("completed_AdamW_step/" + suffix, optimizer.step)
        require(all(torch.isfinite(p).all().item() for _, p in parameters), "Nonfinite updated parameter")
        factory.assert_fixed_identity(torch, decoder, arm)
        row.update(completed_updates=step + 1, latest_training_graph_seed=instance.seed, latest_fugacity=fugacity,
            source_counts=dict(ledger.counts), complete_training_schedules=(step + 1) * 8,
            uniform_values_generated=(step + 1) * 120, backward_calls_completed=step + 1, optimizer_steps_completed=step + 1)
        recorder.event({"event": "update_completed", "fit": label, "update": step + 1,
            "score_objective_without_log_partition": result.objective_without_log_partition.item(), "source_counts": dict(ledger.counts)})
        recorder.save()
        del result, cross, samples, encoded
    path = recorder.output / "checkpoints" / (arm["name"] + "_seed" + str(seed) + "_final.pt")
    row["final_checkpoint"] = recorder.measure("save_final_checkpoint/" + label, lambda: checkpoint_parameters(torch, encoder, decoder, path))
    row.update(status="completed_final_checkpoint", checkpoint_update=freeze["updates_per_fit"], optimizer_state_entries=len(optimizer.state),
        optimizer_state_tensor_elements=sum(v.numel() for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor)))
    recorder.save()


def oracle_check(modules, instance, fugacity, states, partition):
    graph = modules["rational"].CompleteConflictGraph(15, instance.edges, (Fraction(fugacity, 1),) * 15)
    rational = modules["rational"].ExactHardCoreLaw.enumerate(graph, modules["rational"].EnumerationBudget(32768, 64))
    require(partition == rational.partition and tuple((m, Fraction(w)) for m, w in states) == rational.weighted_states,
            "Complete integer/rational oracle mismatch")
    return states, partition


def evaluate_fit(torch, modules, Model, evaluator, factory, metrics, freeze, recorder, fit, development, targets):
    arm = next(arm for arm in freeze["arms_in_order"] if arm["name"] == fit["arm"])
    seed = fit["model_seed"]
    label = arm["name"] + "/seed" + str(seed)
    checkpoint = Path(fit["final_checkpoint"]["path"])
    require(sha(checkpoint) == fit["final_checkpoint"]["sha256"], "Final checkpoint changed")
    state = recorder.measure("load_final_checkpoint/" + label, lambda: torch.load(checkpoint, map_location="cpu"))
    # Explicit fit initialization is not repeated. The fresh source constructor
    # still consumes temporary default RNG work before checkpoint replacement;
    # that construction is timed and charged, and does not change saved values.
    encoder, decoder, ledger, parameters, inventory = recorder.measure("construct_evaluation_model/" + label,
        lambda: factory.construct_system(torch, modules, Model, arm, seed, state["encoder"], state["decoder"],
            graph_audit(modules, development[freeze["development_topology_seeds"][0]]), freeze["source_encoder"]))
    encoder.eval()
    decoder.eval()
    for row in recorder.data["evaluations"]:
        if (row["arm"], row["model_seed"]) != (fit["arm"], seed):
            continue
        row["status"] = "evaluating"
        recorder.save()
        instance, fugacity = development[row["graph_seed"]], row["fugacity"]
        case = graph_case(torch, modules, instance, fugacity)
        weighted, partition = targets[(instance.seed, fugacity)]
        suffix = label + "/dev" + str(instance.seed) + "/lambda" + str(fugacity)
        with torch.no_grad():
            encoded = recorder.measure("source_encoding_evaluation/" + suffix, lambda: encode_case(encoder, ledger, modules, instance, case))
            if instance.seed in freeze["evaluation"]["parity_development_topology_seeds"]:
                row["parity"] = recorder.measure("final_actual_H_scalar_batched_parity/" + suffix, lambda: metrics.actual_H_parity(
                    torch=torch, evaluator=evaluator, decoder=decoder, encoded=encoded, weighted_states=weighted, arm=arm, evaluation=freeze["evaluation"]))
            else:
                row["parity"] = {"status": "not_scheduled_on_this_topology", "rule": "Root-frozen topology subset for implementation parity; complete-law support remains exhaustive"}
            def chunk(progress):
                row["complete_support_progress"] = progress
                # Append-only chunk retention avoids rewriting the growing
                # complete matrix once for every 512-row chunk.
                recorder.event({"event": "complete_law_chunk", "cell": suffix, **progress})
                __ar_diagnostic_meter__.check_budget()
            row["whole_law"] = recorder.measure("complete_law/" + suffix, lambda: metrics.evaluate_complete_law(torch=torch,
                evaluator=evaluator, decoder=decoder, encoded=encoded, weighted_states=weighted, partition=partition,
                arm=arm, evaluation=freeze["evaluation"], on_chunk=chunk))
            count = freeze["deployment_draws"]
            uniform_role = "deployment_uniforms/dev" + str(instance.seed) + "/lambda" + str(fugacity)
            branch_role = "deployment_branches/dev" + str(instance.seed) + "/lambda" + str(fugacity)
            uniforms = recorder.measure("deployment_uniforms/" + suffix, lambda: torch.rand((count, 15), generator=factory.generator(torch, seed, uniform_role)).tolist())
            branches = recorder.measure("deployment_branch_draws/" + suffix, lambda: torch.randint(arm["members"], (count,), generator=factory.generator(torch, seed, branch_role)).tolist())
            cold_H = recorder.measure("deployment_cold_graph_encoding/" + suffix, lambda: encode_case(encoder, ledger, modules, instance, case))
            draw_records = recorder.measure("deployment_warm_complete_generation/" + suffix, lambda: [decoder.generate(cold_H, branch=m, uniforms=uniforms[i])
                for i, m in enumerate(branches)])
            row["deployment"] = {"iid_uniform_mixture_branches": branches, "complete_assignments": [list(sample.assignment) for sample in draw_records],
                "uniform_values_generated": count * 15, "complete_draws": count,
                "timing_scope": "RNG, one cold graph encoding and warm generation measured separately; no quality-selected draws"}
        factory.assert_fixed_identity(torch, decoder, arm)
        row.update(status="completed", checkpoint_sha256=fit["final_checkpoint"]["sha256"], checkpoint_update=fit["checkpoint_update"],
            source_counts_cumulative=dict(ledger.counts), parameters=inventory)
        recorder.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-freeze", type=Path, required=True)
    parser.add_argument("--root-freeze-sha256", required=True)
    parser.add_argument("--root-admission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Read-only contract/pin verification precedes all model imports and draws.
    freeze, admission, pins = verify_contract(args.root_freeze, args.root_freeze_sha256, args.root_admission)
    output = args.output_dir.resolve()
    require(output.is_relative_to(BASE) and output != BASE and output.is_dir(), "Precreated fresh output directory inside packet required")
    require(not (output / "progress.json").exists(), "No overwrite or resume of a prior screen")
    (output / "checkpoints").mkdir()
    (output / "oracles").mkdir()
    recorder = Recorder(output, freeze, args.root_freeze_sha256, admission)
    start = time.perf_counter_ns()
    def timeout(signum, frame):
        raise TimeoutError("Root-frozen whole-operation child alarm; retain incomplete family without retry")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(freeze["runtime"]["internal_alarm_seconds"])
    try:
        torch, pyg, modules, Model, spatial, evaluator, factory, metrics, classical, summary = recorder.measure("runtime_imports", runtime_imports)
        recorder.data["runtime"] = {"python": sys.version, "torch": torch.__version__, "torch_geometric": pyg,
            "CPU_float64_one_thread": True, "source_pins": pins}
        pre_fit_identity_gates(torch, modules, evaluator, factory, metrics, freeze, recorder)
        training = {}
        for seed in freeze["training_topology_seeds"]:
            instance = recorder.measure("generate_training_topology/" + str(seed), lambda seed=seed: spatial.make_complete_spatial_instance(
                instance_id="train_spatial15_seed" + str(seed), seed=seed))
            training[seed] = instance
            recorder.data["train_instances"].append(instance.record())
            recorder.save()
        for fit in recorder.data["fits"]:
            recorder.measure("entire_fit/" + fit["arm"] + "/" + str(fit["model_seed"]), lambda fit=fit: fit_one(torch, modules, Model, factory, freeze, recorder, fit, training))
        require(all(fit["status"] == "completed_final_checkpoint" and fit["completed_updates"] == freeze["updates_per_fit"] for fit in recorder.data["fits"]),
                "All frozen fits/final checkpoints must complete before development draws")
        recorder.data["all_final_checkpoints_complete_before_development"] = True
        recorder.save()
        development, targets = {}, {}
        for seed in freeze["development_topology_seeds"]:
            instance = recorder.measure("generate_untouched_development_topology/" + str(seed), lambda seed=seed: spatial.make_complete_spatial_instance(
                instance_id="development_spatial15_seed" + str(seed), seed=seed))
            development[seed] = instance
            recorder.data["development_instances"].append(instance.record())
            recorder.save()
        for row in recorder.data["classical"]:
            row["status"] = "building_classical_controls"
            recorder.save()
            instance, fugacity = development[row["graph_seed"]], row["fugacity"]
            suffix = str(instance.seed) + "/lambda" + str(fugacity)
            table = recorder.measure("cold_exact_enumeration_and_ticket_build/" + suffix, lambda: classical.exact_build(spatial, instance, fugacity))
            targets[(instance.seed, fugacity)] = recorder.measure("rational_oracle_equality/" + suffix,
                lambda: oracle_check(modules, instance, fugacity, table[0], table[1]))
            target_path = output / "oracles" / ("seed" + str(instance.seed) + "_lambda" + str(fugacity) + ".json")
            def save_target():
                target_path.write_text(json.dumps({"graph_seed": instance.seed, "fugacity": fugacity, "states_considered": 32768,
                    "weighted_states": table[0], "partition": table[1]}, separators=(",", ":")) + "\n")
                return {"path": str(target_path), "sha256": sha(target_path), "bytes": target_path.stat().st_size}
            recorder.data["oracle_files"].append(recorder.measure("retain_complete_integer_oracle/" + suffix, save_target))
            row["exact_draws"] = recorder.measure("exact_ticket_draws/" + suffix, lambda: classical.exact_draws(instance, fugacity, table, freeze["classical_draws"]))
            iid_histogram, iid_record = recorder.measure("matched_exact_IID_histogram_build/" + suffix,
                lambda: classical.exact_iid_histogram(instance, fugacity, table, freeze["exact_iid_histogram_draws"]))
            row["exact_iid_histogram"] = sorted(iid_histogram.items())
            row["exact_iid_record"] = iid_record
            row["exact_iid_whole_cache_law"] = recorder.measure("matched_exact_IID_complete_cache_metrics/" + suffix,
                lambda: classical.empirical_metrics(iid_histogram, targets[(instance.seed, fugacity)][0], targets[(instance.seed, fugacity)][1]))
            row["exact_iid_cache_draws"] = recorder.measure("matched_exact_IID_cache_ticket_draws/" + suffix,
                lambda: classical.cache_draws(instance, fugacity, iid_histogram, freeze["classical_draws"], method="exact_iid_cache"))
            def chain(record):
                recorder.event({"event": "gibbs_chain_completed", "graph_seed": instance.seed, "fugacity": fugacity, **record})
            histogram, record = recorder.measure("gibbs_chain_and_cache_build/" + suffix, lambda: classical.gibbs_build(instance, fugacity, freeze["gibbs"], chain))
            row["gibbs"] = record
            row["gibbs_histogram"] = sorted(histogram.items())
            row["gibbs_whole_cache_law"] = recorder.measure("gibbs_complete_cache_metrics/" + suffix,
                lambda: classical.empirical_metrics(histogram, targets[(instance.seed, fugacity)][0], targets[(instance.seed, fugacity)][1]))
            row["cache_draws"] = recorder.measure("gibbs_cache_ticket_draws/" + suffix, lambda: classical.cache_draws(instance, fugacity, histogram, freeze["classical_draws"]))
            row["status"] = "completed"
            recorder.save()
        for fit in recorder.data["fits"]:
            recorder.measure("entire_final_model_evaluation/" + fit["arm"] + "/" + str(fit["model_seed"]),
                lambda fit=fit: evaluate_fit(torch, modules, Model, evaluator, factory, metrics, freeze, recorder, fit, development, targets))
        require(all(row["status"] == "completed" for row in recorder.data["evaluations"] + recorder.data["classical"]), "Final whole family incomplete")
        recorder.data["status"] = "completed_frozen_fitted_screen"
    except BaseException as error:
        recorder.data.update(status="failed_or_incomplete", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        signal.alarm(0)
        recorder.data["child_wall_seconds_including_imports"] = (time.perf_counter_ns() - start) / 1e9
        recorder.save()
        if "summary" in locals():
            (output / "summary.json").write_text(json.dumps(summary.complete_summary(recorder.data, freeze), indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": recorder.data["status"], "output": str(output)}), flush=True)
    return 0 if recorder.data["status"] == "completed_frozen_fitted_screen" else 1


if __name__ == "__main__":
    raise SystemExit(main())
