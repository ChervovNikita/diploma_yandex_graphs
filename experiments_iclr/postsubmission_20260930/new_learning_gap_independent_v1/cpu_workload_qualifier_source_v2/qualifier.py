"""Prospective one-fixture CPU resource qualifier. Authored, not run/compiled.

Torch, actual source Model and proposal imports occur only in future main after
root admission and hash verification. No development/confirmation fit is here.
"""
import argparse
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
from types import ModuleType

BASE = Path(__file__).resolve().parent
RESEARCH = BASE.parents[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"ru_maxrss_raw": usage.ru_maxrss,
            "ru_maxrss_unit": "bytes" if sys.platform == "darwin" else "KiB_on_Linux; native_raw_otherwise",
            "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime,
            "maxrss_scope": "Cumulative child process high-water mark, not reset per stage"}


class Recorder:
    def __init__(self, output, admission_id):
        self.output = output
        self.data = {"status": "started", "admission_id": admission_id,
                     "stages": [], "probes": [], "full_law_evaluations": [], "full15_parity": [],
                     "source_counter_meaning": "Old backward_trajectories counter means retained score graphs, not completed backwards",
                     "model_quality_scope": "No accuracy, target-law fidelity, promotion or original scientific score is computed/reported"}
        self.save()

    def save(self):
        self.data["latest_raw_child_usage"] = raw_usage()
        temp = self.output / "progress.json.tmp"
        temp.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.output / "progress.json")

    def measure(self, name, function):
        start, cpu = time.perf_counter_ns(), time.process_time_ns()
        stage = {"name": name, "started_utc": datetime.now(timezone.utc).isoformat(),
                 "raw_usage_before": raw_usage(), "status": "started"}
        self.data["stages"].append(stage)
        self.data["active_stage"] = name
        self.save()
        try:
            result = function()
            stage["status"] = "completed"
            return result
        except BaseException:
            stage["status"] = "failed_or_interrupted"
            raise
        finally:
            stage.update(wall_seconds=(time.perf_counter_ns() - start) / 1e9,
                         process_cpu_seconds=(time.process_time_ns() - cpu) / 1e9,
                         finished_utc=datetime.now(timezone.utc).isoformat(), raw_usage_after=raw_usage())
            self.save()


def verify_sources():
    pins = json.loads((BASE / "SOURCE_PINS.json").read_text())
    for row in pins["files"]:
        path = (RESEARCH / row["research_relative_path"]).resolve()
        require(path.is_relative_to(RESEARCH.resolve()), "Pinned dependency escapes research directory")
        require(sha(path) == row["sha256"], "Pinned dependency changed: " + row["research_relative_path"])
    return pins


def file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Pinned module spec unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def role_seed(role):
    return int.from_bytes(hashlib.sha256(("ar_cpu_resource_v1|1729|" + role).encode()).digest()[:8], "big")


def generator(torch, role):
    return torch.Generator(device="cpu").manual_seed(role_seed(role))


def matrices(torch, h, role):
    p = 18
    gx = generator(torch, role + "/input")
    gh = generator(torch, role + "/recurrent")
    go = generator(torch, role + "/output")
    wx, wh, wo = torch.empty(3 * h, p), torch.empty(3 * h, h), torch.empty(2, h)
    for gate in range(3):
        torch.nn.init.xavier_uniform_(wx[gate * h:(gate + 1) * h], generator=gx)
        torch.nn.init.orthogonal_(wh[gate * h:(gate + 1) * h], generator=gh)
    torch.nn.init.xavier_uniform_(wo, generator=go)
    return wx, wh, wo


def decoder_template(torch, arm):
    M, h = arm["members"], arm["hidden"]
    role = "decoder/h" + str(h)
    wx, wh, wo = matrices(torch, h, role)
    ro = torch.empty(M, h).bernoulli_(0.5, generator=generator(torch, role + "/output_r/M" + str(M))).mul_(2).sub_(1)
    if arm["family"] == "shared":
        tensors = {"W_x": wx, "W_h": wh, "b_x": torch.zeros(3 * h),
                   "b_h": torch.zeros(3 * h), "h0": torch.zeros(h), "b_o": torch.zeros(M, 2)}
        if arm["mode"] == "ordinary_heads":
            tensors["W_o"] = torch.stack([wo * ro[m][None, :] for m in range(M)])
        else:
            tensors.update(W_o=wo, r_o=ro, s_o=torch.ones(M, 2))
        return tensors
    tensors = {"W_x": wx, "W_h": wh, "W_o": wo, "b_x": torch.zeros(M, 3 * h),
               "b_h": torch.zeros(M, 3 * h), "b_o": torch.zeros(M, 2), "h0": torch.zeros(M, h)}
    if arm["mode"] == "factored":
        rx = torch.empty(M, 18).bernoulli_(0.5, generator=generator(torch, role + "/input_r/M" + str(M))).mul_(2).sub_(1)
        tensors.update(r_x=rx, s_x=torch.ones(M, 3 * h), r_h=torch.ones(M, h),
                       s_h=torch.ones(M, 3 * h), r_o=ro, s_o=torch.ones(M, 2))
    elif arm["mode"] == "untied":
        private = [matrices(torch, h, role + "/untied/" + str(m)) for m in range(M)]
        tensors.update(W_x=torch.stack([row[0] for row in private]),
                       W_h=torch.stack([row[1] for row in private]), W_o=torch.stack([row[2] for row in private]))
    return tensors


def clone_tensors(tensors):
    return {name: value.detach().clone() for name, value in tensors.items()}


def digest_parameters(torch, parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(bytes(parameter.detach().contiguous().view(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()


def check_storage(torch, parameters):
    parameters = tuple(parameters)
    require(all(p.dtype == torch.float64 and p.device.type == "cpu" and p.is_contiguous() for _, p in parameters), "CPU float64 contiguous parameter storage required")
    addresses = [p.untyped_storage().data_ptr() for _, p in parameters]
    require(len(addresses) == len(set(addresses)), "Parameter tensors accidentally share storage")
    return sum(p.numel() for _, p in parameters)


def make_decoder(torch, modules, arm, template, audit):
    if arm["family"] == "shared":
        ledger = modules["shared_work"].SharedTransitionLedger(audit)
        decoder = modules["shared"].make_shared_transition_decoder(
            torch_api=torch, head_mode=arm["mode"], n_branches=arm["members"],
            embedding_dimension=16, hidden_dimension=arm["hidden"], parameter_tensors=clone_tensors(template), ledger=ledger)
    else:
        ledger = modules["work"].WorkLedger(audit)
        decoder = modules["private"].make_recurrent_decoder(
            torch_api=torch, mode=arm["mode"], n_branches=arm["members"], embedding_dimension=16,
            hidden_dimension=arm["hidden"], parameter_tensors=clone_tensors(template), ledger=ledger)
    return decoder, ledger


def tiny_parity(torch, modules, evaluator, arm, template, audit, freeze):
    decoder, ledger = make_decoder(torch, modules, arm, template, audit)
    cases = (("empty0", 0, (), ()), ("edgeless3", 3, (), (2, 0, 1)),
             ("path3", 3, ((0, 1), (1, 2)), (2, 0, 1)),
             ("triangle_isolate4", 4, ((0, 1), (1, 2), (0, 2)), (2, 3, 0, 1)))
    records = []
    for name, n, edges, order in cases:
        domain = modules["domain"].ScheduleDomain(n, edges, order, "tiny/" + name, "fixed/" + name)
        H = (torch.arange(n * 16).reshape(n, 16).to(torch.float64) % 11 - 5) / 17
        encoded = modules["encoder"].EncodedGraph(H, domain, "arithmetic-parity-proxy-only")
        rows = [tuple((mask >> node) & 1 for node in range(n)) for mask in range(1 << n)]
        rows = [row for row in rows if domain.feasible(row)]
        counts = {}
        batched = evaluator.all_member_log_densities(torch_api=torch, decoder=decoder, encoded=encoded,
                                                   assignments=rows, family=arm["family"], counts=counts)
        scalar_rows = []
        for row in rows:
            if arm["family"] == "shared":
                scalar_rows.append(decoder.full_log_densities(encoded, row, generating_branch=None))
            else:
                scalar_rows.append(torch.stack([decoder.log_density(encoded, row, branch=m, retain_score_gradient=False)
                                                for m in range(arm["members"])]))
        scalar = torch.stack(scalar_rows)
        require(not batched.requires_grad and not scalar.requires_grad, "Evaluation path retained autograd")
        require(torch.allclose(batched, scalar, atol=freeze["evaluation"]["parity_atol"], rtol=0), "Predeclared batched/scalar parity failure")
        records.append({"case": name, "binary_states": 1 << n, "feasible_states": len(rows),
                        "max_absolute_error": (batched - scalar).abs().max().item(), "vector_counts": counts})
    return {"arm": arm["name"], "all_tiny_states_checked": records, "scalar_source_counts": ledger.counts}


def full15_parity(torch, evaluator, decoder, encoded, assignments, arm, ledger, freeze, recorder, fugacity):
    """Frozen ordinal spread on actual full15 H, never selected by scores.

    This parity subset is additional checked work. It never replaces the
    complete support used by all_support or changes any scalar source API.
    """
    F = len(assignments)
    slots = freeze["evaluation"]["full15_parity"]["selection_slots"]
    require(F > 0 and slots == 16, "Pinned complete support and 16 selection slots required")
    ordinals = sorted({j * (F - 1) // (slots - 1) for j in range(slots)})
    rows = [assignments[index] for index in ordinals]
    before = dict(ledger.counts)
    vector_counts = {}
    record = {"arm": arm["name"], "fugacity": fugacity, "status": "started",
            "complete_feasible_states": F, "selection_slots": slots, "selected_unique_count": len(ordinals),
            "selection_rule": freeze["evaluation"]["full15_parity"]["selection_rule"], "selected_ordinals": ordinals,
            "selected_binary_masks": [sum(label << node for node, label in enumerate(row)) for row in rows],
            "all_members_checked": arm["members"], "checked_member_densities_per_engine": len(rows) * arm["members"],
            "scalar_complete_rows_completed": 0, "batched_complete_rows_completed": 0,
            "atol": freeze["evaluation"]["parity_atol"], "rtol": freeze["evaluation"]["parity_rtol"],
            "vector_counts": vector_counts,
            "scope": "Actual full15 initialized source H implementation parity only; complete-law coverage remains exhaustive"}
    recorder.data["full15_parity"].append(record)
    recorder.save()
    try:
        with torch.no_grad():
            batched = evaluator.all_member_log_densities(torch_api=torch, decoder=decoder, encoded=encoded,
                                                       assignments=rows, family=arm["family"], counts=vector_counts)
            record["batched_complete_rows_completed"] = len(rows)
            scalar_rows = []
            for row in rows:
                if arm["family"] == "shared":
                    scalar_rows.append(decoder.full_log_densities(encoded, row, generating_branch=None))
                else:
                    scalar_rows.append(torch.stack([decoder.log_density(encoded, row, branch=m, retain_score_gradient=False)
                                                    for m in range(arm["members"])]))
                record["scalar_complete_rows_completed"] += 1
            scalar = torch.stack(scalar_rows)
        require(not batched.requires_grad and not scalar.requires_grad, "Full15 evaluation retained autograd")
        require(tuple(batched.shape) == (len(ordinals), arm["members"]), "Full15 parity omitted a selected member density")
        record["max_absolute_error"] = (batched - scalar).abs().max().item()
        require(torch.allclose(batched, scalar, atol=freeze["evaluation"]["parity_atol"],
                               rtol=freeze["evaluation"]["parity_rtol"]), "Predeclared full15 scalar/batched parity failure")
        record["status"] = "completed_parity_pass"
        return record
    except BaseException:
        record["status"] = "failed_or_interrupted"
        raise
    finally:
        record["scalar_source_counts_delta"] = {key: ledger.counts.get(key, 0) - before.get(key, 0)
                                                  for key in set(ledger.counts) | set(before)}
        record["counter_completion_scope"] = "Source/vector ledgers charge started paths; completed row counts are separate"
        recorder.save()


def all_support(torch, evaluator, decoder, encoded, assignments, arm, recorder, freeze):
    masses = torch.zeros(arm["members"])
    counts, digest = {}, hashlib.sha256()
    covered = 0
    chunk = freeze["evaluation"]["chunk_rows"]
    recorder.data["active_full_support"] = {"arm": arm["name"], "required_feasible_states": len(assignments), "completed_feasible_states": 0}
    for start in range(0, len(assignments), chunk):
        rows = assignments[start:start + chunk]
        logq = evaluator.all_member_log_densities(torch_api=torch, decoder=decoder, encoded=encoded,
                                                 assignments=rows, family=arm["family"], counts=counts)
        require((logq.exp() > 0).all().item(), "Feasible q mass underflowed to zero")
        masses += logq.exp().sum(dim=0)
        digest.update(json.dumps(logq.tolist(), separators=(",", ":")).encode())
        covered += len(rows)
        recorder.data["active_full_support"].update(completed_feasible_states=covered, vector_counts=dict(counts))
        recorder.save()
    require(covered == len(assignments), "Full support was incomplete")
    require(torch.allclose(masses, torch.ones_like(masses), atol=freeze["evaluation"]["normalization_atol"], rtol=0), "Full member-law normalization failure")
    return {"required_feasible_states": len(assignments), "evaluated_feasible_states": covered,
            "binary_states_accounted": 32768, "structural_zero_states": 32768 - covered,
            "member_mass": masses.tolist(), "full_log_density_stream_sha256": digest.hexdigest(),
            "vector_counts": counts, "scope": "Initialization-state normalization/resource coverage, not target-law fidelity"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--admission-id", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    require(output.is_relative_to(BASE) and output != BASE and output.is_dir(), "Precreated output folder inside this packet required")
    require(args.admission_id.strip(), "Root admission receipt required")
    freeze = json.loads((BASE / "FREEZE.json").read_text())
    recorder = Recorder(output, args.admission_id)
    started = time.perf_counter_ns()
    def timeout(signum, frame):
        raise TimeoutError("Internal 170-second complete-workload alarm; no retry or support trimming")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(freeze["runtime"]["internal_alarm_seconds"])
    try:
        recorder.data["source_pins"] = recorder.measure("dependency_hash_verification", verify_sources)
        def imports():
            import torch
            import torch_geometric
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.set_default_dtype(torch.float64)
            prefix = "new_learning_gap_independent_v1"
            # Bind this namespace to the pinned local packet parent explicitly;
            # no unrelated installed package with the same name is consulted.
            package = ModuleType(prefix)
            package.__path__ = [str(BASE.parent)]
            sys.modules[prefix] = package
            names = {"domain": ".ar_mixture_source_v1.domain", "encoder": ".ar_mixture_source_v1.source_encoder",
                     "work": ".ar_mixture_source_v1.work_accounting", "private": ".ar_mixture_source_v1.ar_decoder",
                     "objective": ".ar_mixture_source_v1.mixture_objective",
                     "shared": ".shared_transition_heads_source_v1.shared_head_decoder",
                     "shared_work": ".shared_transition_heads_source_v1.work_accounting",
                     "rational": ".finite_cache_source_v1.hard_core_reference"}
            modules = {name: importlib.import_module(prefix + suffix) for name, suffix in names.items()}
            core = file_module("root_admitted_original_gcn", RESEARCH / "source_snapshot/models.py")
            spatial = file_module("root_admitted_spatial_v2", RESEARCH / "protocols/ar_qualification_root_v1/spatial_family_source_v2.py")
            evaluator = file_module("root_admitted_batched_evaluation", BASE / "batched_evaluation.py")
            return torch, modules, core.Model, spatial, evaluator, torch_geometric.__version__
        torch, modules, Model, spatial, evaluator, pyg_version = recorder.measure("imports_and_thread_setup", imports)
        recorder.data["runtime"] = {"python": sys.version, "python_executable": sys.executable, "platform": sys.platform,
                                     "torch": torch.__version__, "torch_geometric": pyg_version, "device": "cpu", "dtype": "float64",
                                     "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()}
        require(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1, "Thread pin failed")
        templates = recorder.measure("decoder_initialization", lambda: {arm["name"]: decoder_template(torch, arm) for arm in freeze["arms_in_order"]})
        placeholder_audit = modules["work"].EncoderAudit("resource-gcn-2x16-none", modules["encoder"].AUTHORIZED_SOURCE_SHA256,
                                                        "pinned-source-GCN-no-dropout", "None", 16, 2, 0, 0)
        recorder.data["parity"] = []
        for arm in freeze["arms_in_order"]:
            parity = recorder.measure("tiny_parity/" + arm["name"], lambda arm=arm: tiny_parity(torch, modules, evaluator, arm, templates[arm["name"]], placeholder_audit, freeze))
            recorder.data["parity"].append(parity)
        instance = recorder.measure("one_complete_fixture_generation", lambda: spatial.make_complete_spatial_instance(instance_id=freeze["fixture_id"], seed=freeze["fixture_seed"]))
        recorder.data["fixture"] = instance.record()
        require(len(instance.points) == 15, "Fixture topology/node count changed")
        domain = modules["domain"].ScheduleDomain(15, instance.edges, instance.order, instance.instance_id, "spatial-v2-separate-order-stream")
        edge_pairs = [pair for u, v in instance.edges for pair in ((u, v), (v, u))]
        from types import SimpleNamespace
        graph = SimpleNamespace(edge_index=torch.tensor(edge_pairs, dtype=torch.long).reshape(-1, 2).T.contiguous())
        audit = modules["work"].EncoderAudit("resource-gcn-2x16-none", modules["encoder"].AUTHORIZED_SOURCE_SHA256,
                                            "pinned-GCN/add_self_loops=False/Identity/dropout0", "None", 16, 2, 4 * len(instance.edges), 15 * 2832)
        def initialized_core():
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(role_seed("source_constructor"))
                core = Model(**freeze["source_encoder"]).to(device="cpu", dtype=torch.float64)
            init = generator(torch, "source_encoder_xavier")
            with torch.no_grad():
                for name, parameter in core.named_parameters():
                    if parameter.ndim >= 2:
                        torch.nn.init.xavier_uniform_(parameter, generator=init)
                    else:
                        require(name.endswith("bias"), "Unexpected 1D source parameter in Identity recipe")
                        parameter.zero_()
            require(sum(p.numel() for p in core.parameters()) == 2992, "Actual pinned source encoder parameter count changed")
            return {name: value.detach().clone() for name, value in core.state_dict().items()}
        encoder_state = recorder.measure("source_encoder_initialization", initialized_core)
        uniforms = recorder.measure("schedule_uniform_generation_once", lambda: torch.rand((8, 15), generator=generator(torch, "schedule_uniforms")).tolist())
        recorder.data["external_draw_counts"] = {"fixture_coordinate_uniforms": 30, "fixture_order_shuffle_calls": 1,
                                                 "schedule_uniforms_generated_once": 120, "copied_same_uniforms_to_every_probe": True,
                                                 "default_source_constructor_RNG_and_init_work": "Included in real construction clocks; RNG bits not measured"}
        recorder.data["memory_scope"] = "Raw process maxima include templates, oracle/support tables, tensors, imports and temporary simultaneous live reference/mix/mean systems; not per-model memory"
        recorder.data["binary_head_parameter_scope"] = "Executed two-row sources; ordinary affine head 2h+2 raw parameters has h+1 Bernoulli equivalent. See pinned BINARY_HEAD_PARAMETER_NOTE.md; no compact-head implementation or storage advantage is qualified"
        def system(arm):
            decoder, ledger = make_decoder(torch, modules, arm, templates[arm["name"]], audit)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(role_seed("source_constructor"))
                core = Model(**freeze["source_encoder"]).to(device="cpu", dtype=torch.float64)
            core.load_state_dict(encoder_state)
            encoder = modules["encoder"].make_static_source_encoder(torch_api=torch, source_core=core, audit=audit, ledger=ledger)
            parameters = tuple(("encoder/" + name, p) for name, p in encoder.named_parameters()) + tuple(("decoder/" + name, p) for name, p in decoder.named_parameters())
            count = check_storage(torch, parameters)
            return encoder, decoder, ledger, parameters, count
        recorder.data["oracle"] = []
        for fugacity in freeze["fugacities_in_order"]:
            def oracle_check():
                integer_states, integer_z = spatial.complete_integer_target(instance, fugacity=fugacity)
                rg = modules["rational"].CompleteConflictGraph(15, instance.edges, (Fraction(fugacity, 1),) * 15)
                budget = modules["rational"].EnumerationBudget(freeze["max_states"], freeze["max_rational_bits"])
                rational = modules["rational"].ExactHardCoreLaw.enumerate(rg, budget)
                require(integer_z == rational.partition and tuple((mask, Fraction(weight)) for mask, weight in integer_states) == rational.weighted_states,
                        "Integer full-state oracle differs from sealed rational oracle")
                rows = [tuple((mask >> node) & 1 for node in range(15)) for mask, _ in integer_states]
                return rows, {"fugacity": fugacity, "integer_states_examined": 32768, "rational_states_examined": rational.states_considered,
                              "complete_feasible_states": len(rows), "integer_partition": integer_z,
                              "partition_bits": integer_z.bit_length(), "oracle_tables_exactly_equal": True}
            assignments, oracle_record = recorder.measure("integer_vs_rational_oracle/lambda" + str(fugacity), oracle_check)
            recorder.data["oracle"].append(oracle_record)
            features = torch.full((15, 1), math.log(fugacity))
            log_lambda = torch.full((15,), math.log(fugacity))
            for arm in freeze["arms_in_order"]:
                label = arm["name"] + "/lambda" + str(fugacity)
                encoder, decoder, ledger, params, count = recorder.measure("construct_reference/" + label, lambda arm=arm: system(arm))
                require(count == 2992 + arm["expected_decoder_raw_parameters"], "Actual arm parameter count changed")
                initial_hash = digest_parameters(torch, params)
                with torch.no_grad():
                    encoded = recorder.measure("source_encoding_reference/" + label, lambda: encoder(graph, features, domain, graph_id=domain.graph_id))
                    recorder.measure("full15_scalar_batched_parity/" + label,
                        lambda: full15_parity(torch, evaluator, decoder, encoded, assignments, arm, ledger, freeze, recorder, fugacity))
                    law = recorder.measure("full_support_batched/" + label, lambda: all_support(torch, evaluator, decoder, encoded, assignments, arm, recorder, freeze))
                law.update(arm=arm["name"], fugacity=fugacity, actual_parameters=count,
                           actual_encoder_parameters=2992, actual_decoder_parameters=count - 2992,
                           initial_parameter_sha256=initial_hash, scalar_source_counts=dict(ledger.counts),
                           pairing="One pre-update law evaluation for bit-identical initialized mix/mean systems")
                recorder.data["full_law_evaluations"].append(law)
                baseline_H = encoded.H.detach().clone()
                reference_addresses = {p.untyped_storage().data_ptr() for _, p in params}
                # Compare all three systems while every allocation is live.
                # The scaffold deliberately holds both objective clones here;
                # raw child memory includes them and the reference system.
                paired_systems, live_addresses = {}, [reference_addresses]
                for objective_kind in freeze["objective_order"]:
                    probe_label = label + "/" + objective_kind
                    paired_systems[objective_kind] = recorder.measure("construct_probe/" + probe_label, lambda arm=arm: system(arm))
                    probe_params = paired_systems[objective_kind][3]
                    require(digest_parameters(torch, probe_params) == initial_hash, "Mix/mean/initial-law parameter pairing failed")
                    probe_addresses = {p.untyped_storage().data_ptr() for _, p in probe_params}
                    require(all(not existing & probe_addresses for existing in live_addresses), "Live reference/mix/mean parameter storage aliases")
                    live_addresses.append(probe_addresses)
                del encoder, decoder, encoded, params, probe_params
                for objective_kind in freeze["objective_order"]:
                    probe_label = label + "/" + objective_kind
                    encoder, decoder, ledger, params, count = paired_systems.pop(objective_kind)
                    encoded = recorder.measure("source_encoding_probe/" + probe_label, lambda: encoder(graph, features, domain, graph_id=domain.graph_id))
                    require(torch.equal(encoded.H.detach(), baseline_H), "Paired source encoding differs before updates")
                    M = arm["members"]
                    branches = [m for m in range(M) for _ in range(8 // M)]
                    samples = recorder.measure("complete_generation/" + probe_label,
                                               lambda: tuple(decoder.generate(encoded, branch=m, uniforms=uniforms[index]) for index, m in enumerate(branches)))
                    cross = recorder.measure("complete_cross_densities/" + probe_label, lambda: decoder.cross_member_log_densities(encoded, samples))
                    mixture = modules["domain"].FixedMixture((1.0,) * M)
                    baseline = modules["domain"].IndependentBaseline((0.0,) * M, "predeclared_constant", "root-predeclared-zero-resource-probe", True)
                    result = recorder.measure("source_score_surrogate/" + probe_label,
                        lambda: modules["objective"].fixed_mixture_reverse_kl_surrogate(torch_api=torch, encoded=encoded, samples=samples,
                            cross_log_q=cross, mixture=mixture, log_fugacities=log_lambda, baseline=baseline,
                            sampling_design="stratified_per_branch", objective_kind=objective_kind, ledger=ledger))
                    optimizer = recorder.measure("optimizer_construction/" + probe_label,
                        lambda: torch.optim.AdamW([p for _, p in params], lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0, amsgrad=False, maximize=False, foreach=False,
                            capturable=False, differentiable=False, fused=False))
                    optimizer.zero_grad(set_to_none=True)
                    recorder.measure("one_completed_backward/" + probe_label, result.surrogate.backward)
                    require(all(p.grad is None or torch.isfinite(p.grad).all().item() for _, p in params), "Nonfinite source gradient")
                    recorder.measure("one_completed_optimizer_step/" + probe_label, optimizer.step)
                    require(all(torch.isfinite(p).all().item() for _, p in params), "Nonfinite updated parameter")
                    recorder.data["probes"].append({"arm": arm["name"], "fugacity": fugacity, "objective": objective_kind,
                        "actual_parameters": count, "initial_parameter_sha256": initial_hash,
                        "actual_encoder_parameters": 2992, "actual_decoder_parameters": count - 2992,
                        "optimizer": dict(freeze["optimizer"]), "optimizer_states_initialized_fresh": True,
                        "actual_optimizer_state_entries": len(optimizer.state),
                        "actual_optimizer_state_tensor_elements": sum(value.numel() for state in optimizer.state.values()
                            for value in state.values() if isinstance(value, torch.Tensor)),
                        "paired_live_storage_checked": "Reference, mix and mean were simultaneously live and disjoint before either step",
                        "source_counts": dict(ledger.counts), "actual_backward_calls_completed": 1,
                        "actual_optimizer_steps_completed": 1, "complete_generated_schedules": 8,
                        "complete_cross_density_entries": 8 * M, "uniform_slots_consumed": 120,
                        "raw_child_usage": raw_usage(), "quality_result": "Not computed; training-only resource probe"})
                    recorder.save()
                    del optimizer, result, cross, samples, encoded, params, encoder, decoder
        recorder.data["status"] = "completed_resource_qualification"
        recorder.data["scope"] = "One pinned training-only complete fixture; no scientific fit matrix, accuracy or promotion decision"
    except BaseException as error:
        recorder.data.update(status="failed_or_incomplete", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        signal.alarm(0)
        recorder.data["elapsed_child_wall_seconds_including_imports"] = (time.perf_counter_ns() - started) / 1e9
        recorder.data["final_raw_child_usage"] = raw_usage()
        recorder.save()
        print(json.dumps({"status": recorder.data["status"], "progress": str(output / "progress.json")}), flush=True)
    return 0 if recorder.data["status"] == "completed_resource_qualification" else 1


if __name__ == "__main__":
    raise SystemExit(main())
