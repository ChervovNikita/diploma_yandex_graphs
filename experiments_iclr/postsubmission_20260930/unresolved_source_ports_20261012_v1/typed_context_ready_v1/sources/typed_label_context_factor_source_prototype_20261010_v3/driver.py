"""Disabled complete native comparison: streamed own AMP, selection, state replay.

There is no command-line launcher. execute() rejects the default CLOSED caps
before reading authority or importing a provider. No masked-context/teacher path.
"""
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

from .caps import CLOSED
from .context import half_inputs
from .model import build_bank, CONTEXT_SITES
from .objective import ordered_queries, member_serving
from .provider import prepare_static, prepare_seed, digest
from .runtime_gate import HERE, PROJECT, CONDITIONS, require, sha, read, inside, binding, release_gate, runtime


def write(path, value):
    path = inside(path, exists=False)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def state_digest(torch, model):
    """All native initial values/buffers, not a sample; costs are measured by caller."""
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(str((tuple(value.shape), str(value.dtype))).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def cache_stamp(inputs):
    values = [("feature", inputs.feats)] + [("label" + str(i), v.label_features)
                                            for i, v in enumerate(inputs.paired.views)]
    rows = [(ns, id(mapping), tuple((k, id(v), int(v._version), tuple(v.shape), str(v.dtype), str(v.device))
                                  for k, v in mapping.items())) for ns, mapping in values]
    for i, view in enumerate(inputs.paired.views):
        rows.extend((str(i) + name, id(v), int(v._version), tuple(v.shape), str(v.dtype), str(v.device))
                    for name, v in (("local", view.local_field), ("global", view.global_field)))
    rows.extend((name, id(v), int(v._version), tuple(v.shape), str(v.dtype), str(v.device))
                for name, v in (("TRAIN", inputs.train.targets), ("VALID", inputs.validation.targets)))
    return tuple(rows)


def memory_accounting(torch, bank, optimizer):
    """Logical and unique storage; serialized file bytes recorded separately."""
    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from tensors(item)
    def unique_bytes(values):
        seen, count = set(), 0
        for value in values:
            storage = value.untyped_storage()
            key = (str(value.device), int(storage.data_ptr()))
            if value.numel() and key not in seen:
                seen.add(key)
                count += storage.nbytes()
        return count
    parameters, buffers = tuple(bank.parameters()), tuple(bank.buffers())
    return {"active_parameters": sum(p.numel() for p in parameters if p.requires_grad),
            "total_unique_parameters": sum(p.numel() for p in parameters),
            "unique_parameter_objects": len(parameters),
            "unique_parameter_storage_bytes": unique_bytes(parameters),
            "member_owned_buffer_storage_bytes": unique_bytes(buffers),
            "Adam_state_storage_bytes": unique_bytes(tensors(optimizer.state_dict())),
            "state_dict_logical_parameter_and_buffer_bytes": sum(v.numel()*v.element_size() for v in bank.state_dict().values()),
            "shared_parameters_counted_once": True, "checkpoint_compression_saving_assumed": False}


class Session:
    def __init__(self, rt, engine, inputs, scalar, costs, condition, spec, identity, caps=CLOSED):
        caps.require("source_bound", "model", "data", "runtime")
        self.rt, self.engine, self.inputs, self.scalar, self.costs = rt, engine, inputs, scalar, costs
        self.condition, self.spec, self.identity, self.caps = condition, dict(spec), identity, caps
        torch = rt["torch"]
        self.counters = {"epochs": 0, "TRAIN_native_calls": 0, "TRAIN_backward_calls": 0,
                         "TRAIN_query_rows": 0, "actual_Adam_steps": 0,
                         "serving_native_calls": 0, "serving_query_rows": 0}
        self.initial_checks, self.member_rng = [], []
        self.cache_guard = cache_stamp(inputs)
        self.bank = None
        require(rt["model_class"] is rt["model_module"].SeHGNN
                and rt["model_module"].torch is torch and rt["device"].type == "cuda"
                and isinstance(scalar, torch.cuda.amp.GradScaler) and scalar.is_enabled(), "Exact native CUDA/master AMP interface")
        prototype = engine.make_model(rt, inputs, costs)
        with costs.measure("full_native_initial_state_digest", gpu=True):
            self.native_initial_digest = state_digest(torch, prototype)
        with costs.measure("six_site_factor_context_install_and_deduplicated_Adam", gpu=True):
            before = engine.capture_rng(rt["numpy"], torch)
            self.bank = build_bank(rt["model_module"], prototype,
                PROJECT / "sehgnn_grouped_member_factor_adapter_source_20261009_v1" / "grouped_member_factors.py",
                condition, spec["generator_seed"], caps)
            require(engine.exact(torch, engine.capture_rng(rt["numpy"], torch), before), "Factor/H/U installation consumes no native global RNG")
            del prototype
            slow = tuple(self.bank.slow_parameters())
            private = tuple(p for m in range(len(self.bank.members)) for p in self.bank.private_parameters(m))
            self.all_parameters = slow + private
            require(len({id(p) for p in self.all_parameters}) == len(self.all_parameters)
                    and {id(p) for p in self.all_parameters} == {id(p) for p in self.bank.parameters()}, "Every shared/private/untied parameter in exactly one Adam group")
            require(all(p.dtype == torch.float32 and p.device == rt["device"] for p in self.all_parameters), "Native FP32 parameters before AMP; no body retyping")
            groups = ([{"params": slow, "lr": .001, "weight_decay": 0}] if slow else [])
            groups += [{"params": private, "lr": .001, "weight_decay": 0}]
            self.optimizer = torch.optim.Adam(groups)
            self.verify_optimizer()
        caller = engine.capture_rng(rt["numpy"], torch)
        try:
            for seed in spec["member_rng_seeds"][:len(self.bank.members)]:
                rt["native"].set_random_seed(seed)
                self.member_rng.append(engine.capture_rng(rt["numpy"], torch))
        finally:
            engine.restore_rng(rt["numpy"], torch, caller)
            require(engine.exact(torch, engine.capture_rng(rt["numpy"], torch), caller), "Owned dropout streams preserve loader/master RNG")

    def verify_optimizer(self):
        torch = self.rt["torch"]
        actual = tuple(p for group in self.optimizer.param_groups for p in group["params"])
        require(len({id(p) for p in actual}) == len(actual) == len(self.all_parameters)
                and {id(p) for p in actual} == {id(p) for p in self.all_parameters}, "Deduplicated complete Adam ownership")
        for group in self.optimizer.param_groups:
            require(group["lr"] == .001 and group["weight_decay"] == 0
                    and group["betas"] == (.9, .999) and group["eps"] == 1e-8
                    and group["amsgrad"] is False and group["maximize"] is False, "Unmodified native Adam/default moments")
        self.bank.verify_ownership()
        require(all(p.grad is None or p.grad.shape == p.shape for p in actual), "Native parameter gradient shapes")

    @contextmanager
    def initial_identity_hooks(self):
        """Actual native full-task activations; no toy/dummy forward or provider fixture."""
        torch = self.rt["torch"]
        handles = []
        if self.bank.contextual and self.counters["epochs"] == 0:
            for m, body in enumerate(self.bank.members):
                for site in CONTEXT_SITES:
                    layer = body
                    for part in site.split("."):
                        layer = layer._modules[part]
                    def check(module, args, output, member=m, path=site):
                        with torch.no_grad():
                            x = args[0]
                            native = module._native_forward(module, x*module.input_factor.to(dtype=x.dtype).unsqueeze(0))
                            centered = native-module.bias.to(dtype=native.dtype).unsqueeze(0)
                            base = native+(module.output_factor.to(dtype=native.dtype).unsqueeze(0)-1)*centered
                            require(module.context_U.count_nonzero().item() == 0
                                    and torch.isfinite(output).all().item() and torch.isfinite(base).all().item(),
                                    "Finite initial zero-U same-input native point-factor activations")
                            tolerance = ({"atol": .001, "rtol": .001} if torch.is_autocast_enabled()
                                         else {"atol": 1e-6, "rtol": 1e-5})
                            require(torch.allclose(output, base, **tolerance),
                                    "Initial zero-U activation identity within declared native precision tolerance")
                            field = module._context_field
                            zero = field-field
                            encoded = torch.tanh(torch.einsum("bg,cgh->bch", zero.to(device=x.device, dtype=x.dtype),
                                                module.context_H.to(dtype=x.dtype)))
                            delta = torch.einsum("bch,chk->bck", encoded, module.context_U.to(dtype=encoded.dtype))
                            require(delta.count_nonzero().item() == 0, "Zero observed evidence produces zero modulation")
                            self.initial_checks.append({"member": member, "site": path,
                                "full_task_activation_rows": len(x), "zero_U_same_input_identity": True,
                                "zero_field_no_generator_bias_identity": True,
                                "floating_output_bitwise_equality_required": False, "activation_tolerance": tolerance,
                                "max_abs_activation_drift": float((output-base).abs().max().item())})
                    handles.append(layer.register_forward_hook(check))
        try:
            yield
        finally:
            for handle in reversed(handles):
                handle.remove()

    def train_epoch(self):
        """Stream each half tape; one Adam update after complete own supervision."""
        self.caps.require("source_bound", "model", "data", "runtime")
        if self.counters["epochs"]:
            self.caps.require("scientific")
        torch, engine = self.rt["torch"], self.engine
        require(cache_stamp(self.inputs) == self.cache_guard, "Immutable factual/context/target caches")
        self.bank.train()
        losses, batches = [], 0
        with self.costs.measure("complete_paired_TRAIN_streamed_native_AMP_BCE", gpu=True), self.initial_identity_hooks():
            for train_batch in self.inputs.train_loader:
                batches += 1
                require(batches == 1, "Literal IMDB one external native TRAIN minibatch")
                ordered = ordered_queries(torch, train_batch, self.inputs.train, self.inputs.paired, self.caps)
                self.optimizer.zero_grad()
                for m in range(len(self.bank.members)):
                    caller = engine.capture_rng(self.rt["numpy"], torch)
                    member_loss = 0.0
                    try:
                        engine.restore_rng(self.rt["numpy"], torch, self.member_rng[m])
                        for view, query, positions in ordered:
                            batch, features, labels, field = half_inputs(torch, view, query, self.inputs.feats,
                                                self.condition == "shared_global_mul4", self.caps)
                            device = self.rt["device"]
                            position = torch.tensor(positions, dtype=torch.long, device=self.inputs.train.targets.device)
                            truth = self.inputs.train.targets[position].to(device=device, dtype=torch.float32)
                            with torch.cuda.amp.autocast():
                                output = self.bank.forward_member(m, batch.to(device),
                                    {k: v.to(device) for k, v in features.items()},
                                    {k: v.to(device) for k, v in labels.items()}, field)
                                loss = torch.nn.functional.binary_cross_entropy_with_logits(output, truth)
                                weighted = loss*(len(query)/len(self.inputs.train.ids))
                                own = weighted/len(self.bank.members) if self.bank.shared else weighted
                            require(torch.isfinite(output).all().item() and math.isfinite(float(loss.detach().item())), "Finite five-logit own BCE")
                            self.scalar.scale(own).backward()
                            member_loss += float(weighted.detach().item())
                            self.counters["TRAIN_native_calls"] += 1
                            self.counters["TRAIN_backward_calls"] += 1
                            self.counters["TRAIN_query_rows"] += len(query)
                            del output, loss, weighted, own, features, labels, field, truth, batch, position
                        self.member_rng[m] = engine.capture_rng(self.rt["numpy"], torch)
                        losses.append(member_loss)
                    finally:
                        engine.restore_rng(self.rt["numpy"], torch, caller)
                        require(engine.exact(torch, engine.capture_rng(self.rt["numpy"], torch), caller), "Member forward/backward preserves native loader/master stream")
                self.scalar.unscale_(self.optimizer)
                if self.counters["epochs"] == 0 and self.bank.contextual:
                    h, u = [], []
                    for body in self.bank.members:
                        for name, parameter in body.named_parameters():
                            if name.endswith(".context_H"):
                                require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                                        and parameter.grad.count_nonzero().item() == 0, "Initial H contribution gradient is exactly zero at U0")
                                h.append(name)
                            if name.endswith(".context_U"):
                                require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(), "Finite initially connected U gradient; nonzero not guaranteed")
                                u.append({"name": name, "nonzero_gradient_entries": int(parameter.grad.count_nonzero().item())})
                    self.initial_checks.append({"initial_H_gradient_zero_sites": len(h), "initial_U_gradient_sites": u})
                step = self.optimizer.step
                def observed_step(*args, **kwargs):
                    require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in self.all_parameters), "Finite actual native unscaled Adam gradients")
                    result = step(*args, **kwargs)
                    self.counters["actual_Adam_steps"] += 1
                    return result
                self.optimizer.step = observed_step
                try:
                    self.scalar.step(self.optimizer)
                    self.scalar.update()
                finally:
                    self.optimizer.step = step
        require(batches == 1 and len(losses) == len(self.bank.members), "One complete own paired epoch")
        self.counters["epochs"] += 1
        require(self.counters["TRAIN_native_calls"] == 2*len(self.bank.members)*self.counters["epochs"]
                and self.counters["TRAIN_query_rows"] == len(self.bank.members)*len(self.inputs.train.ids)*self.counters["epochs"], "Exact half-call and complete query-row accounting")
        require(cache_stamp(self.inputs) == self.cache_guard, "No predictor mutation of cached evidence/targets")
        self.verify_optimizer()
        return {"member_own_BCE": losses, "mean_member_own_BCE": sum(losses)/len(losses),
                "optimizer_objective": "mean member own BCE" if self.bank.shared else "sum unscaled private-body own BCE",
                "actual_Adam_steps_so_far": self.counters["actual_Adam_steps"]}

    def evaluate(self, scope="complete_TRAIN_VALID_FP32_paired_serving"):
        torch, engine = self.rt["torch"], self.engine
        before = engine.capture_rng(self.rt["numpy"], torch)
        members_before = engine.cpu_tree(torch, self.member_rng)
        modes = tuple((module, module.training) for module in self.bank.modules())
        prior_calls, prior_rows = self.counters["serving_native_calls"], self.counters["serving_query_rows"]
        ids = self.inputs.train.ids + self.inputs.validation.ids
        require(len(ids) <= 20000, "One complete native known-role eval batch; no evaluation truncation")
        try:
            self.bank.eval()
            with self.costs.measure(scope, gpu=True) as cost, torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                outputs, context_outputs = [], []
                for m in range(len(self.bank.members)):
                    averaged, contexts = member_serving(torch, self.bank, self.inputs, m, self.counters, self.caps)
                    outputs.append(averaged.cpu())
                    context_outputs.append(contexts)
                member_logits = torch.stack(outputs)
                pool = member_logits.mean(dim=0)  # Exact prospective native mean-raw-logit rule.
                probabilities = torch.sigmoid(pool)
                scores, evidence = {}, {}
                n = len(self.inputs.train.ids)
                for role, start, end, truth in (("TRAIN", 0, n, self.inputs.train.targets),
                                              ("VALID", n, len(ids), self.inputs.validation.targets)):
                    current = pool[start:end]
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(current, truth.float()).item()
                    micro, macro = self.rt["native"].evaluator(truth, (probabilities[start:end] > .5).int())
                    member_bce = [float(torch.nn.functional.binary_cross_entropy_with_logits(v[start:end], truth.float()).item()) for v in member_logits]
                    member_f1 = [tuple(float(x) for x in self.rt["native"].evaluator(truth, (torch.sigmoid(v[start:end]) > .5).int())) for v in member_logits]
                    require(all(math.isfinite(x) for x in (bce, float(micro), float(macro), *member_bce)), "Finite complete role selection/competence metrics")
                    scores[role] = {"BCE": float(bce), "micro_F1": float(micro), "macro_F1": float(macro),
                                    "member_BCE": member_bce, "member_micro_macro_F1": member_f1}
                    evidence[role] = {"ids": list(ids[start:end]), "pool_logits": current.clone(),
                        "probabilities": probabilities[start:end].clone(),
                        "member_logits": member_logits[:, start:end].clone(),
                        "context_member_logits": tuple(tuple(v["VALID_logits"].clone() for v in pair)
                                                        for pair in context_outputs) if role == "VALID" else (),
                        "TRAIN_complement_context_evidence": context_outputs if role == "TRAIN" else (),
                        "context_policy": "one excluded-target complement per TRAIN row" if role == "TRAIN"
                                          else "mean raw logits of both frozen TRAIN contexts"}
                calls = self.counters["serving_native_calls"]-prior_calls
                rows = self.counters["serving_query_rows"]-prior_rows
                expected_rows = len(self.bank.members)*(len(self.inputs.train.ids)+2*len(self.inputs.validation.ids))
                require(calls == 2*len(self.bank.members) and rows == expected_rows,
                        "M4 serves eight calls; TRAIN complement once and VALID twice per member")
                cost.update(requested_rows=len(ids), native_model_calls=calls, actual_query_rows=rows,
                            TRAIN_contexts_per_row=1, VALID_contexts_per_row=2,
                            TRAIN_queries_excluded_from_global_field=True,
                            VALID_mean_context_then_mean_raw_member_logits=True,
                            includes_CPU_cache_staging_and_transfers=True)
                return scores, evidence
        finally:
            for module, training in modes:
                module.training = training
            engine.restore_rng(self.rt["numpy"], torch, before)
            require(engine.exact(torch, engine.capture_rng(self.rt["numpy"], torch), before)
                    and engine.exact(torch, self.member_rng, members_before), "Evaluation preserves master and all member streams")
            require(cache_stamp(self.inputs) == self.cache_guard, "Serving preserves evidence/target cache values")

    def snapshot(self, epoch, scores, outputs):
        torch, engine = self.rt["torch"], self.engine
        with self.costs.measure("selected_full_bank_Adam_buffers_scaler_RNG_CPU_snapshot", gpu=True):
            return {"schema": "typed-label-context-native37-selected-state-v3", "identity": self.identity,
                    "condition": self.condition, "seed_spec": self.spec, "epoch": epoch,
                    "role_binding": self.inputs.role_binding, "context_identity_sha256": self.inputs.paired.plan.identity_sha256,
                    "native_initial_digest": self.native_initial_digest,
                    "bank_state": engine.cpu_tree(torch, self.bank.state_dict()),
                    "optimizer_state": engine.cpu_tree(torch, self.optimizer.state_dict()),
                    "scaler_state": engine.cpu_tree(torch, self.scalar.state_dict()),
                    "master_rng": engine.capture_rng(self.rt["numpy"], torch),
                    "member_rng": engine.cpu_tree(torch, self.member_rng),
                    "scores": scores, "outputs": outputs, "known_roles_only": True,
                    "serving_rule": "VALID mean context logits within member then mean raw member logits; TRAIN complement context only"}

    def restore(self, saved):
        torch, engine = self.rt["torch"], self.engine
        require(saved["schema"] == "typed-label-context-native37-selected-state-v3"
                and saved["identity"] == self.identity and saved["condition"] == self.condition
                and saved["seed_spec"] == self.spec and saved["role_binding"] == self.inputs.role_binding
                and saved["context_identity_sha256"] == self.inputs.paired.plan.identity_sha256,
                "Exact selected source/roles/context/condition/seeds identity")
        with self.costs.measure("selected_shared_aliases_full_state_exact_restore", gpu=True):
            self.bank.verify_shared_state_dict(saved["bank_state"])
            self.bank.load_state_dict(saved["bank_state"], strict=True)
            self.optimizer.load_state_dict(saved["optimizer_state"])
            self.scalar.load_state_dict(saved["scaler_state"])
            self.member_rng = engine.cpu_tree(torch, saved["member_rng"])
            require(engine.exact(torch, engine.cpu_tree(torch, self.bank.state_dict()), saved["bank_state"])
                    and engine.exact(torch, engine.cpu_tree(torch, self.optimizer.state_dict()), saved["optimizer_state"])
                    and engine.exact(torch, self.scalar.state_dict(), saved["scaler_state"])
                    and engine.exact(torch, self.member_rng, saved["member_rng"]), "Exact selected native buffers/weights/Adam/scaler/member streams")
            engine.restore_rng(self.rt["numpy"], torch, saved["master_rng"])
            require(engine.exact(torch, engine.capture_rng(self.rt["numpy"], torch), saved["master_rng"]), "Exact selected master stream")
            self.verify_optimizer()


def fit(session, folder, qualify=False):
    if not qualify:
        session.caps.require("scientific")
    best_epoch, best_loss, history = -1, float("inf"), []
    checkpoint = folder / "SELECTED_STATE.pt"
    for epoch in range(1 if qualify else 200):
        with session.costs.measure("native_pre_epoch_GC"):
            gc.collect()
        own = session.train_epoch()
        with session.costs.measure("native_post_TRAIN_cache_cleanup"):
            session.rt["torch"].cuda.empty_cache()
        scores, outputs = session.evaluate()
        improved = scores["VALID"]["BCE"] < best_loss
        if improved:
            best_epoch, best_loss = epoch, scores["VALID"]["BCE"]
            saved = session.snapshot(epoch, scores, outputs)
            with session.costs.measure("selected_checkpoint_serialization") as row:
                temporary = checkpoint.with_suffix(".pt.tmp")
                session.rt["torch"].save(saved, temporary)
                temporary.replace(checkpoint)
                row["serialized_bytes"] = checkpoint.stat().st_size
            del saved
        history.append({"epoch": epoch, "own": own, "scores": scores,
                        "selected": improved, "counters": dict(session.counters)})
        write(folder / "HISTORY.json", history)
        if not qualify and epoch-best_epoch > 50:
            break
    require(best_epoch >= 0 and checkpoint.is_file(), "Complete VALID selector produced selected owned state")
    if qualify:
        require(session.counters["epochs"] == session.counters["actual_Adam_steps"] == 1,
                "Qualification is one complete actual native paired update, not overflow-skipped")
    return {"selected_epoch": best_epoch, "selected_VALID_BCE": best_loss,
            "selection": "strict minimum complete VALID BCE; earliest ties; max200; epoch-best_epoch>50",
            "checkpoint": binding(checkpoint), "history": history,
            "counters": dict(session.counters), "initial_checks": session.initial_checks,
            "native_initial_digest": session.native_initial_digest,
            "storage": memory_accounting(session.rt["torch"], session.bank, session.optimizer)}


def replay(rt, engine, inputs, costs, condition, spec, identity, checkpoint, caps=CLOSED):
    torch = rt["torch"]
    caller = engine.capture_rng(rt["numpy"], torch)
    session = saved = None
    try:
        with costs.measure("weights_only_owned_selected_checkpoint_read"):
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        scalar = torch.cuda.amp.GradScaler()  # Separate from condition's cross-seed master.
        session = Session(rt, engine, inputs, scalar, costs, condition, spec, identity, caps)
        session.restore(saved)
        scores, outputs = session.evaluate("fresh_selected_complete_TRAIN_VALID_serving")
        diagnostics = {}
        for role in ("TRAIN", "VALID"):
            old, new = saved["outputs"][role], outputs[role]
            require(old["ids"] == new["ids"], "Exact replay row order")
            diagnostics[role] = {"max_abs_pool_logit": float((old["pool_logits"]-new["pool_logits"]).abs().max().item()),
                "max_abs_member_logit": float((old["member_logits"]-new["member_logits"]).abs().max().item()),
                "max_abs_context_member_logit": max((float((a-b).abs().max().item()) for pa, pb in zip(old["context_member_logits"], new["context_member_logits"]) for a, b in zip(pa, pb)), default=0.0),
                "max_abs_probability": float((old["probabilities"]-new["probabilities"]).abs().max().item()),
                "prediction_changes": int(((old["probabilities"]>.5)!=(new["probabilities"]>.5)).sum().item())}
        require(all(row["max_abs_pool_logit"] <= .001 and row["max_abs_member_logit"] <= .001
                    and row["max_abs_context_member_logit"] <= .001 and row["max_abs_probability"] <= .001
                    and row["prediction_changes"] == 0 for row in diagnostics.values()), "Complete selected replay drift/prediction gate")
        return {"epoch": saved["epoch"], "scores": scores, "recorded_scores": saved["scores"],
                "exact_model_optimizer_scaler_buffers_RNG_restore": True, "output_bitwise_equality_claim": False,
                "gross_replay_tolerance": .001, "prediction_changes_required": 0, "diagnostics": diagnostics,
                "counters": dict(session.counters)}
    finally:
        session = saved = None
        gc.collect()
        torch.cuda.empty_cache()
        engine.restore_rng(rt["numpy"], torch, caller)
        require(engine.exact(torch, engine.capture_rng(rt["numpy"], torch), caller), "Fresh replay preserves original master stream lifecycle")


def run_one(rt, engine, static, operators, role, release, spec, condition, folder, scalar, identity, caps=CLOSED):
    caps.require("source_bound", "model", "data", "runtime")
    folder = inside(folder, exists=False)
    folder.mkdir(exist_ok=False)
    costs = engine.Costs(folder, rt["torch"], rt["device"])
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    torch = rt["torch"]
    torch.cuda.reset_peak_memory_stats(rt["device"])
    qualify = release["action"] == "qualify_complete_native_context_pair"
    record = {"status": "started", "complete": False, "qualification_passed": False,
              "condition": condition, "seed_spec": spec, "identity": identity,
              "TEST_truth": False, "TEST_membership_known": False, "TEST_file_access": False}
    inputs = session = None
    try:
        inputs = prepare_seed(rt, static, operators, role, release["roles"][str(spec["role_seed"])], spec, costs, caps)
        session = Session(rt, engine, inputs, scalar, costs, condition, spec, identity, caps)
        record["ownership"] = session.bank.verify_ownership()
        record["contexts"] = {"identity_sha256": inputs.paired.plan.identity_sha256,
                              "halves": inputs.paired.plan.halves, "support_audit": inputs.paired.support_audit}
        fitted = fit(session, folder, qualify)
        record.update(fitted)
        master_scaler = engine.cpu_tree(torch, scalar.state_dict())
        session = None  # Free fitted bank before complete fresh constructor.
        gc.collect()
        torch.cuda.empty_cache()
        record["fresh_selected"] = replay(rt, engine, inputs, costs, condition, spec, identity,
                                          folder / "SELECTED_STATE.pt", caps)
        require(engine.exact(torch, scalar.state_dict(), master_scaler), "Condition-owned cross-seed master scaler survives fresh replay")
        record.update(status="complete", complete=True, qualification_passed=qualify,
                      known_roles_only=True, all_full_input_selection_checkpoint_replay_costs_charged=True)
    except BaseException as error:
        record.update(status="failed", complete=False, qualification_passed=False,
                      failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        try:
            torch.cuda.synchronize(rt["device"])
            record.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(rt["device"]),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(rt["device"]))
            session = inputs = None
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException as error:
            record.update(status="failed", complete=False, qualification_passed=False,
                          cleanup_failure={"type": type(error).__name__, "message": str(error)})
        final = resource.getrusage(resource.RUSAGE_SELF)
        record.update(seconds=time.perf_counter()-started, CPU_user_seconds=final.ru_utime-usage.ru_utime,
                      CPU_system_seconds=final.ru_stime-usage.ru_stime,
                      cumulative_process_RSS_peak_bytes=int(final.ru_maxrss*(1 if sys.platform == "darwin" else 1024)),
                      RSS_is_cumulative_process_peak_not_per_condition_increment=True)
        write(folder / "RESULT.json", record)
        write(folder / "COMPLETE.json", {"complete": record["complete"], "status": record["status"],
                                        "qualification_passed": record["qualification_passed"]})


def execute(release_path, caps=CLOSED):
    """External root release only; default call raises before provider or data access."""
    caps.require("source_bound", "model", "data", "runtime")
    release, roles, sources, release_binding = release_gate(release_path, caps)
    output = inside(release["output_directory"], exists=False)
    output.mkdir(parents=True, exist_ok=False)
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    qualify = release["action"] == "qualify_complete_native_context_pair"
    report = {"status": "started", "complete": False, "qualification_passed": False,
              "mode": "qualification" if qualify else "scientific_development_comparison",
              "source_seal_sha256": release["source_seal_sha256"], "protocol_sha256": release["protocol_sha256"],
              "release": release_binding, "roles": release["roles"], "input_files": release["development_files"],
              "conditions": list(CONDITIONS), "seed_specs": release["seed_specs"], "runs": [],
              "TEST_truth": False, "TEST_membership_known": False, "TEST_file_access": False,
              "scientific_novelty_or_superiority_or_acceptance_admitted": False}
    rt = engine = static = operators = scalar = None
    write(output / "COHORT_REPORT.json", report)
    native_initial, context_ids = {}, {}
    try:
        rt, engine, loader, report["runtime"] = runtime(release, sources, caps)
        costs = engine.Costs(output, rt["torch"], rt["device"])
        with costs.measure("full_native_static_preparation"):
            static, operators = prepare_static(rt, engine, loader, release, sources, roles, costs, caps)
        report["feature_shapes"] = static.feature_shapes
        identity = {"source_seal_sha256": release["source_seal_sha256"], "protocol_sha256": release["protocol_sha256"],
                    "release": release_binding, "roles": release["roles"], "input_files": release["development_files"],
                    "schema_receipt": release["schema_receipt"], "runtime": report["runtime"]}
        # One master scaler per condition across its ordered role-seed loop.
        # Do not carry a preceding condition's overflow history into another method.
        for condition in CONDITIONS:
            scalar = rt["torch"].cuda.amp.GradScaler()
            (output / condition).mkdir(exist_ok=False)
            for spec in release["seed_specs"]:
                seed = spec["role_seed"]
                folder = output / condition / ("seed" + str(seed))
                try:
                    run_one(rt, engine, static, operators, roles[seed], release, spec, condition,
                            folder, scalar, identity, caps)
                finally:
                    if (folder / "RESULT.json").exists():
                        report["runs"].append(read(folder / "RESULT.json"))
                        write(output / "COHORT_REPORT.json", report)
                row, done = read(folder / "RESULT.json"), read(folder / "COMPLETE.json")
                require(row["status"] == "complete" and row["complete"] and done["complete"]
                        and (not qualify or row["qualification_passed"]), "Every full fit/replay/cleanup completed; no favorable-subset report")
                if seed in native_initial:
                    require(native_initial[seed] == row["native_initial_digest"]
                            and context_ids[seed] == row["contexts"]["identity_sha256"], "Matched full native initial state and context halves across all six conditions")
                else:
                    native_initial[seed] = row["native_initial_digest"]
                    context_ids[seed] = row["contexts"]["identity_sha256"]
        require(loader.bindings(loader.permitted_files(release["input_root"])) == release["development_files"], "Development files unchanged through full comparison")
        require(binding(release_binding["path"]) == release_binding
                and all(binding(r["path"]) == r for r in release["roles"].values())
                and binding(release["schema_receipt"]["path"]) == release["schema_receipt"]
                and binding(release["native_qualification_receipt"]["path"]) == release["native_qualification_receipt"],
                "Root release/role/schema/native qualification descriptors unchanged through full comparison")
        require(len(report["runs"]) == len(CONDITIONS)*len(release["seed_specs"]), "Complete declared grid, never a selected successful subset")
        report.update(status="complete", complete=True, qualification_passed=qualify,
                      matched_native_initial_state_by_role=native_initial, matched_context_identity_by_role=context_ids,
                      actual_results_do_not_grant_scientific_admission=True)
    except BaseException as error:
        report.update(status="failed", complete=False, qualification_passed=False,
                      failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if rt is not None:
            static = operators = scalar = None
            gc.collect()
            rt["torch"].cuda.empty_cache()
        final = resource.getrusage(resource.RUSAGE_SELF)
        report.update(seconds_including_shared_full_graph_preprocessing=time.perf_counter()-started,
                      CPU_user_seconds=final.ru_utime-usage.ru_utime,
                      CPU_system_seconds=final.ru_stime-usage.ru_stime,
                      cumulative_process_RSS_peak_bytes=int(final.ru_maxrss*(1 if sys.platform == "darwin" else 1024)),
                      nested_cost_event_seconds_must_not_be_summed=True,
                      independent_ordinary_and_same_information_references_not_run_here=True,
                      exploration_grid_alone_cannot_establish_superiority=True)
        write(output / "COHORT_REPORT.json", report)
    return report
