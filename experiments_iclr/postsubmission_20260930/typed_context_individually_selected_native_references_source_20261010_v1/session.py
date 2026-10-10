"""New constructor only; native paired training/selector/state methods inherit V3."""
from typed_label_context_factor_source_prototype_20261010_v3.caps import CLOSED
from typed_label_context_factor_source_prototype_20261010_v3 import driver as v3_driver
from typed_label_context_factor_source_prototype_20261010_v3.runtime_gate import require
from .model import install


class ReferenceSession(v3_driver.Session):
    def __init__(self, rt, engine, inputs, scalar, costs, family, spec, identity, caps=CLOSED):
        caps.require("source_bound", "model", "data", "runtime")
        self.rt, self.engine, self.inputs, self.scalar, self.costs = rt, engine, inputs, scalar, costs
        self.condition, self.spec, self.identity, self.caps = family, dict(spec), identity, caps
        torch = rt["torch"]
        self.counters = {"epochs": 0, "TRAIN_native_calls": 0, "TRAIN_backward_calls": 0,
                         "TRAIN_query_rows": 0, "actual_Adam_steps": 0,
                         "serving_native_calls": 0, "serving_query_rows": 0}
        self.initial_checks, self.member_rng = [], []
        self.cache_guard = v3_driver.cache_stamp(inputs)
        require(rt["model_class"] is rt["model_module"].SeHGNN and rt["model_module"].torch is torch
                and rt["device"].type == "cuda" and isinstance(scalar, torch.cuda.amp.GradScaler)
                and scalar.is_enabled(), "Exact prequalified native CUDA/AMP interface")
        # Common frozen roles/context preparation already happened. Model seeds
        # genuinely initialize each native body; they never resplit its targets.
        rt["native"].set_random_seed(spec["native_initialization_seed"])
        prototype = engine.make_model(rt, inputs, costs)
        with costs.measure("reference_complete_native_initial_digest", gpu=True):
            self.native_initial_digest = v3_driver.state_digest(torch, prototype)
        with costs.measure("reference_own_body_install_Adam", gpu=True):
            before = engine.capture_rng(rt["numpy"], torch)
            self.bank = install(rt["model_module"], prototype, family, spec["generator_seed"], spec["body_index"], caps)
            require(engine.exact(torch, engine.capture_rng(rt["numpy"], torch), before), "No native RNG draw from factor/context installation")
            del prototype
            self.all_parameters = tuple(self.bank.parameters())
            require(all(p.dtype == torch.float32 and p.device == rt["device"] for p in self.all_parameters), "Complete native FP32 body before AMP")
            self.optimizer = torch.optim.Adam(self.all_parameters, lr=.001, weight_decay=0)
            require(len(self.optimizer.state) == 0, "Fresh own Adam; no preceding body's moments")
            self.verify_optimizer()  # Exact V3 native defaults/ownership checks.
        caller = engine.capture_rng(rt["numpy"], torch)
        try:
            rt["native"].set_random_seed(spec["member_rng_seeds"][0])
            self.member_rng.append(engine.capture_rng(rt["numpy"], torch))
        finally:
            engine.restore_rng(rt["numpy"], torch, caller)
            require(engine.exact(torch, engine.capture_rng(rt["numpy"], torch), caller), "Own dropout stream preserves native loader/master stream")
        # With bank.shared=False and one member, inherited train_epoch uses the
        # complete unscaled own loss. Selection is this body's VALID BCE only.


def body_spec(role_spec, member):
    require(member in range(4), "Fixed body order0–3")
    return {**role_spec, "body_index": member,
            "native_initialization_seed": role_spec["optimizer_seed"]+1009*member,
            "member_rng_seeds": [role_spec["member_rng_seeds"][member]]}
