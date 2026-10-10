"""New explicitly scoped native-init constructor and plain own-F forward.

No old alphaF/relationJ guard, replay or qualification is borrowed. Original
libraries stay unchanged. Import is stdlib-only; numerical work is explicit
inside a later root-admitted fresh Session construction.
"""
import ast
import copy
from functools import partial
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from .blocks import make_block
from .accumulated_own import train_step as accumulated_train_step
from .lockstep_forward import BLOCK_NAME, MODEL_CONFIG, NATIVE_SHA, CORE_SHA, forward as coupled_forward

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PORTABLE_SHA = "29beab567eeb4c070c432e2ce31990c88ec2d8c8618aa5dfb85ebedcf4f0ca14"
HOOK = ("portable_private_local_attention_20261008_v1/private_local_attention.py",
        "ffec3e53b59ad43c1343419f5c957a6a89207d08f76d5c0714c8f61d8b9babcb")
INITIALIZER = ("graph_relation_independent_native_local_scorer_source_20261010_v1/initializer.py",
               "e2a9bf05908f48049975e87960fed2c27ce3ab77dc3b65f76ebe3ee01d5acc6e")
SCHEMA = "native-init-live-route-wikics-continuation-v3"
KINDS = ("baseline", "exchange", "separable")


def _load(row, name):
    path = (PHASE/row[0]).resolve(strict=True)
    if not path.is_relative_to(PHASE) or hashlib.sha256(path.read_bytes()).hexdigest() != row[1]:
        raise ValueError("Exact immutable native-scorer dependency required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def descriptor(session):
    stable = {k:v for k,v in session.independent_native_start.items() if k != "elapsed_seconds"}
    return {
        "schema": "native-init-live-route-wikics-source-v3", "kind": session.live_kind,
        "width": 512, "rank": 16, "members": 4, "site": "before_local_block1",
        "added_block_parameters": 0 if session.live_kind == "baseline" else 32768,
        "block_initializer_seed": None if session.live_kind == "baseline" else 870000000+100000*session.seed,
        "native_initializer": stable, "native_hook": HOOK, "initializer_source": INITIALIZER,
        "source_sha256": {n: hashlib.sha256((HERE/n).read_bytes()).hexdigest()
                           for n in ("blocks.py", "lockstep_forward.py", "session_adapter.py", "accumulated_own.py")},
        "original_portable_sha256": PORTABLE_SHA,
        "constructor_AST_sha256": session.live_constructor_AST_sha256,
        "risk": "two-view mean of mean-member TRAIN CE", "pool": "mean_probability",
        "old_relation_credit_and_replay_used": False,
        "view_order": "forward_A; half_F_A_backward; release_A; forward_B; half_F_B_backward; release_B; one_original_Adam",
        "cross_view_auxiliary": False, "bitwise_two_view_trajectory_equivalence": False,
    }


def _snapshot_method(original):
    node = copy.deepcopy(original)
    ds = [n for n in ast.walk(node) if isinstance(n, ast.Dict)
          and any(isinstance(k, ast.Constant) and k.value == "schema" for k in n.keys)]
    if len(ds) != 1:
        raise ValueError("Unique original snapshot identity required")
    d = ds[0]
    i = next(i for i,k in enumerate(d.keys) if isinstance(k, ast.Constant) and k.value == "schema")
    if not isinstance(d.values[i], ast.Constant) or d.values[i].value != "portable-internal-be-continuation-v1":
        raise ValueError("Original snapshot schema changed")
    d.values[i] = ast.Constant(SCHEMA)
    d.keys.append(ast.Constant("live_route"))
    d.values.append(ast.parse("_live_descriptor(self)", mode="eval").body)
    return node


def adapted_public(public, kind):
    if kind not in KINDS:
        raise ValueError("Exactly baseline/exchange/separable; no search")
    source = Path(public.__file__).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != PORTABLE_SHA:
        raise ValueError("Exact original public Session source required")
    tree = ast.parse(source.read_text())
    c = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Session"]
    if len(c) != 1:
        raise ValueError("Unique original Session required")
    methods = {n.name:n for n in c[0].body if isinstance(n, ast.FunctionDef)}
    init = copy.deepcopy(methods["__init__"])
    sites = [i for i,n in enumerate(init.body) if isinstance(n, ast.FunctionDef) and n.name == "optimizer"]
    if len(sites) != 1:
        raise ValueError("Unique original pre-Adam point required")
    init.body.insert(sites[0], ast.parse("_install_native_init_live_route(self)").body[0])
    edited = ast.fix_missing_locations(ast.Module(body=[init,
        _snapshot_method(methods["save_training_state"]),
        _snapshot_method(methods["restore_training_state"])], type_ignores=[]))
    ast_sha = hashlib.sha256(ast.dump(edited, include_attributes=False).encode()).hexdigest()

    def install(session):
        if session.task != "wikics" or session.arm != "be_unit" or session.config["model"] != MODEL_CONFIG:
            raise ValueError("Exact plain-own native WikiCS unit-factor constructor")
        if session.core_provenance != CORE_SHA or session.native_provenance != {"polynormer_model_sha256": NATIVE_SHA}:
            raise ValueError("Exact original public core and native provenance before new installation")
        if session.seed not in (6101, 6203, 6307) or hasattr(session, "optimizers"):
            raise ValueError("Fixed seed and fresh pre-Adam construction required")
        if session.model.independent or session.model.members != 4 or len(session.model.models) != 1:
            raise ValueError("One shared body, four complete private-factor routes")
        wrapper = session.model.models[0]
        if hasattr(wrapper, BLOCK_NAME) or hasattr(session, "private_local_attention"):
            raise ValueError("Fresh installation only")
        hook = _load(HOOK, "_live_route_pinned_native_scorer_hook")
        initializer = _load(INITIALIZER, "_live_route_pinned_native_scorer_initializer")
        controller = hook.install_private_local_attention(wrapper.body, 4)
        session.private_local_attention = controller
        session.independent_native_start = initializer.initialize(session.torch, controller, session.seed)
        original = session.model.member_forward
        session.model.member_forward = partial(controller.forward_member, original)
        session.live_kind = kind
        session.live_constructor_AST_sha256 = ast_sha
        if kind != "baseline":
            torch = session.torch
            cpu = torch.get_rng_state().clone()
            cuda = torch.cuda.get_rng_state(session.cuda_index).clone() if session.cuda_index is not None else None
            block = make_block(torch, kind, 870000000+100000*session.seed).to(session.device)
            wrapper.add_module(BLOCK_NAME, block)
            if not torch.equal(cpu, torch.get_rng_state()) or (cuda is not None and not torch.equal(cuda, torch.cuda.get_rng_state(session.cuda_index))):
                raise ValueError("New projections consumed native/default RNG")
        session.live_work = {"joint_forward_attempts": 0, "joint_forward_completions": 0,
                             "stem_completions": 0, "global_completions": 0,
                             "local_head_completions": 0, "global_head_completions": 0,
                             "block_completions": 0, "local_conv_completions": [0]*7}
        session.live_work.update(backward_attempts=0, backward_completions=0)
        session.live_last_update_events = []
        def finished(name, index=None):
            def hook(_module, _arguments, _output):
                if index is None:
                    session.live_work[name] += 1
                else:
                    session.live_work[name][index] += 1
            return hook
        body = wrapper.body
        session.live_work_hooks = [body.lin_in.register_forward_hook(finished("stem_completions")),
            body.global_attn.register_forward_hook(finished("global_completions")),
            body.pred_local.register_forward_hook(finished("local_head_completions")),
            body.pred_global.register_forward_hook(finished("global_head_completions"))]
        for i, conv in enumerate(body.local_convs):
            session.live_work_hooks.append(conv.register_forward_hook(finished("local_conv_completions", i)))
        if kind != "baseline":
            session.live_work_hooks.append(getattr(wrapper, BLOCK_NAME).register_forward_hook(finished("block_completions")))

    def fresh_forward(session, batch):
        session.live_work["joint_forward_attempts"] += 1
        if session.live_kind == "baseline":
            value = public.Session.forward(session, batch)
        else:
            value = coupled_forward(session, batch)
        session.live_work["joint_forward_completions"] += 1
        return value

    namespace = dict(vars(public))
    namespace.update(_install_native_init_live_route=install, _live_descriptor=descriptor)
    exec(compile(edited, str(source)+":native-init-live-route-v3", "exec"), namespace)
    cls = type("NativeInitLiveRouteSession", (public.Session,), {
        "__init__": namespace["__init__"], "forward": fresh_forward,
        "train_step": accumulated_train_step,
        "save_training_state": namespace["save_training_state"],
        "restore_training_state": namespace["restore_training_state"], "__module__": __name__,
    })
    return SimpleNamespace(Session=cls, recipe=public.recipe, descriptor=descriptor)
