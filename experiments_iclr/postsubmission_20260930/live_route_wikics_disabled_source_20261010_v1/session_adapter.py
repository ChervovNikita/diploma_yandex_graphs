"""Fresh plain-own Session facade; disabled source, no runner or numerical import.

Clone only the pinned constructor and snapshot primitives into a private class.
Register the block inside the sole body BEFORE its original Adam is built.
All original source files/classes and any existing Session stay untouched.
"""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

from .blocks import make_block
from .lockstep_forward import BLOCK_NAME, MODEL_CONFIG, forward

PORTABLE_SHA = "29beab567eeb4c070c432e2ce31990c88ec2d8c8618aa5dfb85ebedcf4f0ca14"
SCHEMA = "live-route-wikics-continuation-v1"


def descriptor(session):
    block = getattr(session.model.models[0], BLOCK_NAME)
    root = Path(__file__).resolve().parent
    return {
        "schema": "fixed-live-route-wikics-source-v1", "kind": block.kind,
        "width": 512, "rank": 16, "members": 4, "site": "before_local_block1",
        "initializer_seed": block.initializer_seed, "added_parameters": 32768,
        "source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                           for name in ("blocks.py", "lockstep_forward.py", "session_adapter.py")},
        "original_portable_sha256": PORTABLE_SHA,
        "constructor_AST_sha256": session.live_constructor_AST_sha256,
        "risk": "two-view mean of mean-member TRAIN CE", "pool": "mean_probability",
        "full_member_replay_used": False, "separate_input_teacher_acquisition": False,
    }


def _snapshot_method(original):
    """Preserve primitive body; add schema/source/kind to save/restore binding."""
    node = copy.deepcopy(original)
    dictionaries = [n for n in ast.walk(node) if isinstance(n, ast.Dict)
                    and any(isinstance(k, ast.Constant) and k.value == "schema" for k in n.keys)]
    if len(dictionaries) != 1:
        raise ValueError("Unique original snapshot identity dictionary required")
    d = dictionaries[0]
    positions = [i for i,k in enumerate(d.keys) if isinstance(k, ast.Constant) and k.value == "schema"]
    i = positions[0]
    if not isinstance(d.values[i], ast.Constant) or d.values[i].value != "portable-internal-be-continuation-v1":
        raise ValueError("Original snapshot schema changed")
    d.values[i] = ast.Constant(SCHEMA)
    d.keys.append(ast.Constant("live_route"))
    d.values.append(ast.parse("_live_descriptor(self)", mode="eval").body)
    return node


def adapted_public(public, kind):
    """Expose an actual callable fixed facade for later root-owned qualification.

    This supports ONLY fresh original plain be_unit WikiCS own training. It
    does not reuse relation-credit parameter guards or the per-member replay.
    Optional private-scorer route contexts are supported by the forward, but
    adding their constructor/credit policy is a separate unreviewed change.
    """
    if kind not in ("exchange", "separable"):
        raise ValueError("One fixed declared block; no budget/site search")
    source = Path(public.__file__).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != PORTABLE_SHA:
        raise ValueError("Exact original public Session source required")
    tree = ast.parse(source.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Session"]
    if len(classes) != 1:
        raise ValueError("Unique original Session class required")
    methods = {n.name:n for n in classes[0].body if isinstance(n, ast.FunctionDef)}
    init = copy.deepcopy(methods["__init__"])
    positions = [i for i,n in enumerate(init.body) if isinstance(n, ast.FunctionDef) and n.name == "optimizer"]
    if len(positions) != 1:
        raise ValueError("Unique original pre-Adam insertion location required")
    init.body.insert(positions[0], ast.parse("_install_live_route_block(self)").body[0])
    edited = ast.fix_missing_locations(ast.Module(body=[init,
        _snapshot_method(methods["save_training_state"]),
        _snapshot_method(methods["restore_training_state"])], type_ignores=[]))
    ast_sha = hashlib.sha256(ast.dump(edited, include_attributes=False).encode()).hexdigest()

    def install(session):
        if session.task != "wikics" or session.arm != "be_unit" or session.config["model"] != MODEL_CONFIG:
            raise ValueError("Only exact plain-own shared WikiCS unit-factor constructor")
        if session.model.independent or session.model.members != 4 or len(session.model.models) != 1:
            raise ValueError("One shared body, four complete routes required")
        if hasattr(session, "optimizers"):
            raise ValueError("Fresh pre-Adam installation only; no optimizer retrofit")
        wrapper = session.model.models[0]
        if hasattr(wrapper, BLOCK_NAME):
            raise ValueError("Block installation is not repeatable")
        seed = 870000000 + 100000 * session.seed
        block = make_block(session.torch, kind, seed).to(session.device)
        wrapper.add_module(BLOCK_NAME, block)
        session.live_constructor_AST_sha256 = ast_sha

    namespace = dict(vars(public))
    namespace.update(_install_live_route_block=install, _live_descriptor=descriptor)
    exec(compile(edited, str(source)+":live-route-fixed-source", "exec"), namespace)
    cls = type("FixedLiveRouteSession", (public.Session,), {
        "__init__": namespace["__init__"], "forward": forward,
        "save_training_state": namespace["save_training_state"],
        "restore_training_state": namespace["restore_training_state"], "__module__": __name__,
    })
    return SimpleNamespace(Session=cls, recipe=public.recipe, descriptor=descriptor)
