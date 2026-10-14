"""Load unchanged pinned author bytes; do not execute their main program."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_sources(redesign_root):
    root = Path(redesign_root).resolve()
    manifest = json.loads((root / "native_sources/MANIFEST.json").read_text())
    result = {}
    for name in ("conv.py", "gnn.py", "main_pyg.py"):
        candidates = [r for r in manifest["files"] if r["family"] == "ogb"
                      and Path(r["snapshot_path"]).name == name]
        if len(candidates) != 1:
            raise ValueError("Exactly one pinned OGB source required: " + name)
        path, row = root / "native_sources/ogb" / name, candidates[0]
        if path.stat().st_size != row["bytes"] or _sha(path) != row["sha256"]:
            raise ValueError("Pinned author bytes changed: " + name)
        result[name] = {"path": str(path), "sha256": row["sha256"], "bytes": row["bytes"]}
    return result


@contextmanager
def _alias(name, module):
    previous, existed = sys.modules.get(name), name in sys.modules
    sys.modules[name] = module
    try:
        yield
    finally:
        if existed:
            sys.modules[name] = previous
        else:
            sys.modules.pop(name, None)


def _execute_source(path, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_native_class(redesign_root):
    sources = pinned_sources(redesign_root)
    suffix = sources["gnn.py"]["sha256"][:12] + sources["conv.py"]["sha256"][:8]
    conv_name, gnn_name = "_ogb_smoke_conv_" + suffix, "_ogb_smoke_gnn_" + suffix
    if gnn_name not in sys.modules:
        conv = _execute_source(Path(sources["conv.py"]["path"]), conv_name)
        with _alias("conv", conv):
            _execute_source(Path(sources["gnn.py"]["path"]), gnn_name)
    return sys.modules[gnn_name].GNN, sources


def load_training_rules(redesign_root):
    path = Path(redesign_root).resolve() / "runtime/training_rules.py"
    name = "_ogb_smoke_shared_rules_" + _sha(path)[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name], {"path": str(path), "sha256": _sha(path)}
