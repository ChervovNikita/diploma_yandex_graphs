"""Confined paths and unchanged source imports for the prospective GINE port."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
SOURCE = HERE / "source"
sys.path.insert(0, str(SOURCE / "sharing_policy"))
sys.path.insert(0, str(SOURCE / "new_tasks"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confined(path, root=HERE):
    path, root = Path(path).resolve(), Path(root).resolve()
    if root != path and root not in path.parents:
        raise ValueError("Path escapes authorized GINE packet/research phase")
    return path


def write_json(path, value):
    path = confined(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_hashes():
    frozen = json.loads((HERE / "SOURCE_PROVENANCE.json").read_text())["source_sha256"]
    for name, expected in frozen.items():
        if sha(SOURCE / name) != expected:
            raise ValueError("Unchanged source snapshot differs: " + name)
    own = ["common.py", "group_folds.py", "gine_collector.py", "probe.py", "verify_cpu.py",
           "make_train_bundle.py", "SOURCE_PROVENANCE.json", "PLAN.json"]
    return {**{"source/" + name: expected for name, expected in frozen.items()},
            **{name: sha(HERE / name) for name in own}}
