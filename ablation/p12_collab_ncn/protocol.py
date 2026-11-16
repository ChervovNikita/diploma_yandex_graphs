"""The published NCN Collab command, without hyperparameter search."""
import hashlib
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "external_sources/ncn"
PIN = "11d597013750da17ce7468e344bec756a7af39a4"
SOURCE_FILES = ("README.md", "NeighborOverlap.py", "ogbdataset.py", "model.py", "utils.py", "env.yaml")
RECIPE = {
    "dataset": "ogbl-collab", "predictor": "cn1", "model": "gcn",
    "epochs": 100, "patience": None, "scheduler": None,
    "hiddim": 64, "mplayers": 1, "nnlayers": 3,
    "batch_size": 65536, "testbs": 131072,
    "gnnlr": 0.0082, "prelr": 0.0037,
    "xdp": 0.25, "tdp": 0.05, "gnnedp": 0.25,
    "preedp": 0.0, "predp": 0.3, "gnndp": 0.1,
    "ln": True, "lnnn": True, "res": True, "jk": False,
    "use_xlin": True, "tailact": True, "twolayerlin": False,
    "maskinput": True, "use_valedges_as_input": True,
    "beta": 1.0, "cndeg": -1,
    "pt": 0.1, "probscale": 2.5, "proboffset": 6.0, "alpha": 1.05,
}
METHODS = ("native0", "native1", "native2", "native3", "gnnm")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_identity():
    receipt = json.loads((Path(__file__).parent / "source_receipt.json").read_text())
    for name, expected in receipt["sha256"].items():
        actual = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Pinned NCN source changed: {name}")
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [REPOSITORY / name for name in (
        "models.py", "ablation/common/worker.py", "ablation/common/resumable.py"
    )]
    hashes = {str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(paths)}
    return {"upstream_commit": PIN, "official_sha256": receipt["sha256"],
            "implementation_sha256": digest_json(hashes), "implementation_files": hashes}


def config_for(cohort, method, smoke_updates, data_dir, identity):
    member = int(method[-1]) if method.startswith("native") else 0
    return dict(RECIPE, cohort=cohort, method=method,
                seed=cohort * 1000 + member, members=4 if method == "gnnm" else 1,
                smoke_updates=smoke_updates, result_role="smoke_only" if smoke_updates else "full",
                data_dir=str(Path(data_dir).resolve()), source=identity,
                test_role="closed_smoke" if smoke_updates else "selected_checkpoint_once",
                route_boundaries=["GCN.convs[0]", "CNLinkPredictor.lin[-1]"]
                if method == "gnnm" else [])


def job_key(cohort, method):
    return f"cohort{cohort}/{method}"
