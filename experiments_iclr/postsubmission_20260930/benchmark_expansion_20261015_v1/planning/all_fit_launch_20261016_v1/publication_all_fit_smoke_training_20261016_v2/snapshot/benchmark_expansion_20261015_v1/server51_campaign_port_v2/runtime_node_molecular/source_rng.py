"""Guarded CPU RNG bookkeeping for one omitted native OGB TEST iterator.

No TEST loader, split, records, labels, model call or metric is constructed.
This preserves only the audited iterator RNG effect, under stated source and
dependency assumptions. It does not establish whole-paper replication.
"""
from __future__ import annotations
import hashlib
import importlib
import inspect
import json
from pathlib import Path

from common import inside, receipt

MODE = "omitted_test_iterator_base_seed_v1"
SCHEMA = "omitted-source-test-iterator-cpu-rng-audit-v1"
ASSUMPTIONS = [
    "map-style nonshuffled DataLoader; num_workers=0; generator=None",
    "no dataset runtime transform, random collation or augmentation",
    "pinned native OGB GIN/GCN (with or without virtual node), full features, mean pooling",
    "native eval sets model.eval and no_grad; dropout uses module.training=False",
    "no stochastic eval extension, model hook, or runtime monkey patch",
    "only omitted iterator RNG consumption is emulated; TEST inference and metrics stay omitted",
    "dependency profile and finite integer-loader RNG semantics must match the bound audit",
]


def one_iterator_seed_draw(torch):
    # Pinned torch.utils.data.dataloader._BaseDataLoaderIter.__init__.
    return torch.empty((), dtype=torch.int64).random_(generator=None).item()


def loader_rng_semantics(torch, include_pyg=False):
    """Finite integer-loader semantics only; restores the caller's CPU RNG."""
    constructors = [("torch", torch.utils.data.DataLoader, False),
                    ("torch_tensor_dataset", torch.utils.data.DataLoader, True)]
    if include_pyg:
        from torch_geometric.loader import DataLoader
        constructors.append(("pyg", DataLoader, False))
    saved = torch.get_rng_state().clone()
    rows = []
    try:
        for label, constructor, tensor_dataset in constructors:
            for seed in (0, 101, 202):
                for size, batch_size in ((0, 1), (1, 32), (2, 1), (33, 32)):
                    torch.random.default_generator.manual_seed(seed)  # CPU only; no CUDA seeding
                    expected = list(range(size))
                    dataset = torch.utils.data.TensorDataset(torch.arange(size)) if tensor_dataset else expected
                    loader = constructor(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=0, generator=None)
                    before = torch.get_rng_state().clone()
                    actual = []
                    for batch in loader:
                        if tensor_dataset:
                            batch = batch[0]
                        actual.extend(batch.tolist())
                    after = torch.get_rng_state().clone()
                    torch.set_rng_state(before)
                    one_iterator_seed_draw(torch)
                    if actual != expected or not torch.equal(torch.get_rng_state(), after):
                        raise ValueError("Installed "+label+" iterator does not match the declared single CPU seed draw")
                    rows.append({"loader": label, "seed": seed, "integer_records": size,
                                 "batch_size": batch_size, "exact_cpu_rng_state_match": True})
    finally:
        torch.set_rng_state(saved)
    return {"kind": "finite integer-loader RNG semantics, not graph/model quality",
            "cases": rows, "model_calls": 0, "task_records_accessed": 0, "TEST_accessed": False}


def check_source_loader_flags(torch, train_loader, valid_loader):
    for loader in (train_loader, valid_loader):
        if loader.num_workers != 0 or loader.generator is not None or loader.batch_size != 32 \
                or loader.drop_last or loader.pin_memory or loader.persistent_workers \
                or isinstance(loader.dataset, torch.utils.data.IterableDataset):
            raise ValueError("Worker/generator/batch/dataset flags differ from the audited native source")
    if type(train_loader.sampler) is not torch.utils.data.RandomSampler \
            or train_loader.sampler.replacement or type(valid_loader.sampler) is not torch.utils.data.SequentialSampler:
        raise ValueError("Audited native TRAIN is shuffled without replacement; VALID/source TEST is nonshuffled")
    if torch.empty(0).device.type != "cpu":
        raise ValueError("The native iterator draw audit assumes the default CPU tensor device")


def dependency_profile(torch, redesign_root):
    """Bind actual installed implementations, rather than assuming a version."""
    import torch_geometric
    import ogb
    modules = [
        "torch.utils.data.dataloader", "torch.utils.data.sampler",
        "torch.utils.data._utils.fetch", "torch.utils.data._utils.collate",
        "torch.nn.functional", "torch.nn.modules.batchnorm",
        "torch.nn.modules.linear", "torch.nn.modules.sparse",
        "torch_geometric.loader.dataloader", "torch_geometric.data.dataset",
        "torch_geometric.data.in_memory_dataset", "torch_geometric.data.data",
        "torch_geometric.data.batch", "torch_geometric.data.collate",
        "torch_geometric.data.separate", "torch_geometric.nn.conv.message_passing",
        "torch_geometric.nn.pool.glob", "ogb.graphproppred.dataset_pyg",
        "ogb.graphproppred.mol_encoder", "ogb.graphproppred.evaluate",
    ]
    files = {}
    for name in modules:
        module = importlib.import_module(name)
        path = inspect.getsourcefile(module)
        if path is None:
            raise ValueError("Auditable installed dependency source is unavailable: "+name)
        files[name] = receipt(Path(path).resolve())
    redesign_root = Path(redesign_root).resolve()
    manifest_path = redesign_root/"native_sources/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    native = {}
    for name in ("main_pyg.py", "gnn.py", "conv.py"):
        rows = [row for row in manifest["files"] if row["family"] == "ogb"
                and Path(row["snapshot_path"]).name == name]
        if len(rows) != 1:
            raise ValueError("Exactly one frozen native OGB file required: "+name)
        actual = receipt(redesign_root/"native_sources/ogb"/name)
        if any(actual[key] != rows[0][key] for key in ("bytes", "sha256")):
            raise ValueError("Native OGB source differs from its frozen manifest: "+name)
        native[name] = actual
    source = inspect.getsource(torch.utils.data.dataloader._BaseDataLoaderIter.__init__)
    if "_base_seed" not in source or ".random_(generator=loader.generator)" not in source:
        raise ValueError("Installed loader base-seed implementation requires a fresh source audit")
    return {"torch_version": str(torch.__version__), "pyg_version": str(torch_geometric.__version__),
            "ogb_version": str(ogb.__version__), "installed_sources": files,
            "native_sources": native, "native_manifest": receipt(manifest_path),
            "base_iterator_init_source_sha256": hashlib.sha256(source.encode()).hexdigest()}


def make_audit(torch, redesign_root):
    return {"schema": SCHEMA, "mode": MODE, "assumptions": ASSUMPTIONS,
            "dependency_profile": dependency_profile(torch, redesign_root),
            "semantic_check": loader_rng_semantics(torch, include_pyg=True),
            "TEST_accessed": False, "model_calls": 0,
            "scope": "installed source/iterator bookkeeping only; no exact whole-paper replication claim"}


class OmittedTestIteratorRNG:
    def __init__(self, torch, audit_receipt, profile, model):
        self.torch, self.audit_receipt, self.profile = torch, audit_receipt, profile
        self.model = model
        self.count = 0

    @classmethod
    def create(cls, torch, adapter, job, policy, root):
        declaration = job.get("source_rng_emulation")
        if declaration is None:
            return None
        if job["family"] != "ogb" or policy.track != "source_reference" \
                or job["procedure"] != "native_single" or adapter.extension_receipt is not None \
                or not policy.evaluate_TRAIN_each_epoch:
            raise ValueError("Omitted TEST iterator emulation is source-reference native OGB M1 only")
        if declaration.get("mode") != MODE or not declaration.get("audit_sha256"):
            raise ValueError("Explicit version/source-bound iterator RNG audit required")
        audit_path = inside(declaration["audit"], root)
        pin = receipt(audit_path)
        if pin["sha256"] != declaration["audit_sha256"]:
            raise ValueError("Declared source RNG audit changed")
        audit = json.loads(audit_path.read_text())
        if audit.get("schema") != SCHEMA or audit.get("mode") != MODE or audit.get("assumptions") != ASSUMPTIONS:
            raise ValueError("Unknown source RNG audit/assumptions")
        profile = dependency_profile(torch, adapter.recipe["_redesign_root"])
        if audit.get("dependency_profile") != profile or audit.get("TEST_accessed") is not False \
                or audit.get("model_calls") != 0:
            raise ValueError("Installed dependency/source profile differs from the explicitly bound audit")
        semantic = audit.get("semantic_check", {})
        cases = semantic.get("cases", [])
        if len(cases) != 36 or {row.get("loader") for row in cases} != {"torch", "torch_tensor_dataset", "pyg"} \
                or any(row.get("exact_cpu_rng_state_match") is not True for row in cases):
            raise ValueError("The bound source RNG audit lacks the finite Torch/PyG loader checks")
        session = adapter.session
        expected_callback = inside(Path(adapter.recipe["_redesign_root"])/"smoke_20261014/ogb_port/session.py", root)
        if Path(adapter.callback_receipt["path"]).resolve() != expected_callback:
            raise ValueError("A different native callback requires a new stochastic-eval audit")
        if Path(inspect.getsourcefile(type(session.model))).resolve() != Path(profile["native_sources"]["gnn.py"]["path"]).resolve():
            raise ValueError("The active native model class differs from the audited author source")
        if session.config.get("feature") != "full" or adapter.train_generator is not None \
                or adapter.valid_generator is not None:
            raise ValueError("Only full-feature global-generator native source loaders are admitted")
        if session.model.graph_pooling != "mean" or getattr(session.train_view.dataset, "transform", None) is not None:
            raise ValueError("A transformed dataset or different pooling needs a new RNG audit")
        check_source_loader_flags(torch, adapter.train_loader, session.valid_loader)
        for module in session.model.modules():
            if any(getattr(module, name, None) for name in ("_forward_hooks", "_forward_pre_hooks")):
                raise ValueError("Model hooks require a fresh stochastic-eval source audit")
        adapter.dependency_receipts.append(pin)
        return cls(torch, pin, profile, session.model)

    def consume(self, completed_epoch):
        if completed_epoch != self.count+1:
            raise ValueError("Exactly one omitted iterator draw must follow each complete TRAIN/VALID epoch")
        if any(module.training for module in self.model.modules()):
            raise ValueError("Source iterator emulation must follow genuine model.eval VALID diagnostics")
        one_iterator_seed_draw(self.torch)
        self.count += 1
        return {"kind": MODE, "completed_epoch": completed_epoch, "emulation_count": self.count,
                "CPU_int64_seed_draws": 1, "TEST_data_accessed": False,
                "TEST_inference_or_metrics_computed": False}

    def metadata(self):
        return {"mode": MODE, "audit_receipt": self.audit_receipt,
                "dependency_versions": {key: value for key, value in self.profile.items() if key.endswith("_version")},
                "assumptions": ASSUMPTIONS, "TEST_accessed": False,
                "scope": "omitted source TEST iterator RNG effect only; not exact whole-paper replication"}
