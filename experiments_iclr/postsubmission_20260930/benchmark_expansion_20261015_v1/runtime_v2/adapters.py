"""Full-epoch TRAIN/VALID drivers around the frozen native smoke sessions.

No native model/operator/optimizer or frozen smoke source is rewritten. Model
libraries are imported only when root executes create_adapter on the allocation.
"""
from __future__ import annotations
import copy
import hashlib
import math
from pathlib import Path

from bookkeeping import epoch_boundary_record
from common import inside, load_module, metadata, receipt


class EpochComplete(Exception):
    pass


class BaseAdapter:
    def __init__(self, torch, job, policy, root):
        self.torch, self.job, self.policy, self.root = torch, job, policy, Path(root).resolve()
        self.recipe = copy.deepcopy(job["recipe"])
        self.recipe["_seed"] = job["seed"]
        self.recipe["seed"] = job["seed"]
        self.recipe["_redesign_root"] = str(inside(self.recipe["_redesign_root"], self.root))
        self.completed_epochs = 0
        source, self.callback_receipt = load_module(job["callback_module"], root, job.get("callback_sha256"))
        extension = job.get("session_transform", job.get("extension_factory", job.get("session_extension")))
        self.base_procedure = job.get("base_procedure", "gnnm_boundary_native" if extension else job["procedure"])
        self.session = source.create_session(self.recipe, self.base_procedure,
                                             inside(job["data_root"], root), job.get("device", "cuda:0"))
        self.extension_receipt = None
        if extension:
            if policy.track == "source_reference":
                raise ValueError("A source-reference baseline cannot have a scientific extension")
            if isinstance(extension, str):
                extension_path, separator, function = extension.partition(":")
                extension = {"path": extension_path, "function": function if separator else "extend_session"}
            module, self.extension_receipt = load_module(extension["path"], root, extension.get("sha256"))
            result = getattr(module, extension.get("function", "extend_session"))(self.session, copy.deepcopy(job))
            if result is not None:
                self.session = result
        self.model = self.session.model
        self.optimizers = self.session.optimizers
        if len(self.optimizers) != 1:
            raise ValueError("Each native body/shared bank is one own fit; I4 is four separate fits")
        self._anchors = dict(self.session.source_parameter_references)
        redesign = Path(self.recipe["_redesign_root"])
        dependencies = [redesign / "runtime/training_rules.py", redesign / "runtime/boundary_core.py"]
        if job["family"] == "node":
            dependencies += [redesign / "native_sources/polyformer/node_classification" / relative
                             for relative in ("utils.py", "mymodels.py", "layers/PolyFormerBlock.py")]
        else:
            dependencies += [redesign / "smoke_20261014/ogb_port" / relative
                             for relative in ("native_loader.py", "categorical_boundary.py")]
            dependencies += [redesign / "native_sources/ogb" / relative for relative in ("gnn.py", "conv.py")]
        self.dependency_receipts = [receipt(inside(path, root)) for path in dependencies]
        self.assert_identity()

    def assert_identity(self):
        current = {id(p) for p in self.model.parameters()}
        if getattr(self.session, "replaced_source_parameter_names", ()) or any(id(p) not in current for p in self._anchors.values()):
            raise ValueError("Original native source parameters must remain in the shared backbone")
        owners = [p for opt in self.optimizers for group in opt.param_groups for p in group["params"]]
        if len({id(p) for p in owners}) != len(owners) or any(id(p) not in current for p in owners):
            raise ValueError("Optimizer ownership changed or aliases parameters")

    def state(self):
        return {"session": self.session.state(), "completed_epochs": self.completed_epochs}

    def load_state(self, value):
        self.session.load_state(copy.deepcopy(value["session"]))
        self.completed_epochs = int(value["completed_epochs"])
        self.assert_identity()

    def metadata(self):
        return {"family": self.job["family"], "callback_receipt": self.callback_receipt,
                "extension_receipt": self.extension_receipt,
                "source_dependency_receipts": self.dependency_receipts,
                "native_session_metadata": metadata(self.session),
                "full_epoch_driver": True, "TEST_accessed": False,
                "checkpoint_scope": "completed_native_epoch_boundary",
                "source_track_fidelity": "native model/settings/horizon with explicitly omitted TEST diagnostics; not exact whole-paper replication"
                    if self.policy.track == "source_reference" else "declared matched training controller; not paper-equivalent"}

    def close(self):
        close = getattr(self.session, "close", None)
        if close:
            close()


class NodeAdapter(BaseAdapter):
    def __init__(self, torch, job, policy, root):
        job = copy.deepcopy(job)
        split = str(job.get("split", 0))
        bindings = job["recipe"]["official_split_bindings"]
        if split not in bindings:
            raise ValueError("Frozen native node split binding missing")
        # The frozen smoke frontend reads key0. Supply the selected source split
        # in this new session recipe, rather than modifying its source file.
        selected = copy.deepcopy(bindings[split])
        job["recipe"]["official_split_bindings"]["0"] = selected
        self.selected_split_binding = selected
        super().__init__(torch, job, policy, root)

    def train_epoch(self):
        result = self.session.train_update()
        loss = float(result["loss"])
        if not math.isfinite(loss):
            raise ValueError("Nonfinite native full-TRAIN node loss")
        self.completed_epochs += 1
        return {"epoch": self.completed_epochs, "complete_native_TRAIN": True,
                "TRAIN_records_traversed": len(self.session.train_ids), "raw_batches": 1,
                "optimizer_maps": 1, "source_skipped_batches": 0, "mean_update_loss": loss,
                "native_work": result.get("operation_counts", result.get("work")),
                "activation_checkpoint_recomputations_not_counted": bool(result.get("activation_checkpointing"))}

    def validate(self):
        torch = self.torch
        result = self.session.validate()
        ids = torch.nonzero(self.session.data.val_mask, as_tuple=False).flatten()
        labels = self.session.data.y[ids].detach().cpu()
        outputs = result["outputs"].detach().cpu()
        if len(outputs) != len(ids) or not torch.isfinite(outputs).all().item():
            raise ValueError("Complete finite official node VALID required")
        accuracy = float(outputs.argmax(-1).eq(labels).float().mean())
        return {"outputs": outputs, "labels": labels, "ids": ids.detach().cpu(),
                "loss": float(result["loss"]), "metrics": {"accuracy": accuracy},
                "role": "VALID", "complete_source_VALID": True, "TEST_accessed": False}

    def train_diagnostic(self):
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            if self.session.core is None:
                outputs = self.model(self.session.data)
            else:
                outputs = torch.stack([r.raw_logits for r in self.session._routes(False)]).mean(0)
            targets = self.session.data.y[self.session.train_ids]
            chosen = outputs[self.session.train_ids]
            return {"accuracy": float(chosen.argmax(-1).eq(targets).float().mean()),
                    "loss": float(torch.nn.functional.cross_entropy(chosen, targets)),
                    "role": "TRAIN", "diagnostic_only": True}

    def close(self):
        if self.session.core is not None:
            self.session.core.remove_hooks()
        super().close()


class OGBAdapter(BaseAdapter):
    def __init__(self, torch, job, policy, root):
        super().__init__(torch, job, policy, root)
        from torch_geometric.loader import DataLoader
        self._iterator = None
        self._seen = []
        self._raw_batches = self._skipped = 0
        self._position = {int(graph_id): i for i, graph_id in enumerate(self.session.train_ids.tolist())}
        self.train_generator = None if policy.track == "source_reference" else \
            torch.Generator(device="cpu").manual_seed(int(job["seed"]) + 70000001)
        self.valid_generator = None if policy.track == "source_reference" else self.session.valid_generator
        self.train_loader = DataLoader(self.session.train_view, batch_size=32, shuffle=True,
                                       num_workers=0, generator=self.train_generator)
        self.session.valid_loader = DataLoader(self.session.valid_view, batch_size=32, shuffle=False,
                                               num_workers=0, generator=self.valid_generator)
        # Replace only the input iterator/cursor interface in this new adapter.
        # The original model, native loss/update, Adam and skip law remain intact.
        self.session._next_train_batch = self._next_train_batch
        from source_rng import OmittedTestIteratorRNG
        self.source_rng_emulator = OmittedTestIteratorRNG.create(torch, self, job, policy, root)

    def _next_train_batch(self):
        if self._iterator is None:
            raise ValueError("Full TRAIN batches can be requested only inside a declared epoch")
        skipped = 0
        while True:
            try:
                batch = next(self._iterator)
            except StopIteration:
                raise EpochComplete() from None
            ids = batch.smoke_graph_id.detach().cpu().reshape(-1).tolist()
            positions = [self._position[int(graph_id)] for graph_id in ids]
            self._seen.extend(positions)
            self._raw_batches += 1
            if batch.x.shape[0] == 1 or int(batch.batch[-1]) == 0:
                self._skipped += 1
                skipped += 1
                continue
            return batch.to(self.session.device), self.torch.tensor(positions, dtype=self.torch.long), skipped

    def train_epoch(self):
        if self._iterator is not None:
            raise ValueError("Nested epochs are forbidden")
        self._seen, self._raw_batches, self._skipped = [], 0, 0
        self._iterator = iter(self.train_loader)
        losses, work = [], []
        try:
            while True:
                try:
                    result = self.session.train_update()
                except EpochComplete:
                    break
                loss = float(result["loss"])
                if not math.isfinite(loss):
                    raise ValueError("Nonfinite real native OGB batch loss")
                losses.append(loss)
                work.append(result.get("operation_counts", result.get("work")))
        finally:
            self._iterator = None
        counts = epoch_boundary_record(len(self.session.train_view), self._seen,
                                       self._raw_batches, len(losses), self._skipped)
        if not losses:
            raise ValueError("A native epoch with zero eligible optimizer maps is not a fit")
        self.completed_epochs += 1
        work_totals = {}
        for row in work:
            for key, value in (row or {}).items():
                if type(value) in (int, float) and math.isfinite(value):
                    work_totals[key] = work_totals.get(key, 0) + value
        return {**counts, "epoch": self.completed_epochs,
                "mean_update_loss": sum(losses) / len(losses),
                "mean_update_loss_is_not_graph_weighted": True,
                "native_work_totals": work_totals,
                "visited_native_TRAIN_order_sha256": hashlib.sha256(
                    self.torch.tensor(self._seen, dtype=self.torch.int64).numpy().tobytes()).hexdigest()}

    def train_diagnostic(self):
        """Native source's shuffled TRAIN eval: genuine loader RNG consumption."""
        torch, session = self.torch, self.session
        self.model.eval()
        predictions, labels = [], []
        with torch.no_grad():
            for batch in self.train_loader:
                batch = batch.to(session.device)
                if batch.x.shape[0] == 1:
                    continue
                if self.base_procedure == "native_single":
                    prediction = self.model(batch)
                else:
                    prediction = torch.stack([self.model.run_member(m, batch)[1] for m in range(4)]).mean(0)
                predictions.append(prediction.detach().cpu())
                labels.append(batch.y.view(prediction.shape).detach().cpu())
        if not predictions:
            raise ValueError("Native TRAIN diagnostic has no valid graph outputs")
        metrics = session.evaluator.eval({"y_true": torch.cat(labels).numpy(), "y_pred": torch.cat(predictions).numpy()})
        return {**{k: float(v) for k, v in metrics.items()}, "role": "TRAIN", "diagnostic_only": True,
                "native_shuffled_TRAIN_loader_RNG_consumed": True}

    def validate(self):
        torch = self.torch
        result = self.session.validate()
        ids = torch.tensor(result["VALID_graph_ids"], dtype=torch.long)
        store = getattr(self.session.valid_view.dataset, "_data", None)
        if store is None:
            store = self.session.valid_view.dataset.data
        labels = store.y.index_select(0, ids).detach().cpu()
        outputs = result["outputs"].detach().cpu()
        if len(ids) != len(outputs) or not torch.isfinite(outputs).all().item():
            raise ValueError("Complete finite source VALID graph outputs required")
        return {"outputs": outputs, "labels": labels, "ids": ids,
                "loss": float(result["loss"]),
                "metrics": {k: float(v) for k, v in result["metrics"].items()},
                "role": "VALID", "complete_source_VALID": True,
                "native_skipped_batches": result["native_skipped_batches"], "TEST_accessed": False}

    def state(self):
        if self._iterator is not None:
            raise ValueError("Checkpoint only after a complete native epoch; live DataLoader is not serializable")
        return {**super().state(), "source_rng_emulation_count": 0 if self.source_rng_emulator is None else self.source_rng_emulator.count,
                "train_generator": None if self.train_generator is None else self.train_generator.get_state().clone(),
                "loader_policy": "source_global_RNG" if self.train_generator is None else "matched_private_persistent_generator"}

    def load_state(self, value):
        super().load_state(value)
        expected = "source_global_RNG" if self.train_generator is None else "matched_private_persistent_generator"
        if value["loader_policy"] != expected:
            raise ValueError("Cannot change loader RNG protocol on resume")
        if self.train_generator is not None:
            self.train_generator.set_state(value["train_generator"].cpu())
        count = int(value["source_rng_emulation_count"])
        if self.source_rng_emulator is None and count != 0:
            raise ValueError("Resume cannot omit the declared source RNG emulation")
        if self.source_rng_emulator is not None:
            if count != self.completed_epochs:
                raise ValueError("Source RNG emulation count differs from completed epochs")
            self.source_rng_emulator.count = count

    def after_epoch_diagnostics(self):
        return None if self.source_rng_emulator is None else self.source_rng_emulator.consume(self.completed_epochs)

    def metadata(self):
        return {**super().metadata(), "batch_size": 32, "native_skip_law_preserved": True,
                "omitted_TEST_iterator_rng_emulation": None if self.source_rng_emulator is None else self.source_rng_emulator.metadata(),
                "sampler": "native PyG DataLoader RandomSampler, complete epoch; no next-epoch crossover",
                "loader_RNG": "global source loader RNG" if self.train_generator is None else "isolated persistent paired training generator",
                "native_TRAIN_evaluation_each_epoch": self.policy.evaluate_TRAIN_each_epoch,
                "native_TEST_diagnostic_omitted": True,
                "native_TEST_iterator_RNG_effect_omitted": self.source_rng_emulator is None,
                "whole_paper_exact_RNG_replication_claim": False}


def create_adapter(torch, job, policy, root):
    if job["family"] == "node":
        return NodeAdapter(torch, job, policy, root)
    if job["family"] == "ogb":
        return OGBAdapter(torch, job, policy, root)
    if job["family"] == "link":
        if not job.get("callback_sha256"):
            raise ValueError("Link entrypoint must be explicitly pinned")
        if any(job.get(key) for key in ("session_transform", "extension_factory", "session_extension")):
            raise ValueError("Use the pinned link integration's explicit candidate IDs, not a generic transform")
        module, pin = load_module(job["callback_module"], root, job["callback_sha256"])
        adapter = module.create_adapter(torch, job, policy, root)
        adapter.callback_receipt, adapter.extension_receipt = pin, None
        meta = adapter.metadata()["native_integration"]
        native = meta["native_binding"]
        adapter.dependency_receipts = [*meta["integration_sources"], *meta["finalized_helper_sources"],
            *native["adapter_source_receipts"].values(), native["adapter_manifest_receipt"]]
        return adapter
    raise ValueError("This adapter supports actual native node and molecular sessions only")
