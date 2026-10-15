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
from common import cpu_clone, digest, inside, load_module, metadata, receipt


class EpochComplete(Exception):
    pass


class BaseAdapter:
    def __init__(self, torch, job, policy, root):
        self.torch, self.job, self.policy, self.root = torch, job, policy, Path(root).resolve()
        self._job_authority = copy.deepcopy(job)
        self._validation_generation = self._validation_serial = 0
        self._same_bank_mean_cache = None
        self.recipe = copy.deepcopy(job["recipe"])
        self.recipe["_seed"] = job["seed"]
        self.recipe["seed"] = job["seed"]
        self.recipe["_redesign_root"] = str(inside(self.recipe["_redesign_root"], self.root))
        declared_root = self.recipe.get("_repository_root")
        if declared_root is not None and inside(declared_root, self.root) != self.root:
            raise ValueError("Recipe repository root differs from the actual execution repository")
        self.recipe["_repository_root"] = str(self.root)
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
        self._invalidate_validation()
        self.session.load_state(copy.deepcopy(value["session"]))
        self.completed_epochs = int(value["completed_epochs"])
        self.assert_identity()

    def _invalidate_validation(self):
        self._validation_generation += 1
        self._same_bank_mean_cache = None

    def _tensor_identity(self, value):
        value = value.detach().cpu().contiguous()
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(memoryview(value.numpy()).cast("B")).hexdigest()}

    def _capture_same_call_mean(self, raw, actual):
        """Only normalized data and optional fields from this exact native call."""
        self._validation_serial += 1
        self._same_bank_mean_cache = None
        keys = ("same_bank_mean_outputs", "same_bank_mean_loss", "same_bank_mean_metrics")
        present = [key in raw for key in keys]
        declarations = [getattr(self.session, name) for name in ("interaction_metadata",
            "conditional_aggregation_node_metadata", "conditional_aggregation_metadata") if hasattr(self.session, name)]
        if not any(present):
            if declarations:
                raise ValueError("Declared native mean provider omitted its same-call diagnostic")
            return  # Original ungated callback: no invented mean or extra forward.
        if not all(present) or raw.get("VALID_scope") != "complete_source_VALID" or raw.get("TEST_accessed") is not False:
            raise ValueError("Complete same-call native VALID mean fields and closed TEST role required")
        if len(declarations) != 1 or self.extension_receipt is None:
            raise ValueError("Mean diagnostics require one explicitly source-bound native extension")
        declaration = declarations[0]
        if declaration.get("method_id") != self._job_authority["procedure"]:
            raise ValueError("Mean provider method differs from immutable JOB")
        manifest_path = inside(Path(self.extension_receipt["path"]).parent / "MANIFEST.json", self.root)
        expected_manifest = self._job_authority.get("method_settings", {}).get("manifest_sha256")
        providers = [row for row in declaration.get("source_receipts", []) if row.get("path") == str(manifest_path)]
        if len(providers) != 1 or providers[0]["sha256"] != expected_manifest or receipt(manifest_path) != providers[0]:
            raise ValueError("Mean provider manifest does not match authenticated extension/JOB")
        torch = self.torch
        raw_ids = raw.get("VALID_node_ids", raw.get("VALID_graph_ids"))
        if isinstance(raw_ids, list) and all(type(v) is int for v in raw_ids):
            raw_ids = torch.tensor(raw_ids, dtype=torch.long)
        if not torch.is_tensor(raw_ids) or raw_ids.dtype != actual["ids"].dtype or not torch.equal(
                raw_ids.detach().cpu(), actual["ids"]):
            raise ValueError("Mean native VALID IDs/order differ from provider-normalized actual output")
        if "VALID_labels" in raw and self._tensor_identity(raw["VALID_labels"]) != self._tensor_identity(actual["labels"]):
            raise ValueError("Mean native VALID labels differ from normalized provider labels")
        mean = raw[keys[0]]
        if not torch.is_tensor(mean) or mean.shape != actual["outputs"].shape or mean.dtype != actual["outputs"].dtype \
                or not torch.isfinite(mean).all().item():
            raise ValueError("Finite same-call mean must retain actual logit shape/dtype")
        loss = float(raw[keys[1]])
        metrics = {key: float(value) for key, value in raw[keys[2]].items()}
        if not math.isfinite(loss) or set(metrics) != set(actual["metrics"]) or any(not math.isfinite(v) for v in metrics.values()):
            raise ValueError("Finite same-call mean loss and native metric keys required")
        authority = {"job_hash": digest(self._job_authority), "state_generation": self._validation_generation,
            "validation_serial": self._validation_serial, "provider_manifest": copy.deepcopy(providers[0]),
            "callback": copy.deepcopy(self.callback_receipt), "extension": copy.deepcopy(self.extension_receipt),
            "ids": self._tensor_identity(actual["ids"]), "labels": self._tensor_identity(actual["labels"]),
            "actual_logits": self._tensor_identity(actual["outputs"]), "mean_logits": self._tensor_identity(mean)}
        sidecar = {"schema": "same-call-native-mean-VALID-v1", "role": "VALID", "complete_source_VALID": True,
            "TEST_accessed": False, "ids": actual["ids"].detach().cpu().clone(),
            "labels": actual["labels"].detach().cpu().clone(), "outputs": mean.detach().cpu().clone(),
            "loss": loss, "metrics": metrics, "authority": authority,
            "selection_uses": "actual serving result only; fixed same-bank mean is diagnostic", "extra_forward_calls": 0}
        self._same_bank_mean_cache = cpu_clone(torch, sidecar)
        actual["same_bank_mean_VALID"] = sidecar

    def selected_mean_export(self, actual, *, job_hash, selected_epoch, selected_checkpoint, runtime_binding):
        """Bind this same-call diagnostic to the finalizer's own loaded selection."""
        value, cache = actual.get("same_bank_mean_VALID"), self._same_bank_mean_cache
        if value is None and cache is None:
            if any(hasattr(self.session, name) for name in ("interaction_metadata",
                    "conditional_aggregation_node_metadata", "conditional_aggregation_metadata")):
                raise ValueError("Declared mean provider has no authenticated current VALID diagnostic")
            return None
        if value is None or cache is None or value["authority"] != cache["authority"]:
            raise ValueError("Missing, stale or modified same-call mean authority")
        authority = cache["authority"]
        if job_hash != digest(self._job_authority) or authority["job_hash"] != job_hash \
                or authority["state_generation"] != self._validation_generation \
                or authority["validation_serial"] != self._validation_serial \
                or self.completed_epochs != selected_epoch:
            raise ValueError("Selected job/epoch/state generation differs from the validated state")
        if receipt(inside(selected_checkpoint["path"], self.root)) != selected_checkpoint:
            raise ValueError("Selected checkpoint bytes changed")
        if runtime_binding["callback_sha256"] != self.callback_receipt["sha256"] or runtime_binding[
                "extension_sha256"] != self.extension_receipt["sha256"]:
            raise ValueError("Selected runtime callback/extension authority differs")
        expected_runtime = {name: receipt(Path(__file__).with_name(name))["sha256"] for name in
            ("run_fit.py", "adapters.py", "bookkeeping.py", "common.py", "controllers.py", "source_rng.py")}
        if runtime_binding["runtime_sha256"] != expected_runtime or runtime_binding["policy_sha256"] is not None \
                or runtime_binding["dependency_sha256"] != {p["path"]: p["sha256"] for p in self.dependency_receipts}:
            raise ValueError("Selected runtime/policy/native dependency authority differs")
        for name, key in (("ids", "ids"), ("labels", "labels"), ("outputs", "actual_logits")):
            if self._tensor_identity(actual[name]) != authority[key]:
                raise ValueError("Actual selected VALID identity/logits changed after same-call capture")
        for name, key in (("ids", "ids"), ("labels", "labels"), ("outputs", "mean_logits")):
            if self._tensor_identity(value[name]) != authority[key]:
                raise ValueError("Mean selected VALID identity/logits changed after capture")
        if value["loss"] != cache["loss"] or value["metrics"] != cache["metrics"]:
            raise ValueError("Mean selected VALID summary changed after capture")
        return {"schema": "own-selected-same-bank-mean-VALID-v1", "role": "VALID", "TEST_accessed": False,
            "job_hash": job_hash, "job_id": self._job_authority["id"], "selected_epoch": selected_epoch,
            "selected_checkpoint": copy.deepcopy(selected_checkpoint), "runtime_binding": copy.deepcopy(runtime_binding),
            "ids": cache["ids"].clone(), "labels": cache["labels"].clone(), "raw_logits": cache["outputs"].clone(),
            "validation": {"role": "VALID", "complete_source_VALID": True, "records": len(cache["ids"]),
                "loss": cache["loss"], "metrics": copy.deepcopy(cache["metrics"]), "TEST_accessed": False},
            "same_call_authority": copy.deepcopy(authority), "extra_forward_calls": 0,
            "selection_changed": False, "diagnostic_only": True}

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
        original_job_authority = copy.deepcopy(job)
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
        self._job_authority = original_job_authority

    def train_epoch(self):
        self._invalidate_validation()
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
        self._same_bank_mean_cache = None
        torch = self.torch
        result = self.session.validate()
        ids = torch.nonzero(self.session.data.val_mask, as_tuple=False).flatten()
        labels = self.session.data.y[ids].detach().cpu()
        outputs = result["outputs"].detach().cpu()
        if len(outputs) != len(ids) or not torch.isfinite(outputs).all().item():
            raise ValueError("Complete finite official node VALID required")
        accuracy = float(outputs.argmax(-1).eq(labels).float().mean())
        value = {"outputs": outputs, "labels": labels, "ids": ids.detach().cpu(),
                "loss": float(result["loss"]), "metrics": {"accuracy": accuracy},
                "role": "VALID", "complete_source_VALID": True, "TEST_accessed": False}
        self._capture_same_call_mean(result, value)
        return value

    def train_diagnostic(self):
        self._invalidate_validation()
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
        self._invalidate_validation()
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
        self._invalidate_validation()
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
        self._same_bank_mean_cache = None
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
        value = {"outputs": outputs, "labels": labels, "ids": ids,
                "loss": float(result["loss"]),
                "metrics": {k: float(v) for k, v in result["metrics"].items()},
                "role": "VALID", "complete_source_VALID": True,
                "native_skipped_batches": result["native_skipped_batches"], "TEST_accessed": False}
        self._capture_same_call_mean(result, value)
        return value

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


class LinkSameCallMeanAdapter:
    """Authenticate an explicit link provider's existing same-call VALID sidecar."""
    def __init__(self, torch, job, policy, root, inner, provider):
        self.torch, self.root, self.inner, self.provider = torch, Path(root).resolve(), inner, provider
        self._job_authority = copy.deepcopy(job)
        self.policy = policy
        self.session, self.model, self.optimizers = inner.session, inner.model, inner.optimizers
        self.callback_receipt = copy.deepcopy(inner.callback_receipt)
        self.extension_receipt = copy.deepcopy(inner.extension_receipt)
        self.dependency_receipts = copy.deepcopy(inner.dependency_receipts)
        self.policy_receipt = receipt(inside(job['link_policy_module'], self.root))
        if self.policy_receipt['sha256'] != job['link_policy_sha256']:
            raise ValueError('Explicit link policy source differs from JOB')
        declaration = self.session.interaction_metadata
        manifest = inside(Path(self.extension_receipt['path']).parent / 'MANIFEST.json', self.root)
        providers = [p for p in declaration['source_receipts'] if p['path'] == str(manifest)]
        if declaration['method_id'] != job['procedure'] or len(providers) != 1 \
                or providers[0] != receipt(manifest) \
                or providers[0]['sha256'] != job['method_settings']['manifest_sha256']:
            raise ValueError('One exact outer method/manifest provider must match immutable JOB')
        self.provider_manifest = copy.deepcopy(providers[0])
        if not callable(getattr(provider, 'selected_mean_sidecar', None)):
            raise ValueError('Explicit link provider lacks a zero-forward same-call sidecar getter')
        self._generation = self._serial = 0
        self._cache = self._loaded_state_signature = self._selection = None

    @property
    def completed_epochs(self):
        return self.inner.completed_epochs

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _invalidate(self):
        self._generation += 1
        self._cache = self._selection = self._loaded_state_signature = None

    def _tensor_identity(self, value):
        if not self.torch.is_tensor(value):
            raise ValueError('Native link identity requires a tensor')
        value = value.detach().cpu().contiguous()
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest()}

    def _state_signature(self, value):
        if self.torch.is_tensor(value):
            return {'tensor': self._tensor_identity(value)}
        if isinstance(value, dict):
            return {'dict': [[type(k).__name__, str(k), self._state_signature(v)]
                             for k, v in sorted(value.items(), key=lambda item: (type(item[0]).__name__, str(item[0])))]}
        if isinstance(value, (list, tuple)):
            return {type(value).__name__: [self._state_signature(v) for v in value]}
        if value is None or type(value) in (str, int, float, bool):
            return {type(value).__name__: repr(value)}
        raise ValueError('Unsupported native checkpoint identity type: ' + type(value).__name__)

    def train_epoch(self):
        self._invalidate()
        return self.inner.train_epoch()

    def state(self):
        return self.inner.state()

    def load_state(self, value):
        self._invalidate()
        expected = self._state_signature(value)
        self.inner.load_state(value)
        if self._state_signature(self.inner.state()) != expected:
            raise ValueError('Exact native link state differs after source restore')
        self._loaded_state_signature = expected

    def _check_runtime(self, runtime_binding):
        expected = {name: receipt(Path(__file__).with_name(name))['sha256'] for name in
                    ('run_fit.py', 'adapters.py', 'bookkeeping.py', 'common.py', 'controllers.py', 'source_rng.py')}
        if runtime_binding != {'runtime_sha256': expected,
                'policy_sha256': self.policy_receipt['sha256'],
                'callback_sha256': self.callback_receipt['sha256'],
                'extension_sha256': self.extension_receipt['sha256'],
                'dependency_sha256': {p['path']: p['sha256'] for p in self.dependency_receipts}}:
            raise ValueError('Selected link runtime/policy/outer extension/dependency authority differs')
        for pin in (self.callback_receipt, self.extension_receipt, self.policy_receipt,
                    self.provider_manifest, *self.dependency_receipts):
            if receipt(inside(pin['path'], self.root)) != pin:
                raise ValueError('Authenticated link source changed')

    def bind_loaded_selection(self, payload, *, selected_checkpoint, job_hash, runtime_binding):
        """Called by the finalizer after it loads this exact selected payload."""
        self._check_runtime(runtime_binding)
        epoch = payload['selector']['selected_epoch']
        if payload['job_hash'] != job_hash or job_hash != digest(self._job_authority) \
                or payload['runtime_binding'] != runtime_binding \
                or self.completed_epochs != epoch or self._loaded_state_signature is None \
                or self._state_signature(payload['adapter']) != self._loaded_state_signature \
                or receipt(inside(selected_checkpoint['path'], self.root)) != selected_checkpoint:
            raise ValueError('Selected payload must be the exact source state just loaded for this JOB')
        self._selection = {'job_hash': job_hash, 'selected_epoch': epoch,
            'selected_checkpoint': copy.deepcopy(selected_checkpoint),
            'runtime_binding': copy.deepcopy(runtime_binding), 'state_generation': self._generation}

    def _source_sidecar(self, actual):
        t = self.torch
        value = self.provider.selected_mean_sidecar(self.session)
        if value.get('role') != 'VALID' or value.get('TEST_accessed') is not False \
                or value.get('method_id') != self._job_authority['procedure'] \
                or value.get('extra_forward_calls') != 0 \
                or value.get('source_updates') != self.session.native.updates \
                or value.get('selected_checkpoint_bound') is not False \
                or value.get('caller_must_bind_loaded_selected_epoch_checkpoint') is not True:
            raise ValueError('Current native source sidecar authority/role differs')
        for source_key, actual_key in (('ids', 'ids'), ('labels', 'labels'), ('query_endpoints', 'query_endpoints'),
                                      ('actual_served_raw_logits', 'outputs'), ('same_bank_mean_raw_logits', 'same_bank_mean_outputs')):
            if self._tensor_identity(value[source_key]) != self._tensor_identity(actual[actual_key]):
                raise ValueError('Same-call source and actual link tensor differ: ' + source_key)
        for source_key, actual_key in (('actual_loss', 'loss'), ('actual_metrics', 'metrics'),
                                      ('same_bank_mean_loss', 'same_bank_mean_loss'), ('same_bank_mean_metrics', 'same_bank_mean_metrics')):
            if value[source_key] != actual[actual_key]:
                raise ValueError('Same-call source and actual link summary differ: ' + source_key)
        if actual.get('role') != 'VALID' or actual.get('complete_source_VALID') is not True \
                or actual.get('TEST_accessed') is not False \
                or (actual.get('complete_VALID_positive_records'), actual.get('complete_VALID_negative_records')) != (60084, 100000):
            raise ValueError('Complete official link VALID role/counts required')
        endpoints = t.cat((self.session.native.valid['edge'], self.session.native.valid['edge_neg'])).cpu()
        if self._tensor_identity(actual['query_endpoints']) != self._tensor_identity(endpoints) \
                or actual['query_endpoints'].shape != (160084, 2) or actual['ids'].shape != (160084,) \
                or actual['labels'].shape != (160084,) or not bool((actual['labels'][:60084] == 1).all()) \
                or not bool((actual['labels'][60084:] == 0).all()):
            raise ValueError('Complete native query endpoints/order/positive-negative roles changed')
        mean, served = actual['same_bank_mean_outputs'], actual['outputs']
        if mean.shape != served.shape or mean.dtype != served.dtype or mean.numel() != 160084 \
                or not bool(t.isfinite(mean).all()) or not bool(t.isfinite(served).all()) \
                or set(actual['same_bank_mean_metrics']) != set(actual['metrics']) \
                or any(not math.isfinite(float(v)) for v in [actual['loss'], actual['same_bank_mean_loss'],
                    *actual['metrics'].values(), *actual['same_bank_mean_metrics'].values()]):
            raise ValueError('Finite same-call native mean/actual values and metric keys required')
        return value

    def validate(self):
        self._serial += 1
        self._cache = None
        before_updates, before_epoch = self.session.native.updates, self.completed_epochs
        actual = self.inner.validate()
        if (before_updates, before_epoch) != (self.session.native.updates, self.completed_epochs):
            raise ValueError('Native TRAIN state advanced during VALID')
        self._source_sidecar(actual)
        authority = {'job_hash': digest(self._job_authority), 'state_generation': self._generation,
            'validation_serial': self._serial, 'source_updates': before_updates,
            'completed_epoch': before_epoch, 'provider_manifest': copy.deepcopy(self.provider_manifest),
            'callback': copy.deepcopy(self.callback_receipt), 'extension': copy.deepcopy(self.extension_receipt),
            'policy': copy.deepcopy(self.policy_receipt),
            **{key: self._tensor_identity(actual[name]) for key, name in
               (('ids', 'ids'), ('labels', 'labels'), ('query_endpoints', 'query_endpoints'),
                ('actual_logits', 'outputs'), ('mean_logits', 'same_bank_mean_outputs'))}}
        sidecar = {'schema': 'same-call-native-link-mean-VALID-v1', 'role': 'VALID',
            'complete_source_VALID': True, 'TEST_accessed': False, 'authority': authority,
            'ids': actual['ids'].detach().cpu().clone(), 'labels': actual['labels'].detach().cpu().clone(),
            'query_endpoints': actual['query_endpoints'].detach().cpu().clone(),
            'outputs': actual['same_bank_mean_outputs'].detach().cpu().clone(),
            'loss': float(actual['same_bank_mean_loss']), 'metrics': copy.deepcopy(actual['same_bank_mean_metrics']),
            'complete_VALID_positive_records': 60084, 'complete_VALID_negative_records': 100000,
            'selection_uses': 'actual declared serving only; same-bank mean is diagnostic', 'extra_forward_calls': 0}
        self._cache = cpu_clone(self.torch, sidecar)
        actual['same_bank_mean_VALID'] = sidecar
        return actual

    def selected_mean_export(self, actual, *, job_hash, selected_epoch, selected_checkpoint, runtime_binding):
        self._check_runtime(runtime_binding)
        selection = {'job_hash': job_hash, 'selected_epoch': selected_epoch,
            'selected_checkpoint': selected_checkpoint, 'runtime_binding': runtime_binding,
            'state_generation': self._generation}
        value, cache = actual.get('same_bank_mean_VALID'), self._cache
        if self._selection != selection or self.completed_epochs != selected_epoch \
                or job_hash != digest(self._job_authority) \
                or receipt(inside(selected_checkpoint['path'], self.root)) != selected_checkpoint \
                or value is None or cache is None or value['authority'] != cache['authority']:
            raise ValueError('Fresh exact loaded selection and current same-call mean authority required')
        self._source_sidecar(actual)  # Getter only; no model or native VALID call.
        authority = cache['authority']
        if (authority['state_generation'], authority['validation_serial'], authority['source_updates']) != \
                (self._generation, self._serial, self.session.native.updates):
            raise ValueError('Stale native link validation generation/update/serial')
        for key, name in (('ids', 'ids'), ('labels', 'labels'), ('query_endpoints', 'query_endpoints'), ('actual_logits', 'outputs')):
            if self._tensor_identity(actual[name]) != authority[key]:
                raise ValueError('Selected actual link VALID tensor changed after capture')
        for key, name in (('ids', 'ids'), ('labels', 'labels'), ('query_endpoints', 'query_endpoints'), ('mean_logits', 'outputs')):
            if self._tensor_identity(value[name]) != authority[key]:
                raise ValueError('Selected link mean tensor changed after capture')
        if value['loss'] != cache['loss'] or value['metrics'] != cache['metrics']:
            raise ValueError('Selected mean summary changed after capture')
        return {'schema': 'own-selected-same-bank-mean-VALID-v1', 'role': 'VALID', 'TEST_accessed': False,
            'job_hash': job_hash, 'job_id': self._job_authority['id'], 'selected_epoch': selected_epoch,
            'selected_checkpoint': copy.deepcopy(selected_checkpoint), 'runtime_binding': copy.deepcopy(runtime_binding),
            'ids': cache['ids'].clone(), 'labels': cache['labels'].clone(), 'query_endpoints': cache['query_endpoints'].clone(),
            'raw_logits': cache['outputs'].clone(), 'loss': cache['loss'], 'metrics': copy.deepcopy(cache['metrics']),
            'complete_VALID_positive_records': 60084, 'complete_VALID_negative_records': 100000,
            'authority': copy.deepcopy(authority), 'extra_forward_calls': 0,
            'selection_uses': 'actual declared serving only; same-bank mean is diagnostic'}


def _link_source_dependencies(adapter, pin, root):
    """Retain all original integration sources and authenticated outer method sources."""
    adapter.callback_receipt = pin
    if not hasattr(adapter, 'extension_receipt'):
        adapter.extension_receipt = None
    meta = adapter.metadata()['native_integration']
    native = meta['native_binding']
    rows = [*meta['integration_sources'], *meta['finalized_helper_sources'],
        *native['adapter_source_receipts'].values(), native['adapter_manifest_receipt']]
    if adapter.extension_receipt is not None:
        if receipt(inside(adapter.extension_receipt['path'], root)) != adapter.extension_receipt:
            raise ValueError('Outer link extension source receipt differs')
        declaration = getattr(adapter.session, 'interaction_metadata', None)
        if declaration is None:
            raise ValueError('Extended link provider needs an explicit authenticated source declaration')
        rows += [adapter.extension_receipt, *declaration['source_receipts']]
    unique = {}
    for row in rows:
        actual = receipt(inside(row['path'], root))
        if actual != row or (row['path'] in unique and unique[row['path']] != row):
            raise ValueError('Native/outer link dependency receipt differs or conflicts')
        unique[row['path']] = row
    adapter.dependency_receipts = [unique[name] for name in sorted(unique)]
    return adapter


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
        adapter = _link_source_dependencies(adapter, pin, root)
        if adapter.extension_receipt is not None:
            provider, _ = load_module(adapter.extension_receipt['path'], root, adapter.extension_receipt['sha256'])
            return LinkSameCallMeanAdapter(torch, job, policy, root, adapter, provider)
        return adapter
    raise ValueError("This adapter supports actual native node and molecular sessions only")
