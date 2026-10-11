"""Use the frozen F Family fit/selector/restore; add only one PE arm.

No numerical imports or CLI. Root must supply the existing owner module and
this packet's packed module, then admit acquisition with a bound receipt.
"""
import hashlib
import json
from pathlib import Path
import time

OWNER_SHA256 = "5ea88c5eba68df512c20ea10ee778d25f46499705059a91cb80104c4fc8943c4"


def family_class(owner, packed):
    if hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest() != OWNER_SHA256:
        raise ValueError("The prepared adapter requires the unchanged frozen F owner")
    class PESAGEFamily(owner.Family):
        def __init__(self, cfg, output, config_sha256, seed_block=None):
            protocol = Path(__file__).with_name("PROTOCOL.json")
            receipt_path = owner.file_receipt(cfg.get("root_fit_admission"))
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("enabled") is not True
                    or receipt.get("scope") != "native_SAGE_PE4_gamma1_BASE_parameter_cap"
                    or receipt.get("protocol_sha256") != hashlib.sha256(protocol.read_bytes()).hexdigest()
                    or receipt.get("full_TRAIN_qualification_pass") is not True
                    or receipt.get("external_finite_supervision_admitted") is not True):
                raise ValueError("This preparation packet has no scientific fit admission")
            fixed = {"stage": "F", "K": None, "k_selection_receipt": None,
                     "depth": 2, "hidden": 128, "dropout": .2, "learning_rate": .001,
                     "max_updates": 1000, "patience_updates": 300,
                     "member_seed_stride": 1000003, "dropout_seed_offset": 3000007,
                     "seeds": [7301, 7403, 7507]}
            if any(cfg.get(k) != v for k, v in fixed.items()):
                raise ValueError("Frozen F/BASE contract or fixed PE roster differs")
            packed.width_contract()
            super().__init__(cfg, output, config_sha256, seed_block)
            for file in (Path(__file__), Path(packed.__file__), protocol):
                self.source_hashes[str(file.resolve())] = hashlib.sha256(file.read_bytes()).hexdigest()
            owner.write_json(output / "SOURCE_HASHES.json", {"config_sha256": config_sha256, "files": self.source_hashes})

        def make(self, seed, member, kind, factorized, members, architecture="shared"):
            if (member, kind, factorized, members, architecture) != (0, "baseline", False, 4, "packed_gamma1"):
                raise ValueError("Only the one frozen PE4 arm is prepared")
            torch, bodies = self.torch, []
            for m in range(4):
                self.set_seed(self.seeds(seed, m)[0])
                bodies.append(self.Model("SAGE", 2, 300, packed.WIDTH, 10, 1, 8, "LayerNorm", .2))
            model = packed.pack_native(torch, bodies)
            del bodies  # Packed coordinates are the sole owner before .to/Adam.
            model.to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
            model.check_optimizer_ownership(optimizer)
            streams = []
            for m in range(4):
                self.set_seed(self.seeds(seed, m)[2])
                stream = {"cpu": torch.get_rng_state()}
                if self.device.type == "cuda":
                    stream["cuda"] = torch.cuda.get_rng_state(self.device)
                streams.append(stream)
            self.packed_model = model
            return model, optimizer, streams

        def body_records(self, seed, member, members, architecture, factorized, checkpoint, selected_step):
            return [{"body_index": m, "native_initialization_seed": self.seeds(seed, m)[0],
                     "factorized": False, "native_parameter_scope": "member-axis slice of every grouped private affine/norm tensor",
                     "packed_member_axis": 0, "packed_member_index": m,
                     "route_factor_dropout_seeds": [{"route": m, "factor_seed": None, "dropout_seed": self.seeds(seed, m)[2]}],
                     "selected_step": selected_step, "selected_state": str(checkpoint), "selection_scope": "coherent_bank"}
                    for m in range(4)]

        def fit_pe_bank(self, folder, seed):
            # Inherited fit is unchanged: F mean own CE, every TRAIN row/member,
            # strict probability-accuracy selector, full model/Adam/streams restore.
            logits, result = super().fit(folder, seed, 0, "baseline", False, 4,
                                         "own_only", False, "packed_gamma1")
            result["packed_width_contract"] = packed.width_contract()
            result["operation_counts"].update(self.packed_model.operation_counts)
            result["temporary_native_construction_and_packing_in_acquisition"] = True
            owner.write_json(folder / "RESULT.json", result)
            del self.packed_model
            return logits, result

        def run(self):
            torch, np, results = self.torch, self.np, []
            owner.write_json(self.output / "CONFIG.json", self.cfg)
            owner.write_json(self.output / "LEARNING_LOCK.json", self.learning_lock)
            for seed in self.seed_block:
                arm = "PE4_gamma1_F"
                self.current.update(arm=arm, seed=seed)
                folder = self.output / f"{arm}_seed{seed}"
                folder.mkdir()
                logits, unit = self.fit_pe_bank(folder / "member0", seed)
                start = time.perf_counter()
                metrics = self.metrics(logits)
                probs, y = logits.softmax(-1), self.valid_y.cpu()
                pool = probs.mean(0)
                costs = dict(unit["costs"])
                costs["selected_pool_seconds"] = time.perf_counter() - start
                export_start = time.perf_counter()
                np.savez(folder / "selected_VALID.npz", ids=self.valid_ids.cpu().numpy(), y=y.numpy(),
                         raw_logits=logits.numpy(), probability_mean=pool.numpy(),
                         member_errors=probs.argmax(-1).ne(y).numpy(), pooled_errors=pool.argmax(-1).ne(y).numpy())
                costs["selected_export_seconds"] = time.perf_counter() - export_start
                costs["selected_serving_readout_seconds"] = costs["selected_forward_and_metrics_seconds"] + costs["selected_pool_seconds"]
                costs["selected_restore_forward_and_pool_seconds"] = costs["selected_restore_seconds"] + costs["selected_serving_readout_seconds"]
                costs["acquisition_restore_selected_readout_seconds"] = costs["acquisition_seconds"] + costs["selected_restore_forward_and_pool_seconds"]
                result = {"arm": arm, "seed": seed, "stage": "F", "K": None,
                          "architecture": "packed_gamma1", "serving": "probability_mean", "valid": metrics,
                          "optimizer_acquisition_bundles": 1, "native_bodies_fitted": 4,
                          "native_trajectories_per_forward": 4, "selection_scope": "coherent_bank",
                          "fits": [unit], "costs": costs}
                owner.write_json(folder / "RESULT.json", result)
                results.append(result)
            complete = self.seed_block == self.cfg["seeds"]
            counts = {k: sum(r["fits"][0]["operation_counts"][k] for r in results)
                      for k in results[0]["fits"][0]["operation_counts"]}
            owner.write_json(self.output / "COMPLETE_PE_BANKS.json", {
                "complete": complete, "completion_scope": "PE arm only; comparison/anchor custody remains root-owned",
                "fixed_seed_roster": self.cfg["seeds"], "completed_seed_block": self.seed_block,
                "fixed_arm_roster": ["PE4_gamma1_F"], "groups": len(results),
                "optimizer_acquisition_bundles": len(results), "native_body_fit_records": 4 * len(results),
                "expected_groups": 3, "expected_bundles": 3, "expected_body_records": 12,
                "config_sha256": self.config_sha256, "source_sha256": self.source_hashes,
                "width_contract": packed.width_contract(), "operation_counts": counts, "results": results,
                "TEST_access": False, "comparison_interpreted": False,
                "quality_reference": "complete full-width genuine_factorized_I4_F",
                "cost_scope": "declared grouped native graph adaptation; all member edge-channel work paid; no speed or exact author-code claim"})

    return PESAGEFamily
