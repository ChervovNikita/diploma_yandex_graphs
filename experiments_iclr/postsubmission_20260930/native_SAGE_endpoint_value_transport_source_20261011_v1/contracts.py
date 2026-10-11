"""Finite metadata admission; import/read metadata only before numerical code."""
import hashlib
import json
from pathlib import Path
import socket

SEEDS = [7301, 7403, 7507]
ARMS = ["BASE", "E4", "N4", "equipped_M1", "equipped_I4"]
FIXED = {"depth": 2, "hidden": 128, "dropout": 0.2, "learning_rate": 0.001,
         "max_updates": 1000, "patience_updates": 300, "rank": 16,
         "member_seed_stride": 1000003, "factor_seed_offset": 2000003,
         "dropout_seed_offset": 3000007, "block_seed_offset": 4000037,
         "stalk_dim": 4, "baseline_policy": "fresh_required",
         "stage": "F", "K": None, "k_selection_receipt": None}
SOURCE_KEYS = {"training_family", "value_installer", "value_helper", "contracts",
               "common_routes", "factors", "models", "run_base", "run_common",
               "sage_conv", "message_passing"}
VALUE_COUNT_KEYS = ("value_conv_calls", "value_builder_calls", "value_builder_rows",
    "value_builder_MACs", "value_pair_matrix_products", "value_pair_matrix_MACs",
    "value_message_rows", "value_message_application_MACs", "value_self_mask_rows",
    "value_self_rows_preserved", "value_gathered_source_coordinates",
    "value_gathered_target_coordinates", "value_context_coordinates",
    "value_generated_incidence_coordinates", "value_generated_transport_coordinates",
    "value_node_transport_gather_coordinates")
PROTOCOL = {"schema_version": 1, "study_kind": "native_SAGE_endpoint_VALUE",
    "arms": ARMS, "seeds": SEEDS, "banks": 15, "optimizer_bundles": 24,
    "native_body_records": 24, "native_routes_per_roster_forward": 51,
    "primary_candidate": "E4", "primary_references": ["BASE", "equipped_I4"],
    "raw_mean_accuracy_gain_pp": 0.2, "raw_all_three_nonnegative": True,
    "raw_at_least_two_positive": True,
    "endpoint_attribution_reference": "N4", "endpoint_mean_accuracy_gain_pp": 0.1,
    "endpoint_all_three_nonnegative": True, "endpoint_at_least_two_positive": True,
    "equipped_M1_reported_without_posthoc_selection": True,
    "calibrated_mean_NLL_deterioration_at_most": 0.02,
    "calibrated_each_seed_NLL_deterioration_at_most": 0.05,
    "primary_accuracy_and_confidence_separate": True,
    "member_quality_is_diagnostic_not_a_veto": True, "costs_are_diagnostic_not_a_gate": True,
    "serving": "probability_mean", "objective": "mean own CE for shared; unscaled own CE for singles/I4",
    "calibration": {"folds": 5, "fold_seed": 11709, "scalar_fits": 75,
        "updates": 500, "optimizer": "CPU float64 Adam", "learning_rate": 0.01,
        "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0,
        "initial_log_T": 0, "endpoint": "fixed final update",
        "operator": "temperature of member log probabilities before probability mean",
        "retries": 0, "grid": False, "all_VALID_refit": False},
    "all_banks_before_comparison": True, "all75_scalar_attempts_before_interpretation": True,
    "TEST_access": False, "automatic_confirmation_launch": False,
    "baseline_policy": "fresh_required", "historical_logit_anchors": False,
    "scope": "Encountered WikiCS development, three optimizer repeats on one complete graph; no novelty or significance claim"}


def receipt(item):
    if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
        raise ValueError("Exact portable path/bytes/sha256 receipt required")
    p = Path(item["path"]).resolve()
    if not p.is_file() or type(item["bytes"]) is not int or p.stat().st_size != item["bytes"]:
        raise ValueError("Source/role receipt size differs: " + str(p))
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        raise ValueError("Source/role digest differs: " + str(p))
    return p


def validate(cfg, *, action, executing_source):
    if action not in ("full_TRAIN_qualification", "fixed_family_fit"):
        raise ValueError("One explicit qualification or finite-family action")
    if cfg.get("seeds") != SEEDS or any(cfg.get(key) != value for key, value in FIXED.items()):
        raise ValueError("Fixed five-arm/three-seed native recipe differs; no grid or rescue")
    limits = cfg.get("family_limits")
    if (not isinstance(limits, dict) or set(limits) != {"wall_seconds", "cuda_bytes", "rss_bytes"}
            or any(type(value) is not int or value <= 0 for value in limits.values())
            or limits["wall_seconds"] > 7200):
        raise ValueError("Finite root full-family wall/CUDA/RSS limits required")
    for key in ("native_repo", "common_routes", "factors", "train_npz", "valid_npz", "device"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise ValueError("Root must supply portable " + key)
    if not cfg["device"].startswith("cuda"):
        raise ValueError("The fixed actual full-graph family uses the admitted CUDA runtime")
    sources, roles = cfg.get("source_receipts"), cfg.get("safe_roles")
    if not isinstance(sources, dict) or set(sources) != SOURCE_KEYS:
        raise ValueError("Pin all native, wrapper, VALUE and actual PyG sources")
    if not isinstance(roles, dict) or set(roles) != {"train", "valid"}:
        raise ValueError("Only safe TRAIN/VALID role receipts permitted")
    paths = {key: receipt(value) for key, value in sources.items()}
    role_paths = {key: receipt(value) for key, value in roles.items()}
    expected = {"training_family": Path(executing_source).resolve(),
                "value_installer": Path(executing_source).resolve().with_name("value_installer.py"),
                "value_helper": Path(executing_source).resolve().with_name("value_transport.py"),
                "contracts": Path(__file__).resolve(),
                "common_routes": Path(cfg["common_routes"]).resolve(),
                "factors": Path(cfg["factors"]).resolve()}
    expected.update({key: (Path(cfg["native_repo"]) / (key + ".py")).resolve()
                     for key in ("models", "run_base", "run_common")})
    if any(paths[key] != value for key, value in expected.items()):
        raise ValueError("Configuration/source receipt path mapping differs")
    if any(role_paths[key] != Path(cfg[key + "_npz"]).resolve() for key in ("train", "valid")):
        raise ValueError("Exact safe role path binding differs")
    admission = json.loads(receipt(cfg.get("root_admission")).read_text())
    if (admission.get("status") != "HOST_GPU_SOURCE_DATA_ADMITTED"
            or admission.get("action") != action
            or admission.get("hostname") != socket.gethostname()
            or not isinstance(admission.get("gpu_uuid"), str)
            or not admission["gpu_uuid"].startswith("GPU-")
            or not isinstance(admission.get("runtime_fingerprint_sha256"), str)
            or len(admission["runtime_fingerprint_sha256"]) != 64
            or admission.get("source_sha256") != {str(paths[key]): sources[key]["sha256"] for key in sources}
            or admission.get("safe_role_sha256") != {key: value["sha256"] for key, value in roles.items()}):
        raise ValueError("Actual root host/GPU/runtime/source/data admission required")
    science_cfg = {key: value for key, value in cfg.items() if key != "root_admission"}
    canonical = hashlib.sha256(json.dumps(science_cfg, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if admission.get("science_config_sha256") != canonical:
        raise ValueError("Admission differs from full fixed non-admission config")
    return {"stage": "F", "K": None, "pilot_provenance_read": False,
            "study_kind": "native_SAGE_endpoint_VALUE", "action": action,
            "source_sha256": {str(paths[key]): sources[key]["sha256"] for key in sources},
            "safe_role_sha256": {key: value["sha256"] for key, value in roles.items()},
            "root_runtime_admission": admission,
            "baseline_policy": "fresh_required"}
