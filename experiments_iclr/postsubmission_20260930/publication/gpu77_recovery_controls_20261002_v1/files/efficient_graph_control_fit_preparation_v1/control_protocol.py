"""Fixed prospective control protocol. Standard library only; no launch action."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass


SEEDS = (17, 29, 43)
CONTEXTS = {
    "polyformer_mono": dict(dataset="squirrel_filtered", nodes=2223,
        features=2089, classes=5, native_width=256, packed_width=116,
        max_epochs=2000, patience=250, parameter_cap=4430644),
    "polynormer_r": dict(dataset="amazon_photo", nodes=7650,
        features=745, classes=8, native_width=512, packed_width=248,
        local_epochs=200, global_epochs=1000, parameter_cap=7773732),
}
ARMS = ("cfg0_single", "cfg0_independent_m4", "fixed_cap_packed_m4",
        "cached_token_mimo_m4")
PROTOCOL = dict(schema="efficient-graph-control-fit-protocol-v1",
    seeds=list(SEEDS), source_splits=[0, 1, 2], contexts=CONTEXTS,
    arms=list(ARMS), configuration="cfg0 only; no HPT",
    primary_pooling="softmax(mean raw member logits)",
    secondary_pooling="mean member softmax probabilities; same selected logits only",
    selector="finite validation NLL; strict improvement; earliest exact tie",
    epoch_zero_eligible=True,
    epoch_zero_status="declared control-protocol adaptation; not native reproduction",
    photo_transition="selected local model AND Adam state; fresh global selector; global-only final",
    member_training_loss="mean member cross entropy",
    independent_m4_selection="joint pooled-validation selector at one common epoch; not four independently selected checkpoints",
    independent_m4_gradient_scale="mean-member CE scales each private gradient by 1/4 versus standalone CE; Adam epsilon prevents an exact-native-equivalence claim",
    independent_m4_status="aligned cfg0 control; not an exact native independent-ensemble reproduction",
    mimo_training_loss="mean tuples, sum matching heads (published scale retained)",
    mimo_tuple_sampling="four independent compact TRAIN permutations; one full exposure per slot per update",
    mimo_inference="repeat each complete 13-token row in all four slots",
    label_scope=["train", "validation"],
    input_policy="caller-supplied existing prepared graph; no preprocessing or data loader",
    packed_backend_policy="explicit sequential or vmap; no automatic fallback",
    scientific_status="source preparation; root decision and qualification required")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


PROTOCOL_SHA256 = digest(PROTOCOL)


@dataclass(frozen=True)
class CompactRole:
    """Only compact node identities and their corresponding role labels."""
    nodes: object
    labels: object


@dataclass(frozen=True)
class PreparedGraph:
    """Repackage known existing fields without preprocessing or label access."""
    teacher_backbone: str
    teacher_input: object
    teacher_edge_index: object


def specification(backbone, arm, seed, *, backend=None):
    if backbone not in CONTEXTS or arm not in ARMS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("Only the two fixed contexts, four declared arms and three seeds are available")
    if arm == "cached_token_mimo_m4" and backbone != "polyformer_mono":
        raise ValueError("Cached-token MIMO is prepared only for PolyFormer")
    if arm == "fixed_cap_packed_m4":
        if backend not in ("sequential", "vmap"):
            raise ValueError("The fixed-cap arm requires an explicit sequential or vmap backend")
    elif backend is not None:
        raise ValueError("Only the fixed-cap arm has a configurable backend")
    context = copy.deepcopy(CONTEXTS[backbone])
    width = (208 if arm == "cached_token_mimo_m4" else context["packed_width"]
        if arm == "fixed_cap_packed_m4" else context["native_width"])
    return dict(schema="efficient-graph-control-cell-v1", backbone=backbone,
        arm=arm, seed=seed, source_split=SEEDS.index(seed), cfg=0,
        members=1 if arm == "cfg0_single" else 4, width=width,
        backend=backend if arm == "fixed_cap_packed_m4" else "native_sequential"
            if arm != "cached_token_mimo_m4" else "cached_token_mimo",
        context=context, protocol_sha256=PROTOCOL_SHA256,
        member_seeds=[seed + 100003 * m for m in range(1 if arm == "cfg0_single" else 4)]
            if arm != "cached_token_mimo_m4" else [seed],
        pooling="common_logit", epoch_zero_eligible=True)


def fixed_plan(*, packed_backend):
    """21 cells for ONE predeclared backend, never an adaptive backend search."""
    return [specification(backbone, arm, seed,
        backend=packed_backend if arm == "fixed_cap_packed_m4" else None)
        for backbone in CONTEXTS for arm in ARMS
        if arm != "cached_token_mimo_m4" or backbone == "polyformer_mono"
        for seed in SEEDS]


def canonical_spec(spec):
    expected = specification(spec["backbone"], spec["arm"], spec["seed"],
        backend=spec["backend"] if spec["arm"] == "fixed_cap_packed_m4" else None)
    if spec != expected:
        raise ValueError("Cell specification was modified outside the frozen protocol")
    return expected


def qualification_checks(spec):
    required = ["actual_module_parameter_count", "finite_loss_and_nonzero_gradients",
        "validation_selection_and_earliest_ties", "selected_state_restore",
        "prepared_input_and_compact_role_binding"]
    if spec["backbone"] == "polynormer_r":
        required += ["selected_local_model_and_adam_handoff", "both_photo_stages"]
    if spec["arm"] == "fixed_cap_packed_m4":
        required += ["private_member_parameters", "fixed_complete_source_cap"]
        if spec["backend"] == "vmap":
            required += ["output_gradient_two_step_adam_equivalence", "member_isolation",
                "distinct_training_dropout_streams", "vmap_forward_and_backward_supported"]
    if spec["arm"] == "cached_token_mimo_m4":
        required += ["compact_tuple_token_label_correspondence", "one_exposure_per_slot",
            "complete_row_repetition_inference", "summed_head_loss_and_gradient",
            "fixed_complete_source_cap"]
    return required


def require_sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA256 identity")


def validate_admission(spec, admission, *, mode, manifest_sha256):
    """Validate root assertions; never issues a scientific decision itself.

    Receipt verification and authenticity are root duties. This gate rejects
    missing, differently bound or incomplete assertions before scientific import.
    """
    canonical_spec(spec)
    if mode not in ("qualify", "full"):
        raise ValueError("Declare qualify or full")
    require_sha(manifest_sha256, "manifest_sha256")
    if admission.get("schema") != "efficient-graph-control-root-admission-v1":
        raise ValueError("Root admission schema missing")
    if admission.get("scientific_decision") != "frozen_admitted":
        raise ValueError("Root has not frozen and admitted the scientific control decision")
    if admission.get("mode") != mode or admission.get("spec_sha256") != digest(spec):
        raise ValueError("Root admission does not bind this mode/cell")
    if admission.get("packet_manifest_sha256") != manifest_sha256:
        raise ValueError("Root admission does not bind this source packet")
    if admission.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("Root admission does not bind this fixed protocol")
    for key in ("decision_id", "attempt_id", "runtime_binding"):
        if not admission.get(key):
            raise ValueError(f"Missing root {key}")
    for key in ("prepared_graph_sha256", "train_pack_sha256", "validation_pack_sha256"):
        require_sha(admission.get(key), key)
    closure = admission.get("prior_graph_init_closure", {})
    if closure.get("closed") is not True:
        raise ValueError("Full-context qualification/fitting requires current graph-init study closure")
    require_sha(closure.get("receipt_sha256"), "prior_graph_init_closure.receipt_sha256")
    runtime = admission["runtime_binding"]
    if set(runtime) != {"torch_version", "pyg_version", "device", "environment_sha256"} or not all(
            isinstance(v, str) and v for v in runtime.values()):
        raise ValueError("Explicit Torch/PyG versions, device and environment identity required")
    require_sha(runtime["environment_sha256"], "runtime_binding.environment_sha256")
    if mode == "full":
        qualification = admission.get("root_qualification", {})
        if qualification.get("passed") is not True or qualification.get("spec_sha256") != digest(spec):
            raise ValueError("Full fit requires root qualification of this exact cell")
        if qualification.get("runtime_binding") != runtime:
            raise ValueError("Qualification runtime differs from full-fit runtime")
        require_sha(qualification.get("receipt_sha256"), "qualification.receipt_sha256")
        if not set(qualification_checks(spec)) <= set(qualification.get("passed_checks", [])):
            raise ValueError("Root qualification lacks required checks")
    return copy.deepcopy(admission)
