"""Source-only lookahead contract; stdlib hashes and explicit admission refusal."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE_ONLY = True
MODEL_SHA = "7416ddddd5a4057ec80624c648f32a5c8369c8c2628e8b58c450c82fe22e3c86"
BOUNDARY_SHA = "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303"
PLACEMENTS = ("tied", "first", "middle", "last")
LOSSES = ("own", "pooled")
RECIPE = {"members":4,"hidden_dim":128,"num_layers":3,"hidden_dim_multiplier":2,
          "output_dim":1,"dropout":0.1,"dtype":"float32","optimizer":"AdamW",
          "lr":0.001,"weight_decay":0.0,"eps":1e-8,"betas":[0.9,0.999],
          "steps":8,"batch_size":128,"threads":1,"interop_threads":1,
          "update_order_seed":"21109+1000*source_seed+cycle",
          "dropout_seed":"27431+1000*source_seed+step",
          "Q_group_split_seed":704,"P_checkpoint_epoch":25}
RULE = {"primary_fraction":0.1,"adverse_fraction":1.0,
        "numerical_margin":1e-6,"primary_each_seed_minimum_gain":0.001,
        "primary_mean_minimum_gain":0.001,"full_label_each_seed_maximum_harm":0.001,
        "minimum_groups_per_update_and_screen_fold":8,
        "ranking":"largest mean primary gain; ties first, middle, last",
        "loss_families_separate":True,"accuracy_promotion_allowed":False,
        "practical_margin_rationale":"One thousandth of mean graph BCE, fixed without MolHIV predictive outcomes; a mechanism continuation threshold, not a statistical significance threshold"}
CODE = ("lookahead_contract.py","lookahead_core.py","runner.py","verify_cpu.py","PROTOCOL.md","README.md")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def inside(path,root=PHASE):
    p=Path(path).resolve();root=Path(root).resolve()
    if root not in p.parents:raise ValueError("Path outside authorized project phase")
    return p


def checked(reference):
    p=inside(PHASE/reference["path"])
    if sha(p)!=reference["sha256"]:raise ValueError("Input hash differs: "+reference["path"])
    return p


def verify_model_sources():
    files={"new_tasks/gine_adapters.py":MODEL_SHA,"source_snapshot/models.py":BOUNDARY_SHA}
    for rel,digest in files.items():
        if sha(PHASE/rel)!=digest:raise ValueError("Unchanged GINE source differs: "+rel)
    return files


def sources():
    return {**verify_model_sources(),**{str((HERE/name).relative_to(PHASE)):sha(HERE/name) for name in CODE}}


def write_new(path,value):
    p=inside(path,HERE)
    with p.open("x") as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")


def refuse_real_input_admission(binding_path=None):
    """No CPUv3/v5 bundle or unfrozen future checkpoint can enter this packet."""
    if SOURCE_ONLY:
        raise ValueError("v1 remains SOURCE_ONLY: no binding, dataset or checkpoint is opened; require a separately versioned coordinator-approved admission stage after successful export/CPU/P25 replay and grouped fold/mask/state verification")
    if binding_path is None:raise ValueError("No real-input binding supplied; v1 is source-only")
    p=inside(binding_path)
    if not p.is_file():raise FileNotFoundError("Missing future lookahead input binding")
    value=json.loads(p.read_text())
    required={"schema","source_sha256","approved_CPU_runtime_freeze","CPU_execution_gate",
              "successful_grouping_export","bundle","fold_and_mask_lock","six_P25_cells"}
    if not required<=set(value):raise ValueError("Incomplete future lookahead binding")
    if value["schema"]!="gnnm-molhiv-lookahead-future-input-binding-v1":raise ValueError("Wrong future binding schema")
    if value["source_sha256"]!=sources():raise ValueError("Future binding source hashes differ")
    for key in ("approved_CPU_runtime_freeze","CPU_execution_gate","successful_grouping_export","bundle","fold_and_mask_lock"):
        checked(value[key])
    cells=value["six_P25_cells"]
    if [(c["seed"],c["label_fraction"]) for c in cells]!=[(s,f) for s in range(3) for f in (0.1,1.0)]:
        raise ValueError("All six common P25 cells required")
    for c in cells:
        if c["epoch"]!=25:raise ValueError("Require exact P25 source")
        for key in ("checkpoint","probe_contract","probe_protocol","probe_replay","fit_trace","artifact_manifest"):
            checked(c[key])
    raise ValueError("v1 remains SOURCE_ONLY: a separately versioned coordinator-approved admission stage must verify successful export/CPU/P25 replay, grouped fold/mask/state identities before loading data or model weights")
