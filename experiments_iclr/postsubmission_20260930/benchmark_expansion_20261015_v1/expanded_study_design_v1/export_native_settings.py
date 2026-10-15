"""Export the unchanged native settings for the scientific roster proposal."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent.parent / "reference_wrapper_redesign_20261013_v1/PLAN.json"
plan = json.loads(PARENT.read_text())
source, matched = {}, {}
for job in plan["jobs"]:
    if job["kind"] != "fit":
        continue
    c = job["contract"]
    if c["track"] == "native_replica":
        source.setdefault(c["recipe_id"], c)
    elif c["model_procedure"] == "gnnm_boundary_native":
        matched.setdefault(c["recipe_id"], c)
assert set(source) == set(matched) and len(source) == 27
rows = []
for name in sorted(source):
    s, m = source[name], matched[name]
    assert s["native_config"] == m["native_config"]
    assert s["training"]["optimizer"] == m["training"]["optimizer"]
    rows.append(dict(preset=name, dataset=s["dataset"],
                     native_config=s["native_config"],
                     native_optimizer=s["training"]["optimizer"],
                     source_reference_controller=s["training"],
                     matched_controller=m["training"],
                     source_seed_fidelity=s["source_seed_fidelity"],
                     published_comparison=s["source_bindings"]["literature_and_release_row"],
                     official_evaluator=s["official_evaluator"],
                     source_snapshot=s["source_bindings"]["author_snapshot_manifest"],
                     changes_to_native_hyperparameters=False,
                     native_operator_changes=False,
                     HPO=False))
value=dict(schema="expanded-roster-native-settings-v1",parent_PLAN_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
           rows=rows,source_reference_fits=270,main_native_I4_body_fits=1080,
           main_M1_extra_training_fits=0,full_campaign_GO=False)
out=HERE/"NATIVE_SETTINGS.json"
raw=json.dumps(value,sort_keys=True,indent=2)+"\n"
assert not out.exists() or out.read_text()==raw
out.write_text(raw)
print(json.dumps(dict(presets=len(rows),native_config_and_optimizer_parity=True,HPO=False)))
