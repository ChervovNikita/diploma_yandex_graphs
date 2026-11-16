"""Bounded GPU coverage for the owner update; never a scientific campaign."""
import argparse
import json
from pathlib import Path

from .plan import Builder, MODELS, DATASETS, ACTIVE_ROMAN_VARIANTS, node_spec, historical_spec
from .schema import SCHEMA, validate_plan


def build():
    b = Builder()
    specs = [historical_spec("roman-empire", model, 5, 0, "gnnm", contrast=True)
             for model in MODELS]
    specs += [historical_spec(dataset, "SAGE", 5, 0, "gnnm", contrast=True)
              for dataset in DATASETS if dataset != "roman-empire"]
    specs += [node_spec("roman-empire", "GAT-sep", 1, 0, method)
              for method in ACTIVE_ROMAN_VARIANTS]
    specs += [node_spec("roman-empire", model, 1, 0, "all_layer_be")
              for model in MODELS if model != "GAT-sep"]
    specs += [node_spec("roman-empire", "GAT", 1, 0, method) for method in ("native0", "gnnm")]
    for index, spec in enumerate(specs):
        fit = b.fit(spec, "engineering_smoke", 1)
        score = "valid_smoke/" + str(index)
        b.add(score, "node_score", dict(fits=[fit], setting=spec["method"],
              dataset=spec["dataset"], model=spec["model"], split=0,
              stage="engineering_smoke", roles=["val"]), [fit], 2, "gpu", memory=64)
        if index in (0, 17, 26):
            b.add("diversity_smoke/" + str(index), "diversity",
                  dict(score=score), [score], 3, "gpu", memory=64)
    for pair, module, methods in [
        ("molbace_gcn", "p09_molecular", ["native0", "native1", "native2", "native3",
         "gnnm", "gnnm_contrast", "private_bn", "gnnm_private_bn_contrast"]),
        ("collab_buddy", "p11_collab_buddy", ["native0", "native1", "native2", "native3",
         "gnnm", "gnnm_contrast", "private_bn", "link_same_query_global"]),
    ]:
        for method in methods:
            b.add("transfer_smoke/" + pair + "/" + method, "transfer_fit",
                  dict(pair=pair, module=module, method=method, cohort=0, min_epochs=0,
                       epochs=1000, patience=50, continuation_epochs=2000, defer_test=True),
                  priority=4, kind="gpu", memory=40 if module=="p11_collab_buddy" else 12)
    for task in b.tasks.values():
        if task["metadata"]["action"] in {"node_fit", "transfer_fit"}:
            task["argv"] += ["--smoke-updates", "2"]
    plan = dict(schema=SCHEMA, campaign_id="owner-engineering-20261112",
                role="engineering_smoke", scientific_results=False, heldout_scored=False,
                tasks=list(b.tasks.values()))
    validate_plan(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(build(), indent=2)+"\n")


if __name__ == "__main__":
    main()
