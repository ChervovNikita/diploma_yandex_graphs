"""Verify the completed bounded owner GPU DAG without any additional training."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch

from .schema import load_plan
from .store import CampaignStore, atomic_json
from .tasks import output, result


def verify(root, plan_path):
    plan = load_plan(plan_path)
    states = CampaignStore(root, plan).snapshot()
    statuses = Counter(row["status"] for row in states.values())
    if statuses != {"complete":len(plan["tasks"])}:
        raise ValueError("Engineering queue is not fully successful: "+str(statuses))
    node_rows, transfer_rows = [], []
    actions = Counter()
    for task in plan["tasks"]:
        name, meta = task["id"], task["metadata"]
        action = meta["action"]
        actions[action] += 1
        if states[name]["attempts"] != 1:
            raise ValueError("Unexpected retry: "+name)
        if action == "node_fit":
            row = result(root,name)
            if row["role"] != "engineering_smoke" or row["stopped_step"] != 2:
                raise ValueError("Node smoke exceeded bounded role: "+name)
            payload = torch.load(output(root,name)/"predictions.pt",map_location="cpu",weights_only=False)
            if set(payload["roles"]) != {"train","val"}:
                raise ValueError("Unexpected node scoring roles: "+str(payload["roles"].keys()))
            for role in payload["roles"].values():
                if not torch.isfinite(role["logits"]).all():
                    raise ValueError("Nonfinite engineering predictions")
            node_rows.append(dict(task=name,dataset=row["dataset"],model=row["model"],
                                  variant=row.get("variant"),num_layers=row["num_layers"],
                                  stopped_step=2,validation_nodes=len(payload["roles"]["val"]["indices"])))
        elif action == "transfer_fit":
            row = result(root,name)
            fit = row["fit"]
            if row["role"] != "engineering_smoke" or fit["test_evaluated"]:
                raise ValueError("Transfer engineering TEST boundary violated")
            updates = fit.get("global_step",fit.get("updates"))
            if updates != 2:
                raise ValueError("Expected exactly two transfer batches")
            directory = Path(row["fit_dir"])
            payload = torch.load(directory/"predictions.pt",map_location="cpu",weights_only=False)
            if meta["module"] == "p09_molecular":
                if set(payload["splits"]) != {"valid"}:
                    raise ValueError("Molecular heldout scoring occurred")
                selected = torch.load(directory/"model.pt",map_location="cpu",weights_only=False)["model"]
                valid = payload["splits"]["valid"]
                n = len(valid["indices"])
                finite = torch.isfinite(valid["member_logits"]).all()
            else:
                if set(payload)-{"config"} != {"valid"}:
                    raise ValueError("BUDDY heldout scoring occurred")
                selected = torch.load(directory/"best_model.pt",map_location="cpu",weights_only=False)
                valid = payload["valid"]
                n = len(valid["scores"])
                finite = torch.isfinite(valid["scores"]).all()
            if not finite:
                raise ValueError("Nonfinite transfer validation predictions")
            private = meta["method"] in {"private_bn","gnnm_private_bn_contrast","link_same_query_global"}
            running = [tensor for key,tensor in selected.items() if key.endswith("running_mean")]
            if not running or any(t.ndim != (2 if private else 1) for t in running):
                raise ValueError("Incorrect route-private/shared BN buffers")
            contrast = meta["method"] in {"gnnm_contrast","gnnm_private_bn_contrast","link_same_query_global"}
            diagnostic = fit.get("last_contrast_diagnostic")
            if contrast and (not diagnostic or diagnostic.get("anchors",0) < 1):
                raise ValueError("Contrastive GPU path was not active")
            transfer_rows.append(dict(task=name,pair=meta["pair"],method=meta["method"],
                                      updates=updates,validation_objects=n,private_bn=private,
                                      contrastive_active=contrast))
    return dict(passed=True,role="engineering_smoke",scientific_results=False,
                heldout_scored=False,statuses=dict(statuses),actions=dict(actions),
                node_fits=node_rows,transfer_fits=transfer_rows,
                target_cross_host_filesystem="pending destination qualification",
                target_driver="pending destination qualification")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    report=verify(args.root,args.plan)
    atomic_json(args.out,report)
    print(json.dumps(dict(passed=report["passed"],statuses=report["statuses"],actions=report["actions"])))


if __name__ == "__main__":
    main()
