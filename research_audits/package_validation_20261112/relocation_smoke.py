"""Bounded checks using only the extracted repository and relocated environment."""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch

from ablation.campaign.tasks import REPO, node_fit, transfer_fit
from ablation.campaign.plan import historical_spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository",type=Path,required=True)
    parser.add_argument("--environment",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    assert REPO.resolve()==args.repository.resolve()
    assert Path(sys.prefix).resolve()==args.environment.resolve()
    assert Path(torch.__file__).resolve().is_relative_to(args.environment.resolve())
    assert torch.cuda.is_available()
    root=REPO/".campaign_test_tmp/relocated_gpu_smoke"
    torch.cuda.reset_peak_memory_stats()
    spec=historical_spec("minesweeper","SAGE",1,0,"gnnm",contrast=True)
    node=node_fit(dict(spec=spec),root/"node",root,"cuda:0",smoke_updates=2)
    assert node["stopped_step"]==2 and node["role"]=="engineering_smoke" and node["test_deferred"]
    node_peak=torch.cuda.max_memory_allocated()/1024**3
    gc.collect()
    torch.cuda.empty_cache()
    transfer=[]
    for pair,module,method in (
        ("molbace_gcn","p09_molecular","gnnm_private_bn_contrast"),
        ("collab_buddy","p11_collab_buddy","gnnm_contrast")):
        meta=dict(pair=pair,module=module,method=method,cohort=0,min_epochs=0,
                  epochs=1000,patience=50,continuation_epochs=2000,defer_test=True)
        row=transfer_fit(meta,root/pair,"cuda:0",smoke_updates=2)
        fit=row["fit"]
        assert row["role"]=="engineering_smoke" and not fit["test_evaluated"]
        assert fit.get("global_step",fit.get("updates"))==2
        assert fit["last_contrast_diagnostic"]["anchors"]>0
        transfer.append(dict(pair=pair,method=method,updates=2,test_evaluated=False))
    report=dict(passed=True,role="engineering_smoke",scientific_results=False,heldout_scored=False,
                repository=str(REPO),environment=sys.prefix,torch_source=torch.__file__,
                gpu=torch.cuda.get_device_name(0),node_updates=node["stopped_step"],
                node_peak_allocated_gib=node_peak,transfer=transfer,
                scope="Fresh archive source and relocated fresh environment on the preparation GPU; read-only data/cache hardlinks conserve NFS space")
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
