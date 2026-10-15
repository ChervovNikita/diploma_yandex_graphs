"""Immutable diagnostic of already completed fits; no model or TEST import."""
import csv
import hashlib
import json
from pathlib import Path


def run(expansion):
    root = Path(expansion).resolve()
    pilot = root / "results/molbace_pilot_v1"
    target = root / "analysis_outputs/molbace_first_seed_curves_v1"
    target.mkdir(parents=True, exist_ok=False)
    names = {
        "native_single": "Native single",
        "gnnm_boundary_native": "Base GNNM",
        "gnnm_graph_alignment_variance": "Alignment + variance",
        "gnnm_private_input_residual": "Private input correction",
    }
    rows, sources, summary = [], [], []
    for result_path in sorted((pilot / "fits").glob("*/RESULT.json")):
        result = json.loads(result_path.read_text())
        if result["procedure"] not in names or "seed-101" not in result["job_id"]:
            continue
        assert result["TEST_accessed"] is False
        curve_path = result_path.with_name("CURVE.jsonl")
        curve = [json.loads(line) for line in curve_path.read_text().splitlines()]
        assert len(curve) == result["completed_epochs"]
        for event in curve:
            assert event["train"]["complete_native_TRAIN"]
            valid = event["validation"]
            assert valid["complete_source_VALID"] and valid["TEST_accessed"] is False
            rows.append(dict(method=result["procedure"], epoch=event["train"]["epoch"],
                             VALID_AUC=valid["metrics"]["rocauc"], VALID_BCE=valid["loss"],
                             VALID_graphs=valid["records"]))
        for f in (result_path, curve_path):
            sources.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        summary.append({k:result[k] for k in ("procedure","selected_epoch","best_validation_metric",
                                              "completed_epochs","stop_reason","invocation_wall_seconds")})
    assert len(summary) == 4 and len(rows) == 596
    with (target/"CURVES.csv").open("x") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    receipt=dict(scope="One selected-VALID seed; incomplete planned3-seed/I4 cohort; descriptive only",
                 sources=sources, summary=summary, TEST_accessed=False, new_optimizer_steps=0,
                 convergence_interpretation="Patience met the frozen VALID controller; not proof of optimization stationarity")
    (target/"RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        receipt["figure_status"]="matplotlib unavailable; numerical CSV retained, no dependency installed"
        (target/"RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
        return receipt
    colors=["#36393b","#457b9d","#e76f51","#6a994e"]
    figure,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
    for (method,label),color in zip(names.items(),colors):
        selected=next(v for v in summary if v["procedure"]==method)
        values=[v for v in rows if v["method"]==method]
        for ax,key,title in zip(axes,["VALID_AUC","VALID_BCE"],["Whole-VALID ROC-AUC","Whole-VALID BCE"]):
            ax.plot([v["epoch"] for v in values],[v[key] for v in values],color=color,label=label,linewidth=1,alpha=.85)
            point=next(v for v in values if v['epoch']==selected['selected_epoch'])
            ax.scatter([point['epoch']],[point[key]],color=color,s=30,zorder=3,edgecolor='white',linewidth=.5)
            ax.set(xlabel='Complete TRAIN epoch',ylabel=title)
            ax.grid(alpha=.15)
            ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8)
    figure.suptitle('MolBACE / native GIN / seed101 — development diagnostic',fontsize=11)
    figure.savefig(target/'CURVES.svg')
    plt.close(figure)
    receipt['figure_status']='saved SVG; dots are own AUC-selected checkpoints; no cross-seed uncertainty'
    receipt['figure_sha256']=hashlib.sha256((target/'CURVES.svg').read_bytes()).hexdigest()
    (target/"RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--expansion',required=True)
    r=run(p.parse_args().expansion)
    print(json.dumps({k:r[k] for k in ['scope','figure_status','new_optimizer_steps']}))
