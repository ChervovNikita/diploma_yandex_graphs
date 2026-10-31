"""Apply the predeclared development gate to all24 new fits; stdlib only."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from prepare_v2 import ARMS, checked, inside, ref, sha, source_hashes, verify_seal, write_new


def summarize(protocol_path, fits, replays):
    v=json.loads(inside(protocol_path).read_text()); verify_seal(v)
    if v["status"]!="runtime_frozen_no_predictive_fit" or v["source_sha256"]!=source_hashes():
        raise ValueError("Require the exact current runtime registration")
    fits=inside(fits); replays=inside(replays); cells={}; budgets=[]; evidence=[]
    for label in (10,100):
        for seed in (0,1,2):
            row={}
            for arm in ARMS:
                name=f"seed{seed}_label{label}_{arm}_run01"; folder=fits/name
                result=json.loads((folder/"result.json").read_text())
                replay_path=replays/(name+".json"); replay=json.loads(replay_path.read_text())
                contract=json.loads((folder/"contract.json").read_text())
                if (result["status"]!="completed_predictive_development_fit" or result["fit_graph_steps"]!=2000
                        or replay["status"]!="PASS" or replay["fit_artifact_manifest_sha256"]!=sha(folder/"artifact_manifest.json")
                        or replay["contract_sha256"]!=sha(folder/"contract.json") or replay["protocol_sha256"]!=v["protocol_sha256"]
                        or contract["protocol_sha256"]!=v["protocol_sha256"] or contract["seed"]!=seed
                        or contract["label_fraction"]!=label/100 or contract["arm"]!=arm):
                    raise ValueError("Incomplete/mismatched fitted and replayed cell: "+name)
                # Only a full replay authorizes aggregation; changes since replay also fail.
                kept=json.loads((folder/"artifact_manifest.json").read_text())
                actual={str(p.relative_to(folder)):sha(p) for p in folder.rglob("*") if p.is_file() and p.name!="artifact_manifest.json"}
                if kept!=actual: raise ValueError("Changed replayed fit: "+name)
                with (folder/"validation_trace.csv").open() as f: trace=list(csv.DictReader(f))
                if [int(r["epoch"]) for r in trace]!=list(range(101)): raise ValueError("Incomplete budget")
                old=[r for r in trace if 91<=int(r["epoch"])<=95]; new=[r for r in trace if 96<=int(r["epoch"])<=100]
                f1_gain=statistics.mean(float(r["micro_f1"]) for r in new)-statistics.mean(float(r["micro_f1"]) for r in old)
                bce_drop=statistics.mean(float(r["pooled_bce"]) for r in old)-statistics.mean(float(r["pooled_bce"]) for r in new)
                limited=result["selected"]["epoch"]>=90 and (f1_gain>0.0001 or bce_drop>0.00001)
                budget={"fit":name,"selected_epoch":result["selected"]["epoch"],"late_mean_f1_gain":f1_gain,
                        "late_mean_pooled_bce_drop":bce_drop,"budget_limited":limited}
                budgets.append(budget); row[arm]={"selected":result["selected"],"epoch100":result["epoch100"],"budget":budget}
                evidence.append({"result":ref(folder/"result.json"),"replay":ref(replay_path),"artifact_manifest":ref(folder/"artifact_manifest.json")})
            cells[f"seed{seed}_label{label}"]=row
    contrasts={}; checks=[]
    for label in (10,100):
        for control in ("tied","random","raw"):
            differences=[cells[f"seed{s}_label{label}"]["selected"]["selected"]["micro_f1"]-
                         cells[f"seed{s}_label{label}"][control]["selected"]["micro_f1"] for s in (0,1,2)]
            bce_differences=[cells[f"seed{s}_label{label}"]["selected"]["selected"]["pooled_bce"]-
                             cells[f"seed{s}_label{label}"][control]["selected"]["pooled_bce"] for s in (0,1,2)]
            final_differences=[cells[f"seed{s}_label{label}"]["selected"]["epoch100"]["micro_f1"]-
                              cells[f"seed{s}_label{label}"][control]["epoch100"]["micro_f1"] for s in (0,1,2)]
            contrasts[f"label{label}_selected_minus_{control}"]={"selected_micro_f1":differences,
                "mean_selected_micro_f1":statistics.mean(differences),"selected_pooled_bce":bce_differences,
                "mean_selected_pooled_bce":statistics.mean(bce_differences),"epoch100_micro_f1":final_differences}
            if label==10:
                checks.append({"control":control,"mean_f1_gate":statistics.mean(differences)>=0.005,
                               "nonzero_rank_seeds_positive":differences[0]>0 and differences[2]>0,
                               "minimum_difference_gate":min(differences)>=-0.002,
                               "mean_pooled_loss_gate":statistics.mean(bce_differences)<=0})
    late=any(b["budget_limited"] and "label10" in b["fit"] for b in budgets)
    passed=not late and all(all(value for key,value in c.items() if key!="control") for c in checks)
    return {"status":"DEVELOPMENT_CONTINUATION_GATE_PASS" if passed else "DEVELOPMENT_CONTINUATION_GATE_FAIL",
            "protocol":ref(protocol_path),"fit_count":24,"cells":cells,"paired_contrasts":contrasts,
            "primary_gate_checks":checks,"budget_audit":budgets,"primary_budget_limited":late,
            "evidence":evidence,"adverse_F2_evidence":v["prior_evidence"]["adverse_F2_summary"],
            "CI_or_significance_claimed":False,"independent_confirmation_still_required":True,
            "test_scored":False,"method_superiority_claimed":False}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--fits",type=Path,required=True); p.add_argument("--replays",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(); result=summarize(args.protocol,args.fits,args.replays); write_new(args.output,result)
    print(json.dumps({"status":result["status"],"primary_budget_limited":result["primary_budget_limited"]}))
