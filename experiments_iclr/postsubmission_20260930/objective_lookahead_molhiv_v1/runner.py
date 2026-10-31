"""Freeze source readiness or explicitly refuse real inputs for lookahead v1."""
import argparse
from lookahead_contract import (SOURCE_ONLY, RECIPE, RULE, PLACEMENTS, LOSSES,
                                sources, write_new, refuse_real_input_admission)


def preparation():
    return {"schema":"gnnm-molhiv-lookahead-source-preparation-v1",
        "status":"SOURCE_ONLY_NOT_EXECUTION_APPROVED","source_only":SOURCE_ONLY,
        "sources":sources(),"recipe":RECIPE,"rule":RULE,
        "placements":list(PLACEMENTS),"training_losses":list(LOSSES),
        "future_cell_order":[{"seed":s,"label_fraction":f,"source_epoch":25}
                             for s in range(3) for f in (.1,1.)],
        "future_binding_schema":"gnnm-molhiv-lookahead-future-input-binding-v1",
        "future_required_artifacts":["approved_CPU_runtime_freeze","CPU_execution_gate",
            "successful_grouping_export","bundle","fold_and_mask_lock","six_P25_cells"],
        "future_required_P25_artifacts":["checkpoint","probe_contract","probe_protocol",
            "probe_replay","fit_trace","artifact_manifest"],
        "future_required_semantic_checks":["coordinator-approved successful exporter",
            "correct authorized one-GPU allocation verified; wrong seven-GPU allocation evidence excluded",
            "frozen matching CPU runtime and passed execution gate",
            "all six P-only epoch25 fits, contracts and deterministic replays complete",
            "accepted label masks, common source states and group identities match",
            "group-disjoint P, Q-update and Q-screen; train-only labels",
            "full screening population, class/group counts and update coverage recorded"],
        "prohibited_current_bindings":["CPUv3","failed grouping v5 exporter",
            "runtime, data, checkpoints or results acquired on the unauthorized seven-GPU allocation"],
        "real_input_admission":False,"predictive_accuracy_claimed":False,
        "novelty_or_superiority_claimed":False,
        "next_stage":"A separately versioned coordinator-approved admission packet is required."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare",help="Write a stdlib-only source manifest; no experiment")
    prep.add_argument("--output",required=True)
    real=sub.add_parser("real-cell",help="Always refuses before opening binding, data or checkpoint")
    real.add_argument("--binding",required=True)
    args=parser.parse_args()
    if args.command=="prepare":
        write_new(args.output,preparation())
        print("Source-only packet prepared; no execution or real-input admission approved.")
    else:
        refuse_real_input_admission(args.binding)


if __name__=="__main__":main()
