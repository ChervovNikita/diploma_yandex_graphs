"""Count a prospective scientific roster, without constructing executable jobs.

The predecessor's source references, paired cohorts and native training
contracts are immutable inputs. Missing integrations never become runnable
because this script names a scientifically meaningful method.
"""
from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
PREDECESSOR = EXPANSION.parent / "reference_wrapper_redesign_20261013_v1/PLAN.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method(name, families, question, rule, controls, **extra):
    return dict(id=name, families=families, question=question, rule=rule,
                controls=controls, **extra)


def catalogue():
    all_tasks = ["node", "ogb", "link"]
    labelled = ["node", "ogb"]
    return [
        method("gnnm_boundary_native", all_tasks, "How does original shared GNNM compare?",
               "Shared native body, original signed input factors and neutral remaining factors; mean own native risk F and mean raw logits.",
               ["native_M1", "native_I4"], predecessor="gnnm_boundary_native"),
        method("boundary_private_half_pool", all_tasks, "Does pool credit help private factors?",
               "Shared gradient F; private gradient .5F+.5L_mean. One native optimizer transition.",
               ["gnnm_boundary_native", "aggregation_mean_globalJ_v1"], predecessor="boundary_private_half_pool"),
        method("boundary_private_fixed_responsibility", labelled, "Does generic persistent reweighting help?",
               "Preserve predecessor fixed Exp weights and native class/group normalization. Shared F; private weighted W.",
               ["gnnm_boundary_native", "structural_class_quartile_private_v1"], predecessor="boundary_private_fixed_responsibility"),
        method("boundary_own_within_route_supcon", ["node"], "Does the earlier within-route class objective help?",
               "Exact predecessor node within-route SupCon recipe; do not alias it to the new cross-route geometry.",
               ["gnnm_boundary_native", "task_geometry_alignment"], predecessor="boundary_own_within_route_supcon"),
        method("task_geometry_alignment", labelled, "Does supervised cross-route class geometry help?",
               "All-block F+.05A, temperature .2. Class-balanced anchors/denominator; same-class existing route preheads. Full native task population retained.",
               ["gnnm_boundary_native", "task_geometry_alignment_variance"]),
        method("task_geometry_alignment_variance", labelled, "Does within-class sample variation add value to alignment?",
               "Same alignment plus conditional within-route, within-class variance floor .5/sqrt(width); coefficient .05 on the declared combined auxiliary.",
               ["gnnm_boundary_native", "task_geometry_alignment"]),
        method("link_same_query_cosine", ["link"], "Does same-query cross-route alignment alone help?",
               "Shared F; private F+.05A. Existing authenticated query pairs, no task-negative invention, same endpoint-eligible anchors as contrast controls.",
               ["gnnm_boundary_native", "link_same_query_global", "link_same_query_endpoint"]),
        method("link_same_query_global", ["link"], "Does sampled discrimination improve on cosine alignment?",
               "Shared F; private F+.05InfoNCE, temperature .2. Same-query positive; same-relation opposite source-label alternatives; endpoint-matched counts/support.",
               ["link_same_query_cosine", "link_same_query_endpoint"]),
        method("link_same_query_endpoint", ["link"], "Does endpoint context improve the contrast?",
               "Same rule and source query reductions as global, but alternatives share an endpoint. Max1024 anchors and16 alternatives; disclose zero support.",
               ["gnnm_boundary_native", "link_same_query_global"]),
        method("structural_class_quartile_private_v1", labelled, "Does meaningful persistent responsibility help?",
               "TRAIN classwise incoming-degree/atom-count fractional quartiles, w=.5+2q. Shared F; private W. Preserve native reductions, no minibatch renormalization.",
               ["gnnm_boundary_native", "structural_class_quartile_permuted_private_v1"]),
        method("structural_class_quartile_permuted_private_v1", labelled, "Is any effect attributable to the structural association?",
               "One whole-vector within-class permutation of the exact quartile weight table, fit seed+7030031, no redraw; same block updates.",
               ["structural_class_quartile_private_v1"]),
        method("link_endpoint_responsibility_v2", ["link"], "Does persistent endpoint-correlated credit help ranking?",
               "Corrected v2 symmetric positive endpoint field; shared F/private W; exact native group normalization. Normalized marginals need not match query control.",
               ["gnnm_boundary_native", "link_query_responsibility_v2"]),
        method("link_query_responsibility_v2", ["link"], "Does endpoint association matter beyond matched raw noise?",
               "Corrected v2 query field: two Exp draws for nonloops, one for loops, matching endpoint raw marginal; group normalization retained.",
               ["link_endpoint_responsibility_v2"]),
        method("context_affinity_common", all_tasks, "Does task-compatible fixed context geometry help?",
               "All-block F+.05A; existing prehead donor distribution at temperature .2 learns .5q_A+.5q_B for every route. TRAIN-only target labels; same donor support/work as split.",
               ["gnnm_boundary_native", "context_affinity_split"], requires="TASK_CONTEXT_SPEC.md and native binding"),
        method("context_affinity_split", all_tasks, "Does assigning different information targets help more than their common mixture?",
               "Same loss/donors/budget as common. Routes0/1 use q_A, routes2/3 q_B. No extra inference input, head, teacher or VALID target.",
               ["context_affinity_common"], requires="TASK_CONTEXT_SPEC.md and native binding"),
        method("portable_input_residual_be_v1", all_tasks, "Can nonlinear private corrections improve boundary GNNM?",
               "BE plus h+U_m GELU(V_m h) at the declared encoded input; rank=min(8,width), zero U and fixed-law orthonormal V frames; one shared native body.",
               ["gnnm_boundary_native", "portable_input_residual_only_v1"]),
        method("portable_input_residual_only_v1", all_tasks, "Are diagonal BE factors needed with the private residual?",
               "Four private residual routes around one shared native body, BE hooks/banks removed before training; same native constructor and optimizer semantics.",
               ["portable_input_residual_be_v1", "native_M1", "native_I4"]),
        method("aggregation_mean_globalJ_v1", all_tasks, "Does joint pool training improve the mean?",
               "All-block J=.5F+.5L_actual. Uniform mean, zero added parameters; alias of predecessor global half-pool, one fit only.",
               ["gnnm_boundary_native", "aggregation_global_globalJ_v1", "aggregation_conditional_globalJ_v1"], predecessor="boundary_global_half_pool"),
        method("aggregation_global_globalJ_v1", all_tasks, "Do learned route weights improve on a uniform mean?",
               "Same J, four TRAIN-learned biases, weights .05+.8softmax(score), exact uniform start.",
               ["aggregation_mean_globalJ_v1", "aggregation_conditional_globalJ_v1"]),
        method("aggregation_conditional_globalJ_v1", all_tasks, "Can observable prediction information choose a better pool?",
               "Same J,20 parameters, detached entropy confidence/JS descriptors, hidden4 GELU scorer and route biases; same positive weight floor/uniform start.",
               ["aggregation_mean_globalJ_v1", "aggregation_global_globalJ_v1", "same_selected_bank_fixed_mean"]),
        method("interaction_geometry_input", all_tasks, "Do task geometry and private nonlinear capacity complement each other?",
               "Combine BE input residual with labelled alignment+variance or link endpoint contrast, using each task's exact preceding gradient rule; no coefficient search.",
               ["gnnm_boundary_native", "portable_input_residual_be_v1", "task_geometry_alignment_variance_or_link_endpoint"]),
        method("interaction_responsibility_private_pool", all_tasks, "Does pooled private credit complement persistent task responsibility?",
               "Shared F; private .5W_structured+.5L_mean. Compare G0, structured W and private half-pool to complete the2x2.",
               ["gnnm_boundary_native", "structural_class_quartile_private_or_link_endpoint_v2", "boundary_private_half_pool"]),
        method("interaction_input_meanJ", all_tasks, "Does the adapter help under the controlled pooling objective?",
               "BE input residual with all-block .5F+.5L_mean; same adapter initialization/site and J as its ingredient controls.",
               ["aggregation_mean_globalJ_v1", "portable_input_residual_be_v1", "interaction_input_conditionalJ"]),
        method("interaction_input_conditionalJ", all_tasks, "Does conditional pooling use adapter alternatives better?",
               "BE input residual plus conditional20-parameter aggregator, all-block .5F+.5L_actual; compare same selected bank under actual gate and mean.",
               ["aggregation_conditional_globalJ_v1", "aggregation_mean_globalJ_v1", "interaction_input_meanJ"]),
        method("gnnm_private_bn_state_v3", ["ogb", "link"], "Does shared running state partly explain weak route evaluation?",
               "Prospective route-private BN running buffers/counters; affine and native weights shared, original BN operators/batches. No BN introduced into BN-free models.",
               ["gnnm_boundary_native"], only_native_BN=True),
    ]


def build():
    predecessor = json.loads(PREDECESSOR.read_text())
    bindings_path = EXPANSION / "job_binding_preparation_v1/CELL_BINDINGS.json"
    bindings = json.loads(bindings_path.read_text())["cells"]
    source = [j for j in predecessor["jobs"] if j["kind"] == "fit" and j["contract"]["track"] == "native_replica"]
    main = [j for j in predecessor["jobs"] if j["kind"] == "fit" and j["contract"]["track"] == "long_convergence"]
    base = [j for j in main if j["contract"]["model_procedure"] == "gnnm_boundary_native"]
    assert len(source) == len(base) == 270
    cohorts = collections.Counter(j["contract"]["recipe_id"] for j in base)
    assert len(cohorts) == 27 and set(cohorts.values()) == {10}
    methods = catalogue()
    assert len({m["id"] for m in methods}) == len(methods)
    cells, by_task = [], collections.defaultdict(collections.Counter)
    for cell_id, count in sorted(cohorts.items()):
        binding = bindings[cell_id]
        family = binding["family"]
        has_bn = family == "ogb" or cell_id == "collab-buddy"
        applicable = [m for m in methods if family in m["families"] and (not m.get("only_native_BN") or has_bn)]
        ids = [m["id"] for m in applicable]
        # Reproduction is a separate physical fit. M1 reuses native I4 body0.
        cells.append(dict(id=cell_id, dataset=binding["dataset"], task_family=family,
                          cohorts=count, source_reference_fits=count,
                          matched_native_body_fits=4*count, matched_single_extra_fits=0,
                          shared_method_ids=ids, shared_method_fits=len(ids)*count,
                          total_fits=(5+len(ids))*count,
                          status="prospective_design_not_executable_or_approved"))
        by_task[family].update(source=count, native=4*count, shared=len(ids)*count)
    totals = dict(source=sum(c["source_reference_fits"] for c in cells),
                  matched_native=sum(c["matched_native_body_fits"] for c in cells),
                  shared=sum(c["shared_method_fits"] for c in cells))
    totals["physical_fits"] = sum(totals.values())
    assert totals["physical_fits"] == sum(c["total_fits"] for c in cells)
    previous_counts = collections.Counter((j["contract"]["recipe_id"], j["contract"]["model_procedure"]) for j in main)
    by_id = {c["id"]:c for c in cells}
    for (cell, proc), count in previous_counts.items():
        if proc == "native_single":
            assert count == 4*cohorts[cell]
        else:
            mapped = "aggregation_mean_globalJ_v1" if proc == "boundary_global_half_pool" else proc
            assert mapped in by_id[cell]["shared_method_ids"]
    result = dict(schema="gnnm-expanded-scientific-roster-proposal-v1", parent_PLAN_sha256=sha(PREDECESSOR),
                  parent_binding_sha256=sha(bindings_path), full_campaign_GO=False,
                  giant_launch_enabled=False, executable_job_count=0,
                  proposal_is_ready_to_launch=False, HPO=False,
                  original_paper_reruns=False, original_paper_scores_changed=False,
                  datasets=sorted({c["dataset"] for c in cells}), preset_count=len(cells),
                  scientific_method_definition_count=len(methods),
                  architecture_count_convention="8 author families: PolyFormer;GCN;GCN-VN;GIN;GIN-VN;NCN;NCNC;BUDDY. Mono/Cheb and source dropout presets remain separate settings.",
                  totals=totals, task_totals={k:dict(v) for k,v in by_task.items()},
                  methods=methods, cells=cells,
                  not_included=["C7 joint-pattern: native observed target/capture not yet frozen",
                                "probabilistic factors/flow matching: no defensible frozen distribution/target yet",
                                "private-BN x adapter: may be a later explicit study, no implicit Cartesian grid",
                                "discarded F1/equipped/joint-untied/capacity/internal-placement campaigns"],
                  required_before_launch=["actual native task ports and ingredient composition",
                                          "runtime full-epoch/selection/stop/resume qualification",
                                          "all source split/provider/serving bindings",
                                          "measured per-job CPU/GPU/disk cost and two-child admission",
                                          "prospective confirmation and multiplicity policy",
                                          "user review/GO and later explicit reauthorization of77"])
    return result


if __name__ == "__main__":
    result = build()
    target = HERE / "PROPOSED_ROSTER.json"
    raw = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n"
    if target.exists() and target.read_text() != raw:
        raise ValueError("Preserve this proposal; changed scientific design needs another version")
    target.write_text(raw)
    print(json.dumps({k:result[k] for k in ["totals", "task_totals", "preset_count", "scientific_method_definition_count", "executable_job_count"]}))
