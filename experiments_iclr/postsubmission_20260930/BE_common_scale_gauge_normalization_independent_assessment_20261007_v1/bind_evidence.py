"""Bind selected literature/source text; stdlib only, no scientific execution."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


records = [
    ("BE", "2002.06715v2", "BatchEnsemble: An alternative approach to Efficient Ensemble and Lifelong Learning",
     "continuous_graph_efficiency_gap_v1/gnnm_vectorized_eval_source_v1/sources/batchensemble.txt",
     [(420, 534)], "retained_primary_targeted_revisit", "Section3.1 Eqs1–5 and opening of3.2; no result-table audit"),
    ("Rank1BNN", "2005.07186v2", "Rank-1 Bayesian Neural Networks",
     "graph_ensemble_gap_skeptic_v1/efficient_controls_v1/sources/2005.07186v2.txt",
     [(640, 820)], "retained_primary_targeted_revisit", "Section3.1 factor model, expected likelihood, factor KL and W prior; saved initialization conclusions reused"),
    ("TabM4", "2410.24210v3", "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling",
     "tabm_deep_competence_graph_limits_20261006_v1/primary/tabm_page_04.txt",
     [(1, None)], "retained_primary_targeted_revisit", "PDFp4 Section3.2 and3.3 architecture, outside-bias convention"),
    ("TabM5", "2410.24210v3", "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling",
     "tabm_deep_competence_graph_limits_20261006_v1/primary/tabm_page_05.txt",
     [(1, None)], "retained_primary_targeted_revisit", "PDFp5 method continuation; figure/table text incidental and numerical outcomes not audited"),
    ("TabM6", "2410.24210v3", "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling",
     "tabm_deep_competence_graph_limits_20261006_v1/primary/tabm_page_06.txt",
     [(1, None)], "retained_primary_targeted_revisit", "PDFp6 first-adapter/later-unit initialization and method summary; experimental-section opening incidental"),
    ("WeightNorm", "1602.07868v3", "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks",
     "continuous_graph_efficiency_gap_v1/round13_neural_enkf_qualification_v1/sources/weightnorm.txt",
     [(270, 394)], "retained_primary_targeted_revisit", "Sections2 and2.1 Eqs2–4, normalization vs reparameterization, gradient projection, raw-radius and Adam caveat"),
    ("DoRA", "2402.09353v1", "DoRA: Weight-Decomposed Low-Rank Adaptation",
     str(ROOT.relative_to(PHASE) / "primary/2402.09353v1.txt"),
     [(106, 140)], "fresh_primary_scoped_methods", "Sections4.1–4.3 Eqs5–11; differentiated norm vs detached norm; not full paper or numerical result audit"),
    ("PathSGD", "1506.02617v1", "Path-SGD: Path-Normalized Optimization in Deep Neural Networks",
     str(ROOT.relative_to(PHASE) / "primary/1506.02617v1.txt"),
     [(113, 150), (210, 240)], "fresh_primary_scoped_methods", "Section2 rescaling/equivalent functions/SGD sensitivity; Section4 Eqs6–9 and Theorem4.1 proof; not whole paper"),
    ("LoRAPro", "2407.18242v1", "LoRA-Pro: Are Low-Rank Adapters Properly Optimized?",
     str(ROOT.relative_to(PHASE) / "primary/2407.18242v1.txt"),
     [(111, 209)], "fresh_primary_scoped_methods", "Sections3.1–3.3 main method Eqs1–14; proofs/Adam appendix and numerical tables not audited"),
    ("LoRARITE", "2410.20625v1", "LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization",
     str(ROOT.relative_to(PHASE) / "primary/2410.20625v1.txt"),
     [(102, 166), (176, 270)], "fresh_primary_scoped_methods", "Section2.1–2.2, Section2.3 opening; Section3.1–3.2 Eqs9–17 and algorithm opening; convergence/proofs/tables not audited"),
    ("NaturalGradient", "1301.3584v1", "Revisiting Natural Gradient for Deep Networks",
     str(ROOT.relative_to(PHASE) / "primary/1301.3584v1.txt"),
     [(97, 165)], "fresh_primary_scoped_methods", "Section2 density-manifold/KL/Fisher geometry Eqs1–4; other sections not audited"),
]

scopes = []
passages = []
for key, version, title, relative, ranges, status, meaning in records:
    path = PHASE / relative
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines(keepends=True)
    selected = []
    for first, last in ranges:
        last = len(lines) if last is None else last
        assert 1 <= first <= last <= len(lines), (key, first, last, len(lines))
        excerpt = "".join(lines[first - 1:last])
        selected.append({"first_line": first, "last_line": last, "sha256": sha(excerpt.encode()), "text": excerpt})
    scopes.append({"key": key, "version": version, "title": title, "source_path": str(path),
                   "source_sha256": sha(raw), "read_status": status, "semantic_scope": meaning,
                   "ranges": [{k: v for k, v in x.items() if k != "text"} for x in selected]})
    passages.append({"key": key, "version": version, "source_path": str(path), "passages": selected})

save("READ_SCOPES.json", {"schema": "BE-common-scale-gauge-read-scopes-v1", "date": "2026-10-07",
    "policy": "Selected primary method reads only; raw source retention, metadata discovery and keyword locators are not whole-paper reads.",
    "records": scopes, "fresh_primary_identities_semantically_read": 5,
    "retained_primary_identities_targetedly_revisited": 4, "full_paper_reads": 0,
    "author_implementation_audits": 0, "numerical_results_reproduced": 0})
save("INSPECTED_PASSAGES.json", {"schema": "BE-common-scale-gauge-selected-passages-v1", "records": passages})

inputs = [
    "graph_fast_factor_uncertainty_literature_v1/REUSED_CONCLUSIONS.json",
    "tabm_deep_competence_graph_limits_20261006_v1/CONCLUSIONS.json",
    "tabm_deep_competence_graph_limits_20261006_v1/READ_SCOPES.json",
    "tabm_deep_competence_graph_limits_20261006_v1/INPUT_BINDINGS.json",
    "continuous_graph_efficiency_gap_v1/gnnm_vectorized_eval_source_v1/PRIMARY_CONCLUSIONS.json",
    "continuous_graph_efficiency_gap_v1/round13_neural_enkf_qualification_v1/REPORT.md",
    "continuous_graph_efficiency_gap_v1/round13_neural_enkf_qualification_v1/SOURCE_MAP.json",
    "portable_internal_be_public_interface_20261007_v2/core/factors.py",
]
bindings = []
for relative in inputs:
    path = PHASE / relative
    raw = path.read_bytes()
    bindings.append({"path": str(path), "bytes": len(raw), "sha256": sha(raw), "role": "saved-note_or_static-source_binding"})
save("INPUT_BINDINGS.json", {"schema": "BE-common-scale-gauge-input-bindings-v1", "files": bindings,
    "policy": "Bindings certify the selected files, not every reference contained in them. Referenced data/checkpoint/internal-outcome payloads were not opened."})

save("CONCLUSIONS.json", {"schema": "BE-common-scale-gauge-independent-conclusions-v1", "date": "2026-10-07",
    "decision": "reject_normalization_alone_as_novelty_or_established_competence_remedy; retain_narrow_credit_attribution_question",
    "exact_algebra": "r'_mj=r_mj/a_j; s'_mi=s_mi/b_i; W'=diag(b)Wdiag(a), a/b=member-RMS, positive coordinate norms required",
    "bias": "outside bias unchanged; inside bias q'=diag(b)q",
    "post_affine_LN": "c'=c/d; gamma'=d*gamma; beta'=d*beta for whole affine output",
    "main_qualifications": [
        "Normalization fixes only common positive coordinate scale, not all gauge freedom.",
        "Private radial updates alone change functions; pure gauge also compensates shared parameters.",
        "Function-only loss gradient is orthogonal to the joint gauge direction; RMS drift is not a failure diagnosis.",
        "Forward-normalized, sphere/retraction and post-update canonicalization rules are different optimizers.",
        "Project actual preconditioned displacement, not only gradient, for tangent corrections.",
        "Post-own tangent correction with retraction preserves column norm, not pooled function.",
        "Regularization and Rank1BNN prior/KL distributions need explicit treatment.",
        "Common-only correction is exactly common shared feature reweighting at each eligible affine map.",
        "No descent, competence, calibration, diversity, graph-specific or compute guarantee established."],
    "representative_credit_controls": ["own_only", "unrestricted", "common_only_Qd", "relative_only_retract_Pd"],
    "adoption_or_source_change": False, "training_or_numerical_execution": False,
    "private_server_access": False, "root_ledger_or_prior_seal_change": False})

# Complete raw-retention metadata for the successful bounded Path-SGD v1 fallback.
record = json.loads((ROOT / "PATH_SGD_ACCESS.json").read_text())
for suffix in ("html", "txt"):
    path = ROOT / "primary" / ("1506.02617v1." + suffix)
    raw = path.read_bytes()
    record["sha256_" + suffix] = sha(raw)
    record["bytes_" + suffix] = len(raw)
record["status"] = "retrieved_then_scoped_primary_method_read"
save("PATH_SGD_ACCESS.json", record)

print(json.dumps({"scoped_primary_identities": len({r[1] for r in records}),
                  "fresh_primary_identities": 5, "retained_primary_identities": 4,
                  "selected_passage_records": len(records), "bound_inputs": len(bindings)}))
