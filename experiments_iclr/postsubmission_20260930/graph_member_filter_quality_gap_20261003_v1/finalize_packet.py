"""Build custody records for the bounded literature review. No model imports."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PACKET = Path(__file__).resolve().parent
RESEARCH = PACKET.parent
NOW = datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (PACKET / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def binding(relative_path):
    path = RESEARCH / relative_path
    return {"path": relative_path, "absolute_path": str(path), "sha256": sha(path), "size": path.stat().st_size}


index_path = RESEARCH / "literature_memory/index_v30/LITERATURE_INDEX.json"
index = json.loads(index_path.read_text())
expected_index_sha = "780f42d770fb73ef890bd1b4b6fd48a9eeb07f481c757d206c7c496ff8c70d70"
assert sha(index_path) == expected_index_sha
assert len(index["paper_records"]) == 127
new_ids = ["arXiv:2106.09910v1", "arXiv:2303.01028v1", "DOI:10.24963/ijcai.2025/685"]
for item in index["paper_records"]:
    assert not any(new_id.lower().split("v1")[0] in item["canonical_id"].lower() for new_id in new_ids)

# Select substantive saved conclusions, rather than treating later shallow reuse
# records or an early metadata-only MORGAN record as a new primary reading.
selected_positions = [3, 6, 45, 46, 48, 49, 50, 52, 53, 66, 96, 113, 120]
reused_papers = []
external_bindings = {"literature_memory/index_v30/LITERATURE_INDEX.json": binding("literature_memory/index_v30/LITERATURE_INDEX.json")}
for position in selected_positions:
    record = index["paper_records"][position]
    b = binding(record["conclusion_file"])
    assert b["sha256"] == record["conclusion_file_sha256"], record["conclusion_file"]
    external_bindings[b["path"]] = b
    reused_papers.append({
        "index_position_zero_based": position,
        "read_kind_this_packet": "saved_conclusion_reused_no_primary_reread",
        "saved_record": record,
        "verified_conclusion_binding": b,
    })

assessment_paths = [
    "graph_extension_distinct_ideas_20261002_v1/QUALITY_LANE_ASSESSMENT.md",
    "graph_ensemble_gap_skeptic_v1/ASSESSMENT.md",
    "member_subspace_messages_v1/REPORT.md",
]
reused_assessments = []
for path in assessment_paths:
    b = binding(path)
    external_bindings[path] = b
    reused_assessments.append({"read_kind": "saved_assessment_reused_no_source_or_outcome_reread", **b})
write("REUSED_REFERENCES.json", {
    "schema": "bounded-literature-reuse-v1",
    "created_UTC": NOW,
    "index": external_bindings["literature_memory/index_v30/LITERATURE_INDEX.json"],
    "memory_accounting": {"conclusion_records": 127, "normalized_paper_ids": 83, "software_ids": 2, "not_full_paper_read_counts": True},
    "paper_records": reused_papers,
    "assessments": reused_assessments,
    "specific_saved_assessment_claims": [
        {"path": assessment_paths[0], "claim": "Graph-regime residual-Gram pooling is simplex least-squares stacking; flexible routing is graph-conditioned MoE. Shared-step private compensation reduces to a rescaled ordinary private gradient."},
        {"path": assessment_paths[1], "locator": "Section 4 prior table, GPNet and ADaMoRE rows; exact-member and all-layer BE controls", "claim": "Multiple-hop/signed graph filters and structural expert diversity are established; exact member messages are the competent control."},
        {"path": assessment_paths[2], "claim": "The failed edge-sampled residual branch remains failed; fixed member contrast projections are information restrictions, not unbiased transport."},
    ],
})

closed_specs = [
    ("spectral_privacy/ppi_training_v1/analysis_v1/RESULT_SUMMARY_v1.md",
     "049bc608e5a4aa779818fab4df84b670d374cf48fec78fa422233e61962642d1",
     {"fits_or_replays": 24, "primary_disposition": "All original primary 10%-label gates failed against tied/random/raw", "limits": "Full-label results cannot rescue the primary failure; convergence unestablished after late budget warnings; no evidence more epochs reverse losses. Four full-width paths and 12 propagations remain. Particular SAGE output-correction recipe, not universal graph-filter impossibility."}),
    ("paired_squirrel_completed_scientific_review_20261003_v1/REPORT.md",
     "e07435707448be57b9074d3238e4f38005b10002f846b68f60f30685e881ed10",
     {"fits": 12, "primary_disposition": "Arithmetic trigger true, practical tie at tens of micro-nats", "limits": "All arms selected after one/four continuation updates. No practical/general superiority or mechanism confirmation."}),
    ("graph_heterogeneous_dblp_execution_root_v1/CLOSED_DEVELOPMENT_DECISION_ROOT_v1.md",
     "68dbbf9f004388a9607f4dc70a17aad949ee050cbd2d4d3d5672f25f2e6507e4",
     {"study": "complete35 relation CP", "primary_disposition": "Fails both global-BE and shared-relation development gates", "cp_minus_global_be_mean_nll": -0.0003424, "wins_global_be": "3/5", "cp_minus_shared_relation_mean_nll": 0.0002235, "wins_shared_relation": "2/5", "original_practical_threshold": 0.005, "limits": "No useful improvement in the tested recipe; no universal relation-conditioning impossibility."}),
]
closed_records = []
for path, expected, conclusion in closed_specs:
    b = binding(path)
    assert b["sha256"] == expected
    external_bindings[path] = b
    closed_records.append({**b, "authorization": "Parent explicitly authorized this closed conclusion only", "scope": "Closed summary read before compaction; byte hashing only during finalization; no raw artifacts", "retained_closed_conclusion": conclusion})
write("CLOSED_DECISION_BINDINGS.json", {"schema": "authorized-closed-conclusions-v1", "created_UTC": NOW, "records": closed_records, "raw_artifacts_accessed": False})

retrieval = json.loads((PACKET / "primary/RETRIEVAL.json").read_text())
by_key = {r["key"]: r for r in retrieval}
source_retrieval = json.loads((PACKET / "author_source/SOURCE_RETRIEVAL.json").read_text())
for record in retrieval:
    for source in record["sources"]:
        assert sha(PACKET / source["path"]) == source["sha256"]
for source in source_retrieval:
    assert sha(PACKET / source["local_path"]) == source["sha256"]

paper_scopes = [
    {"key": "bankgcn", "canonical_id": new_ids[0], "header_identification_pages": [1], "method_scope": [{"pages": [4], "scope": "Spectral preliminaries and limitation discussion"}, {"pages": [5, 6, 7], "scope": "Complete Section 4, equations 7–20; subspaces, learned graph filters, message passing, nonlinearity and filter diversity"}, {"pages": [8], "scope": "Section 5.1 architecture and CE plus filter-diversity objective only"}], "visual_pages_inspected": [6], "renders_saved": ["primary/bankgcn_method-05.png", "primary/bankgcn_method-06.png", "primary/bankgcn_method-07.png"], "incidental_exposure": "Surrounding dataset table on page 8; no result score adopted", "not_read_or_certified": ["Full paper", "Complete proofs", "Complete results/evaluation", "Author implementation"]},
    {"key": "specformer", "canonical_id": new_ids[1], "header_identification_pages": [1], "method_scope": [{"pages": [3, 4, 5], "scope": "Complete Sections 4.1–4.4, equations 2–6, stated properties, complexity and scalability"}], "visual_pages_inspected": [4], "renders_saved": ["primary/specformer_method-04.png", "primary/specformer_method-05.png"], "incidental_exposure": None, "not_read_or_certified": ["Full paper", "Appendix proofs", "Complete recipe/results", "Author implementation"]},
    {"key": "hgen", "canonical_id": new_ids[2], "header_identification_pages": [1], "method_scope": [{"pages": [1], "scope": "Abstract/introduction and secondary GENN locator"}, {"pages": [2, 3, 4], "scope": "Section 2 and main Sections 3.1–3.4, equations 1–9, theorem/remarks"}, {"pages": [5], "scope": "Theorem continuation before Section 4"}], "visual_pages_inspected": [3], "renders_saved": ["primary/hgen_method-3.png", "primary/hgen_method-4.png"], "incidental_exposure": "Complexity empirical prose/figure captions were incidentally exposed; no numerical utility adopted", "not_read_or_certified": ["Full paper", "Independent theorem proof audit", "Complete evaluation/results", "Referenced Appendix Algorithm 1 and Supplement D (not in the official eight-page bytes; not retrieved)"]},
]
for scope in paper_scopes:
    r = by_key[scope["key"]]
    pdf = next(s for s in r["sources"] if s["path"].endswith(".pdf"))
    scope.update({"full_paper_read": False, "read_status": "new_scoped_primary_method_not_full_paper", "numeric_results_adopted_or_reproduced": False, "pdf": {**pdf, "absolute_path": str(PACKET / pdf["path"])}, "extraction": {"path": f"primary/{scope['key']}.pages.json", "sha256": sha(PACKET / f"primary/{scope['key']}.pages.json")}, "page_count": r["page_count"]})

code_scope = {"paper_key": "hgen", "official_repository": "https://github.com/Chrisshen12/HGEN", "commit": "3caba805b2c3e16ee2dfd7d56d7e79405f66fd01", "commit_date": "2026-08-29T16:59:24Z", "read_kind": "bounded_static_source_read_no_import_or_execution", "source_files": source_retrieval, "semantic_line_scopes_inclusive": [{"path": "author_source/model.py", "ranges": [[1, 365]], "note": "Pre-compaction scope 1–201,250–295,314–365 completed with 201–250 and 250–365 after resume"}, {"path": "author_source/train.py", "ranges": [[185, 224]]}], "locator_scope": "Broad keyword locator scanned both source files; not a whole-trainer semantic audit", "not_accessed": ["dataset.py", "datasets", "labels", "checkpoints", "logs", "Other source modules", "Code imports or execution"]}
parse_record = {"source": "author_source/model.py", "source_sha256": sha(PACKET / "author_source/model.py"), "operation": "ast.parse of saved text only; no import or execution", "time_UTC": NOW}
try:
    ast.parse((PACKET / "author_source/model.py").read_text())
    parse_record["result"] = "success"
except SyntaxError as error:
    parse_record.update({"result": "syntax_failure", "exception": type(error).__name__, "line": error.lineno, "offset": error.offset, "message": error.msg})
assert parse_record["exception"] == "TabError" and parse_record["line"] == 355
write("HGEN_SOURCE_CHECK.json", {"schema": "static-author-source-check-v1", "static_parse": parse_record, "findings": [{"source_lines": [188, 198], "finding": "Separate GCN_embed instance per path/member, with separate attention and path decoders"}, {"source_lines": [314, 345], "finding": "Separate feature encoder and GCNConv layers; active feature dropout; edge/node dropout commented"}, {"source_lines": [201, 292], "finding": "Learners called in initial unused-stack pass and again in used attention-fusion pass; path logits summed then log-softmaxed"}, {"source_lines": [154, 162], "finding": "No visible minmax epsilon; uniform residual missing for member0 versus printed equation"}, {"source": "author_source/train.py", "source_lines": [193, 222], "finding": "One optimizer and fused NLL plus lambda*L1(Gram)^2, one backward/step; printed paper has unsquared L1"}], "runnable_reproduction_qualified": False, "performance_or_runtime_measured": False, "scientific_limit": "Implementation discrepancies do not erase graph-neighborhood ensemble ancestry or establish novelty"})
write("READ_SCOPES.json", {"schema": "bounded-primary-read-scope-v1", "created_UTC": NOW, "new_primary_method_scopes": 3, "new_full_paper_certifications": 0, "retained_primary_paper_rereads": 0, "new_bounded_author_source_audits": 1, "code_execution_or_training": 0, "papers": paper_scopes, "author_source": code_scope, "scope_policy": "Machine extraction of every PDF page and whole-file keyword scanning do not count as reading/certifying every page, method or result", "post_compaction_note": "Resume completed the already active bounded assignment; no fourth primary method fetched or read"})

conclusions = [
    {"key": "bankgcn", "canonical_id": new_ids[0], "verified_title": "Message Passing in Graph Convolution Networks via Adaptive Filter Banks", "authors": ["Xing Gao", "Wenrui Dai", "Chenglin Li", "Junni Zou", "Hongkai Xiong", "Pascal Frossard"], "version_date": "2021-06-18", "saved_takeaway": "Learned subspace projections and Chebyshev graph-filter coefficients, residual filtered features, concatenation and ReLU, stacked subspace mixing, max absolute coefficient-cosine penalty, one graph-level CE predictor.", "operation_level_overlap": "Learnable graph-filter diversity before nonlinear processing is established", "bounded_difference": "Inspected method is one graph-classification filter-bank predictor, not predictive members on a tied node feature backbone with member losses", "limits": ["Coefficient cosine diversity need not imply response or complementary error diversity", "No graph-classification scores transferred to target node ensembles", "No source/native reproduction audit"]},
    {"key": "specformer", "canonical_id": new_ids[1], "verified_title": "Specformer: Spectral Graph Neural Networks Meet Transformers", "authors": ["Deyu Bo", "Chuan Shi", "Lele Wang", "Renjie Liao"], "version_date": "2023-03-02", "publication": "ICLR 2023", "saved_takeaway": "Spectral transformer learns multiple eigenvalue filters and reconstructs bases; feature-channel operators combine bases before learned feature mixing and nonlinear processing, with varying layer sharing by architecture.", "operation_level_overlap": "Learned nonlinear multiple-filter representations and channel-specific graph operators are established", "bounded_difference": "Single predictor, not independently supervised predictive members", "limits": ["Full eigendecomposition/preprocessing and dense bases/attention costs count", "Truncation is a changed control", "Proofs, recipes/results and code not certified", "No calibration or target-quality guarantee"]},
    {"key": "hgen", "canonical_id": new_ids[2], "verified_title": "HGEN: Heterogeneous Graph Ensemble Networks", "authors": ["Jiajun Shen", "Yufei Jin", "Kaibu Feng", "Yi He", "Xingquan Zhu"], "publication": "IJCAI 2025", "canonical_doi": "10.24963/ijcai.2025/685", "saved_takeaway": "Meta-path graph neighborhoods, multiple GNNs with feature dropout per path, within-path residual attention, summed path logits, fused CE and graph-pooled embedding Gram penalty. Pinned source gives each learner its own feature encoder and GCN layers.", "operation_level_overlap": "Ensembles receiving distinct learned/selected graph-neighborhood evidence before prediction are established", "bounded_difference": "Separate feature encoders and graph learners, rather than shared feature maps with only private scalar filters", "limits": ["Uncentered Gram includes norm effects and is not an error-complementarity guarantee", "Supplement/Appendix algorithm not retrieved; theorem not independently validated", "Pinned source penalty/attention differ from printed paper; static model.py parse fails at line355", "No executable reproduction, results or runtime qualification"]},
]
for conclusion, scope in zip(conclusions, paper_scopes):
    conclusion.update({"read_status": "new_scoped_primary_method_plus_bounded_source_for_hgen_not_full_paper", "full_read": False, "primary_url": scope["pdf"]["url"], "actual_download_url": scope["pdf"]["actual_response_url"], "source_file": scope["pdf"]["path"], "source_sha256": scope["pdf"]["sha256"], "read_scope_reference": "READ_SCOPES.json", "exact_read_scope": scope["method_scope"], "numeric_results_adopted": False, "disposition": "Attribute graph operation family; retain unresolved sharing/utility comparison; no new pilot", "global_absence_or_novelty_certificate": False})
write("PAPER_CONCLUSIONS.json", {"schema": "scoped-primary-paper-conclusions-v1", "created_UTC": NOW, "new_scoped_papers": 3, "full_paper_certifications": 0, "papers": conclusions})

discovery = json.loads((PACKET / "discovery/OPENALEX.json").read_text())
rows = []
for query_position, query in enumerate(discovery):
    for result_position, result in enumerate(query["results"]):
        title = result["title"]
        low = title.lower()
        if "adaptive filter banks" in low:
            disposition = "new_bankgcn_primary_method_scope_read"
        elif low == "specformer: spectral graph neural networks meet transformers":
            disposition = "new_specformer_primary_method_scope_read"
        elif low == "graph ensemble neural network":
            disposition = "closest_unresolved_primary_access_403_no_method_claim"
        elif "triple filter ensembles" in low:
            disposition = "reused_saved_tfe_gnn_primary_conclusion_no_reread"
        elif low == "is homophily a necessity for graph neural networks?":
            disposition = "metadata_only_acm_candidate_not_method_read_under_three_scope_cap"
        elif "from node interaction to hop interaction" in low:
            disposition = "metadata_only_hop_interaction_candidate_query_title_differs_no_method_claim"
        else:
            disposition = "metadata_only_not_primary_assessed_broad_query_or_other_candidate"
        rows.append({"query_position_zero_based": query_position, "result_position_zero_based": result_position, "id": result["id"], "title": title, "doi": result.get("doi"), "publication_date": result.get("publication_date"), "disposition": disposition})
crossref = json.loads((PACKET / "discovery/GENN_CROSSREF.json").read_text())["message"]
write("CANDIDATE_DISPOSITIONS.json", {"schema": "bounded-metadata-candidate-dispositions-v1", "created_UTC": NOW, "openalex_queries": [{"query": q["query"], "url": q["url"], "retrieved_UTC": q["retrieved_UTC"], "reported_total": q["total"], "rows_retained": len(q["results"])} for q in discovery], "returned_rows_including_duplicates": len(rows), "rows": rows, "unresolved_closest_lead": {"title": crossref["title"][0], "doi": crossref["DOI"], "authors": [a["given"] + " " + a["family"] for a in crossref["author"]], "published": crossref["published"], "primary_access": json.loads((PACKET / "discovery/GENN_ACCESS.json").read_text()), "secondary_locator": "HGEN PDF page1 introduction describes GEN integrating ensemble operations throughout GNN training", "not_established": ["Primary operator", "Parameter sharing", "Training objective", "Pooling", "Exact target exclusion"], "disposition": "Unresolved primary access; no fourth method read; missing access is not novelty evidence"}, "direct_primary_not_from_these_search_rows": [{"key": "hgen", "locator": "Official IJCAI2025 proceedings and linked author repository"}], "coverage_or_novelty_claim": False})

write("EVIDENCE_MAP.json", {"schema": "claim-source-coordinate-map-v1", "created_UTC": NOW, "claims": [
    {"claim": "Learned subspace graph filters before nonlinear processing and coefficient diversity penalty", "source": "primary/bankgcn.pdf", "pdf_pages": [5, 6, 7], "section": "4", "equations": "7–20"},
    {"claim": "One graph-level classifier and CE plus filter-diversity objective", "source": "primary/bankgcn.pdf", "pdf_pages": [8], "section": "5.1"},
    {"claim": "Multiple learned spectral bases and feature-channel operators followed by feature mixing/nonlinearity", "source": "primary/specformer.pdf", "pdf_pages": [3, 4, 5], "section": "4.1–4.4", "equations": "2–6"},
    {"claim": "Meta-path neighborhoods, feature-dropout GNNs, residual attention, summed outputs and Gram penalty", "source": "primary/hgen.pdf", "pdf_pages": [2, 3, 4], "section": "3.1–3.4", "equations": "1–9"},
    {"claim": "Separate feature encoders and private graph layers", "source": "author_source/model.py", "line_ranges_inclusive": [[188, 198], [314, 345]]},
    {"claim": "Unused initial pass, actual path fusion and raw-logit summation", "source": "author_source/model.py", "line_ranges_inclusive": [[201, 292]]},
    {"claim": "Fused NLL plus squared L1 Gram penalty and one optimizer step", "source": "author_source/train.py", "line_ranges_inclusive": [[193, 222]]},
    {"claim": "Paper/code attention discrepancy", "source": "author_source/model.py", "line_ranges_inclusive": [[154, 162]], "paper_source": "primary/hgen.pdf", "paper_equation": "6"},
    {"claim": "Static source TabError", "source": "HGEN_SOURCE_CHECK.json", "source_line": 355},
    {"claim": "Head information bottleneck, linear raw-logit collapse, complete-basis absorption and response-Gram identity", "source": "REPORT.md", "sections": ["1", "2"], "kind": "Elementary algebra for explicitly stated families, not primary theorem or experiment"},
    {"claim": "Three closed development failures retained", "source": "CLOSED_DECISION_BINDINGS.json", "kind": "Authorized closed summaries only"},
    {"claim": "Modern backbones, adjacent expert/filter priors and access limits", "source": "REUSED_REFERENCES.json", "kind": "Saved substantive conclusion reuse, no new primary reread"},
]})

write("PILOT_DECISION.json", {"schema": "bounded-quality-pilot-decision-v1", "created_UTC": NOW, "pilots_proposed": 0, "operation_distinction_against_weak_head_only_control": "Private neighborhood filtering before a discarded common propagation channel and before nonlinearity can preserve different evidence", "strong_control_reductions": ["Linear mean raw logits collapse to a single graph-basis predictor", "One-stage polynomial filtering is exactly absorbable into a capable head on the complete same shared basis", "Repeated nonlinear multiple-filter channels already have BankGCN/Specformer ancestry"], "unresolved_quality_question": "Does predictive-member allocation with tied feature maps improve pooled quality and measured total cost over equally capable multiscale heads, nonlinear filter-bank singles, all-layer exact-message BE and independent ensembles?", "reason_for_zero_pilot": "Current family changes sharing, member allocation and objectives around established graph operations; no specified new operation survives closest-prior review strongly enough for a new-method pilot", "not_claimed": ["Architecture universally useless", "All nonlinear adaptive variants exactly equivalent", "No empirical utility possible", "Global novelty exclusion", "Resource availability determines the scientific disposition"], "execution_authorized_by_this_packet": False, "existing_frozen_studies_changed": False})

write("PROVENANCE.json", {"schema": "bounded-quality-gap-custody-v1", "created_UTC": NOW, "packet": PACKET.name, "task": "Bounded review of graph-specific member filter/neighborhood diversity on a shared feature backbone versus final-classifier diversity", "external_unchanged_bindings": list(external_bindings.values()), "retrieval_receipts": ["primary/RETRIEVAL.json", "author_source/METADATA_RETRIEVAL.json", "author_source/SOURCE_RETRIEVAL.json", "discovery/OPENALEX.json", "discovery/GENN_CROSSREF.json", "discovery/GENN_ACCESS.json"], "primary_version_policy": "Unversioned arXiv download URLs; exact v1 verified from PDF footer and landing history. Official IJCAI2025 PDF for HGEN.", "scope_counts": {"new_primary_method_scopes": 3, "new_full_paper_certifications": 0, "author_source_audits": 1, "openalex_metadata_queries": 7, "openalex_returned_rows_with_duplicates": 78, "new_pilots": 0}, "constraints_observed": {"training_or_experiments": False, "gpu_requests": False, "current_outcomes_accessed": False, "initializer_outcomes_accessed": False, "mixed40_outcomes_accessed": False, "native15_outcomes_accessed": False, "buddy_outcomes_accessed": False, "closed_summary_raw_artifacts_accessed": False, "canonical_ledger_or_status_edits": False, "frozen_study_changes": False, "index_append": False}, "source_execution_note": "Only retrieval/extraction/custody scripts and static ast.parse; no author-source imports, model/dataset code execution or training", "new_write_scope": str(PACKET), "not_established": ["Target predictive superiority", "Runtime/cost reproduction", "Theory certification", "Every-page reading", "Exact complete-prior equivalence", "Global novelty"]})

payloads = []
for path in sorted(PACKET.rglob("*")):
    if path.is_file() and path.name != "MANIFEST.json":
        payloads.append({"path": str(path.relative_to(PACKET)), "sha256": sha(path), "size": path.stat().st_size})
write("MANIFEST.json", {"schema": "sha256-payload-manifest-v1", "created_UTC": NOW, "packet": PACKET.name, "algorithm": "SHA256", "excludes": ["MANIFEST.json (self)"], "payload_file_count": len(payloads), "files": payloads})
print(json.dumps({"payload_files": len(payloads), "report_sha256": sha(PACKET / "REPORT.md"), "manifest_sha256": sha(PACKET / "MANIFEST.json"), "new_primary_scopes": 3, "full_paper_certifications": 0, "pilots": 0}, indent=2))
