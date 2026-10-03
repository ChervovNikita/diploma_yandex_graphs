"""Build literature-only receipts/excerpts/custody; no modeling or data work."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

P = Path(__file__).resolve().parent
R = P.parent
PC = R / "cached_structure_link_ensemble_feasibility_v1"


def digest(path):
    b = path.read_bytes()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def write(name, value):
    (P / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


pencil_blocks = json.loads((PC / "sources/pencil_2602.01553v4_blocks.json").read_text())
prior = json.loads((PC / "READ_SCOPES.json").read_text())
prior_ids = next(s["paragraph_equation_ids"] for s in prior["scopes"] if s.get("id") == "pencil_2602.01553v4")
pencil_ids = [
    "S3.E1", "S3.E2", "S3.E3",
    "S3.SS3.p1.1", "S3.SS3.p2.1", "S3.SS3.p3.1", "S3.SS3.p4.1",
    "A1.SS0.SSS0.Px1.p1.1", "A1.E6", "A1.SS0.SSS0.Px1.p1.2",
    "A1.SS0.SSS0.Px1.p2.1", "A1.SS0.SSS0.Px1.p3.1", "A1.E7", "A1.SS0.SSS0.Px1.p3.2",
    "A1.SS0.SSS0.Px2.p1.1", "A1.E8", "A1.SS0.SSS0.Px2.p1.2",
    "A2.SS9.p1.1", "A2.SS9.p2.1", "A2.SS9.p3.1", "A2.SS9.p4.1", "A2.SS9.p5.1", "A2.SS9.p6.1", "A2.SS9.p7.1", "A2.SS9.p8.1", "A2.SS9.p8.2", "A2.SS9.p9.1", "A2.SS9.p10.1",
]
assert not set(pencil_ids) & set(prior_ids)
new_methods = [
    {
        "key": "iecnc", "canonical_id": "doi:10.1007/s41019-024-00267-6",
        "verified_title": "Common Neighbor Completion with Information Entropy for Link Prediction in Social Networks",
        "authors": ["Zhengyun Zhou", "Guojia Wan", "Bo Du"],
        "version": "Final Data Science and Engineering 10:40-53 article; published online 2025-01-24",
        "primary_url": "https://link.springer.com/content/pdf/10.1007/s41019-024-00267-6.pdf",
        "primary_file": "primary/common_neighbor_entropy2025.pdf",
        "source_binding": digest(P / "primary/common_neighbor_entropy2025.pdf"),
        "read_status": "first_scoped_primary_method_read_relative_to_retained_index_and_saved_scope_map",
        "full_paper_read": False, "author_code_read": False,
        "saved_takeaway": "Learned residual-neighbor common-adjacency probabilities are combined with contextual MPNN features; Eq11 uses -P ln P weighted aggregation. Direct current learned-completion/features association ancestry.",
        "operational_overlap": "Missing-link/common-neighbor probabilities alter graph feature aggregation before final score. Such within-predictor association precedes candidate.",
        "remaining_delta": "Exact own-member versus post-clamp-mean weights under shared encoder/factorized native NCNC decoder bank not established by this scope.",
        "qualification_limits": ["-p ln p individual-event uncertainty/maximum claim conflicts with printed scalar function; not Bernoulli entropy", "Observed-CN P=1 gives zero Eq11 weight", "q/output dimensions and softmax axis not qualified", "MPNN terminology inconsistent; no author code read", "No graph-collab quality/cost result adopted"],
        "exact_read_scope": [{"pdf_page": 4, "line_start": 52, "line_end": 118}, {"pdf_page": 5, "lines": "all"}, {"pdf_page": 6, "lines": "all"}, {"pdf_page": 7, "line_start": 1, "line_end": 63}],
        "visual_checks": ["evidence/iecnc-05.png", "evidence/iecnc-06.png", "evidence/iecnc-07.png"],
    },
    {
        "key": "egae", "canonical_id": "doi:10.7717/peerj-cs.2648",
        "verified_title": "Ensemble graph auto-encoders for clustering and link prediction",
        "authors": ["Chengxin Xie", "Jingui Huang", "Yongjiang Shi", "Hui Pang", "Liting Gao", "Xiumei Wen"],
        "version": "PeerJ Computer Science 11:e2648 final article, 2025-01-22; PMC11784894 fulltext",
        "primary_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11784894/",
        "primary_file": "primary/ensemble_gae_pmc.html",
        "source_binding": digest(P / "primary/ensemble_gae_pmc.html"),
        "read_status": "first_scoped_primary_method_read_relative_to_retained_index_and_saved_scope_map",
        "full_paper_read": False, "author_code_read": False,
        "saved_takeaway": "Three autoencoder embedding matrices adaptively combined before common adjacency decoder; nested GCN/GAT/SuperGAT feature fusion. Explicit pre-decoder pooling precedent.",
        "operational_overlap": "Graph ensemble fusion location before versus after decoding is already a design choice.",
        "remaining_delta": "No query-dependent native NCNC probability bank, factorized private completion/decoder association or post-clamp twin identified in scoped source.",
        "qualification_limits": ["Adaptive weight dimensions/optimization not source-code certified", "Scalar-weight inner-product expansion is a stated special case, not an implementation reproduction", "Cora/Citeseer/PubMed claims do not qualify official collab or full cost"],
        "exact_read_scope": {"block_start_inclusive": 36, "block_end_inclusive": 66, "primary_section_ids": ["sec3", "sec4", "sec5", "sec6", "sec7", "sec8", "sec9"], "equation_ids": ["eqn-1", "eqn-2", "eqn-3", "eqn-4", "eqn-5"], "algorithm_table_section_ids": ["table-10", "table-11"]},
    },
]
retained = {
    "key": "pencil_changed_question_scope", "canonical_id": "arxiv:2602.01553", "versioned_id": "arXiv:2602.01553v4",
    "verified_title": "Plain Transformers are Surprisingly Powerful Link Predictors", "version_date": "2026-09-28",
    "authors": ["Quang Truong", "Yu Song", "Donald Loveland", "Mingxuan Ju", "Tong Zhao", "Neil Shah", "Jiliang Tang"],
    "read_status": "retained_identity_targeted_primary_scope_extension_for_changed_question",
    "full_paper_read": False, "author_code_read": False,
    "prior_scope_reference": "cached_structure_link_ensemble_feasibility_v1/READ_SCOPES.json",
    "previous_ids_not_repeated": prior_ids,
    "new_selected_ids": pencil_ids,
    "source_file": "cached_structure_link_ensemble_feasibility_v1/sources/pencil_2602.01553v4.html",
    "source_binding": digest(PC / "sources/pencil_2602.01553v4.html"),
    "saved_takeaway": "Propagation adjacency recovered from observed adjacency/identifier fields, not learned missing-edge weights. Learned transformer remains pair-context capacity. B9 theorem concerns LRP-PENCIL versus SEAL under same sampling; LRP is absent from experiments and does not establish candidate/NCNC dominance or fixed-cost merit.",
    "complete_published_hypothesis_equality_or_absence_certified": False,
}
write("PAPER_CONCLUSIONS.json", {"schema": "ncnc-pairing-bounded-current-priors-v1", "first_scoped_primary_method_identity_count": 2, "retained_targeted_method_revisit_count": 1, "full_paper_read_count": 0, "new_records": new_methods, "retained_scope_extension": retained, "global_novelty_quality_or_acceptance_verdict": False})

write("READ_SCOPES.json", {
    "schema": "bounded-current-lp-reading-v1", "public_discovery_calls": 6, "discovery_call_limit": 6,
    "exact_selected_source_retrieval_attempts": 5, "retrieval_attempts_are_not_discovery_search_calls": True,
    "first_scoped_method_identities": 2, "new_method_identity_limit": 3,
    "retained_pencil_scope_extensions": 1, "total_scoped_method_read_events": 3,
    "full_primary_papers_read": 0, "author_code_read_events": 0,
    "methods": [{"key": s["key"], "version": s["version"], "exact_scope": s["exact_read_scope"]} for s in new_methods],
    "pencil": {"version": retained["versioned_id"], "prior_ids": prior_ids, "new_ids": pencil_ids, "selected_ids_overlap_prior_scope": False, "purpose": "Resolve adjacency semantics and current-framework comparison, not repeat saved full scope"},
    "extraction_not_reading": "All 14 IECNC PDF pages mechanically extracted; only declared pages/lines read. Whole PMC HTML mechanically parsed to blocks; only declared method blocks and metadata/abstract read.",
    "incidental_primary_exposure": ["IECNC p1 title/authors/abstract and global keyword locators, including printed empirical rows/claims; p7 experiment dataset description displayed and full visual p7", "E-GAE abstract empirical claims, intro/related-work decoder locator snippets and Table1 in selected method block output; no evaluation audit", "PENCIL new3.3 heuristic-regression paragraphs, B9 theoretical proof sketch/empirical caveats, bibliography titles/links and title/abstract/intro opening metadata inspection; no full empirical review"],
    "retained_scope_reuse": "Starting LP quality report/conclusions/citations, cached-structure LP conclusions/scopes, saved LPFormer/Link-MoE/NCNC source map; no original NCNC/BUDDY/LPFormer/Link-MoE method or prototype reopen.",
    "incidental_historical_context": "Starting quality report contains historical outcome-summary sentences; no original/live outcome content opened.",
    "unread_primary_lead": "Bridging Theory and Practice in Link Representation with Graph Neural Networks: selected OpenReview PDF/exact note route403; no method read.",
    "other_discovered_metadata_not_primary_method_reads": True,
})

excerpts = []
for page, start, end, claim in [(5, 42, 75, "Learned residual-neighbor completion probabilities"), (6, 72, 105, "Probabilistic completion times contextual MPNN features; Eq9"), (7, 1, 63, "Printed entropy-weight aggregation, scalar-uncertainty claims and Eq11")]:
    file = f"primary/common_neighbor_entropy2025_page{page}.txt"
    text = "\n".join((P / file).read_text().splitlines()[start-1:end])
    excerpts.append({"key": "iecnc", "coordinate_kind": "PDF_page_extracted_line_range", "file": file, "file_binding": digest(P/file), "pdf_page": page, "line_start": start, "line_end": end, "claim": claim, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
for i, b in enumerate(pencil_blocks):
    if b.get("id") in pencil_ids:
        excerpts.append({"key": "pencil", "coordinate_kind": "saved_HTML_block_and_id", "base": "research_root", "file": "cached_structure_link_ensemble_feasibility_v1/sources/pencil_2602.01553v4_blocks.json", "file_binding": digest(PC / "sources/pencil_2602.01553v4_blocks.json"), "block_index": i, "html_id": b["id"], "text": b["text"], "text_sha256": hashlib.sha256(b["text"].encode()).hexdigest()})
egae_blocks = json.loads((P / "primary/ensemble_gae_blocks.json").read_text())
for i in [45, 46, 47, 59, 60, 61, 62, 63, 64]:
    b = egae_blocks[i]
    excerpts.append({"key": "egae", "coordinate_kind": "PMC_HTML_block_and_section", "file": "primary/ensemble_gae_blocks.json", "file_binding": digest(P / "primary/ensemble_gae_blocks.json"), "block_index": i, "html_id": b["id"], "section_id": b["section_id"], "text": b["text"], "text_sha256": b["sha256"]})
write("PRIMARY_EXCERPTS.json", {"schema": "exact-primary-scope-excerpts-v1", "excerpts": excerpts, "equation_typography_note": "IECNC p5-7 visually verified; PENCIL and E-GAE retained primary HTML equation blocks contain duplicate math representations/spacing artifacts. Raw primary bytes remain authoritative."})

inputs = [
    "link_prediction_quality_gap_20261003_v1/REPORT.md", "link_prediction_quality_gap_20261003_v1/PAPER_CONCLUSIONS.json", "link_prediction_quality_gap_20261003_v1/CITATIONS.json", "link_prediction_quality_gap_20261003_v1/READ_SCOPES.json", "link_prediction_quality_gap_20261003_v1/MANIFEST.json",
    "cached_structure_link_ensemble_feasibility_v1/READ_SCOPES.json", "cached_structure_link_ensemble_feasibility_v1/PAPER_CONCLUSIONS.json", "cached_structure_link_ensemble_feasibility_v1/RETRIEVAL_LOG.json", "cached_structure_link_ensemble_feasibility_v1/sources/pencil_2602.01553v4.html", "cached_structure_link_ensemble_feasibility_v1/sources/pencil_2602.01553v4_blocks.json",
    "graph_link_structural_uncertainty_literature_v1/PAPER_CONCLUSIONS.json", "graph_link_competing_family_source_20261003_v1/REUSED_CONCLUSIONS.json",
    "literature_memory/index_v31/LITERATURE_INDEX.json", "literature_memory/index_v32/LITERATURE_INDEX.json", "literature_memory/index_v32/IDENTITY_AND_SCOPE_ACCOUNTING.json", "literature_memory/index_v32/APPEND_RECEIPT.json", "literature_memory/index_v32/VERIFICATION.json",
    "conditional_graph_response_closest_priors_20261003_v1/REPORT.md", "conditional_graph_response_closest_priors_20261003_v1/MANIFEST.json", "conditional_graph_response_closest_priors_20261003_v1/PRIOR_REPORT_CUSTODY_BEFORE.json",
]
write("INPUT_BINDINGS.json", {"research_root": str(R), "inputs": [{"path": f, **digest(R/f)} for f in inputs], "read_kind": "Reports/conclusions/receipts and targeted retained PENCIL passages; some files hash-bound only. No outcomes/prototypes/model files opened."})
write("REUSED_CONCLUSIONS.json", {"schema": "saved-lp-prior-map-reuse-v1", "source": "link_prediction_quality_gap_20261003_v1/PAPER_CONCLUSIONS.json", "source_binding": digest(R / "link_prediction_quality_gap_20261003_v1/PAPER_CONCLUSIONS.json"), "source_primary_reopens": 0, "prior_operations": {"NCNC": "Native missing-link scores detached before clamp/weighted sparse feature sum; own predictor association exists; independent complete members retain it", "Link-MoE": "Complete expert scores combined; structure-conditioned gate training/supervision and whole expert work remain charged", "LPFormer": "Learned pair-context/structural reference; retained source scope and unverified utility", "BUDDY": "Independent nonlinear members can reuse deterministic structural cache; same query input is an information boundary, no useful collision measured"}})
write("DISCOVERY_DISPOSITIONS.json", {
    "schema": "bounded-lp-discovery-dispositions-v1", "public_calls_used": 6,
    "first_two_calls": "Noisy general metadata; no method source inferred from their rankings.",
    "method_selected": [s["canonical_id"] for s in new_methods],
    "retained_method_selected_for_missing_passages": "arxiv:2602.01553",
    "unread_selected_lead": {"title": "Bridging Theory and Practice in Link Representation with Graph Neural Networks", "OpenReview_id": "WYnvP3DePZ", "venue": "NeurIPS 2025", "locator": "Saved PENCIL bibliography; https://openreview.net/forum?id=WYnvP3DePZ", "PDF_and_exact_record_status": "403", "primary_method_read": False},
    "not_selected": "Other title/abstract candidates remain metadata-only; no additional retrieval/search after bounded selection.",
    "no_novelty_or_absence_certificate": True,
})
write("PROVENANCE.json", {"schema": "source-only-lp-comparison-provenance-v1", "UTC": datetime.now(timezone.utc).isoformat(), "external_requests": "6 discovery metadata requests + 5 selected exact-source/record retrieval attempts. No blocked URL repeated.", "selection_scope": "2 genuinely new primary identities + 1 retained PENCIL missing-passage/changed-question scope. No full-paper or author-source certification.", "source_extraction": "Bundled pypdf; bundled pdftoppm; stdlib HTMLParser after unavailable bs4; no installation/model imports.", "non_actions": ["no prototype edits", "no numerical/model execution", "no data/labels/live or raw outcomes", "no cache/checkpoint/tensor/server", "no canonical ledger/status/old-index edits", "no new experiment or acceptance verdict"], "separate_index_task": "index_v32 appends only the previous sealed conditional-response packet; this LP packet remains separate pending root review."})

before = json.loads((R / "conditional_graph_response_closest_priors_20261003_v1/PRIOR_REPORT_CUSTODY_BEFORE.json").read_text())
custody = []
for b in before:
    observed = digest(R/b["path"])
    custody.append({"path": b["path"], "expected": {k:b[k] for k in ["bytes","sha256"]}, "observed": observed, "unchanged": observed == {k:b[k] for k in ["bytes","sha256"]}})
write("PRIOR_REPORT_CUSTODY.json", {"targets": len(custody), "all_unchanged": all(v["unchanged"] for v in custody), "files": custody, "note": "Old 270 snapshot targets rehashed without parsing outcomes; separate bindings preserve new prior packet report/index files. Concurrent newer files outside snapshot not certified by this list."})
excluded = {"MANIFEST.json", "VERIFICATION.json"}
write("MANIFEST.json", {"schema": "bounded-source-review-manifest-v1", "excludes": sorted(excluded), "files": [{"path": str(f.relative_to(P)), **digest(f)} for f in sorted(P.rglob('*')) if f.is_file() and f.name not in excluded]})
print(json.dumps({"discovery_calls": 6, "new_methods": 2, "retained_scope_extensions": 1, "passages": len(excerpts), "older_reports_preserved": all(v["unchanged"] for v in custody)}))
