"""Build local source/custody metadata only; no modeling or data operations."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

PACKET = Path(__file__).resolve().parent
RESEARCH = PACKET.parent


def digest(path):
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def write(name, value):
    (PACKET / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


sources = [
    {
        "key": "repulsive",
        "canonical_id": "arxiv:2106.11642",
        "version": "arXiv:2106.11642v3",
        "title": "Repulsive Deep Ensembles are Bayesian",
        "authors": ["Francesco D'Angelo", "Vincent Fortuin"],
        "revision_date": "2023-03-28",
        "url": "https://arxiv.org/pdf/2106.11642v3",
        "pdf": "primary/repulsive_v3.pdf",
        "pdf_pages": 21,
        "method_scopes": [{"page": 3, "lines": "all"}, {"page": 4, "lines": "all"}, {"page": 7, "lines": "all"}],
        "other_reading": "Page1 title/authors/abstract opening; global keyword locator lines including incidental printed results/recipes. No full experiments, proofs or appendix review.",
        "visual_checks": ["evidence/repulsive_v3-04.png", "evidence/repulsive_v3-07.png"],
        "operational_overlap": "Finite function evaluation, kernel-based interaction and Jacobian-transpose pullback already published. Candidate degree-two energy identity is independently algebraic, not equality to the published KDE/SGE/SSGE posterior update.",
        "remaining_delta": "Exact conditional graph-removal probability map, private-only restricted update, per-member CE/energy/backtracking feasibility; full published overlap not excluded.",
        "qualification_limit": "Read Eq20 and kernel requirements; no general posterior/convergence inheritance for arbitrary degree-two energy or constrained private adaptation.",
    },
    {
        "key": "cfgnn",
        "canonical_id": "arxiv:2102.03322",
        "version": "arXiv:2102.03322v4",
        "title": "CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks",
        "authors": ["Ana Lucic", "Maartje ter Hoeve", "Gabriele Tolomei", "Maarten de Rijke", "Fabrizio Silvestri"],
        "revision_date": "2022-02-22",
        "url": "https://arxiv.org/pdf/2102.03322v4",
        "pdf": "primary/cfgnn_v4.pdf",
        "pdf_pages": 13,
        "method_scopes": [{"page": 3, "lines": "all"}, {"page": 4, "lines": "all"}, {"page": 5, "lines": "all"}],
        "other_reading": "Page1 title/authors/abstract opening; global keyword locator lines; complete displayed p5 includes complexity and experimental setup/runtime/accuracy text incidentally. No evaluation audit or whole-paper review.",
        "visual_checks": ["evidence/cfgnn_v4-04.png", "evidence/cfgnn_v4-05.png"],
        "operational_overlap": "Probability predictor under finite adjacency edge deletion, fixed W, retained selfloops and recomputed degrees.",
        "remaining_delta": "Source optimizes local mask for minimal prediction flip; candidate fixes supervised masks and adapts private paths to retain competence and diversify conditional responses.",
        "qualification_limit": "p4 minimization text vs p5 plus-gradient printed update and unresolved threshold-gradient details. No author code read.",
    },
    {
        "key": "adgcl",
        "canonical_id": "arxiv:2106.05819",
        "version": "arXiv:2106.05819v4",
        "title": "Adversarial Graph Augmentation to Improve Graph Contrastive Learning",
        "authors": ["Susheel Suresh", "Pan Li", "Cong Hao", "Jennifer Neville"],
        "revision_date": "2021-11-03",
        "url": "https://arxiv.org/pdf/2106.05819v4",
        "pdf": "primary/adgcl_v4.pdf",
        "pdf_pages": 25,
        "method_scopes": [{"page": 3, "line_start": 63, "line_end": 66}, {"page": 4, "lines": "all"}, {"page": 5, "lines": "all"}, {"page": 6, "lines": "all"}, {"page": 18, "line_start": 1, "line_end": 45}],
        "other_reading": "Page1 title/authors/abstract opening; global keyword locator lines including empirical tables/results/recipe exposure; p4 motivating example, p5 theorem statement, p6 related-work opening, p18 post-algorithm results text exposed. Full visual p18 includes further regularization-result text. No proof or empirical evaluation review.",
        "visual_checks": ["evidence/adgcl_v4-06.png", "evidence/adgcl_v4-18.png"],
        "operational_overlap": "Learned edge-dropping family, graphwise drop-ratio regularization, min-max view/encoder training and InfoNCE on projected graph embeddings.",
        "remaining_delta": "Not an inter-member centered class-probability response objective. Candidate has fixed label-derived masks, node/class/neighborhood grouping and frozen common/head/private-only guarded updates.",
        "qualification_limit": "Visually confirmed retained/dropped omega/p semantics inconsistency, relaxed formula semantics and Algorithm1 z1/z2 assignment inconsistency. No author implementation pin/read or fidelity certification.",
    },
]

for source in sources:
    source.update(digest(PACKET / source["pdf"]))
    source["read_status"] = "first_scoped_primary_method_read_relative_to_index_v31"
    source["full_paper_read"] = False
    source["author_code_read"] = False
    source["extraction_note"] = "All pages mechanically extracted; only declared scopes read. Coordinates are pypdf page-text lines; PDF visuals decide equation typography."

write("CONCLUSIONS.json", {
    "schema": "conditional-graph-response-source-conclusions-v1",
    "sources": sources,
    "implementation_identity": {
        "status": "exact_algebraic_identity",
        "kernel": "k(f,g)=mean_s inner(T_s(f),T_s(g))^2",
        "feature_map": "Concat_s[T_s(f) tensor_product T_s(f)]/sqrt(number_active_groups)",
        "candidate_energy": "mean_unordered_member_pairs k(f_m,f_n)",
        "gradient": "Same complete chain-rule private-coordinate pullback, including centering/normalization Jacobians.",
        "matched_comparator": "Collapses to candidate if response, coefficient, CE, coordinates, optimizer, active set and guard rules match; no duplicate fit recommended.",
        "published_equality_claim": False,
        "posterior_convergence_claim": False,
    },
    "remaining_delta": "Conditional finite graph-removal probability-response map and competence/energy constrained private update setting; no new cosine primitive claimed.",
    "unresolved_metadata_only_lead": {
        "title": "Adversarial Contrastive Graph Augmentation with Counterfactual Regularization",
        "canonical_id": "doi:10.1609/aaai.v39i18.34101",
        "venue": "AAAI 2025", "publication_date": "2025-04-11", "pages": "19086-19094",
        "method_scope_read": False, "primary_method_access": "failed_bounded_routes",
        "metadata_files": ["primary/acgcr_crossref.json", "search/search_counterfactual_views.json"],
        "scope_limit": "Abstract metadata cannot resolve exact objective/conditioning/update permissions or establish absence of complete overlap.",
    },
    "no_acceptance_verdict": True,
})

write("READ_SCOPES.json", {
    "schema": "bounded-primary-reading-accounting-v1",
    "new_primary_method_scopes": 3,
    "maximum_authorized_new_primary_method_scopes": 4,
    "new_primary_method_identities": [s["canonical_id"] for s in sources],
    "full_primary_papers_read": 0,
    "author_code_scopes_read": 0,
    "saved_dice_forde_adp_primary_reopens": 0,
    "retained_index_counts_not_full_paper_counts": {"conclusion_records": 130, "normalized_paper_identifiers": 85, "software_identifiers": 2},
    "selected_method_scopes": [{key: s[key] for key in ["key", "version", "method_scopes", "other_reading", "visual_checks", "extraction_note"]} for s in sources],
    "metadata_read": "arXiv title/author/version/abstract metadata for three sources; OpenAlex/Crossref discovery metadata, including ACGA abstract. No ACGA method source read.",
    "retained_scope_reuse": "Quality-gap report and saved REUSED_REFERENCES/READ_SCOPES; no prior primary PDF/HTML reopened. Archived quotations in reused JSON incidentally displayed.",
    "initial_input_order": ["quality-gap REPORT.md", "quality-gap CANDIDATE.json", "index_v31 README/accounting/metadata"],
    "comprehensive_search_or_full_paper_total_certified": False,
})

excerpts = []
ranges = [
    ("repulsive", "repulsive_v3", 4, 44, 78, "Finite canonical projection and Jacobian pullback; Definition1 and Eq5"),
    ("repulsive", "repulsive_v3", 7, 43, 70, "KDE-normalized functional update and prior requirement; Eq20"),
    ("repulsive", "repulsive_v3", 7, 71, 78, "Generic kernel repulsion versus posterior consistency constraints"),
    ("cfgnn", "cfgnn_v4", 4, 20, 42, "Sparse adjacency edge-removal parameterization"),
    ("cfgnn", "cfgnn_v4", 4, 43, 77, "Selfloops, degree recomputation and fixed predictor weights; Eqs3-4"),
    ("cfgnn", "cfgnn_v4", 4, 78, 96, "Prediction-flip NLL, distance and intermediate parameter gradient; Eq5"),
    ("cfgnn", "cfgnn_v4", 5, 2, 27, "Algorithm1 including printed plus-gradient update"),
    ("adgcl", "adgcl_v4", 4, 55, 78, "Augmentation family and adversarial min-max objective; Eq6"),
    ("adgcl", "adgcl_v4", 5, 39, 50, "Learnable edge perturbation encoder/augmenter roles"),
    ("adgcl", "adgcl_v4", 6, 11, 29, "Edge parameterization and relaxation; Eq7"),
    ("adgcl", "adgcl_v4", 6, 30, 63, "Drop regularization and InfoNCE objective; Eqs8-9"),
    ("adgcl", "adgcl_v4", 18, 1, 39, "Algorithm1 update/regularizer/z-assignment print"),
]
for key, stem, page, start, end, claim in ranges:
    file = f"primary/{stem}_page{page}.txt"
    lines = (PACKET / file).read_text().splitlines()
    text = "\n".join(lines[start - 1:end])
    excerpts.append({"source": key, "pdf_page": page, "page_text": file,
                     "page_text_binding": digest(PACKET / file),
                     "line_start": start, "line_end": end, "claim": claim,
                     "extracted_text": text,
                     "excerpt_utf8_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "note": "Verbatim pypdf extraction, including ligatures/line-wrap artifacts; visually verified equations in associated source pages."})
write("PRIMARY_EXCERPTS.json", {"schema": "page-line-primary-excerpts-v1", "passages": excerpts})

inputs = [
    "graph_contrastive_private_paths_quality_gap_20261003_v1/REPORT.md",
    "graph_contrastive_private_paths_quality_gap_20261003_v1/CANDIDATE.json",
    "graph_contrastive_private_paths_quality_gap_20261003_v1/PILOT_SPEC.json",
    "graph_contrastive_private_paths_quality_gap_20261003_v1/REUSED_REFERENCES.json",
    "graph_contrastive_private_paths_quality_gap_20261003_v1/READ_SCOPES.json",
    "literature_memory/index_v31/README.md",
    "literature_memory/index_v31/LITERATURE_INDEX.json",
    "literature_memory/index_v31/READ_ACCOUNTING_CORRECTION.json",
    "literature_memory/index_v31/HGEN_ALIAS_APPEND.json",
    "literature_memory/index_v31/ADOPTION_RECEIPT.json",
]
write("INPUT_BINDINGS.json", {"base": str(RESEARCH), "inputs": [{"path": f, **digest(RESEARCH / f)} for f in inputs], "pilot_spec_read_kind": "hash/custody binding only; no edit"})

write("REUSED_CONCLUSIONS.json", {
    "schema": "saved-prior-interpretation-reuse-v1",
    "source_packet": "graph_contrastive_private_paths_quality_gap_20261003_v1",
    "custody": "INPUT_BINDINGS.json pins REPORT.md, REUSED_REFERENCES.json and READ_SCOPES.json; their retained source/coordinate receipts remain authoritative.",
    "primary_reopens": 0,
    "sources": [
        {"id": "arXiv:2101.05544v1", "name": "DICE", "passages": "Section2.1 Eq1", "takeaway": "Label-conditioned feature MI/redundancy with competence; not candidate cosine statistic."},
        {"id": "arXiv:2306.02775v3", "name": "FoRDE", "passages": "Sections3.2,3.4,3.5", "takeaway": "Normalized true-label input-gradient kernel, particle repulsion, minibatch bias and extra differentiation; image cost does not transfer to graphs."},
        {"id": "arXiv:1901.08846v3", "name": "ADP", "passages": "Sections3.1-3.3", "takeaway": "Non-target probability diversity, entropy and CE; determinant rank limit and probability-pooling scope."},
    ],
    "incidental_exposure": "Archived primary quotations embedded in saved reuse records were displayed; no original primary source reopened or new scope claimed.",
})

write("CONTROL_RECOMMENDATIONS.json", {
    "schema": "distinct-prospective-source-controls-v1",
    "saved_pilot_modified": False,
    "fit_authorization_or_execution": False,
    "duplicate_comparator": {"name": "Matched degree-two response-kernel energy", "disposition": "exact candidate identity; source/algebra check only; do not duplicate a fit"},
    "distinct_kernel_energy_family": {
        "kernel": "k_h=mean_active_groups exp(-squared_norm(U_m,s-U_n,s)/(2*h^2))",
        "energy": "D_h=mean_unordered_member_pairs k_h",
        "fixed_matched_parts": ["exact finite graph response", "groups/active set", "centering/normalization", "native/probe CE", "private-only permissions", "warm state", "optimizer/backtracking", "competence and original response-energy guards", "native inference/pool"],
        "changed_part": "similarity geometry; RBF distinguishes opposite directions, squared cosine identifies their similarity",
        "qualification": "Declare and fix bandwidth/strength or qualify under a prospective source-label budget without current pilot outcomes; equal candidate qualification effort; full extra costs charged.",
        "label": "Functional-kernel-informed RBF energy adaptation; not RDE posterior update, no Bayesian convergence claim",
    },
    "other_distinct_families": ["native probability versus graph-removal response object", "global/class/class-neighborhood conditioning", "source-qualified KDE normalized force with explicit prior/score/denominator", "existing augmentation-only and degree/count-matched label-permuted masks", "separately source-qualified AD-GCL-inspired node/private view learner"],
    "execution_limit": "Any new fitted arm requires a prospective amendment; no change or added fits to the saved eight-arm pilot are made here.",
})

write("PROVENANCE.json", {
    "schema": "bounded-source-review-provenance-v1",
    "created_UTC": datetime.now(timezone.utc).isoformat(),
    "tools": "Public HTTP urllib; OpenAlex/Crossref metadata; bundled pypdf extraction; bundled pdftoppm rendering; filesystem/hash/stdlib verification.",
    "initial_unretained_searches": [
        "Google graph counterfactual augmentation contrastive learning edge removal class conditional: HTTP429",
        "DuckDuckGo graph counterfactual contrastive augmentation: HTTP202 bot challenge, no results",
        "Bing RSS ensemble functional diversity response regularization: unrelated results, ignored",
        "OpenAlex graph counterfactual contrastive augmentation",
        "OpenAlex ensemble function space diversity regularization",
        "OpenAlex ensemble functional diversity probability repulsive",
        "OpenAlex ensemble class conditional diversity regularization",
        "OpenAlex CF-GNNExplainer counterfactual explanations graph neural networks",
        "OpenAlex Repulsive Deep Ensembles are Bayesian",
        "OpenAlex graph supervised contrastive class neighborhood",
        "OpenAlex conditional ensemble diversity neural networks",
        "OpenAlex title-search supervised contrastive graph",
    ],
    "unretained_search_limit": "Metadata-only discovery; not reproducible retained search evidence or primary method reading. Selected queries were repeated and saved with receipts.",
    "retained_receipts": sorted(p.name for p in PACKET.glob("RETRIEVAL*.json")),
    "new_primary_scope_selection": "Three exact-version sources after retained-index identity/title check. Fourth slot unused.",
    "non_actions": ["no pilot/fit", "no datasets/tensors", "no compute-server/current-live-outcome access", "no canonical ledger/index/manuscript edits", "no acceptance verdict", "no subagents"],
    "preservation": "Rehash pre-existing 270 REPORT.md/REVIEW.json snapshot targets without parsing outcomes. Concurrent new files outside snapshot do not change preservation scope.",
    "implementation_comparison": "Mathematical identity only. No candidate or author implementation was executed or numerically compared.",
})

before = json.loads((PACKET / "PRIOR_REPORT_CUSTODY_BEFORE.json").read_text())
after = []
for item in before:
    target = RESEARCH / item["path"]
    observed = digest(target) if target.is_file() else {"missing": True}
    after.append({"path": item["path"], "expected": {k: item[k] for k in ["bytes", "sha256"]},
                  "observed": observed, "unchanged": observed == {k: item[k] for k in ["bytes", "sha256"]}})
write("PRIOR_REPORT_CUSTODY_AFTER.json", {"targets": len(after), "all_unchanged": all(v["unchanged"] for v in after), "files": after})

excluded = {"MANIFEST.json", "VERIFICATION.json"}
write("MANIFEST.json", {"schema": "source-packet-hash-manifest-v1", "manifest_excludes": sorted(excluded),
    "files": [{"path": str(f.relative_to(PACKET)), **digest(f)} for f in sorted(PACKET.rglob("*")) if f.is_file() and f.name not in excluded]})
print(json.dumps({"new_scoped_methods": 3, "full_primary_reads": 0, "prior_reports": len(after), "prior_reports_unchanged": all(v["unchanged"] for v in after), "manifest_files": len(json.loads((PACKET / "MANIFEST.json").read_text())["files"])}))
