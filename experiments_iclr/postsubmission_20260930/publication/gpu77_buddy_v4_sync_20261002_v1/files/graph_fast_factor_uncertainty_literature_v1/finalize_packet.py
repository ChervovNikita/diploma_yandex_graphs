"""Build documentation and custody records using only standard-library file operations.

This script never imports or executes author source, models, scientific code or data.
Run once before sealing; it refuses to overwrite an existing seal.
"""
from pathlib import Path
import datetime
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PACKET = Path(__file__).resolve().parent
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(name, value):
    (PACKET / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def bind(path, inspection):
    p = ROOT / path
    body = p.read_bytes()
    return {"path": path, "bytes": len(body), "sha256": digest(body), "inspection": inspection}


if (PACKET / "SEAL.json").exists():
    raise RuntimeError("Existing sealed packet must be preserved.")

index_path = "literature_memory/index_v16/LITERATURE_INDEX.json"
index = json.loads((ROOT / index_path).read_text())
assert digest((ROOT / index_path).read_bytes()) == "abce7373be0605391b91623b9917d3ee2b44603b2e9b3d482139850d015e68a4"
old = ROOT / "graph_initializer_cache_literature_followup_v1"
assert digest((old / "MANIFEST.json").read_bytes()) == "28023b88fc66f2ee7ff14ed10c0496acc2a6220490e40b836386fd04798b7c99"
old_manifest = json.loads((old / "MANIFEST.json").read_text())
for r in old_manifest["files"]:
    p = old / r["path"]
    assert p.stat().st_size == r["bytes"] and digest(p.read_bytes()) == r["sha256"]

inputs = [
    (index_path, "saved index consulted first in carried working state; hash reverified"),
    ("literature_memory/index_v16/README.md", "saved index context"),
    ("graph_ensemble_gap_skeptic_v1/efficient_controls_v1/REPORT.md", "saved Rank-1 sections and comparator qualifications reused"),
    ("graph_ensemble_gap_skeptic_v1/efficient_controls_v1/INSPECTED_PASSAGES.json", "custody binding for saved scoped primary passages; no new primary reread"),
    ("continuous_graph_efficiency_gap_v1/round04_graph_adaptation/REPORT.md", "saved GVBLL table and protocol/cost qualifiers reused"),
    ("continuous_graph_efficiency_gap_v1/round04_graph_adaptation/SOURCE_MAP.json", "saved exact GVBLL scope retained"),
    ("closest_graph_ensemble_sources_v1/REPORT.md", "saved shallow/Laplace graph ensemble table and source limitations reused"),
    ("new_learning_gap_independent_v1/literature_gap_round9_equivariant_v1/REPORT.md", "saved BLoB table and conditional uncertainty qualifications reused"),
    ("literature_root_followup_20261002_v1/graph_uncertainty_collapse_v1/PAPER_CONCLUSIONS.json", "indexed conclusion reused byte-for-byte"),
    ("graph_initializer_cache_literature_followup_v1/PAPER_CONCLUSIONS.json", "saved He conclusion reused byte-for-byte"),
    ("graph_initializer_cache_literature_followup_v1/READ_SCOPES.json", "saved He scope binding; no new primary reread"),
    ("graph_initializer_cache_literature_followup_v1/MANIFEST.json", "sealed predecessor hash and complete payload verification"),
    ("graph_initializer_cache_literature_followup_v1/SEAL.json", "sealed predecessor integrity"),
]
bindings = [bind(p, s) for p, s in inputs]
write("INPUT_BINDINGS.json", {
    "schema": "graph-fast-factor-uncertainty-inputs-v1", "created_utc": NOW,
    "consulted_saved_index_first": True,
    "index_accounting": index["read_accounting"],
    "index_counts_are_full_read_counts": False,
    "bindings": bindings,
})

passages = []
scopes = []


def html_scope(key, stem, ranges, limits):
    source = PACKET / "primary" / (stem + ".html")
    parsed = PACKET / "primary" / (stem + "_blocks.json")
    blocks = json.loads(parsed.read_text())
    ids = sorted({i for lo, hi in ranges for i in range(lo, hi + 1)})
    for i in ids:
        b = blocks[i]
        passages.append({
            "key": key, "kind": "primary_html_block",
            "source": str(source.relative_to(PACKET)),
            "source_sha256": digest(source.read_bytes()),
            "parsed_source": str(parsed.relative_to(PACKET)),
            "zero_based_block": i, "html_id": b["id"],
            "text": b["text"], "text_sha256": digest(b["text"].encode()),
        })
    scopes.append({
        "key": key, "read_status": "new_primary_scoped_methods_not_full_paper",
        "source": str(source.relative_to(PACKET)), "source_sha256": digest(source.read_bytes()),
        "zero_based_html_block_ranges_inclusive": ranges,
        "section_scope": limits,
        "full_paper_read": False,
    })


html_scope("matern", "matern_v4",
           [(0, 1), (3, 4), (11, 11), (13, 119), (129, 130), (139, 139), (148, 154)],
           "Sections 2–3 methods; Section 4.2 classification paragraphs; Appendix A Cora recipe; erratum and title/abstract/incidental introduction. No complete results, figure or proof audit.")
html_scope("laplora", "laplora_v5",
           [(0, 3), (16, 55), (58, 59), (70, 70), (77, 85), (101, 101), (111, 142), (168, 168)],
           "Sections 3–4; setup/comparators and cost prose; Appendix A.1 MC probability averaging; Appendix E low-rank precision/Woodbury; Appendix F initial prior-selection paragraph; Appendix C recipe prose and code links. No complete tables/figure/proof audit.")

pdf = PACKET / "primary/grf_v2.pdf"
pages = json.loads((PACKET / "primary/grf_v2_pages.json").read_text())
pdf_scope = {
    "key": "grf", "read_status": "new_primary_pdf_scoped_methods_not_full_paper",
    "source": "primary/grf_v2.pdf", "source_sha256": digest(pdf.read_bytes()),
    "one_based_PDF_pages": [
        {"pages": [1, 2, 3, 4, 5, 6], "scope": "main setup, feature estimator, Algorithm 1, assumptions, CG/pathwise method; no complete appendix proof audit"},
        {"pages": [9], "scope": "Section 4.4 classification limitation; incidental page conclusion text"},
        {"pages": [10], "scope": "Section 6 reproducibility; incidental funding/reference text not method evidence"},
        {"pages": [22, 23], "scope": "Appendix C.7 graph-only dataset, SVGP model and classification recipe/Table 7; Algorithm 3 on page 22 incidental, not audited"},
    ],
    "visual_inspection": [
        {"pages": [5, 6], "images": ["primary/grf_method-05.png", "primary/grf_method-06.png"], "scope": "previously inspected method formulas/layout"},
        {"pages": [22, 23], "images": ["primary/grf_classification-22.png", "primary/grf_classification-23.png"], "scope": "classification formulas/recipe/table and page layout; no plot-data reconstruction"},
    ],
    "unread_scope": ["complete remaining experiments", "Appendix A/B proof audit", "other figure pixels", "author implementation"],
    "full_paper_read": False,
}
scopes.append(pdf_scope)
for n in [1, 2, 3, 4, 5, 6, 9, 10, 22, 23]:
    text = pages[n - 1]["text"]
    assert pages[n - 1]["PDF_page"] == n
    if n == 9:
        text = text[text.index("4.4  Future work:"):text.index("5   Conclusion")]
    elif n == 10:
        text = text[text.index("Reproducibility."):text.index("7   Funding")]
    elif n == 22:
        text = text[text.index("C.7"):]
    passages.append({
        "key": "grf", "kind": "primary_pdf_text",
        "source": "primary/grf_v2.pdf", "source_sha256": digest(pdf.read_bytes()),
        "parsed_source": "primary/grf_v2_pages.json", "one_based_PDF_page": n,
        "text": text, "text_sha256": digest(text.encode()),
    })

source_files = [
    ("matern", "matern_README.md", 1, 64),
    ("matern", "matern_graph_matern_kernels_graph_matern_kernel.py", 1, 111),
    ("matern", "matern_graph_matern_svgp.py", 1, 41),
    ("matern", "matern_setup.py", 1, 33),
    ("laplora", "laplora_README.md", 1, 148),
    ("laplora", "laplora_pyproject.toml", 1, 96),
    ("laplora", "laplora_bayesian_lora_main.py", 1, 312),
]
source_scopes = []
for key, name, lo, hi in source_files:
    p = PACKET / "author_source" / name
    lines = p.read_text().splitlines(keepends=True)
    assert len(lines) == hi
    text = "".join(lines[lo - 1:hi])
    rec = {
        "key": key, "kind": "pinned_author_source_text",
        "source": str(p.relative_to(PACKET)), "source_sha256": digest(p.read_bytes()),
        "one_based_line_range_inclusive": [lo, hi],
        "text": text, "text_sha256": digest(text.encode()),
        "imported_or_executed": False,
    }
    passages.append(rec)
    source_scopes.append({k: v for k, v in rec.items() if k not in ("text", "text_sha256")})
write("READ_SCOPES.json", {
    "schema": "graph-fast-factor-uncertainty-read-scopes-v1",
    "primary_scopes": scopes, "author_source_scopes": source_scopes,
    "retrieval_metadata_does_not_count_as_method_read": True,
    "new_scoped_primary_method_reads": 3, "new_full_primary_reads": 0,
    "new_primary_paper_limit": 3, "scientific_execution": False,
})
write("INSPECTED_PASSAGES.json", {
    "schema": "graph-fast-factor-uncertainty-passages-v1",
    "passage_locators": "HTML indices are zero-based; PDF pages and source lines are one-based; source HTML IDs retained where available.",
    "passages": passages,
})

paper_records = [
    {
        "key": "matern", "canonical_id": "arXiv:2010.15538v4",
        "verified_title": "Matérn Gaussian Processes on Graphs",
        "authors": ["Viacheslav Borovitskiy", "Iskander Azangulov", "Alexander Terenin", "Peter Mostowsky", "Marc Peter Deisenroth", "Nicolas Durrande"],
        "version_date": "2026-05-15", "original_method_venue": "AISTATS 2021",
        "primary_url": "https://arxiv.org/html/2010.15538v4",
        "read_status": "new_primary_scoped_methods_plus_pinned_author_source_not_full_paper",
        "exact_read_scope": "READ_SCOPES.json:matern",
        "saved_takeaway": "Graph Matérn/diffusion covariance, sparse integer-order GMRF precision, Bayesian finite Fourier features and categorical VI are established. Finite truncation can cause variance starvation.",
        "relationship": "Graph energy in a fast-factor Jacobian is a restricted graph Gaussian prior, not a newly supplied covariance principle or selective-risk guarantee.",
        "official_repository": "https://github.com/spbu-math-cs/Graph-Gaussian-Processes",
        "official_commit": "781be70c8913b96c365475889f1f39629a5c9d0f",
        "commit_date": "2025-11-17T09:57:28Z",
        "source_findings": ["README marks repository deprecated and points to GeometricKernels.", "Kernel normalizes spectral weights to an average-variance scale and can multiply a point-feature kernel.", "Source read only; no notebooks/data or runtime."],
        "limits": ["Graph-only Cora LCC, 140 labels, 500 eigenpairs; categorical robust-max SVGP recipe.", "No finite BE study or selective-risk bound established.", "v4 erratum corrects regression RMSE labels/scaling; version date does not make the method new."],
        "disposition": "Existing graph-prior/finite-basis construction; no candidate promotion.",
    },
    {
        "key": "laplora", "canonical_id": "arXiv:2308.13111v5",
        "verified_title": "Bayesian Low-rank Adaptation for Large Language Models",
        "authors": ["Adam X. Yang", "Maxime Robeyns", "Xi Wang", "Laurence Aitchison"],
        "version_date": "2024-02-05", "primary_url": "https://arxiv.org/html/2308.13111v5",
        "read_status": "new_primary_scoped_methods_plus_pinned_author_source_not_full_paper",
        "exact_read_scope": "READ_SCOPES.json:laplora",
        "saved_takeaway": "Frozen-base adapter Laplace, low-rank KFAC, linearized output covariance and prior tuning by model evidence precede this candidate. Correct predictive MC averages probabilities.",
        "relationship": "Direct compact posterior approximation prior; applying it to graph factors is an adaptation with a graph prior, not a supported distinct uncertainty principle.",
        "official_repository": "https://github.com/MaximeRobeyns/bayesian_lora",
        "official_commit": "4e99db8df6860eaf0e4bc1a168fb3faa45a094f0",
        "commit_date": "2024-06-22T09:04:57Z",
        "source_findings": ["main.py 234–312 returns per-example class covariance, not cross-example/node blocks.", "README example averages sampled logits; paper main paragraph/Appendix A.1 specifies probability averaging.", "Default output helpers reference external configuration assumptions; no runtime qualification."],
        "limits": ["Language tasks, not graph/selective-risk evidence.", "Adapter-only uncertainty conditions on a fitted frozen base.", "Low-rank curvature necessary for cost; compact factor count alone insufficient.", "Use negative-log-posterior precision convention; printed main sign/Fisher notation not an unqualified implementation recipe."],
        "disposition": "Direct prior and bounded reusable source; no candidate promotion.",
    },
    {
        "key": "grf", "canonical_id": "arXiv:2509.03691v2",
        "verified_title": "Graph Random Features for Scalable Gaussian Processes",
        "authors": ["Matthew Zhang", "Jihao Andreas Lin", "Krzysztof Choromanski", "Adrian Weller", "Richard E. Turner", "Isaac Reid"],
        "version_date": "2025-09-25", "observed_status": "Preprint. Under review.",
        "primary_url": "https://arxiv.org/pdf/2509.03691v2",
        "read_status": "new_primary_pdf_scoped_methods_not_full_paper",
        "exact_read_scope": "READ_SCOPES.json:grf",
        "saved_takeaway": "Random graph features already support coherent graph-wide function draws and pathwise Gaussian conditioning with implicit solves. Complete scalable classification treatment explicitly deferred.",
        "relationship": "Recent graph covariance/sampling baseline; classification recipe does not establish uncertainty/selective-risk utility for this finite factor construction.",
        "official_repository_verified": False,
        "source_findings": ["PDF page 10 promises code after double-blind review; no official implementation verified in this bounded pass."],
        "limits": ["O(N^1.5) inference statement is conditional on feature/noise/solver assumptions and does not apply to the categorical example.", "Cora LCC graph-only, 80/20, SVGP/softmax, 1000-iteration cap, five seeds.", "16384 walkers/node and 22.17% kernel nonzeros in reported Cora setting.", "No selective-risk or current-host qualification; HTML endpoint returned 404."],
        "disposition": "Recent graph baseline with explicit classification limits; no candidate promotion.",
    },
]
for r in paper_records:
    r["primary_sha256"] = next(s["source_sha256"] for s in scopes if s["key"] == r["key"])
write("PAPER_CONCLUSIONS.json", {
    "schema": "graph-fast-factor-uncertainty-primary-conclusions-v1",
    "records": paper_records,
    "read_accounting": {"new_primary_paper_limit": 3, "genuinely_new_primary_papers_retrieved": 3,
                        "new_scoped_primary_method_reads": 3, "new_full_primary_reads": 0,
                        "retained_primary_reread_increment": 0,
                        "saved_conclusions_are_not_new_primary_reads": True},
    "overall_conclusion": "No-go for distinct learner promotion; no pilot. Established components give a conditional factor-basis graph Gaussian approximation, but no finite-model calibration/selective-risk guarantee or measured useful-cost gain.",
})

legacy_heuristic_records = [
    {
        "canonical_id": "arXiv:2005.07186v2", "verified_title": "Rank-1 Bayesian Neural Networks",
        "source": "graph_ensemble_gap_skeptic_v1/efficient_controls_v1/REPORT.md",
        "inherited_scope": "Sections 3.1–3.4 and Appendices B/C, as explicitly recorded in saved report/passages.",
        "inherited_read_status": "saved scoped primary/source inspection; no whole-paper status inferred",
        "saved_takeaway": "Distributions over r/s with point-estimated W, expected likelihood + factor KL + W prior; initialization/KL schedule and coherent K-component samples. All sampled GNN trajectories must be charged.",
        "limits": ["Image native recipe; graph port requires a declared VI recipe.", "Noise on deterministic factors alone is not a reproduction."],
    },
    {
        "canonical_id": "arXiv:2609.13655v1",
        "verified_title": "Online Bayesian Node Classification on Inductive Graphs under Distribution Shift",
        "source": "continuous_graph_efficiency_gap_v1/round04_graph_adaptation/REPORT.md",
        "inherited_scope": next(r["scope"] for r in json.loads((ROOT / "continuous_graph_efficiency_gap_v1/round04_graph_adaptation/SOURCE_MAP.json").read_text())["primary_sources"] if r["id"] == "GVBLL"),
        "inherited_read_status": "saved scoped primary method, not promoted to full read",
        "saved_takeaway": "Deterministic encoder, variational Bayesian final layer; frozen online encoder, diagonal Laplace/power prior/anchor; predict before receiving later labels.",
        "limits": ["MAP for accuracy/NLL and MC for uncertainty differ.", "Head-update cost excludes encoder; not unlabeled TTA."],
    },
    {
        "canonical_id": "arXiv:2602.15747v1", "verified_title": "How to Train a Shallow Ensemble",
        "source": "closest_graph_ensemble_sources_v1/REPORT.md",
        "inherited_scope": "Sections II/IV common feature backbone, final-layer Gaussian-NLL, Laplace/rigidity committees and fine-tuning; saved table also states full text inspected.",
        "inherited_read_status": "saved report says full text inspected; no new read or independent recertification",
        "saved_takeaway": "Graph/atomistic final-layer Laplace/rigidity committees are prior.",
        "limits": ["Energy/force setting, not finite node-classification selective-risk evidence.", "Saved paper-linked author repository requests returned 404; preserve unresolved access."],
    },
    {
        "canonical_id": "arXiv:2406.11675v5", "verified_title": "BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models",
        "source": "new_learning_gap_independent_v1/literature_gap_round9_equivariant_v1/REPORT.md",
        "inherited_scope": "Section 3.1 and Appendix A.3 as saved: fixed W0, deterministic B, Gaussian A, variational free energy; multiplicative Rademacher flipout masks.",
        "inherited_read_status": "saved scoped methods; no full-paper status inferred",
        "saved_takeaway": "Frozen-base Bayesian low-rank adapters and correlated full-weight perturbations precede the candidate.",
        "limits": ["Language task; no graph calibration or selective-risk guarantee inherited."],
    },
]
for r in legacy_heuristic_records:
    r["source_sha256"] = digest((ROOT / r["source"]).read_bytes())
    r["current_pass_status"] = "saved_conclusion_reuse_no_primary_reread"
collapse = [r for r in index["paper_records"] if r.get("canonical_id") == "arXiv:2605.22593v1"]
assert len(collapse) == 1
assert digest((ROOT / collapse[0]["conclusion_file"]).read_bytes()) == collapse[0]["conclusion_file_sha256"]
he = next(r for r in json.loads((old / "PAPER_CONCLUSIONS.json").read_text())["records"] if r["canonical_id"] == "arXiv:2007.05864v2")
write("REUSED_CONCLUSIONS.json", {
    "schema": "graph-fast-factor-uncertainty-reused-conclusions-v1",
    "policy": "Preserve saved read scopes/status/limitations; no source reacquisition and zero new primary read credit. Index counts are not full-paper read counts.",
    "saved_report_records": legacy_heuristic_records,
    "index_records_preserved_verbatim": collapse,
    "predecessor_records_preserved_verbatim": [he],
    "predecessor_record_source": "graph_initializer_cache_literature_followup_v1/PAPER_CONCLUSIONS.json",
    "new_primary_read_increment": 0,
})

discovery = []
for p in sorted((PACKET / "discovery").glob("*.json")):
    obj = json.loads(p.read_text())
    for r in obj.get("results", []):
        discovery.append({
            "source": str(p.relative_to(PACKET)), "source_sha256": digest(p.read_bytes()),
            "id": r.get("id"), "title": r.get("title"), "date": r.get("publication_date"),
            "doi": r.get("doi"),
            "disposition": "metadata only; no new primary method read; relevance/claims not certified",
        })
ns = {"a": "http://www.w3.org/2005/Atom"}
for p in sorted((PACKET / "discovery").glob("*.xml")):
    for e in ET.fromstring(p.read_text()).findall("a:entry", ns):
        ident = e.findtext("a:id", namespaces=ns)
        disp = "metadata-only title lead; not assessed under three-new-primary cap"
        if "2509.03691" in ident:
            disp = "promoted as new scoped primary; see PAPER_CONCLUSIONS.json:grf"
        elif "2609.13655" in ident:
            disp = "saved GVBLL scope reused; no primary reacquisition/read credit"
        elif "2510.06181" in ident:
            disp = "unresolved risk/dependence lead; title only, no bound or relevance claim"
        elif "2603.17569" in ident:
            disp = "unresolved graph-transformer GP-limit lead; metadata only, no finite-model result inferred"
        discovery.append({
            "source": str(p.relative_to(PACKET)), "source_sha256": digest(p.read_bytes()),
            "id": ident, "title": e.findtext("a:title", namespaces=ns),
            "updated": e.findtext("a:updated", namespaces=ns), "disposition": disp,
        })
write("DISCOVERY_DISPOSITIONS.json", {
    "schema": "graph-fast-factor-uncertainty-discovery-v1",
    "scope": "Four bounded OpenAlex query returns and three bounded arXiv query returns; metadata/abstract endpoints are not full method evidence.",
    "selected_primary_ids": ["2010.15538v4", "2308.13111v5", "2509.03691v2"],
    "candidates": discovery, "exhaustive_search": False,
})

retrieval_checks = []
for logname in ["DISCOVERY_RETRIEVAL.json", "IDENTITY_RETRIEVAL.json", "PRIMARY_RETRIEVAL.json", "AUTHOR_RETRIEVAL.json"]:
    for r in json.loads((PACKET / logname).read_text()):
        if "path" not in r:
            retrieval_checks.append({"log": logname, "url": r["url"], "outcome": r.get("error", "no saved body")})
            continue
        p = ROOT.parent / r["path"]
        assert p.is_file(), str(p)
        assert p.stat().st_size == r["bytes"] and digest(p.read_bytes()) == r["sha256"], str(p)
        retrieval_checks.append({"log": logname, "path": str(p.relative_to(PACKET)), "sha256": r["sha256"], "verified": True})
for r in passages:
    p = PACKET / r["source"]
    assert digest(p.read_bytes()) == r["source_sha256"]
    assert digest(r["text"].encode()) == r["text_sha256"]
    if r["kind"] == "primary_html_block":
        assert json.loads((PACKET / r["parsed_source"]).read_text())[r["zero_based_block"]]["text"] == r["text"]
    elif r["kind"] == "primary_pdf_text":
        assert r["text"] in pages[r["one_based_PDF_page"] - 1]["text"]
    elif r["kind"] == "pinned_author_source_text":
        lo, hi = r["one_based_line_range_inclusive"]
        assert "".join(p.read_text().splitlines(keepends=True)[lo - 1:hi]) == r["text"]
for r in bindings:
    p = ROOT / r["path"]
    assert p.stat().st_size == r["bytes"] and digest(p.read_bytes()) == r["sha256"]
write("VERIFICATION.json", {
    "schema": "graph-fast-factor-uncertainty-verification-v1",
    "verified_at_utc": NOW, "retrieval_checks": retrieval_checks,
    "passage_count": len(passages), "all_passage_source_hashes_and_locators_verified": True,
    "all_input_hashes_unchanged": True, "sealed_predecessor_payload_files_verified": len(old_manifest["files"]),
    "json_parsed": True, "scientific_runtime_or_author_import": False,
    "new_scoped_primary_methods": 3, "new_full_primary_reads": 0,
    "no_go": True, "pilot_recommended": False,
})
