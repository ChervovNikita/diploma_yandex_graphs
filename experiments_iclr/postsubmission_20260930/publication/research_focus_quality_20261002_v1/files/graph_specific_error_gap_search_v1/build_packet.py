"""Seal a bounded literature-only packet; no scientific runtime work."""
from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
import re

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
BASE = ROOT / "postsubmission_research_20260930"


def h(data):
    return hashlib.sha256(data).hexdigest()


def dump(name, value):
    (P / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


expected = {
    "literature_memory/index_v19/LITERATURE_INDEX.json": "831d760fb3bbe12fb88203a51b251398fb50eb3263938bf117c604ac4a84d109",
    "graph_error_specialization_gap_search_v1/MANIFEST.json": "5ebb479bbce602bc28c2ff8f5c921c3bdf879e96dbb8c6b32f5b42b9b57c6588",
    "graph_error_specialization_gap_search_v1/PAPER_CONCLUSIONS.json": "9ca590c1e408cf9f539cd52a6252e61ded3e6fa6bc57f00e78388c55ca26a682",
    "graph_error_specialization_gap_search_v1/CANDIDATE_SPEC.json": "f7162ece09f14b73c08e7167c10b12af9e1e779ad4c41a33e18ac9a990434844",
    "graph_init_local_prediction_objective_review_v1/MANIFEST.json": "4a0b95253fa73bb06442a836e08b4650d267103dfc59a72144748b7dfb60376b",
    "graph_error_specialization_pulse_state_availability_v1/MANIFEST.json": "d258eaa813656c1ea10a8abd5ea665703378a6dfec4946db7f00fcc27c2fe1e7",
    "graph_error_specialization_pulse_state_availability_v1/STATE_AVAILABILITY.md": "6d3b1d7144b34db9c7eff431723d81c2aeb54c81e1f9b726b19d9041f881a821",
    "graph_error_specialization_pulse_state_availability_v1/ACQUISITION_AMENDMENT.json": "958419c7e8af27ab6f73cc8af6ab132d3df4f7f159fe7f7d867cd9c65f8935cc",
}
bindings = []
for rel, sha in expected.items():
    data = (BASE / rel).read_bytes()
    assert h(data) == sha, f"Bound prior changed: {rel}"
    bindings.append({"path": "postsubmission_research_20260930/" + rel, "bytes": len(data), "sha256": sha})

index = json.loads((BASE / "literature_memory/index_v19/LITERATURE_INDEX.json").read_text())
prior_gap = json.loads((BASE / "graph_error_specialization_gap_search_v1/PAPER_CONCLUSIONS.json").read_text())
prior_ids = {x["normalized_identifier"] for x in index["canonical_identifier_normalization"]["groups"]}
prior_ids.update(x["canonical_id"] for x in prior_gap["paper_records"])
new_ids = ["arxiv:1908.05081", "doi:10.18653/v1/2021.naacl-main.229", "arxiv:2005.11079", "arxiv:1909.07578"]
assert len(set(new_ids)) == 4 and not set(new_ids).intersection(prior_ids)
dump("PRIOR_LEDGER_CHECK.json", {
    "prior_normalized_identifiers": sorted(prior_ids),
    "new_primary_identifiers": new_ids,
    "no_prior_identifier_overlap": True,
    "normalization": "Prior index's exact groups plus SEA/sigma canonical IDs; new DOI case normalized; arXiv versions retained in source scope rather than used to evade deduplication.",
    "retained_primary_rereads": 0,
})
dump("INPUT_BINDINGS.json", {
    "schema": "graph-specific-error-gap-input-bindings-v1", "inputs": bindings,
    "prior_use": "Saved conclusions/identifiers and current sealed candidate/acquisition analysis only; no retained primary bodies reread.",
    "mutation_scope": "This separate new packet only; parent owns index/status/adoption.",
    "scientific_execution_authorized": False,
})

keys = ["PreGS", "MORGAN", "FAGEL", "Local Learning with Boosting", "Bayesian Deep Ensembles", "Input-gradient", "Matérn", "Graph Neural Networks for Link Prediction with Subgraph Sketching", "Mixture of Link Predictors"]
reuse = []
for row in index["paper_records"]:
    title = row["conclusion"].get("verified_title", row["conclusion"].get("title", ""))
    if any(key.lower() in title.lower() for key in keys):
        reuse.append(row)
dump("REUSED_CONCLUSIONS.json", {"index_v19_records": reuse, "SEA_sigma_records": prior_gap["paper_records"], "primary_rereads": 0, "scope": "Conclusion reuse only; duplicate index notes retained without reinterpreting them as new papers."})


class Meta(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "meta" and d.get("name", "").startswith("citation_"):
            self.meta.append(d)


metas = {}
for name in ["adagcn_abs.html", "trees_landing.html", "grand_abs.html", "stacking_abs.html"]:
    m = Meta()
    m.feed((P / "primary" / name).read_text())
    metas[name] = m.meta
dump("VERIFIED_PRIMARY_METADATA.json", {"citation_metadata": metas, "limits": "Metadata proves the identified landing-page identity; GRAND read is its exact NeurIPS2020 PDF, not a claim to read the latest arXiv version."})
assert (P / "primary/adagcn_latest.pdf").read_bytes() == (P / "primary/adagcn_v3.pdf").read_bytes()
assert (P / "primary/stacking_latest.pdf").read_bytes() == (P / "primary/stacking_v1.pdf").read_bytes()

methods = [
    {
        "canonical_id": new_ids[0], "versioned_id": "1908.05081v3", "verified_title": "AdaGCN: Adaboosting Graph Convolutional Networks into Deep Models",
        "publication": "ICLR 2021; source header verifies conference publication",
        "primary_url": "https://arxiv.org/pdf/1908.05081v3", "source_file": "primary/adagcn_v3.pdf", "extraction_file": "primary/adagcn_latest_pages.json", "pages_read": [3, 4, 5, 6],
        "exact_read_scope": "PDF pages3-6: nonlinear hop classifiers, error/node-weight updates, warm-start difference from simultaneous sharing, limited propositions as stated (proofs not read), Section3 and complete SAMME.R Algorithm1. Page headers1-2 and incidental setup/table context included in locator accounting.",
        "saved_takeaway": "Cached normalized-adjacency powers feed sequential nonlinear classifiers; previous classifier parameters initialize the next; updated TRAIN-node weighted loss and centered log-probability SAMME.R combination guide later specialization. Successive classifier parameters differ despite common architecture.",
        "equivalence_risk": "Graph-aware error specialization, warm transfer, hop diversity and propagation outside repeated training are prior. A hop/reweight/warm-start rewrite is insufficient as a new mechanism.",
        "scoped_delta": "Inspected method does not construct graph-filtered pooled CE cotangent VJPs on existing simultaneously shared/private routes or the proposed finite joint-projected pulse.",
        "decisive_next_step": "Keep topology-attribution and full-acquisition/cost falsifiers; no AdaGCN control execution or recipe grid admitted.",
        "visual_pages": [6],
    },
    {
        "canonical_id": new_ids[1], "versioned_id": "NAACL2021 official published PDF", "verified_title": "Graph Ensemble Learning over Multiple Dependency Trees for Aspect-level Sentiment Classification",
        "publication": "NAACL-HLT 2021; official ACL metadata DOI and pages2884-2894", "doi": "10.18653/v1/2021.naacl-main.229",
        "primary_url": "https://aclanthology.org/2021.naacl-main.229.pdf", "source_file": "primary/trees.pdf", "extraction_file": "primary/trees_pages.json", "pages_read": [2, 3, 4],
        "exact_read_scope": "PDF pages2-4: GraphMerge rationale/context and Sections3.1-3.4 edge union, typed RGAT, BERT/position input and ordinary CE classifier; Figure2 pixels and page4 typed-attention equation. Page1/5 headers and incidental table/setup context not full-paper/result certification.",
        "saved_takeaway": "GraphMerge unions multiple parser edge sets over the same words and trains one typed relational-attention model on BERT features with ordinary CE; it does not maintain complementary learned member predictions.",
        "equivalence_risk": "Compact exploitation of multiple topology hypotheses is prior; graph-input union is a different object from ensemble error diversity. Parsing/union edge work must be accounted separately.",
        "scoped_delta": "No member-private graph-error cotangent/Jacobian pulse or ensemble complementarity loss in inspected Sections3.1-3.4.",
        "decisive_next_step": "Do not use title-level graph-ensemble similarity as exact-method equivalence or inherit its cost claim; retain topology-vs-generic control.",
        "visual_pages": [3, 4],
    },
    {
        "canonical_id": new_ids[2], "versioned_id": "NeurIPS2020 official published PDF", "verified_title": "Graph Random Neural Networks for Semi-Supervised Learning on Graphs",
        "arxiv_landing_title": "Graph Random Neural Network for Semi-Supervised Learning on Graphs", "publication": "NeurIPS 2020", "doi": None,
        "primary_url": "https://proceedings.neurips.cc/paper/2020/file/fb4c835feb0a65cc39739320d7a51c02-Paper.pdf", "source_file": "primary/grand.pdf", "extraction_file": "primary/grand_pages.json", "pages_read": [3, 4, 5],
        "exact_read_scope": "PDF pages3-5: Sections3.1-3.2, DropNode/mixed-order propagation, one shared MLP, mean supervised CE and sharpened-center squared consistency, Algorithm1, inference/complexity and explicit homophily limitation. Page1/2/6 header previews not full theory/result reads.",
        "saved_takeaway": "Stochastic feature masks plus fixed mixed-order graph propagation create views for one shared MLP; supervised mean CE plus sharpened-average agreement trains it. Inference is one deterministic propagated-feature prediction, not a learned member ensemble.",
        "equivalence_risk": "Topology-aware views with shared parameters and prediction regularization are prior. Agreement training differs from complementary route errors; homophily-based behavior and runtime do not transfer.",
        "scoped_delta": "No restricted member-private parameter initialization, graph-filtered error VJP or finite paired pulse in inspected method.",
        "decisive_next_step": "If the proposal becomes graph-view augmentation plus agreement, attribute this mechanism; no homophily transfer or calibration claim.",
        "visual_pages": [4, 5],
    },
    {
        "canonical_id": new_ids[3], "versioned_id": "1909.07578v1 plus published main text", "verified_title": "Stacking models for nearly optimal link prediction in complex networks",
        "publication": "PNAS 2020", "doi": "10.1073/pnas.1914950117",
        "primary_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7519231/", "source_file": "primary/stacking_pmc.html", "extraction_file": "primary/stacking_pmc_blocks.json", "inclusive_block_ranges": [[29, 35]],
        "supplementary_source": {"url": "https://arxiv.org/pdf/1909.07578v1", "file": "primary/stacking_v1.pdf", "extraction_file": "primary/stacking_latest_pages.json", "pages_read": [8, 9, 15]},
        "exact_read_scope": "Published main Methods and Materials zero-based blocks29-35; arXivv1 SI pages8-9 predictors/stacking/meta-training and page15 missingness/negative-example method with incidental neighboring context. Mechanical full-document locator snippets separately preserved, not a full-paper read. Published SI attempt was non-PDF and not inspected.",
        "saved_takeaway": "Topological/model/embedding scores become features for a supervised stacked forest; distinct predictor errors motivate mixing. The accessible SI describes TRAIN edge removal, fixed score inputs, fivefold meta-model selection and observed nonedges as noisy negatives.",
        "equivalence_risk": "Complementary graph predictors and trained combinations of fixed graph scores are prior; cache sharing is available to independent predictors too. A new learned aggregator changes the fixed-pool estimand and selection budget.",
        "scoped_delta": "No compact shared/private route tangent intervention or graph-filtered TRAIN error pulse in inspected stacking recipe.",
        "decisive_next_step": "Same-cache independent-predictor comparator and declared query-space error operator are necessary; do not copy missingness assumptions or add pooling/gate grids.",
        "visual_pages": [9],
        "version_limit": "Published main and arXiv SI scopes distinguished; no claim of exact final-SI recipe identity.",
    },
]
for row in methods:
    row["read_status"] = "new_scoped_primary_method_read_not_full_paper"
    row["full_read"] = False
    row["author_source_read_or_execution"] = False
    row["source_sha256"] = h((P / row["source_file"]).read_bytes())
    row["extraction_sha256"] = h((P / row["extraction_file"]).read_bytes())
    if "supplementary_source" in row:
        s = row["supplementary_source"]
        s["source_sha256"] = h((P / s["file"]).read_bytes())
        s["extraction_sha256"] = h((P / s["extraction_file"]).read_bytes())
    row["not_claimed_read"] = ["Full paper/proofs/results/references audit", "Native author implementation", "Runtime/memory/cost qualification"]
dump("READ_SCOPES.json", {
    "schema": "graph-specific-error-gap-read-scopes-v1", "new_primary_limit": 4,
    "new_scoped_primary_method_reads": 4, "new_full_primary_reads": 0, "retained_primary_rereads": 0,
    "scope_convention": "PDF page numbers one-based inclusive; saved HTML blocks zero-based inclusive. Downloads/extractions of whole sources are not full-read certification.",
    "papers": methods,
    "additional_header_context": {
        "adagcn": {"pages": [1, 2], "first_extracted_lines": 8}, "trees": {"pages": [1, 5], "first_extracted_lines": 8},
        "grand": {"pages": [1, 2, 6], "first_extracted_lines": 6}, "stacking_arxiv": {"pages": [1, 2, 3, 4, 5, 6], "first_extracted_lines": 6},
    },
    "locator_limit": "Mechanical locator previews include incidental caption/result words; displayed output was truncated. These are preserved as generated locator context, not fully inspected passages or performance evidence.",
})
dump("PAPER_CONCLUSIONS.json", {"schema": "graph-specific-error-gap-primary-conclusions-v1", "paper_records": methods, "read_accounting": {"new_scoped_primary_method_reads": 4, "new_full_primary_reads": 0, "retained_primary_rereads": 0, "author_source_reads_or_execution": 0, "cumulative_project_read_totals_certified": False}, "decision": "Only conditional finite graph-error mechanism question survives; broad ingredient/cache claims removed; no execution or persistent schedule admission."})

passages = []


def pdf_passage(paper, page, begin, end, claim, supplement=False):
    m = methods[paper]
    source = m["supplementary_source"] if supplement else m
    extraction = source["extraction_file"]
    text = json.loads((P / extraction).read_text())["pages"][page - 1]["text"]
    a = text.index(begin)
    b = text.index(end, a) + len(end)
    quote = text[a:b]
    passages.append({"canonical_id": m["canonical_id"], "source_file": source.get("file", source.get("source_file")), "extraction_file": extraction, "page_1based": page, "character_start": a, "character_end_exclusive": b, "exact_text": quote, "text_sha256": h(quote.encode()), "supports": claim})


pdf_passage(0, 4, "One crucial point", "just similar to a recurrent neural network.", "One recursively fitted classifier architecture; previous parameters initialize later hop learners.")
pdf_passage(0, 5, "In practice, we employ SAMME.R", "rather than predicted hard labels in SAMME, in the", "Implemented confidence-rated SAMME.R differs from preceding SAMME introduction.")
pdf_passage(0, 6, "Due to the fact that dense tensor", "whiletrainingtheneuralnetwork.", "Graph features computed in advance; no sparse propagation inside repeated classifier passes.")
pdf_passage(1, 3, "takes the union of the edges from all parses, and", "takes the union of the edges from all parses, and", "Graph-input edge union, not member prediction-diversity loss; full formula and paragraph checked in page3 render.")
pdf_passage(1, 4, "we ﬁrst add reciprocal edges for each edge in the", "we ﬁrst add reciprocal edges for each edge in the", "Reciprocal-edge construction; parent/child/self details checked in page4 render.")
pdf_passage(1, 4, "minimize the standard cross entropy loss function,", "minimize the standard cross entropy loss function,", "Ordinary CE; classifier and weight decay checked in page4 render.")
pdf_passage(2, 4, "Notethatthesamplingprocedure", "set˜X as the original feature matrixX.", "Training stochasticity and deterministic original-feature inference.")
pdf_passage(2, 4, "After performing random propagation", "corresponding outputs:", "Multiple graph-feature augmentations feed a shared MLP.")
pdf_passage(2, 5, "we ﬁrst calculate the label distribution", "average distributions.", "Consistency target is sharpened mean prediction rather than complementary error.")
pdf_passage(2, 5, "Limitations.", "GRAND may not succeed on graphs with less homophily.", "Explicit homophily limitation.")
blocks = json.loads((P / "primary/stacking_pmc_blocks.json").read_text())["blocks"]
for idx, begin, end, claim in [
    (31, "In total, we consider 203", "missing link “predictor.”", "Graph-derived features and predictor scores share the supervised input boundary."),
    (32, "In particular, stacked generalization", "made by individual predictors.", "Supervised aggregation learns the error/predictor relationship."),
    (32, "but only if those predictors", "signals they exploit ( 27 ).", "Distinct errors and sufficiently different signals motivate combining predictors."),
]:
    text = blocks[idx]["text"]
    a = text.index(begin)
    b = text.index(end, a) + len(end)
    quote = text[a:b]
    passages.append({"canonical_id": new_ids[3], "source_file": "primary/stacking_pmc.html", "extraction_file": "primary/stacking_pmc_blocks.json", "block_0based": idx, "character_start": a, "character_end_exclusive": b, "exact_text": quote, "text_sha256": h(quote.encode()), "supports": claim})
pdf_passage(3, 9, "supervisedrandomforestalgorithmforthemeta-learningstep", "supervisedrandomforestalgorithmforthemeta-learningstep", "Accessible arXiv SI uses a forest meta-model; input score and TRAIN split details checked in page9 render.", supplement=True)
dump("INSPECTED_PASSAGES.json", {"schema": "graph-specific-error-gap-short-passages-v1", "passages": passages, "scope_limit": "Short exact passages support method conclusions; read boundaries are wider and explicitly listed in READ_SCOPES.json. Extraction has spacing/rotated-text limitations; selected renders checked."})

locator = []
for row in json.loads((P / "primary/stacking_latest_pages.json").read_text())["pages"]:
    if row["page_1based"] < 8:
        continue
    lines = row["text"].splitlines()
    hits = [i for i, line in enumerate(lines) if re.search(r"random forest|training data|stacked|cross.validation|Section D|D\. |D Training|SVM", line, re.I)]
    for i in hits[:5]:
        locator.append({"page_1based": row["page_1based"], "match_line_1based": i + 1, "preview_line_start_1based": max(0, i - 1) + 1, "preview_line_end_1based": min(len(lines), i + 2), "text": " ".join(lines[max(0, i - 1):i + 2])})
dump("LOCATOR_CONTEXT.json", {"source": "primary/stacking_v1.pdf", "generated_previews": locator, "read_status": "Only locator context; original tool output truncated, not all generated previews certified inspected; no results audit or performance conclusion."})

prior_titles = {row["conclusion"].get("verified_title", row["conclusion"].get("title", "")).casefold() for row in index["paper_records"]}
prior_titles.update(row["verified_title"].casefold() for row in prior_gap["paper_records"])
new_title_map = {row["verified_title"].casefold(): row["canonical_id"] for row in methods}
new_title_map[methods[2]["arxiv_landing_title"].casefold()] = new_ids[2]
works = {}
for file in sorted((P / "discovery").glob("*.json")):
    v = json.loads(file.read_text())
    for row in v.get("results", []):
        key = row.get("doi") or row["id"]
        if key not in works:
            title = row["display_name"]
            if title.casefold() in new_title_map:
                status = "NEW_SCOPED_PRIMARY_METHOD_READ"
                reason = "Read only the exact scopes in READ_SCOPES.json; not a full read."
            elif title.casefold() in prior_titles:
                status = "REUSED_PRIOR_LEDGER_NO_PRIMARY_REREAD"
                reason = "Already indexed conclusion; retained primary excluded."
            elif any(x in title.casefold() for x in ["graph ensemble neural network", "boosting-gnn", "graph ensemble boosting", "an ensemble approach to link prediction", "boosting graph neural", "spatial-temporal graph boosting"]):
                status = "UNREAD_ADJACENT_LEAD_NOT_QUALIFIED"
                reason = "Metadata-only lead outside the bounded four-method pass; not evidence for exact equivalence or absence."
            else:
                status = "METADATA_ONLY_OUTSIDE_BOUNDED_METHOD_SCOPE"
                reason = "Not selected for this four-source mechanism review; no primary method conclusion."
            works[key] = {"discovery_identifier": key, "title": title, "publication_year": row.get("publication_year"), "doi": row.get("doi"), "disposition": status, "reason": reason, "occurrences": []}
        works[key]["occurrences"].append(str(file.relative_to(P)))
failures = []
for file in sorted(P.glob("*REQUESTS_run*.json")) + sorted(P.glob("PRIMARY_RETRIEVAL_run*.json")):
    v = json.loads(file.read_text())
    for row in v.get("requests", []):
        if "error" in row or row.get("primary_method_access"):
            failures.append({"record_file": file.name, "record": row})
dump("DISCOVERY_DISPOSITIONS.json", {"schema": "graph-specific-error-gap-discovery-dispositions-v1", "unique_metadata_works": len(works), "works": list(works.values()), "access_failures_and_limits": failures, "search_limit": "Metadata search is not exhaustive; four new primary methods only; unread leads and access failures cannot justify absence/novelty claims."})

dump("PILOT_KILL_CRITERIA.json", {
    "status": "ANALYSIS_ONLY_NO_ADMISSION_OR_PROTOCOL_MUTATION",
    "criteria": [
        {"id": "K1", "trigger": "Original topology fails frozen later-quality screen against both common-only and node-permuted controls", "decision": "Stop graph-specific promotion; no topology/pool grid", "basis": [new_ids[0], new_ids[1], new_ids[2], "sealed candidate"]},
        {"id": "K2", "trigger": "Implementation reduces to hop features, scalar boosting weights, merged inputs or agreement regularization; or link/query error operator and TRAIN lift remain undefined", "decision": "Attribute existing mechanism or stop graph-error extension before pilot", "basis": new_ids},
        {"id": "K3", "trigger": "Future qualification/joint rank/finite signed realization/paired TRAIN-CE guards fail", "decision": "Retain all failure reasons; do not reseed, retime or add refreshes", "basis": ["sealed local-objective review", "sealed gap/pulse candidate"]},
        {"id": "K4", "trigger": "Fresh actual-update200 state, Adam/RNG/history/patience resume or full acquisition budget cannot be qualified", "decision": "Stop fixed-time pulse admission; selected endpoint is a different estimand", "basis": ["sealed state availability amendment"]},
        {"id": "K5", "trigger": "Cache varies with private learned states, unfair preprocessing comparator, or benefit requires a new learned pool/gate", "decision": "Stop cache/graph-mechanism claim; learned aggregation measures another capacity/budget", "basis": [new_ids[3], "retained BUDDY and Link-MoE conclusions"]},
        {"id": "K6", "trigger": "Frozen practical quality screen fails or whole operation time/peak memory defeats small extension", "decision": "Stop pulse/persistent promotion; retain full paid acquisition/trial/continuation/serving charges", "basis": ["sealed candidate", "sealed acquisition amendment"]},
    ],
    "no_new_hyperparameter_grid": True,
    "no_persistent_refresh_admission": True,
})

for rel, sha in expected.items():
    assert h((BASE / rel).read_bytes()) == sha
for file in P.rglob("*.json"):
    if file.name not in {"MANIFEST.json", "SEAL.json"}:
        json.loads(file.read_text())
dump("VERIFICATION.json", {
    "schema": "graph-specific-error-gap-verification-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
    "prior_inputs_verified_unchanged": len(bindings), "new_primary_identifiers": len(new_ids), "primary_identifier_overlap": 0,
    "new_scoped_method_reads": 4, "new_full_primary_reads": 0, "retained_primary_rereads": 0,
    "exact_short_passages": len(passages), "passage_offsets_and_hashes_verified_by_builder": True,
    "versioned_PDF_aliases_byte_identical": {"AdaGCN_latest_equals_v3": True, "link_stacking_latest_equals_v1": True},
    "visually_inspected_method_renders": ["renders/adagcn_v3_page6.png", "renders/trees_page3.png", "renders/trees_page4.png", "renders/grand_page4.png", "renders/grand_page5.png", "renders/stacking_v1_page9.png"],
    "published_stacking_SI_limit": "HTTP200 returned non-PDF HTML; saved failure response, not read as SI",
    "JSON_parse": "PASS", "runtime_scientific_reads_or_execution": False,
    "writes_confined_to_separate_packet": True,
    "limits": "Integrity/scope only, no full proof/result/source audit, native runtime, exact-method equivalence proof or global absence/novelty inference.",
})

payload = []
for file in sorted(P.rglob("*")):
    if file.is_file() and file.name not in {"MANIFEST.json", "SEAL.json"} and "__pycache__" not in file.parts:
        data = file.read_bytes()
        payload.append({"path": str(file.relative_to(P)), "bytes": len(data), "sha256": h(data)})
dump("MANIFEST.json", {"schema": "graph-specific-error-gap-search-manifest-v1", "created_utc": datetime.now(timezone.utc).isoformat(), "bounded_literature_analysis_only": True, "scientific_execution_authorized": False, "payload": payload})
manifest = (P / "MANIFEST.json").read_bytes()
for row in payload:
    data = (P / row["path"]).read_bytes()
    assert len(data) == row["bytes"] and h(data) == row["sha256"]
dump("SEAL.json", {"schema": "graph-specific-error-gap-search-seal-v1", "manifest_sha256": h(manifest), "manifest_bytes": len(manifest), "payload_count": len(payload), "payload_verified": True, "prior_inputs_preserved": True, "scientific_execution_authorized": False})
print(json.dumps({"packet": str(P), "manifest_sha256": h(manifest), "payloads": len(payload), "new_scoped_method_reads": 4, "full_reads": 0, "retained_primary_rereads": 0, "short_passages": len(passages)}))
