"""Build provenance and seal this literature packet using only the standard library.

This script does not import scientific runtimes or open datasets/checkpoints.
"""
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NOW = datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


specs = {
    "lpformer": {
        "id": "2310.11009v4",
        "title": "LPFormer: An Adaptive Graph Transformer for Link Prediction",
        "sha256": "d03994afa65cbf82e935ec6f4d44a2ac3ea461ad1ad84ab7575e3ca21fb64543",
        "ranges": [[18, 91], [102, 110], [180, 193]],
        "sections": ["3.1–3.5 main method", "4.1–4.3 setup and factor diagnostics", "4.6–4.7 HeaRT and epoch timing", "E.2–E.5 DDI policy, PPR and factor assignment"],
        "excerpts": [19, 23, 85, 86, 187],
        "unread": ["Complete introduction/related-work text", "Complete remaining experimental sections and appendix", "Algorithm listing beyond its caption and described procedure", "Figure pixels", "Author source"],
    },
    "clp": {
        "id": "2406.18763v2",
        "title": "Conformalized Link Prediction on Graph Neural Networks",
        "sha256": "4e6c517b101408ccf3c9b6c475b667907e6bf4f16ac5cfac06e1b2c617a1f0ff",
        "ranges": [[27, 83], [90, 118], [130, 135]],
        "sections": ["3.1–3.4 exchangeability, CQR and sampling", "4.1–4.4 setup/results/ablation text", "Appendix A recipe tables"],
        "excerpts": [30, 31, 33, 34, 40, 42, 44, 51, 63, 96, 134],
        "unread": ["Complete introduction/related work", "Algorithm 1 listing beyond caption/references", "Figure pixels", "Full remaining paper", "Author source", "PDF formula comparison"],
    },
    "fdr_missing": {
        "id": "2507.07025v2",
        "title": "Conformal Network Link Prediction with False Discovery Rate Control under Unstructured Missingness",
        "sha256": "c0d007a882b16fb23d4fbd8cc22765235d005999f23a6f2cbc2cd7db76d1337c",
        "ranges": [[5, 87], [105, 113], [147, 176]],
        "sections": ["Introduction assumptions and problem (5–19)", "2.1–2.5 full main method/assumption text", "4–5 real-data setup and concluding limitations", "Appendix D undirected/bipartite statements and construction text"],
        "excerpts": [6, 7, 23, 27, 41, 42, 50, 61, 81, 82, 83, 84, 86, 106, 153, 156],
        "unread": ["Complete simulation section", "Full proofs in Appendices A–C", "Algorithm listings 1–4 beyond captions/descriptions", "Appendix E", "Figure pixels", "Author source"],
    },
}
blocks = {}
scope_records = []
excerpts = []
for name, spec in specs.items():
    html = HERE / "primary" / (name + ".html")
    assert sha(html) == spec["sha256"], name
    block_path = HERE / "primary" / (name + "_blocks.json")
    block_list = json.loads(block_path.read_text())
    assert all(b["index"] == i for i, b in enumerate(block_list))
    blocks[name] = block_list
    indices = sorted({i for lo, hi in spec["ranges"] for i in range(lo, hi + 1)})
    scope_records.append({
        "canonical_id": "arxiv:" + spec["id"].split("v")[0],
        "versioned_id": spec["id"], "title": spec["title"],
        "primary_url": "https://arxiv.org/html/" + spec["id"],
        "primary_html_sha256": sha(html), "parsed_blocks_sha256": sha(block_path),
        "read_kind": "new scoped primary method read",
        "complete_paper_read": False, "author_source_read_or_execution": False,
        "inclusive_block_ranges": spec["ranges"], "unique_scoped_blocks": len(indices),
        "section_description": spec["sections"],
        "read_passages": [{
            "block_index": i, "tag": block_list[i]["tag"], "id": block_list[i].get("id"),
            "section": block_list[i].get("section"),
            "text_sha256": hashlib.sha256(block_list[i]["text"].encode()).hexdigest(),
        } for i in indices],
        "incidental_abstract_block": 2,
        "incidental_metadata_and_headings": "Abstract/heading inventories were seen; they add no method-read or full-read credit.",
        "not_claimed_read": spec["unread"],
    })
    for i in spec["excerpts"]:
        excerpts.append({"versioned_id": spec["id"], **block_list[i]})


class MathNodes(HTMLParser):
    def __init__(self, ids):
        super().__init__()
        self.ids = ids
        self.depth = 0
        self.active = None
        self.out = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if self.active:
            self.depth += 1
        elif attrs.get("id") in self.ids:
            self.active = attrs["id"]
            self.depth = 1
            self.out[self.active] = {"tokens": [], "alttext": attrs.get("alttext")}

    def handle_endtag(self, tag):
        if self.active:
            self.depth -= 1
            if self.depth == 0:
                self.active = None

    def handle_data(self, data):
        if self.active and data.strip():
            self.out[self.active]["tokens"].append(data.strip())


math_ids = {"S3.E7.m1", "S3.Thmtheorem2.p1.m3", "S3.Thmtheorem2.p1.m5"}
parser = MathNodes(math_ids)
parser.feed((HERE / "primary/clp.html").read_text())
assert set(parser.out) == math_ids
for record in scope_records:
    if record["versioned_id"] == specs["clp"]["id"]:
        record["additional_inspected_mathml_ids"] = sorted(math_ids)
save("READ_SCOPES.json", {
    "schema": "graph-link-structural-uncertainty-read-scopes-v1",
    "generated_UTC": NOW,
    "accounting": {"new_scoped_primary_method_reads": 3, "new_full_primary_reads": 0,
                   "retained_primary_revisits": 0, "new_primary_paper_limit": 3,
                   "new_primary_papers_used": 3, "cumulative_project_read_totals_certified": False},
    "scope_convention": "Zero-based parsed block indices; inclusive ranges. Repeated/overlapping inspections of these new sources are deduplicated. Full retrieval/parsing is not a full read. Captions are not figure-pixel or pseudocode-listing inspection.",
    "papers": scope_records,
})
save("PRIMARY_EXCERPTS.json", {"source": "Exact selected parsed primary passages", "excerpts": excerpts})
save("CLP_FORMULA_VERIFICATION.json", {
    "primary_version": specs["clp"]["id"], "html_sha256": specs["clp"]["sha256"],
    "nodes": parser.out,
    "verified_boundary": "MathML operators, TeX annotations and alttext in the pinned HTML agree. No PDF or author implementation inspected.",
    "inversion_of_eq6": "max(lower-y,y-upper)<=q iff lower-q<=y<=upper+q",
    "printed_interval_issue": "Eq7 and Theorem3.2 print upper endpoint q-upper, inconsistent with inversion of Eq6.",
    "printed_quantile_issue": "Theorem3.2 puts /K inside ceil, while proof block51 uses order index ceil((K+1)(1-alpha)).",
})

cal_features = [0, 1] * 10
cal_labels = cal_features[:]
future_features = [0, 1] * 10
future_labels = [1 - x for x in future_features]
cal_residuals = [abs(x - y) for x, y in zip(cal_features, cal_labels)]
future_residuals = [abs(x - y) for x, y in zip(future_features, future_labels)]
k = math.ceil((len(cal_features) + 1) * 0.9)
q = sorted(cal_residuals)[k - 1]
coverage = sum(abs(x - y) <= q for x, y in zip(future_features, future_labels)) / len(future_features)
assert q == 0 and coverage == 0 and sum(cal_labels) == sum(future_labels) == 10
lower, upper, q2, y = 0.2, 0.8, 0.1, 0.5
assert max(lower - y, y - upper) <= q2
assert lower - q2 > q2 - upper
save("ANALYTIC_CHECKS.json", {
    "execution_kind": "standard-library arithmetic on explicitly synthetic binary sequences; no graph/data/model runtime",
    "order_invariance_not_exchangeability": {
        "predictor": "f(z)=z; case-order invariant; lower=upper=f(z)",
        "calibration": {"features": cal_features, "labels": cal_labels, "residuals": cal_residuals},
        "future": {"features": future_features, "labels": future_labels, "residuals": future_residuals},
        "nominal_coverage": 0.9, "calibration_order_index": k, "q": q,
        "future_coverage": coverage, "both_positive_label_prevalences": 0.5,
        "conclusion": "Case-order invariance and unchanged label prevalence alone do not imply chronological conformal coverage. No claim about the paper's experimental failure.",
    },
    "printed_interval_counterexample": {"lower": lower, "upper": upper, "q": q2, "y": y,
        "nonconformity": max(lower-y, y-upper),
        "printed_interval": [lower-q2, q2-upper], "inverted_interval": [lower-q2, upper+q2],
        "conclusion": "Pinned HTML interval disagrees with its own score inversion; implementation not audited."},
})

conclusions = [
    {
        "name": "lpformer", "closest_prior_role": "Pair-conditioned structural learner and factor-wise diagnostic",
        "primary_text_verified": [
            "GCN node features plus target-pair context attention and learned PPR relative encoding.",
            "CN, one-hop and more distant positional encoders; PPR thresholds limit context; counts enter final scalar decoder.",
            "Factor partition already uses CN, PPR and cosine-feature heuristics; collab uses 80% in Appendix E.4.",
            "Many baselines are borrowed from published tables; epoch timing and HeaRT are narrower comparisons than complete official collab cost/performance."],
        "author_claims_unverified": ["Strong/SOTA benchmark performance", "Consistency across factors", "Training efficiency"],
        "adoption": "Use as an attributed structural/diagnostic prior and possible later capable control. Do not claim new attention or factor-stratification principle.",
    },
    {
        "name": "clp", "closest_prior_role": "CQR uncertainty on GNN edge embeddings",
        "primary_text_verified": [
            "Balanced label subsets, topology from training plus validation positives, neural quantile fitting and calibration.",
            "Degree-guided sampling targets power-law alignment for shorter intervals.",
            "Listed experiments exclude collab and repeat randomized calibration/test splits.",
            "Pinned HTML has interval/quantile inconsistencies verified against its MathML, annotations and alttext.",
            "Fixed/order-invariant scoring alone does not imply chronological score exchangeability; explicit analytic counterexample saved."],
        "author_claims_unverified": ["Marginal coverage and improved interval efficiency in experiments", "Correctness of author implementation"],
        "adoption": "Treat as calibration prior, not proof of conditional or temporal GNNM uncertainty. Printed equations require source/implementation clarification before direct reproduction.",
    },
    {
        "name": "fdr_missing", "closest_prior_role": "Structured graph conformal inference and dependent-link FDR aggregation",
        "primary_text_verified": [
            "Weighted graphon complete-network exchangeability and Assumption 1.1 M independent of realized A.",
            "Paper explicitly excludes latent-variable MAR and realized-link MNAR; missing rates may otherwise be unknown/nonuniform.",
            "Multi-splitting and shared fully observed rows construct exchangeable local scores; BH decisions become averaged e-values for e-BH.",
            "Local theorem states no ties almost surely or random tie-breaking.",
            "Undirected extension aggregates upper triangle; static trade example uses random 10% masking.",
            "Basic theorem does not establish the paper's practical inflation factor; paragraph 83 states further theory is open."],
        "author_claims_unverified": ["Finite-sample FDR theorems were not fully independently proof-audited", "Empirical FDR/power improvements", "Scalability of source implementation"],
        "adoption": "Retain explicit assumption boundary. This specialized construction does not confer its guarantee on a generic GNNM score wrapper or chronological future-edge prediction.",
    },
]
paper_records = []
for conclusion in conclusions:
    name = conclusion.pop("name")
    spec = specs[name]
    paper_records.append({
        "canonical_id": "arxiv:" + spec["id"].split("v")[0],
        "versioned_id": spec["id"], "title": spec["title"],
        "url": "https://arxiv.org/html/" + spec["id"], "html_sha256": spec["sha256"],
        "read_kind": "new scoped primary method read", "full_read": False,
        "source_code_inspected": False, "results_reproduced": False, **conclusion,
    })
save("PAPER_CONCLUSIONS.json", {
    "schema": "graph-link-structural-uncertainty-conclusions-v1", "created_UTC": NOW,
    "decision": "no-go distinct learner/calibration guarantee; retain prospective fixed diagnostic",
    "paper_records": paper_records,
    "novelty_boundary": "Bounded search supplies no exhaustive absence result or novelty claim.",
})

reuse_specs = [
    ("link_covariance_idea_v1/REPORT.md", ["NCN/NCNC, BUDDY/ELPH, Neo-GNN and SEAL already read", "Mean dot-member pooling is a wider deterministic embedding", "Auxiliary BCE convexity gap is not uncertainty", "Sampled BCE does not certify deployment probability"]),
    ("cached_structure_link_ensemble_feasibility_v1/REPORT.md", ["BUDDY shared cache used equally for independent predictors", "Link-MoE structural/feature gate and 80% validation gate-fitting supervision", "PENCIL already read"]),
    ("gine_closest_control_v1/link_recommendation_gap_v1/REPORT.md", ["DivNS/SRNS/BNS/AHNS/ConCF uncertainty or member-guided negative sampling already assessed"]),
    ("adaptive_sharing_novelty_v1/REPORT.md", ["Avoid repeating layer/rank allocation and lookahead sharing"]),
    ("continuous_method_gap_search_v1/round11_joint_risk_decisions/REPORT.md", ["Coherent dependence/scenario-risk proposals already assessed"]),
]
save("REUSED_CONCLUSIONS.json", {
    "schema": "graph-link-structural-uncertainty-reuse-v1",
    "records": [{"path": p, "sha256": sha(ROOT/p), "reused_conclusions": c,
                 "new_primary_read_credit": 0, "original_scope_is_authoritative": True}
                for p, c in reuse_specs],
    "indexed_lead_not_reread": "2605.22593v1 node-classification deep-ensemble uncertainty conclusion was identified in index_v18 but supplies no new primary credit and is not needed for the retained test.",
})

ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
discovery_xml = ET.fromstring((HERE/"discovery/link_uncertainty.txt").read_bytes())
selected_ids = {x["id"].split("v")[0] for x in specs.values()}
arxiv_entries = []
for entry in discovery_xml.findall("a:entry", ns):
    ident = entry.findtext("a:id", namespaces=ns)
    normalized = ident.rsplit("/",1)[-1].split("v")[0]
    arxiv_entries.append({"id": ident, "title": " ".join(entry.findtext("a:title", namespaces=ns).split()),
        "disposition": "selected for recorded primary scope" if normalized in selected_ids else "discovery title/abstract metadata only; no method inspection"})
openalex = json.loads((HERE/"discovery/lpformer.txt").read_text())
openalex_entries = [{"id": e.get("id"), "title": e.get("display_name"), "doi": e.get("doi"),
    "disposition": "discovery metadata only; LPFormer selected independently by pinned primary ID, other entries not inspected"}
    for e in openalex.get("results", [])]
assert len(arxiv_entries) == 15 and len(openalex_entries) == 5
save("DISCOVERY_DISPOSITIONS.json", {
    "schema": "graph-link-structural-uncertainty-discovery-v1",
    "arxiv_returned_entries": len(arxiv_entries),
    "arxiv_total_results_metadata": discovery_xml.findtext("o:totalResults", namespaces=ns),
    "arxiv_entries": arxiv_entries, "openalex_returned_entries": len(openalex_entries),
    "openalex_entries": openalex_entries,
    "selected_primary_metadata": "SELECTED_METADATA.json (three exact versioned IDs)",
    "venue_boundary": "KDD 2024/DOI is OpenAlex metadata; publisher not independently checked.",
    "read_credit_rule": "Discovery metadata is not a scoped method read or a full-paper read.",
})
save("DEDUP_CHECK.json", {
    "memory_consulted_before_acquisition": "literature_memory/index_v18/{README.md,ADOPTION_RECEIPT.json,LITERATURE_INDEX.json}",
    "preacquisition_result_from_continuation_record": "Zero exact selected-ID/full-title matches in index and phase Markdown/conclusion/read-scope files. Bibliographic mentions of LPFormer do not certify a prior primary read.",
    "postwrite_verification": {"UTC": NOW, "tool": "rg", "exit_code": 1,
        "excluded": "graph_link_structural_uncertainty_literature_v1/**",
        "globs": ["*.md", "*CONCLUSIONS*.json", "*READ_SCOPE*.json", "LITERATURE_INDEX.json"],
        "query": "three exact arXiv IDs and three exact complete titles", "matching_lines": 0},
    "limits": "Index_v18 catalog is not a complete primary-read inventory; project text search supplements it. Counts certify this packet only, not cumulative full/scoped totals.",
})

input_paths = ["literature_memory/index_v18/README.md", "literature_memory/index_v18/ADOPTION_RECEIPT.json",
               "literature_memory/index_v18/LITERATURE_INDEX.json"] + [p for p,c in reuse_specs] + [
    "buddy_shared_cache_execution_v4/README.md", "buddy_shared_cache_execution_v4/CONFIG.json",
    "buddy_shared_cache_execution_v4/SOURCE_MANIFEST.json", "buddy_shared_cache_execution_v4/SEAL.json",
    "buddy_complete_data_cache_preparation_v2/SOURCE_MANIFEST.json", "buddy_complete_data_cache_preparation_v2/SEAL.json",
]
bindings = [{"path": p, "bytes": (ROOT/p).stat().st_size, "sha256": sha(ROOT/p)} for p in input_paths]
assert sha(ROOT/"literature_memory/index_v18/LITERATURE_INDEX.json") == "c3a77a96dd82092879a4ce78a2e258ac67d9b445a9b47c00976d1ea8894d36d0"
sealed_expected = {
    "buddy_shared_cache_execution_v4": "ba528e95d90be0ec32e5aacb0fb3d5445eca9a68d38ee215138e72521ea67ff8",
    "buddy_complete_data_cache_preparation_v2": "1f0856c5316d791220741d4f89a9058c65876a0a929921fe159deb41c592ce27",
}
verified = []
for packet, expected in sealed_expected.items():
    source_manifest = ROOT/packet/"SOURCE_MANIFEST.json"
    assert sha(source_manifest) == expected
    manifest = json.loads(source_manifest.read_text())
    for item in manifest["files"]:
        source = ROOT/packet/item["path"]
        assert sha(source) == item["sha256"] and source.stat().st_size == item["bytes"], source
    verified.append({"packet": packet, "manifest_sha256": expected,
                     "payload_files_verified": len(manifest["files"]), "unchanged": True})
save("INPUT_BINDINGS.json", {"schema": "graph-link-structural-uncertainty-input-bindings-v1",
    "bound_UTC": NOW, "files": bindings,
    "sealed_source_payload_verification": verified,
    "scope": "Research text/provenance bytes only. No active data, checkpoint, run artifact or scientific runtime opened."})

save("PAIRED_TEST_SPEC.json", {
    "schema": "prospective-gnnm-structural-uncertainty-diagnostic-v1",
    "status": "proposal only; not executed or added to active study",
    "question": "Does compact member disagreement preserve independent-ensemble added error information for CN=0 future pairs?",
    "arms": ["factorized4", "independent4"], "anchors": ["single256", "matched_single", "native1024"],
    "optimizer_seeds": [0,1,2], "pairing": "same complete train graph, fixed native cache, examples, official negative pool and locked family checkpoints",
    "primary_strata": ["exact train CN=0", "exact train CN>0"],
    "secondary_strata": "training-node degree quartiles plus isolated-node bin; descriptive only",
    "pooling": "mean raw member logits", "error_target": "binary sampled label error at pooled logit>0",
    "primary_uncertainty": "variance across four member sigmoid scores",
    "score_only_control": "abs(sigmoid(mean raw logits)-0.5)",
    "selection": "lowest uncertainty vs highest pooled confidence; same ceiling(50% candidate count) within each CN stratum; stable official pair-row tie breaking; no labels used for selection",
    "primary_risk": "equal-label-mass weighted sampled binary risk on retained set; report true-label counts/errors separately",
    "primary_effect": "T=(R_I,C-R_I,U)-(R_F,C-R_F,U) in CN=0",
    "falsifier": {"mean_T_at_least":0.005,"T_positive_in_each_seed":True,"mean_independent_A_at_least":0.005},
    "competence_interpretation_gate": "absolute full-CN0 sampled risk difference F vs I <=0.01 in every seed; otherwise pooled-competence confounding",
    "fixed_secondary": ["75% and 100% retention", "member negative-pool rank percentile spread saturation check", "CN>0 table", "unanimous strict Hits@50 positive misses", "u<=0.001 and confidence>=0.45 event"],
    "no_significance_or_coverage_claim": True,
    "failure_interpretation": "Independent disagreement with no benefit gives no evidence for compact uncertainty extension. Similar unanimous errors are compatible with several causes and are not causal proof of missing structure.",
    "strong_mechanism_control": "Separate later NCNC or LPFormer comparison with identical graph/negative policy and complete added cost accounting; not authorized/executed here.",
    "chronology_boundary": "Future-cohort behavior alone does not identify temporal shift as causal; no conformal guarantee assumed.",
    "prospective_boundary": "Must freeze before diagnostic outcomes are inspected; otherwise exploratory and require separately reserved confirmation cohort.",
})

for item in bindings:
    assert sha(ROOT/item["path"]) == item["sha256"], item["path"]
save("VERIFICATION.json", {"UTC": NOW, "primary_html_hashes_match_retrieval": True,
    "input_bindings_rechecked_unchanged": True, "sealed_source_payloads_unchanged": verified,
    "parsed_scopes_in_bounds_and_deduplicated": True, "analytic_counterexample_assertions_passed": True,
    "new_paper_limit_satisfied": True, "report_claims_and_read_scopes_reviewed": True,
    "active_study_execution": False, "author_source_execution": False,
    "rendered_layout_verified": False, "layout_boundary": "Saved Markdown content inspected as text; no visual preview claimed."})
payload = [{"path": str(p.relative_to(HERE)), "bytes": p.stat().st_size, "sha256": sha(p)}
           for p in sorted(HERE.rglob("*")) if p.is_file() and p.name not in {"MANIFEST.json", "SEAL.json"}]
save("MANIFEST.json", {"schema": "graph-link-structural-uncertainty-manifest-v1", "created_UTC": NOW, "files": payload})
save("SEAL.json", {"schema": "graph-link-structural-uncertainty-seal-v1", "sealed_UTC": NOW,
    "manifest_sha256": sha(HERE/"MANIFEST.json"), "payload_files": len(payload),
    "report_sha256": sha(HERE/"REPORT.md"),
    "decision": "no-go new learner/calibration guarantee; fixed prospective paired diagnostic only",
    "new_scoped_primary_method_reads": 3, "new_full_primary_reads": 0,
    "retained_primary_revisits": 0, "author_source_read_or_execution": False,
    "active_study_compute_or_source_edits": False, "memory_index_edits": False,
    "cumulative_project_read_totals_certified": False})
for item in payload:
    assert sha(HERE/item["path"]) == item["sha256"]
print(json.dumps({"packet": str(HERE), "payload_files": len(payload),
                  "manifest_sha256": sha(HERE/"MANIFEST.json"), "decision": "no-go learner; prospective diagnostic",
                  "read_count": {"scoped_new":3,"full_new":0,"retained_rereads":0}}, indent=2))
