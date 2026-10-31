"""Assemble literature custody records; standard-library file operations only."""
from pathlib import Path
import datetime
import hashlib
import json
import xml.etree.ElementTree as ET

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parent
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def text_sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def save(name, obj):
    (PACKET / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


assert not (PACKET / "SEAL.json").exists(), "Preserve an existing seal."
index_path = ROOT / "literature_memory/index_v17/LITERATURE_INDEX.json"
assert sha(index_path) == "95d5b18e089d625435f8ca499f39df09dac3a035e29b0ca0691c4e023548c843"
index = json.loads(index_path.read_text())
input_scopes = [
    ("literature_memory/index_v17/LITERATURE_INDEX.json", "Consulted first; relevant indexed conclusions and packet paths, no primary reread."),
    ("literature_memory/index_v17/README.md", "Index read/count policy."),
    ("literature_memory/index_v17/ADOPTION_RECEIPT.json", "Index adoption custody."),
    ("heterogeneous_factor_gap_v1/REPORT.md", "Complete saved assessment, algebra/comparator boundary and existing no-go."),
    ("heterogeneous_factor_gap_v1/PAPER_CONCLUSIONS.json", "HG-Adapter and temporal heterogeneous pretraining scopes reused."),
    ("heterogeneous_factor_gap_v1/REUSED_CONCLUSIONS.json", "Saved type-indexed diagonal-adapter equivalence and prior dispositions."),
    ("heterogeneous_factor_gap_v1/DERIVATIONS.json", "Saved extraction provenance; no source reacquisition."),
    ("shared_message_closest_prior_v1/REPORT.md", "Saved common-message consensus/transport algebra and source scopes reused."),
    ("learned_router_be_literature_v1/REPORT.md", "Saved shared transport/source-pooling, graph MoE and adapter-routing assessment reused."),
    ("shared_message_closest_prior_v1/contrastive_repulsion_prior_v1/REPORT.md", "Saved CDLG/DICE/HGEN/SuGAr pairing/diversity findings and limitations reused."),
    ("graph_initializer_cache_literature_followup_v1/PAPER_CONCLUSIONS.json", "Saved initializer/JVP/scale-control conclusions reused, no primary reread."),
    ("graph_initializer_cache_literature_followup_v1/REPORT.md", "Saved existing initializer/Gram-matched control disposition."),
    ("graph_fast_factor_uncertainty_literature_v1/REUSED_CONCLUSIONS.json", "Saved Rank-1 BNN conditional factor inference scope reused."),
]
bindings = []
for relative, scope in input_scopes:
    p = ROOT / relative
    bindings.append({"path": relative, "bytes": p.stat().st_size, "sha256": sha(p), "inspection": scope})
save("INPUT_BINDINGS.json", {
    "schema": "relation-conditioned-factor-inputs-v1", "created_utc": NOW,
    "saved_index_consulted_first": True, "bindings": bindings,
    "index_counts_are_full_paper_read_counts": False,
    "index_read_accounting_preserved": index["read_accounting"],
})

scopes = [
    {
        "key": "rgcn", "canonical_id": "arXiv:1703.06103v4", "stem": "rgcn_v4",
        "ranges": [[0, 0], [12, 39], [52, 56], [87, 90]],
        "scope": "Sections 2.1–3 complete propagation/basis/block/classification method; selected entity-classification setup and supplementary preprocessing paragraphs. No full link-prediction method, proof/table/figure audit.",
    },
    {
        "key": "hgt", "canonical_id": "arXiv:2003.01332v1", "stem": "hgt_v1",
        "ranges": [[0, 0], [36, 94], [96, 108]],
        "scope": "Sections 3–4 typed attention/messages/aggregation/temporal method and sampling/timestamp prose; selected schema/task/setup. Algorithm tables not independently audited; no full results/figure audit.",
    },
    {
        "key": "sehgnn", "canonical_id": "arXiv:2207.02547v3", "stem": "sehgnn_v3",
        "ranges": [[0, 2], [29, 72], [99, 106]],
        "scope": "Motivation and complete main aggregation/projection/semantic-fusion method including Algorithm 1 parameter/input prose; Appendix A commutation and selected native recipe. No full experiments/figure/table audit.",
    },
]
passages = []
primary_scope_records = []
for s in scopes:
    raw = PACKET / "primary" / (s["stem"] + ".html")
    parsed = PACKET / "primary" / (s["stem"] + "_blocks.json")
    blocks = json.loads(parsed.read_text())
    primary_scope_records.append({
        "key": s["key"], "canonical_id": s["canonical_id"],
        "read_status": "new_primary_scoped_methods_not_full_paper",
        "source": str(raw.relative_to(PACKET)), "source_sha256": sha(raw),
        "zero_based_HTML_block_ranges_inclusive": s["ranges"], "exact_scope": s["scope"],
        "full_paper_read": False,
    })
    for i in sorted({i for lo, hi in s["ranges"] for i in range(lo, hi + 1)}):
        b = blocks[i]
        passages.append({
            "key": s["key"], "kind": "primary_html_block",
            "source": str(raw.relative_to(PACKET)), "source_sha256": sha(raw),
            "parsed_source": str(parsed.relative_to(PACKET)),
            "zero_based_block": i, "html_id": b["id"],
            "text": b["text"], "text_sha256": text_sha(b["text"]),
        })

source_ranges = [
    ("rgcn", "rgcn_README.md", [[1, 105]], "Complete README; coauthor-owned entity-classification repository identification/dependency/recipe."),
    ("rgcn", "rgcn_rgcn_layers_graph.py", [[1, 125]], "Complete source text; aggregation-before-composed-map/basis and activation."),
    ("rgcn", "rgcn_setup.py", [[1, 21]], "Package/dependency metadata."),
    ("hgt", "hgt_README.md", [[1, 86]], "Complete README; primary-linked repo, sampling/preparation/dependencies."),
    ("hgt", "hgt_pyHGT_conv.py", [[1, 139]], "HGTConv body fully inspected. DenseHGTConv registration lines 143–180 were incidentally viewed, not method-audited; remaining classes unread."),
    ("sehgnn", "sehgnn_Readme.md", [[1, 42]], "Complete README; correct-case author-linked repository/requirements."),
    ("sehgnn", "sehgnn_hgb_Readme.md", [[1, 38]], "Complete native HGB commands; performance image pixels not retrieved."),
    ("sehgnn", "sehgnn_ogbn_Readme.md", [[1, 58]], "Complete staged recipe and optional ComplEx acquisition; performance image pixels not retrieved."),
    ("sehgnn", "sehgnn_hgb_model.py", [[1, 228]], "Complete HGB model source; per-metapath maps/semantic fusion and sparse/dense input branches."),
    ("sehgnn", "sehgnn_ogbn_utils.py", [[1, 120]], "Typed mean propagation 49–84 and removal helper; adjacent metric-helper beginning incidental, no complete diagnostic/trainer/data-loader audit."),
    ("sehgnn", "sehgnn_ogbn_main.py", [[1, 220], [248, 291]], "Feature-cache construction before stages; staged pseudo-label inputs and MAG label propagation/diagonal removal. No complete training/selection/argument audit; OGB model source not acquired."),
]
source_scope_records = []
for key, filename, ranges, note in source_ranges:
    p = PACKET / "author_source" / filename
    lines = p.read_text().splitlines(keepends=True)
    source_scope_records.append({
        "key": key, "source": str(p.relative_to(PACKET)), "source_sha256": sha(p),
        "one_based_source_line_ranges_inclusive": ranges,
        "scope": note, "file_line_count": len(lines), "imported_or_executed": False,
    })
    for lo, hi in ranges:
        assert hi <= len(lines)
        body = "".join(lines[lo - 1:hi])
        passages.append({
            "key": key, "kind": "pinned_author_source_text",
            "source": str(p.relative_to(PACKET)), "source_sha256": sha(p),
            "one_based_line_range_inclusive": [lo, hi],
            "text": body, "text_sha256": text_sha(body),
        })
save("READ_SCOPES.json", {
    "schema": "relation-conditioned-factor-read-scopes-v1",
    "primary_scopes": primary_scope_records, "author_source_scopes": source_scope_records,
    "new_scoped_primary_method_reads": 3, "new_full_primary_reads": 0,
    "retained_primary_revisit_increment": 0, "new_primary_paper_limit": 3,
    "wrong_identifier_abstract_metadata_is_not_a_method_read": True,
    "scientific_imports_or_execution": False,
})
save("INSPECTED_PASSAGES.json", {
    "schema": "relation-conditioned-factor-passages-v1",
    "locator_policy": "HTML block indices zero-based; source lines one-based; exact extracted bytes/text hashes retained.",
    "passages": passages,
})

records = [
    {
        "key": "rgcn", "canonical_id": "arXiv:1703.06103v4",
        "verified_title": "Modeling Relational Data with Graph Convolutional Networks",
        "authors": ["Michael Schlichtkrull", "Thomas N. Kipf", "Peter Bloem", "Rianne van den Berg", "Ivan Titov", "Max Welling"],
        "version_date": "2017-10-26", "primary_url": "https://arxiv.org/html/1703.06103v4",
        "saved_takeaway": "Relation-specific linear propagation with shared relation bases and block decompositions is prior. Relation-resolved linear aggregates can precede map composition for common inputs/operators; basis sharing does not collapse private member trajectories.",
        "relationship": "Direct relation/channel-sharing prior and exact computation boundary; typed rank-one factors are additional diagonal adapters.",
        "official_repository": "https://github.com/tkipf/relational-gcn",
        "repository_identification": "Coauthor-owned README identifies paper/entity-classification code; not linked from inspected primary HTML.",
        "official_commit": "4bec1341dd46b72bf482f7ed26c2dca4533577f6",
        "commit_date": "2018-03-15T14:45:55Z",
        "source_findings": ["graph.py 81–113 aggregates supports, concatenates, composes relation weights from bases and applies activation.", "README specifies legacy Keras 1.2.1/Theano 0.9.0, CPU and compilation constraints; no runtime qualification."],
        "limits": ["No member ensemble or predictive-complementarity guarantee.", "Selected entity-classification recipe; link-prediction body/results not fully read.", "Paper basis/block parameter reduction is not a graph edge-channel timing claim."],
    },
    {
        "key": "hgt", "canonical_id": "arXiv:2003.01332v1",
        "verified_title": "Heterogeneous Graph Transformer",
        "authors": ["Ziniu Hu", "Yuxiao Dong", "Kuansan Wang", "Yizhou Sun"],
        "version_date": "2020-03-03", "primary_url": "https://arxiv.org/html/2003.01332v1",
        "saved_takeaway": "Meta-relation factorization shares type and relation attention/message parameters. Attention depends on current source/target states and normalizes over target neighborhoods; private factors/states generally induce private operators.",
        "relationship": "Rules out transferring one-common-operator commutation to arbitrary private HGT routes; frozen/common attention is a declared architectural restriction.",
        "official_repository": "https://github.com/acbull/pyHGT",
        "repository_identification": "Primary-linked author repository.",
        "official_commit": "85eaccd482bc1d1af56c2de297b6e3a88b96d5cd",
        "commit_date": "2023-09-08T16:46:18Z",
        "source_findings": ["HGTConv 60–111 derives relation scores and messages from current endpoints and softmaxes by target ID.", "114–134 applies GELU, typed output map, gated residual and normalization.", "Pinned code's relation-prior tensor shape differs from printed full meta-relation prior; no exact manuscript-version claim."],
        "limits": ["Remaining convolution classes/source paths not method-audited.", "Sampling/timestamp prose scoped; algorithm tables/results not completely audited.", "Sharing stored relation matrices alone does not share edge scores or transport; no host cost test."],
    },
    {
        "key": "sehgnn", "canonical_id": "arXiv:2207.02547v3",
        "verified_title": "Simple and Efficient Heterogeneous Graph Neural Network",
        "authors": ["Xiaocheng Yang", "Mingyu Yan", "Shirui Pan", "Xiaochun Ye", "Dongrui Fan"],
        "version_date": "2023-09-01", "primary_url": "https://arxiv.org/html/2207.02547v3",
        "saved_takeaway": "Fixed typed/metapath mean aggregation is explicitly moved to preprocessing and reused before learned semantic MLP/fusion. This supplies the feasible shared-cache alternative to recurrent private graph transport.",
        "relationship": "An ensemble of private relation/channel maps on a common typed cache is a compact semantic-adapter/BE composition, not a new transport operation.",
        "official_repository": "https://github.com/ICT-GIMLab/SeHGNN",
        "repository_identification": "Primary-linked author repository.",
        "official_commit": "e92bd37d0b803457339555684f139b4c8f3e160d",
        "commit_date": "2023-10-16T16:25:34Z",
        "source_findings": ["ogbn/utils.py 49–84 builds typed metapath means; main.py 77–105 extracts/cache-reorders before stage training.", "HGB model.py 71–92 defines per-metapath linear maps; 196–228 consumes/fuses semantic inputs.", "HGB sparse branch 196–201 multiplies learned embeddings inside forward; not universally a free dense cache.", "OGB labels/pseudo-labels are propagated within stages; stronger README recipe adds 500000-step ComplEx acquisition."],
        "limits": ["Empirical attention/depth findings are task-specific, not a universal sharing theorem.", "Paper label propagation and optional extra embedding acquisition must be included in competence/cost accounting.", "HGB model and OGB preprocessing scopes are different branches; OGB model/source trainer not fully audited.", "No new factor ensemble, dataset acquisition or runtime qualification."],
    },
]
for r in records:
    scope = next(s for s in primary_scope_records if s["key"] == r["key"])
    r.update(read_status="new_primary_scoped_methods_plus_pinned_author_source_not_full_paper",
             exact_read_scope="READ_SCOPES.json:" + r["key"],
             primary_sha256=scope["source_sha256"],
             disposition="Established typed-sharing/computation prior; no distinct method/pilot admission.")
save("PAPER_CONCLUSIONS.json", {
    "schema": "relation-conditioned-factor-primary-conclusions-v1", "records": records,
    "read_accounting": {"new_primary_paper_limit": 3, "genuinely_new_primary_methods": 3,
                        "new_scoped_primary_method_reads": 3, "new_full_primary_reads": 0,
                        "retained_primary_revisit_increment": 0},
    "overall_conclusion": "No-go. Fixed typed aggregates plus private maps are an established decoupled semantic ensemble; exact recurrent private transport costs member channels, and pooling changes the model. No new contrast-retention/closure mechanism identified.",
    "pilot_recommended": False,
})

reuse_ids = ["2609.26310", "2605.15888", "2609.08709", "2412.11085", "2411.01155", "2609.35219", "2410.24210", "2306.02775", "2002.06715"]
reused_index_records = [r for r in index["paper_records"] if any(i in str(r["canonical_id"]) for i in reuse_ids)]
for r in reused_index_records:
    assert sha(ROOT / r["conclusion_file"]) == r["conclusion_file_sha256"]
typed_reuse = json.loads((ROOT / "heterogeneous_factor_gap_v1/REUSED_CONCLUSIONS.json").read_text())
rank1_reuse = json.loads((ROOT / "graph_fast_factor_uncertainty_literature_v1/REUSED_CONCLUSIONS.json").read_text())
save("REUSED_CONCLUSIONS.json", {
    "schema": "relation-conditioned-factor-reused-conclusions-v1",
    "policy": "Preserve inherited scopes/status/limitations; zero new primary read credit. Full-text-read claims in older reports are not independently recertified.",
    "index_records_preserved_verbatim": reused_index_records,
    "typed_prior_reuse_file_preserved_verbatim": typed_reuse,
    "rank1_saved_report_record_preserved_verbatim": [r for r in rank1_reuse["saved_report_records"] if "2005.07186" in r["canonical_id"]],
    "saved_report_dispositions": [
        {"source": "shared_message_closest_prior_v1/REPORT.md", "scope": "Complete saved common-message assessment", "takeaway": "Fixed consensus discards transported member-difference modes but retains private local/root states; cross-stitch mixing, BE and cached propagation are established primitives."},
        {"source": "learned_router_be_literature_v1/REPORT.md", "scope": "Complete saved routing/source-pooling assessment", "takeaway": "Graph MoE, shared/low-rank experts, source pooling and common transport are prior ingredients; no-go for promotion of that assembly."},
        {"source": "shared_message_closest_prior_v1/contrastive_repulsion_prior_v1/REPORT.md", "scope": "Complete saved graph diversity assessment", "takeaway": "CDLG/DICE/SuGAr/HGEN supply pairing/class-aware/evidence/meta-path diversity precedents; generic repulsion not a new principle."},
        {"source": "graph_initializer_cache_literature_followup_v1/REPORT.md", "scope": "Saved initializer/control disposition", "takeaway": "No distinct initializer; graph-directed direction partitioning must survive known Jacobian/output-scale and Gram-matched random controls, without active-study amendment."},
    ],
    "new_primary_read_increment": 0,
})

discovered = []
ns = {"a": "http://www.w3.org/2005/Atom"}
for p in sorted((PACKET / "discovery").glob("*.xml")):
    for e in ET.fromstring(p.read_text()).findall("a:entry", ns):
        ident = e.findtext("a:id", namespaces=ns)
        discovered.append({
            "source": str(p.relative_to(PACKET)), "source_sha256": sha(p),
            "id": ident, "title": e.findtext("a:title", namespaces=ns),
            "disposition": "Exact-title selected primary" if "2207.02547" in ident else "Metadata only; mostly off-topic; no method or absence-of-prior inference",
        })
save("DISCOVERY_DISPOSITIONS.json", {
    "schema": "relation-conditioned-factor-discovery-v1",
    "exhaustive_search": False, "bounded_metadata_queries": 3,
    "candidates": discovered,
    "wrong_identifier": {"url": "https://arxiv.org/abs/2211.12740", "saved_path": "primary/sehgnn_abs.html", "verified_title": "Masked Autoencoding for Scalable and Generalizable Decision Making", "disposition": "Unrelated abstract-page metadata only; excluded without primary-method read; exact-title discovery corrected SeHGNN to 2207.02547v3."},
    "README_case_failure": "SeHGNN README.md returned 404; pinned tree established Readme.md and corrected retrieval succeeded.",
    "new_primary_cap": 3, "new_primary_methods_used": 3,
})

retrieval_checks = []
for log in sorted(PACKET.glob("*RETRIEVAL.json")):
    for r in json.loads(log.read_text()):
        if "path" not in r:
            retrieval_checks.append({"log": log.name, "url": r["url"], "outcome": r.get("error", "no body"), "failure_is_negative_literature_evidence": False})
            continue
        p = ROOT.parent / r["path"]
        assert p.stat().st_size == r["bytes"] and sha(p) == r["sha256"]
        retrieval_checks.append({"log": log.name, "source": str(p.relative_to(PACKET)), "sha256": r["sha256"], "verified": True})
for r in passages:
    p = PACKET / r["source"]
    assert sha(p) == r["source_sha256"] and text_sha(r["text"]) == r["text_sha256"]
    if r["kind"] == "primary_html_block":
        assert json.loads((PACKET / r["parsed_source"]).read_text())[r["zero_based_block"]]["text"] == r["text"]
    else:
        lo, hi = r["one_based_line_range_inclusive"]
        assert "".join(p.read_text().splitlines(keepends=True)[lo - 1:hi]) == r["text"]
for r in bindings:
    p = ROOT / r["path"]
    assert p.stat().st_size == r["bytes"] and sha(p) == r["sha256"]
save("VERIFICATION.json", {
    "schema": "relation-conditioned-factor-verification-v1", "verified_at_UTC": NOW,
    "retrieval_checks": retrieval_checks,
    "passage_count": len(passages), "passage_locators_and_hashes_verified": True,
    "input_hashes_unchanged": True, "reused_index_conclusion_file_hashes_verified": True,
    "new_scoped_primary_method_reads": 3, "new_full_primary_reads": 0,
    "scientific_imports_or_execution": False, "pilot_recommended": False,
    "decision": "no-go for distinct method/pilot lane",
})
