"""Build a source-only literature packet using retained text and stdlib metadata."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
STAMP = datetime.now(timezone.utc).isoformat()
INDEX_SHA = "9a451fedaaa11d5533f4255c061f49404e667f52d9c7e94629d6541861d4730e"


def desc(path, base=ROOT):
    data = path.read_bytes()
    return {"path": str(path.relative_to(base)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def dump(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


index_path = PHASE / "literature_memory/index_v40/LITERATURE_INDEX.json"
assert desc(index_path, PHASE)["sha256"] == INDEX_SHA
index = json.loads(index_path.read_text())
records = index["paper_records"]
all_cached_ids = {r["canonical_id"] for r in records}
new = {
    "graphlora": ("2409.16670v2", [[34, 80]], "Sections4.1-4.5: cross-graph alignment, residual low-rank GNN, contrastive/structure/task objectives and theorem statement; proof not audited."),
    "graph_lora_generation": ("2606.22429v1", [[36, 75]], "Sections4.1-4.4: task graph/input, graph weight generation, low-rank factors, norm rescaling and token likelihood training; theorem/corollary statements, no proof audit."),
    "mose": ("2509.09337v1", [[18, 73]], "Main method: anonymous-walk patterns, topology context, top-K gate, trainable hidden-graph experts, fusion and objectives; proposition statement, no proof or appendix algorithm audit."),
    "laplora_graph": ("2602.07278v1", [[16, 52]], "Method, spectral filter statements and implementation paragraph; single spectral low-rank propagation residual. Proofs and results not audited."),
    "x_lora": ("2402.07148v2", [[161, 184]], "Sections4.1-4.2.3: token/layer adapter scalings, Eq5, dual pass, frozen-base/frozen-adapter default, layer execution."),
    "graphlora_rec": ("2606.07526v1", [[31, 64]], "Sections4.1-4.5: collaborative input, graph encoder, low-rank bottleneck injection and source complexity discussion. No experiments/results section read."),
}
for version, _, _ in new.values():
    assert not any(version.split("v")[0] in cid for cid in all_cached_ids)

ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
metadata = {}
query_records = []
dispositions = []
selected_by_version = {v[0]: k for k, v in new.items()}
for file in sorted((ROOT / "discovery").glob("*.xml")):
    tree = ET.fromstring(file.read_bytes())
    receipt = json.loads(file.with_name(file.stem + "_RECEIPT.json").read_text())
    entries = tree.findall("a:entry", ns)
    total = int(tree.findtext("o:totalResults", default="0", namespaces=ns))
    query_records.append({"key": file.stem, "query": receipt["query"],
                          "URL": receipt["URL"], "start": 0, "max_results": 15,
                          "sortBy": "relevance", "sortOrder": "descending",
                          "total_reported": total, "returned": len(entries),
                          "truncated": total > len(entries), "receipt": desc(file.with_name(file.stem + "_RECEIPT.json"))})
    for ent in entries:
        url = ent.findtext("a:id", namespaces=ns)
        version = url.rsplit("/", 1)[-1]
        item = {"canonical_id": "arXiv:" + version, "version": version,
                "title": " ".join(ent.findtext("a:title", namespaces=ns).split()),
                "authors": [a.findtext("a:name", namespaces=ns) for a in ent.findall("a:author", ns)],
                "first_posted": ent.findtext("a:published", namespaces=ns),
                "version_updated": ent.findtext("a:updated", namespaces=ns),
                "abs_url": "https://arxiv.org/abs/" + version,
                "discovery_source": desc(file)}
        metadata[version] = item
        if version in selected_by_version:
            disposition = "new_scoped_primary_method_read"
        elif any(version.split("v")[0] in cid for cid in all_cached_ids):
            disposition = "indexed_identity_metadata_only_here_no_new_version_method_read"
        else:
            disposition = "metadata_only_not_selected_in_bounded_scout"
        dispositions.append({"query": file.stem, "canonical_id": item["canonical_id"],
                             "title": item["title"], "disposition": disposition,
                             "negative_novelty_evidence": False})

citations = []
scopes = []
navigation = []
for key, (version, ranges, note) in new.items():
    receipt = json.loads((ROOT / "primary" / (key + "_RECEIPT.json")).read_text())
    assert receipt["status"] == 200
    assert receipt["output"] == desc(ROOT / receipt["output"]["path"])
    blocks_file = ROOT / "primary" / (key + ".blocks.json")
    blocks = json.loads(blocks_file.read_text())
    equations_file = ROOT / "primary" / (key + ".equations.json")
    selected = [b for b in blocks if any(lo <= b["index"] <= hi for lo, hi in ranges)]
    title = next(b["text"] for b in blocks if b["tag"] == "h1")
    citations.append({"key": key, **metadata[version], "html_title": title,
                      "primary_url": receipt["URL"], "primary": receipt["output"],
                      "read_status": "new_scoped_primary_method_not_full_paper",
                      "publication_status": "Pinned arXiv version; venue/latest-version status not qualified."})
    scopes.append({"key": key, "canonical_id": "arXiv:" + version,
                   "primary": receipt["output"], "blocks": desc(blocks_file),
                   "equations": desc(equations_file),
                   "ranges_zero_based_inclusive": ranges, "scope_note": note,
                   "read_blocks": selected,
                   "full_paper_read": False, "proof_audit": False,
                   "author_source_inspected": False, "numeric_results_adopted": False,
                   "special_exposure": {
                       "graph_lora_generation": "Initial blocks36-75 display truncated in the middle; exact blocks49-70 subsequently displayed completely in two bounded reads (49-59 and60-70). Method coverage claim is the union after repair.",
                       "graphlora_rec": "Block63 includes a source parameter-overhead example (~1.67%); it is incidental source exposure, not adopted measurement or quality evidence.",
                   }.get(key)})
    navigation.append({"key": key, "scope": "Extractor heading/equation previews used for navigation; previews are not substantive whole-section/proof/results reads.",
                       "previews": [{"index": b["index"], "tag": b["tag"], "text_prefix": b["text"][:130]} for b in blocks if b["tag"].startswith("h") or b["tag"] == "table"],
                       "additional_heading_only": {"graphlora": [81], "mose": [156]}.get(key, [])})
    text = "\n\n".join("BLOCK " + str(b["index"]) + " [" + str(b["id"]) + "] " + b["heading_locator"] + "\n" + b["text"] for b in selected)
    (ROOT / "primary" / (key + ".read_scope.txt")).write_text(text + "\n")

reuse_ids = ["arXiv:2411.01155v1", "arXiv:2608.16384v1", "arXiv:2405.14438v5",
             "arXiv:2412.11085v1", "arXiv:2412.00418v1", "arXiv:2605.15888v1"]
reused = []
for cid in reuse_ids:
    matched = next(r for r in records if r["canonical_id"] == cid)
    reused.append({"canonical_id": cid, "scope_here": "Cached index conclusion reused; no primary reread/retrieval here.",
                   "index_record": matched})
dump("REUSED_CONCLUSIONS.json", {"schema": "structural_lowrank_reused_conclusions_v1", "UTC": STAMP,
                                 "index": desc(index_path, PHASE), "cached_records": reused})

paper_conclusions = {
    "graphlora": {
        "operation": "Frozen pretrained GNN plus same-architecture low-rank residual GNN; structure-weighted MMD, label-aware contrast, predicted-label edge similarity and classification objectives.",
        "relationship": "Direct graph-structural low-rank adaptation prior; no distinct predictive members or input-generated private member adapters established in read scope.",
        "source_limits": ["PPR transition prose and printed symmetric-normalization expression differ.", "Eq12 prints positive log-likelihood in a minimized structural loss; Eq13 uses an argmax label where Eq14 then logs it. These are retained printed ambiguities, not silently repaired.", "Theorem statement read; proof/empirical results not audited."]},
    "graph_lora_generation": {
        "operation": "Whole-graph GNN representation h=MaxPool(GNN(X,A)) generates a low-rank factor; frozen-decoder weight becomes W_hat+(L+g(h;Theta))R^T. GaRA rescales generated G to match average column L1 norm of L; token conditional-likelihood training.",
        "relationship": "Direct graph-conditioned low-rank weight-generation prior. Whole-graph/task adaptation in an LLM differs from local structural member specialization on a strong graph backbone.",
        "source_limits": ["Graph representation is whole-graph; query task subgraph is a separate encoder/input component.", "Theorem4.1/Corollary4.2 norm-amplification statements read; AppendixA proofs not audited or transported as guarantees.", "Norm ratio denominator edge cases/code implementation and results not qualified."]},
    "mose": {
        "operation": "Anonymous random-walk patterns yield structural subgraphs; topology-aware context feeds noisy top-K expert gates. Trainable hidden graphs and random-walk kernels define experts; gated fusion plus task/balance objective.",
        "relationship": "Direct structural-regime expert prior for node/graph classification, without a frozen strong-backbone low-rank member construction in read scope.",
        "source_limits": ["Printed Eq14 reduces importance over nodes and experts to a scalar whereas a per-expert vector is needed for usual CV balance interpretation; preserve ambiguity.", "Reconstruction proposition statement read; proof/appendix algorithm not audited.", "No link-prediction method or benchmark quality adopted."]},
    "laplora_graph": {
        "operation": "Low-rank correction of GCN propagation in leading Laplacian eigenvectors, learned eigenvalue MLP and depth-annealed coefficient; shared residual branch across layers.",
        "relationship": "Single structural spectral adapter; no context-conditioned expert assignment or predictive member bank established.",
        "source_limits": ["Printed strict-contraction statement over all lambda in(0,2] cannot be strict at lambda=1 where both filters are zero. Boundary lambda=2 requires the stated stability condition.", "Bounded sigmoid theta alone does not establish all stability conditions; no theorem or quality guarantee transported.", "Implementation paragraph describes width64 GCN without BN/residual backbone; no strong-backbone transfer demonstrated here."]},
    "x_lora": {
        "operation": "Hidden-state scaling head predicts token/layer softmax weights across additive low-rank adapters; Eq5 mixes B_i A_i(x lambda_i)alpha_i in one transformer trajectory. Default freezes base and adapters and trains gate; dual pass obtains scalings then logits.",
        "relationship": "Closest direct low-rank adapter-bank routing prior. It demonstrates context-dependent low-rank composition, not independently retained/served predictive members tied to measured graph structure.",
        "source_limits": ["Paper permits opening adapters/base as alternatives but read scope does not establish target strong graph-backbone recipe.", "Printed A_i dimensions in block164 are not conformable with general W0 d-by-k unless d=k; Eq5 operation is retained, no silent general-dimension repair.", "Dual-pass/cache handling is a source design claim; no author code, runtime, scientific results or graph-quality transfer qualified."]},
    "graphlora_rec": {
        "operation": "Trainable collaborative embeddings feed LLM tokens and a sampled graph encoder; bottleneck z_n=W_neck e_n^L enters one LoRA layer as B(lambda_lora A x_t+lambda_gnn z_n) for mapped user/item token anchors.",
        "relationship": "Direct node-structural context injection inside a low-rank pathway, but printed Eq10 is an additive structural bottleneck signal with shared factors, not generated/private member factors or separately served members.",
        "source_limits": ["GNN/W_neck are jointly optimized; exact frozen/base optimizer/label protocol not qualified from method alone.", "Source overhead example and sampled-subgraph complexity are not current-project work or quality measurements.", "Same GraphLoRA name as cross-graph transfer paper; identities kept separate."]},
}
dump("PAPER_CONCLUSIONS.json", {"schema": "structural_lowrank_paper_conclusions_v1", "UTC": STAMP,
                                "papers": [{"key": k, "canonical_id": "arXiv:" + new[k][0], "read_scope_reference": "READ_SCOPES.json", **v} for k, v in paper_conclusions.items()]})
dump("CITATIONS.json", {"schema": "structural_lowrank_citations_v1", "UTC": STAMP,
                        "new_scoped_sources": citations, "cached_prior_reference": "REUSED_CONCLUSIONS.json"})
dump("READ_SCOPES.json", {"schema": "structural_lowrank_exact_read_scopes_v1", "UTC": STAMP,
                          "indexing": "Deterministic extractor zero-based inclusive block indices.", "sources": scopes})
dump("NAVIGATION_SCOPES.json", {"schema": "structural_lowrank_navigation_v1", "UTC": STAMP, "sources": navigation})
dump("DISCOVERY_QUERIES.json", {"schema": "structural_lowrank_discovery_queries_v1", "UTC": STAMP, "queries": query_records})
dump("DISCOVERY_DISPOSITIONS.json", {"schema": "structural_lowrank_discovery_dispositions_v1", "UTC": STAMP, "entries": dispositions,
                                    "policy": "Unselected metadata, indexed versions and unread latest versions do not count as method reads or absence evidence."})
dump("SEARCH_LIMITS.json", {"schema": "structural_lowrank_search_limits_v1", "UTC": STAMP,
                           "query_limits": query_records, "returned_entries_including_duplicates": len(dispositions),
                           "unique_returned_version_ids": len({d["canonical_id"] for d in dispositions}),
                           "limitations": ["Bounded relevance-ranked arXiv retrieval, not a systematic review or global novelty/absence certificate.",
                                           "Context-lowrank and structural-expert queries are truncated at15; no later pages fetched.",
                                           "Six selected pinned versions method-read; other results retained at metadata/disposition level.",
                                           "CHoE v2 appears in discovery; cached v1 conclusion was reused and v2 method was not read.",
                                           "No code/reproduction, matched benchmark superiority, theorem audit or acceptance prediction."]})
dump("READ_ACCOUNTING.json", {"schema": "honest_structural_lowrank_read_accounting_v1", "UTC": STAMP,
                             "index_conclusions_consulted_before_primary_reads": True,
                             "prior_index_paper_records": len(records),
                             "prior_normalized_paper_identities": 119,
                             "new_identities_with_scoped_primary_method_read": 6,
                             "new_HTML_method_reads": 6, "full_papers_certified": 0, "proof_audits": 0,
                             "author_implementations_inspected": 0, "benchmark_results_adopted": 0,
                             "retrieval_is_not_read": True, "extraction_is_not_read": True,
                             "navigation_is_not_full_method_read": True,
                             "truncation_handling": "Initial GaRA blocks36-75 display was truncated;49-70 later completely displayed. Other certified ranges follow inherited scoped-read record and two new bounded method displays.",
                             "operations": ["stdlib public arXiv metadata/HTML retrieval", "deterministic stdlib HTML/MathML-alttext extraction", "text reading", "stdlib metadata/JSON/AST/hash validation"],
                             "model_training_or_numerical_prediction_executed": False,
                             "downloaded_content_executed": False,
                             "GPU_SSH_upload_runtime_mutation": False,
                             "arrays_checkpoints_labels_runtime_binaries_read": False,
                             "canonical_or_index_mutation": False, "read_scopes": "READ_SCOPES.json"})

prior = PHASE / "ncnc_coherent_completion_recent_literature_scout_20261003_v1"
dump("INPUT_BINDINGS.json", {"schema": "structural_lowrank_inputs_v1", "UTC": STAMP,
                            "scope": "Source/literature/metadata scout only. No review of own V4 repair.",
                            "bindings": {"index_v40": {n: desc(index_path.parent / n, PHASE) for n in ["LITERATURE_INDEX.json", "MANIFEST.json", "SEAL.json"]},
                                         "recent_coherent_completion_scout": {n: desc(prior / n, PHASE) for n in ["REPORT.md", "CONCLUSIONS.json", "MANIFEST.json", "SEAL.json"]}},
                            "parent_supplied_context": {"Mixed40": "Complete Mixed40 is negative; preserve result, no outcome rescue. Supplied by parent; runtime outcomes not independently reread here.",
                                                        "question": "One quality-oriented extension beyond NCNC joint-pattern supervision: graph-conditioned low-rank member adaptation on a strong backbone, specialization tied to measurable structural context."},
                            "canonical_or_index_mutation": False})
dump("CONCLUSIONS.json", {"schema": "structural_lowrank_scientific_scout_synthesis_v1", "UTC": STAMP,
                         "decision": "NO_WARRANTED_NEW_EXPERIMENT_AT_PRESENT",
                         "hypotheses_or_paired_comparisons_proposed": 0,
                         "reason": "Graph-structural low-rank adaptation, graph-conditioned factor generation, low-rank adapter mixing and structural-regime expert routing are established nearby operations. No reviewed evidence establishes structural-context-conditioned complementary errors on a capable target backbone or incremental quality from separately retained predictive members.",
                         "own_algebraic_observation": {"statement": "For conformable rank-r factors applied to the same layer state x, sum_m a_m(c)B_m A_m x equals [B_1 ... B_M] [a_1(c)A_1; ...; a_M(c)A_M] x, a single conditional adapter of rank at most Mr.",
                                                       "scope": "Representation identity for additive linear updates sharing x; not a whole-method/training/resource equivalence or a result for separately evolving nonlinear model trajectories.",
                                                       "consequence": "Gaining over an unconditioned lower-rank baseline can reflect context access, rank/capacity or optimization. It does not establish predictive-member necessity."},
                         "unresolved_scientific_gap": "If a competent strong baseline later shows TRAIN-only structural context predicting reproducible complementary errors, separate-member utility would still require a capable same-context single and a test that breaks member-context association while retaining marginal resources. This is a relevance criterion, not a specified/adopted/authorized experiment here.",
                         "retained_negative": "Complete Mixed40 negative result remains negative. No new objective policy, rescue, tuning or acceptance promise.",
                         "claims_not_established": ["novel generic mixture/LoRA/context principle", "quality transfer to target strong backbone", "need for multiple retained members", "latent ambiguity/calibration", "benchmark superiority", "exact target-method equivalence to any paper", "global literature absence"],
                         "execution": False, "canonical_or_index_mutation": False})

report = """# Structural-context low-rank adaptation: scoped scientific scout

**Decision: no warranted new experiment at present.** The reviewed sources establish the nearby mechanisms, while leaving the target question—quality added by distinct predictive members on a capable strong backbone—unanswered. Complete Mixed40 remains negative. This packet provides attribution and a scientific limit; it supplies no run, tuning change or acceptance promise.

## What the new sources establish

| Pinned source | Operation read | Consequence for this extension |
|---|---|---|
| [GraphLoRA, cross-graph transfer](https://arxiv.org/html/2409.16670v2), Yang et al. | Frozen GNN plus trainable low-rank residual GNN; diffusion-weighted distribution alignment, contrast and structural/task objectives (§4.1–4.5, blocks34–80). | Structure-aware low-rank graph adaptation is prior. Separate predictive members and generated per-context member factors are not established in this scope. |
| [GaRA](https://arxiv.org/html/2606.22429v1), Sun et al. | Whole-graph GNN representation generates a low-rank factor: `W = W_hat + (L + g(h)) R^T`; generated-factor column norms are rescaled to the reference factor (§4.1–4.4, blocks36–75). | Direct graph-conditioned low-rank weight-generation prior. Whole-graph LLM adaptation differs from local structural member specialization. |
| [MoSE](https://arxiv.org/html/2509.09337v1), Ye et al. | Anonymous-walk patterns construct structural subgraphs; topology context routes to trainable hidden-graph/random-walk-kernel experts with task and balance objectives (blocks18–73). | Measurable topology-conditioned specialization is prior. This scope establishes neither low-rank strong-backbone members nor link-prediction quality transfer. |
| [Laplacian-LoRA](https://arxiv.org/html/2602.07278v1), Alisetti | Leading Laplacian eigenvectors support a learned, depth-dependent low-rank propagation correction (blocks16–52). | A structural adapter can be a single model. The described width64 GCN is not evidence for improvement on the target strong backbone. |
| [X-LoRA](https://arxiv.org/html/2402.07148v2) | A hidden-state gate mixes low-rank adapters per token/layer; one pass generates scalings and another produces logits. Default trains the gate with base/adapters frozen (§4.1–4.2.3, blocks161–184). | Direct conditional low-rank adapter-bank prior. It produces one composed trajectory; it does not establish independently retained graph-conditioned member predictions. |
| [GraphLoRA, LLM recommendation](https://arxiv.org/html/2606.07526v1) | Node structural representation enters a LoRA bottleneck as `B(lambda_lora A x + lambda_gnn z_node)` at selected user/item token anchors (§4.1–4.5, blocks31–64). | Node structural context in a low-rank pathway is prior. The printed equation injects a context signal through shared factors; it does not generate a private predictive member. |

The two GraphLoRA papers are different identities. X-LoRA and recommendation GraphLoRA were selected from the already retained discovery results as closer attribution checks after the initial four reads. Their metadata and exact versions are in `CITATIONS.json`.

## Why the member claim remains open

For the same layer state `x`, an additive bank satisfies

`sum_m a_m(c) B_m A_m x = [B_1 ... B_M] [a_1(c)A_1; ...; a_M(c)A_M] x`.

With conformable rank-r factors, the right-hand side is a single conditional adapter of rank at most `M r`. This is our algebraic observation, with a narrow scope: it does not equate training dynamics, regularization, gate work or separately evolving nonlinear member trajectories. It shows why a gain over an unconditioned smaller adapter would not identify a benefit from retaining distinct predictive members. Context access, capacity and optimization remain explanations.

Separate adapted backbone trajectories need not collapse to that layer identity, but they incur their trajectory work. Their quality claim therefore needs a capable single with the same structural information and disclosed rank/work. The reviewed methods and cached conclusions do not supply target evidence that TRAIN-only structural context predicts reproducible complementary errors beyond such a single. No concrete paired experiment is warranted by this scout now. A future finding of that error structure would make the member-context association question relevant; it would still need the same-context single and an association-breaking falsifier. This is a relevance criterion, not a run proposal.

## Prior memory and limits

Index_v40 was consulted first (168 records,119 normalized paper identities; SHA `9a451fedaaa11d5533f4255c061f49404e667f52d9c7e94629d6541861d4730e`). HG-Adapter, Self-Routed Tensor Adapters, LoRA-Ensemble, GraphMoRE, MoE-NP and CHoE conclusions were reused without primary rereads. They already cover compact structural adapters, conditional factor/core composition, private low-rank ensembles, topology routing and relation/meta-path experts. The immediately preceding coherent-completion scout also requires capable shared-context singles before inferring necessity of retained hypotheses. `REUSED_CONCLUSIONS.json` and `INPUT_BINDINGS.json` bind this evidence.

Printed ambiguities remain explicit in `PAPER_CONCLUSIONS.json`: GraphLoRA transfer's PPR/structural-loss notation, MoSE's scalar importance expression, Laplacian-LoRA's strict-contraction/stability boundary, and X-LoRA's general factor dimensions. GaRA's theorem/corollary statements were read; proofs and denominator edge cases were not qualified. None is transported as a theorem or quality guarantee. The recommendation paper's source overhead example is incidental exposure, not a current-project measurement.

Three arXiv discovery queries returned only relevance-ranked prefixes (max15 each):2/2 GraphLoRA results,15/273 graph low-rank/context results and15/841 structural-expert results. These are search coverage counts, not evidence of absence or progress. Unselected leads remain metadata-only, including newer versions of cached identities. No whole-paper certification, proof audit, author-code inspection, benchmark outcome adoption, numerical/model execution, GPU/SSH/upload, arrays/checkpoints/labels/runtime-binary access or canonical/index change occurred. The initial GaRA display was truncated; blocks49–70 were subsequently displayed completely. Exact scoped passages and navigation exposures are separately recorded.
"""
(ROOT / "REPORT.md").write_text(report)

json_files = sorted(ROOT.rglob("*.json"))
for p in json_files:
    json.loads(p.read_text())
py_files = sorted(ROOT.rglob("*.py"))
for p in py_files:
    source = p.read_text()
    ast.parse(source, filename=str(p))
    compile(source, str(p), "exec")
dump("VERIFICATION.json", {"schema": "structural_lowrank_stdlib_verification_v1", "UTC": STAMP,
                          "status": "PASSED_TEXT_METADATA_ONLY", "json_files_parsed_before_verification": len(json_files),
                          "python_files_AST_compile_checked_without_execution": [desc(p) for p in py_files],
                          "primary_receipts_bound": len(new),
                          "index_sha256_unchanged": desc(index_path, PHASE)["sha256"],
                          "scientific_execution": False})
payloads = [desc(p) for p in sorted(ROOT.rglob("*")) if p.is_file() and p.name not in {"MANIFEST.json", "SEAL.json"}]
dump("MANIFEST.json", {"schema": "structural_lowrank_scout_manifest_v1", "UTC": STAMP,
                      "status": "SEALED_SCOPED_PRIMARY_READ_SOURCE_ONLY_NO_INDEX_INTEGRATION",
                      "payload": payloads, "payload_count": len(payloads),
                      "payload_bytes": sum(p["bytes"] for p in payloads),
                      "inputs": "INPUT_BINDINGS.json", "read_scopes": "READ_SCOPES.json", "report": "REPORT.md"})
dump("SEAL.json", {"schema": "structural_lowrank_scout_seal_v1", "UTC": STAMP,
                  "status": "SCOPED_PRIMARY_SOURCE_PACKET_SEALED",
                  "manifest": desc(ROOT / "MANIFEST.json"), "report": desc(ROOT / "REPORT.md"),
                  "read_scopes": desc(ROOT / "READ_SCOPES.json"), "verification": desc(ROOT / "VERIFICATION.json"),
                  "payload_count": len(payloads), "payload_bytes": sum(p["bytes"] for p in payloads),
                  "scoped_new_papers": 6, "proposed_hypotheses_or_comparisons": 0,
                  "decision": "NO_WARRANTED_NEW_EXPERIMENT_AT_PRESENT",
                  "index_sha256_unchanged": INDEX_SHA, "scientific_execution": False,
                  "canonical_or_index_edit": False})
for p in ROOT.rglob("*"):
    if p.is_file():
        p.chmod(0o444)
print(json.dumps({"report": desc(ROOT / "REPORT.md"), "manifest": desc(ROOT / "MANIFEST.json"),
                  "seal": desc(ROOT / "SEAL.json"), "payload_count": len(payloads),
                  "payload_bytes": sum(p["bytes"] for p in payloads)}, indent=2))
