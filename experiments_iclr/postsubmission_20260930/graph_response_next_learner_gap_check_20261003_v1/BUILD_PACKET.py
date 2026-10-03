"""Build source-only custody records. No numerical libraries or experiment access."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PHASE = ROOT / "postsubmission_research_20260930"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def binding(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(data), "sha256": sha(data)}


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


inputs = [
    "literature_memory/index_v35/LITERATURE_INDEX.json",
    "graph_structural_response_gap_literature_20261003_v1/REPORT.md",
    "graph_structural_response_gap_literature_20261003_v1/THEORY_AND_DISCRIMINATING_CHECK.md",
    "graph_structural_response_gap_literature_20261003_v1/PROSPECTIVE_DIAGNOSTIC.json",
    "graph_structural_response_gap_literature_20261003_v1/PRIMARY_EXCERPTS.json",
    "conditional_graph_response_closest_priors_20261003_v1/REPORT.md",
    "conditional_graph_response_closest_priors_20261003_v1/PRIMARY_EXCERPTS.json",
    "graph_contrastive_private_paths_quality_gap_20261003_v1/CANDIDATE.json",
    "graph_contrastive_private_paths_quality_gap_20261003_v1/CITATIONS.json",
    "graph_contrastive_private_paths_quality_gap_20261003_v1/READ_SCOPES.json",
    "conditional_graph_response_dice_forde_source_recipes_20261003_v2/REPORT.md",
    "graph_conditional_response_native_source_preparation_20261003_v4/REPORT.md",
    "conditional_graph_response_pilot_proposal_20261003_v1/PROPOSAL.md",
    "heterogeneous_label_relevant_diversity_theory_20261003_v1/REPORT.md",
    "heterogeneous_label_relevant_diversity_theory_20261003_v1/PAPER_CONCLUSIONS.json",
    "heterogeneous_label_relevant_diversity_theory_20261003_v1/READ_SCOPES.json",
    "heterogeneous_label_relevant_diversity_theory_20261003_v1/primary/unified_pages.json",
    "member_subspace_messages_v1/contrastive_be_diversity_literature_v1/SOURCES_AND_EVIDENCE.json",
    "amazon_ratings_baseline_context_20261003_v1/REPORT.md",
    "amazon_ratings_baseline_context_20261003_v1/AUTHOR_HYPERPARAMETERS.json",
    "amazon_ratings_baseline_context_20261003_v1/READ_SCOPES.json",
    "amazon_ratings_baseline_context_20261003_v1/SOURCE_BINDINGS.json",
    "coordinate_source_independent_review_v1/strong_backbones_v1/sources/polynormer_code/run.sh",
]
save("INPUT_BINDINGS.json", {"schema": "bounded_saved_input_custody_v1", "inputs": [binding(PHASE / p) for p in inputs], "live_original_data_or_score_artifacts_read": []})

excerpts = []
gncl_path = PHASE / "member_subspace_messages_v1/contrastive_be_diversity_literature_v1/SOURCES_AND_EVIDENCE.json"
gncl = next(r for r in json.loads(gncl_path.read_text())["own_primary_records"] if r["name"] == "gncl")
for q in gncl["inspected_passages"]:
    excerpts.append({"source": "GNCL", "public_url": gncl["primary"]["url"], "version": gncl["version"], "saved_archive": binding(gncl_path), "locator": {"html_anchor": q["html_anchor"], "block_index": q["block_index"]}, "exact_text": q["text"], "text_sha256": sha(q["text"].encode()), "read_type": "retained_archived_primary_excerpt_reuse"})
for q in gncl["equations"]:
    excerpts.append({"source": "GNCL", "public_url": gncl["primary"]["url"], "version": gncl["version"], "saved_archive": binding(gncl_path), "locator": q["id"], "exact_math_strings": q["math"], "read_type": "retained_archived_equation_string_reuse_no_new_visual_check"})

wood_path = PHASE / "heterogeneous_label_relevant_diversity_theory_20261003_v1/primary/unified_pages.json"
wood = json.loads(wood_path.read_text())
for page, start, end in [(10, 35, 52), (14, 2, 22)]:
    txt = next(p["text"] for p in wood if p["page"] == page)
    exact = "\n".join(txt.splitlines()[start-1:end])
    excerpts.append({"source": "Wood_et_al", "public_url": "https://jmlr.org/papers/volume24/23-0041/23-0041.pdf", "version": "JMLR24(359),2023", "saved_archive": binding(wood_path), "locator": {"pdf_page": page, "extraction_line_start": start, "extraction_line_end": end}, "exact_text": exact, "text_sha256": sha(exact.encode()), "read_type": "retained_saved_page_excerpt_reuse", "equation_extraction_limit": "PDF extraction contains ligatures and garbled equation ordering. No equation is silently repaired here; report equations are attributed/local derivations."})

rep_path = PHASE / "conditional_graph_response_closest_priors_20261003_v1/PRIMARY_EXCERPTS.json"
for q in json.loads(rep_path.read_text())["passages"]:
    if q["source"] == "repulsive" and q["pdf_page"] in [4, 7]:
        excerpts.append({"source": "Repulsive_Deep_Ensembles", "public_url": "https://arxiv.org/pdf/2106.11642v3", "version": "2106.11642v3", "saved_archive": binding(rep_path), "locator": {"pdf_page": q["pdf_page"], "line_start": q["line_start"], "line_end": q["line_end"]}, "exact_text": q["extracted_text"], "text_sha256": sha(q["extracted_text"].encode()), "read_type": "retained_archived_primary_excerpt_reuse", "inherited_primary_page_binding": q["page_text_binding"]})

spectral_path = PHASE / "graph_structural_response_gap_literature_20261003_v1/PRIMARY_EXCERPTS.json"
for q in json.loads(spectral_path.read_text())["excerpts"]:
    if q["arxiv_version"] == "2511.11928v2" and q["index"] in [96, 97, 101, 174]:
        excerpts.append({"source": "Interpolated_Spectral_Augmentation", "public_url": q["primary_url"], "version": q["arxiv_version"], "saved_archive": binding(spectral_path), "locator": {"html_id": q["html_id"], "block_index": q["index"]}, "exact_text": q["text"], "text_sha256": sha(q["text"].encode()), "inherited_primary_sha256": q["primary_sha256"], "read_type": "retained_archived_primary_excerpt_reuse"})

command_path = PHASE / "coordinate_source_independent_review_v1/strong_backbones_v1/sources/polynormer_code/run.sh"
exact = command_path.read_text().splitlines()[4]
excerpts.append({"source": "Polynormer_author_Amazon_command", "public_url": "https://github.com/cornell-zhang/Polynormer/blob/fc8c276c9c5dfbd616d83f65338a3392188a5e08/run.sh#L5", "version": "fc8c276c9c5dfbd616d83f65338a3392188a5e08", "saved_archive": binding(command_path), "locator": {"source_line": 5}, "exact_text": exact, "text_sha256": sha(exact.encode()), "read_type": "retained_author_source_single_line_reuse_not_an_audit"})
save("PUBLIC_SOURCE_EXCERPTS.json", {"schema": "exact_retained_public_source_excerpts_v1", "fresh_retrievals": 0, "archival_source_notice": "These are byte-exact excerpts retained from previously retrieved primary material. This child does not certify current online availability, new whole-paper reading, equation visual QA or author-source parity.", "excerpts": excerpts})

save("READ_SCOPES.json", {
    "schema": "retained_source_and_analytic_review_scope_v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "fresh_public_queries": 0, "fresh_primary_method_scopes": 0, "new_primary_identities": 0,
    "full_paper_reads": 0, "author_code_audits": 0, "new_visual_equation_checks": 0,
    "retained_excerpt_exposure": [
        {"source": "GNCL2011.02952v2", "scope": "Four archived inspected paragraphs and Eq4/5 strings; existing method conclusion reused."},
        {"source": "Wood_JMLR23-0041", "scope": "Saved page10 extracted lines35-52 and page14 lines2-22; locator output also exposed page14 lines23-37 example prose. No full page or proof audit."},
        {"source": "Repulsive2106.11642v3", "scope": "Previously saved p4/p7 quotation blocks in retained closest-prior report/JSON; no original PDF reopened."},
        {"source": "Interpolated2511.11928v2", "scope": "Saved excerpts96/97/101/174 retained; bounded search output exposed excerpts84/157 as well. No whole method/proof reread."},
        {"source": "Polynormer_author_commit", "scope": "Previously saved run.sh line5 retained; neighboring lines4-8 displayed by bounded source excerpt read. No other author source newly inspected."}
    ],
    "saved_report_scopes": "Bound INPUT_BINDINGS reports read as saved conclusions. Corrected FoRDE/DICE formulas and native composition use source-spec reports, not newly opened author implementations.",
    "index_scope": "v35 top-level keys/accounting, selected diversity/decomposition records; automated keyword selection also exposed duplicate backbone and other matching records. Identity/record counts are not whole-paper totals.",
    "incidental_exposure_limit": "Public benchmark values and other saved-paper summaries appeared in baseline/index output; no new ranking or gate adopted. No live original-score artifact was read.",
    "parent_results": "Root-supplied message stored in ROOT_CONTEXT.json; not independently replayed by this child.",
    "scientific_execution": {"datasets": 0, "checkpoints": 0, "original_scores": 0, "model_imports": 0, "numerical_or_tensor_checks": 0, "fits": 0, "GPU": 0, "remote_execution": 0, "installation": 0, "PDF_compilation": 0, "subagents": 0},
    "permitted_local_operations": "Filesystem reading/writing, stdlib JSON selection, hashing and packet verification only."
})

save("ROOT_CONTEXT.json", {"schema": "parent_supplied_context_not_child_replay_v1", "source": "Root collaboration message during this task", "verification_by_child": False, "reported": {"source_defaults_mean_native_validation_accuracy_percent": 42.7024, "Roman_transfer_mean_native_validation_accuracy_percent": 44.0851, "Roman_accuracy_wins_all_three_blocks": True, "Roman_validation_NLL_rounded_nats": [2.437, 3.932, 3.080], "Roman_fit_accuracy_percent_range_rounded": [78.6, 83.7]}, "interpretation_limit": "Recipe winner within the bounded two-transfer comparison; no comparable 80%-TRAIN published competence threshold. No original-score or checkpoint artifact accessed."})

save("CONCLUSIONS.json", {"schema": "structural_response_utility_boundary_v1", "new_learner_loss_proposed": False, "novelty_certified": False, "beyond_FoRDE_DICE_NCL_spectral_augmentation_superiority_certified": False, "analytic_results": ["Native task risk can stay exactly invariant as partial weighted-edge responses change, for both pools in the agreement witness.", "For the one-interior-plus-complete-deletion witness, normalized centered responses remain collinear and squared-cosine D stays1 with zero b-direction gradient.", "Symmetric lazy-walk interior probes t and1-t are exact response duplicates in the degree-two witness.", "Native loss-matched ambiguity is not structural-response cosine; endpoint/pooling must be named.", "Exact binomial token probes apply only to fixed graph-independent bank plus row-local head; nonlinear graph trajectories do not inherit the shortcut."], "decision": "Move first to source-authored Amazon Polynormer-r using the already specified GNNM-versus-independent-ensemble pair; no more PolyFormer tuning or new response loss recommended.", "preserves_existing_packets": True, "launch_authority": False, "unresolved_access_overlap": ["GENN method inaccessible in prior packet", "AAAI2025 counterfactual regularization method access gap retained"]})

save("FALSIFIABLE_TEST.json", {
    "schema": "one_reused_source_backbone_pair_not_launched_v1", "status": "prospective_recommendation_only",
    "origin": binding(PHASE / "amazon_ratings_baseline_context_20261003_v1/REPORT.md"),
    "arm_count": 2, "representative_graph": "Amazon Ratings, existing80%-of-official-TRAIN adaptation",
    "arms": ["M4 boundary GNNM Polynormer-r", "M4 independent native Polynormer-r ensemble"],
    "no_extra_loss_or_tuning": True, "author_pin": "fc8c276c9c5dfbd616d83f65338a3392188a5e08",
    "recipe": json.loads((PHASE / "amazon_ratings_baseline_context_20261003_v1/AUTHOR_HYPERPARAMETERS.json").read_text())["polynormer_release"]["dataset_settings"]["amazon-ratings"],
    "blocks": [{"official_split": j, "block_seed": s, "independent_member_seeds": [s+1009*m for m in range(4)]} for j,s in [(0,17),(1,29),(2,43)]],
    "role_selection": "Preserve exact existing hash fit/control roles; native officialVAL accuracy selects checkpoints; officialTEST untouched.",
    "pooling": "Arithmetic mean member probabilities, both arms; no temperature or weighting fit.",
    "primary_contrast": "Paired TRAIN-control pooledNLL(GNNM)-pooledNLL(independent), per block and mean at their prospectively declared selected states.",
    "companions": ["control accuracy", "control Brier", "individual member NLL", "selected local/global stage", "complete paid training/preparation/serving time", "peak memory/stored parameter bytes"],
    "falsifiers": ["Independent source models remain weak: no competent-base interpretation.", "Competent independent ensemble is better and no measured cost trade-off supports sharing: source GNNM sharing claim unsupported.", "A quality/cost advantage disappears when all member graph trajectories, initialization/stage restoration and selection work are charged."],
    "conditional_followup_boundary": "Matching competent native independent quality with measured sharing savings only makes a later separately frozen structural-response comparator worthwhile. It does not establish such utility or superiority now.",
    "required_before_launch": ["source/runtime parity of saved unexecuted boundary adapter", "explicit local/global checkpoint flag custody", "exact role/split source identity", "complete measured representative resource accounting", "separate parent authorization"],
    "numeric_pass_threshold_invented": False, "existing_sealed_conditional_response_pilot_amended": False,
    "graph_response_probe_warning": "Polynormer-r requires complete graph-dependent nonlinear trajectories; fixed-bank binomial probes are not exact for its whole predictor."
})

baseline = (PHASE / "amazon_ratings_baseline_context_20261003_v1/REPORT.md").read_text()
section = baseline.split("## 4. Exactly one prospectively fixed next comparison\n", 1)[1].split("## 5. Custody and read accounting", 1)[0]
(HERE / "REUSED_PAIRED_SPEC.md").write_text("# Exact retained prospective comparison\n\nSource: amazon_ratings_baseline_context_20261003_v1/REPORT.md, section4. This is retained specification text, not launch/adoption.\n\n" + section)

save("MANIFEST.json", {"schema": "new_child_packet_manifest_v1", "created_utc": datetime.now(timezone.utc).isoformat(), "files": [binding(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in ["MANIFEST.json", "VERIFICATION.json"]], "mutation_scope": str(HERE.relative_to(ROOT)), "prior_files_written": []})
