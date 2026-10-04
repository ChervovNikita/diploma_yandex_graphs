"""Bind bounded source/literature conclusions without data, training or network."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    with (ROOT / name).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def descriptor(relative, scope, kind):
    path = PHASE / relative
    result = dict(path=relative, bytes=path.stat().st_size, sha256=digest(path), kind=kind, scope=scope)
    if isinstance(scope, list):
        lines = path.read_text().splitlines(keepends=True)
        result["excerpt_bindings"] = [dict(first_line=a, last_line=b, sha256=hashlib.sha256("".join(lines[a-1:b]).encode()).hexdigest()) for a, b in scope]
    return result


def main():
    sources = [
        descriptor("literature_memory/index_v53/LITERATURE_INDEX.json", "Selected retained conclusions and scopes; no repeated paper retrieval", "memory_reuse"),
        descriptor("amazon_ratings_baseline_context_20261003_v1/REPORT.md", "Existing complete scoped baseline report", "conclusion_reuse"),
        descriptor("amazon_ratings_baseline_context_20261003_v1/AUTHOR_HYPERPARAMETERS.json", "Authored recipe and paper/release separation", "conclusion_reuse"),
        descriptor("accuracy_first_graph_view_reference_source_preparation_20261004_v2/PROTOCOL.json", "Ordinary reference recipe, all-TRAIN role, selector and split contract", "source_protocol_reuse"),
        descriptor("accuracy_first_native_gnnm_reference_source_preparation_20261004_v2/PROTOCOL.json", "Native GNNM objective/exposure differences and full-TRAIN role", "source_protocol_reuse"),
        descriptor("coordinate_source_independent_review_v1/strong_backbones_v1/sources/polynormer_code/run.sh", [[4, 8]], "previously_scoped_author_source_reinspection"),
        descriptor("coordinate_source_independent_review_v1/strong_backbones_v1/sources/polynormer_code/parse.py", "Entire; already scoped defaults", "previously_scoped_author_source_reinspection"),
        descriptor("coordinate_source_independent_review_v1/strong_backbones_v1/sources/polynormer_code/main.py", [[24, 145]], "previously_scoped_author_source_reinspection"),
        descriptor("amazon_ratings_baseline_context_20261003_v1/sources/yandex_code/train.py", [[1, 150]], "previously_scoped_author_source_reinspection"),
        descriptor("amazon_ratings_baseline_context_20261003_v1/sources/yandex_code/model.py", "Entire; small native model wrapper", "retained_author_source_inspection"),
        descriptor("amazon_ratings_baseline_context_20261003_v1/sources/yandex_code/datasets.py", [[1, 73]], "previously_scoped_author_source_reinspection"),
        descriptor("amazon_ratings_baseline_context_20261003_v1/sources/yandex_code/scripts/run_all_experiments.sh", [[59, 105]], "previously_scoped_author_source_reinspection"),
        descriptor("amazon_accuracy_calibration_scout_20261004_v1/source/modules.py", [[1, 120]], "new_author_source_scope_at_retained_paper_identity"),
        descriptor("amazon_accuracy_calibration_scout_20261004_v1/source/utils.py", "Entire; exact selector, optimizer groups and no-warmup schedule", "new_author_source_scope_at_retained_paper_identity"),
        descriptor("coordinate_source_independent_review_v1/strong_backbones_v1/sources/pitfalls_paper.txt", [[598, 629], [900, 934], [1206, 1246], [1490, 1523], [4773, 4788], [4855, 4880]], "retained_primary_incremental_Amazon_table_scope"),
        descriptor("amazon_accuracy_calibration_scout_20261004_v1/primary/feature_sensitivity_v1.txt", [[110, 131], [235, 488], [662, 797], [992, 1011], [3419, 3542]], "new_primary_scoped_method_protocol_Amazon_result_not_full_paper"),
    ]
    write("SOURCE_BINDINGS.json", sources)
    records = [
        dict(canonical_id="arxiv:2409.05755", version="v3", verified_title="Re-evaluating the Advancements of Heterophilic Graph Learning", primary_url="https://arxiv.org/html/2409.05755v3", source_file=sources[-2]["path"], source_sha256=sources[-2]["sha256"], read_kind="retained_primary_incremental_Amazon_tables", full_paper_read=False, exact_read_scope=sources[-2]["scope"], saved_takeaway="The exact Amazon table confirms substantial validation-tuning sensitivity for simple GCN/MLP controls; the larger published search budget and TEST endpoints are not competence gates for three selected-VALIDATION development blocks.", new_identity=False, numeric_project_results_opened=False),
        dict(canonical_id="arxiv:2609.33764", versioned_id="2609.33764v1", version_date="2026-09-27", verified_title="Beyond Fixed Features: Architecture-Dependent Sensitivity to Node Representations under Heterophily", authors=["Priyanath Maji", "Sidharth Gaur", "Rajavinoth Paul Durai"], primary_url="https://arxiv.org/html/2609.33764v1", source_file=sources[-1]["path"], source_sha256=sources[-1]["sha256"], read_kind="new_scoped_primary_protocol_method_Amazon_results", full_paper_read=False, exact_read_scope=sources[-1]["scope"], saved_takeaway="Architecture and input-representation effects interact; reconstructed fastText/SBERT/MPNet graphs are evaluated primarily by Macro F1 with independently tuned architectures. They do not establish a stronger original-raw-feature Amazon accuracy baseline or GNNM gain.", disposition="Retain as a later representation-control lead; do not change current comparison or claim novelty", public_source_urls=["https://github.com/priyanathmaji/graphdatasets", "https://github.com/priyanathmaji/neighborsGNN"], author_source_inspected=False, dataset_or_mask_bytes_verified=False, numeric_results_reproduced=False, new_identity=True, limits=["Preprint, no verified conference acceptance", "No graph/mask byte equivalence established", "Macro F1 selector differs from accuracy", "Richer dimensions/pretraining and independent tuning confound a direct benchmark comparison"]),
    ]
    write("PAPER_CONCLUSIONS.json", dict(schema="amazon-accuracy-calibration-scoped-conclusions-v1", paper_records=records))
    write("READ_ACCOUNTING.json", dict(new_paper_scoped_reads=1, retained_paper_incremental_scopes=1, new_primary_paper_scopes_total=2, full_paper_reads=0, latest_metadata_queries=4, new_author_source_scopes_retained_identity=2, discovery_only_not_paper_reads=True, project_data_or_scores_or_checkpoints_opened=False, scientific_execution=False, canonical_literature_index_modified=False, failed_requirement_retrieval_preserved=True))
    write("CALIBRATION_PROPOSAL.json", dict(status="PROPOSAL_PARENT_DECISION_REQUIRED_NO_EXECUTION", anchor="Polynormer-r author Amazon release", same_protocol_native_fits="Use exact complete member0 all-TRAIN ordinary fits already in proposed 30-fit comparison", optional_reference="Pinned Yandex residual GraphSAGE", optional_depths=[1,2,3,4,5], optional_split0_depth_selection="Best VALIDATION correct count; earliest update within fit, shallower depth for ties", optional_additional_physical_fits=7, optional_schedule="1000 updates, fixed source defaults; no broad optimizer search", competence_interpretation="If mean same-protocol native Polynormer accuracy is below the source-qualified residual SAGE, do not claim a competent strong-modern-single comparison without investigating the deficiency", published_TEST_threshold=None, new_TEST_reader=False, original_30_fit_budget_changed=False, source_or_CUDA_qualification_claimed=False, predictive_quality_claimed=False, SOTA_or_novelty_claimed=False, no_manuscript_or_canonical_ledger_edits=True))
    for record in sources:
        assert digest(PHASE / record["path"]) == record["sha256"]
    retrievals=json.loads((ROOT / "source/RETRIEVAL.json").read_text())
    for entry in retrievals:
        if entry["status"] == "RETRIEVED":
            assert digest(ROOT / "source" / entry["path"]) == entry["sha256"]
    write("VERIFICATION.json", dict(UTC=datetime.now(timezone.utc).isoformat(), status="SCOPES_AND_CUSTODY_VERIFIED", external_source_bindings_unchanged=True, source_bindings=len(sources), manuscript_or_ledger_or_original_scores_changed=False, scientific_execution=False, predictive_data_access=False))
    files=[]
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name not in ("MANIFEST.json", "SEAL.json") and "__pycache__" not in path.parts:
            assert not path.is_symlink()
            files.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest(path)))
    write("MANIFEST.json", dict(schema="amazon-accuracy-calibration-scout-manifest-v1", files=files))
    write("SEAL.json", dict(schema="amazon-accuracy-calibration-scout-seal-v1", manifest_sha256=digest(ROOT / "MANIFEST.json"), immutable=True))
    for record in files:
        assert digest(ROOT / record["path"]) == record["sha256"]
    print(json.dumps(dict(status="SEALED_AND_VERIFIED", manifest_sha256=digest(ROOT / "MANIFEST.json"), seal_sha256=digest(ROOT / "SEAL.json"), paper_conclusions_sha256=digest(ROOT / "PAPER_CONCLUSIONS.json"), files=len(files))))


if __name__ == "__main__":
    main()
