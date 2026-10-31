"""Local text/provenance sealing only; never imports models or scientific libraries."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def write(name, obj):
    p = HERE / name
    if p.exists():
        raise RuntimeError(f'Never replace sealed/provenance output: {p}')
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def main():
    index_path = PHASE / 'literature_memory/index_v19/LITERATURE_INDEX.json'
    index_sha = sha(index_path)
    assert index_sha == '831d760fb3bbe12fb88203a51b251398fb50eb3263938bf117c604ac4a84d109'
    index = read(index_path)
    retained_ids = {g['normalized_identifier'] for g in index['canonical_identifier_normalization']['groups']}
    scopes = read(HERE / 'READ_SCOPES.json')
    assert scopes['new_scoped_primary_method_reads'] == scopes['new_primary_limit'] == 2
    assert scopes['new_full_primary_reads'] == scopes['retained_primary_rereads'] == 0
    assert len(scopes['papers']) == 2
    assert sum(s['unique_scoped_blocks'] for s in scopes['papers']) == 102
    for s in scopes['papers']:
        assert s['canonical_id'] not in retained_ids
        assert sha(HERE / s['blocks_file']) == s['blocks_sha256']
        key = 'sea' if s['canonical_id'] == 'arxiv:2508.04948' else 'sigma'
        assert sha(HERE / 'primary' / f'{key}_v1.html') == s['primary_html_sha256']
        assert ('arXiv:' + s['versioned_id']) in (HERE / 'primary' / f'{key}_v1.html').read_text()

    reuse = read(HERE / 'REUSED_CONCLUSIONS.json')
    assert reuse['index_sha256'] == index_sha and reuse['retained_primary_rereads'] == 0
    reused_files = {}
    for r in reuse['records']:
        p = PHASE / r['conclusion_file']
        assert sha(p) == r['conclusion_file_sha256']
        reused_files[r['conclusion_file']] = {'path': r['conclusion_file'],
            'sha256': sha(p), 'bytes': p.stat().st_size,
            'inspection': 'Only saved index conclusion content reused; no primary file opened'}
    inputs = [{'path': str(index_path.relative_to(PHASE)), 'sha256': index_sha,
               'bytes': index_path.stat().st_size, 'inspection': 'Index first, normalized identities/scopes/catalog/prior saved conclusions'},
              *reused_files.values()]
    for rel in ['graph_init_local_prediction_objective_review_v1/DERIVATION.md',
                'graph_init_local_prediction_objective_review_v1/DIAGNOSTIC_PLAN.md',
                'graph_init_local_prediction_objective_review_v1/MANIFEST.json',
                'graph_init_local_prediction_objective_review_v1/SEAL.json']:
        p = PHASE / rel
        inputs.append({'path': rel, 'sha256': sha(p), 'bytes': p.stat().st_size,
                       'inspection': 'Retained source-bound mathematical interpretation reused; no new primary read'})
    math_manifest = PHASE / 'graph_init_local_prediction_objective_review_v1/MANIFEST.json'
    assert sha(math_manifest) == '4a0b95253fa73bb06442a836e08b4650d267103dfc59a72144748b7dfb60376b'
    write('INPUT_BINDINGS.json', {'schema': 'graph-error-specialization-inputs-v1',
        'created_UTC': datetime.now(timezone.utc).isoformat(), 'bindings': inputs,
        'index_consulted_before_acquisition': True,
        'outcomes_labels_arrays_checkpoints_read': False,
        'registered_running_sources_or_protocols_modified': False})

    discovery = read(HERE / 'DISCOVERY_METADATA_run01.json') + read(HERE / 'DISCOVERY_METADATA_run02.json')
    selected = {s['canonical_id'] for s in scopes['papers']}
    dispositions = []
    for r in discovery:
        match = re.search(r'arxiv.org/abs/([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?', r['id'])
        canonical = 'arxiv:' + match.group(1) if match else None
        if canonical in selected:
            status = 'Selected exact primary version for one of two new scoped method reads'
        elif canonical in retained_ids:
            status = 'Retained identity: metadata screened only; primary not fetched or reread'
        elif any(k in r['title'].lower() for k in ['ensemble', 'multi-head', 'diversity', 'batchensemble', 'neurotrails']):
            status = 'Abstract/metadata-only screen; not selected within two-paper budget; no method takeaway or novelty absence inferred'
        else:
            status = 'Irrelevant/broad application or survey result: metadata only; no method read'
        dispositions.append({'query': r['query'], 'id': r['id'], 'canonical_id_if_arxiv': canonical,
                             'title': r['title'], 'disposition': status})
    write('DISCOVERY_DISPOSITIONS.json', {'schema': 'graph-error-specialization-discovery-v1',
        'arxiv_metadata_queries': 6, 'openalex_queries': 1, 'entries': len(discovery),
        'unique_arxiv_identifiers': len({x['canonical_id_if_arxiv'] for x in dispositions if x['canonical_id_if_arxiv']}),
        'exhaustive_search_claim': False, 'records': dispositions})

    paper_records = [
        {'canonical_id': 'arxiv:2508.04948', 'versioned_id': '2508.04948v1',
         'verified_title': scopes['papers'][0]['title'], 'authors': scopes['papers'][0]['authors'],
         'version_date': '2025-08-07', 'primary_url': scopes['papers'][0]['primary_url'],
         'primary_html_sha256': scopes['papers'][0]['primary_html_sha256'],
         'read_kind': 'new scoped primary method read', 'full_read': False,
         'exact_read_scope': 'READ_SCOPES.json:arxiv:2508.04948, blocks18-42 and147-160',
         'saved_takeaway': 'Squared-error ensemble self-error/complementary-target adjustment is direct quality/diversity prior. The inspected classification recipe uses one-hot regression. Dropping other-member constants requires separate-coordinate treatment; shared-body gradients need explicit target/detachment semantics.',
         'relationship': 'Motivates task-error directions rather than arbitrary spread; does not prove a CE/shared-factor graph refresh or validate its cost.',
         'limits': ['No full boundary/proof audit', 'Complete Algorithm1 listing not inspected; only caption/equations', 'Author source not inspected', 'No shared-factor/graph execution or results reproduction'],
         'disposition': 'Attribute complementary error-directed objectives; do not directly transfer regression bounds or the literal printed loss to shared CE models'},
        {'canonical_id': 'arxiv:2607.23860', 'versioned_id': '2607.23860v1',
         'verified_title': scopes['papers'][1]['title'], 'authors': scopes['papers'][1]['authors'],
         'version_date': '2026-07-26', 'primary_url': scopes['papers'][1]['primary_url'],
         'primary_html_sha256': scopes['papers'][1]['primary_html_sha256'],
         'read_kind': 'new scoped primary method read', 'full_read': False,
         'exact_read_scope': 'READ_SCOPES.json:arxiv:2607.23860, blocks19-50,52-59,108-130',
         'saved_takeaway': 'Shared-backbone per-member normalization with bounded sigmoid scales and a continuing temperature-controlled log-softmax parameter regularizer is direct persistence prior. Each member still executes a complete forward; parameter/predictive-diversity correspondence is empirical.',
         'relationship': 'Persistence and compact modulation are not novel. A current graph-error-cotangent/private-slice pulse differs from generic parameter-importance repulsion, but global novelty/utility remain unestablished.',
         'limits': ['No full proof/result audit or author source read', 'Vision/text rather than graph qualification', 'Positive-scale conversion and reinitialized heads are not a drop-in preserved GNNM warm function', 'Author-reported compute/uncertainty claims do not transfer', 'Finite temperature does not eliminate all owner corners; joint allocation is empirically task balanced'],
         'disposition': 'Nearest continuous shared-representation diversification prior; no parameter sharing implies saved whole-member compute or calibrated graph uncertainty'},
    ]
    write('PAPER_CONCLUSIONS.json', {'schema': 'graph-error-specialization-primary-conclusions-v1',
        'paper_records': paper_records,
        'read_accounting': {'new_scoped_primary_method_reads': 2, 'new_full_primary_reads': 0,
            'retained_primary_rereads': 0, 'author_source_reads_or_execution': 0,
            'cumulative_project_read_totals_certified': False},
        'decision': 'One conditional, prospective graph-error refresh screen survives; no new learner/theory/calibration/utility claim or scientific admission'})

    write('CANDIDATE_SPEC.json', {'schema': 'graph-error-specialization-prospective-pulse-v1',
        'status': 'ANALYSIS_ONLY_NOT_IMPLEMENTED_OR_ADMITTED',
        'trigger': 'Consider only after whole running cohort closes and the existing orientation diagnostic shows a meaningful finite signal whose persistence is worth testing; do not assume those outcomes.',
        'population': 'One already-bound PolyFormer-Mono/Squirrel cfg0 family, 512-dimensional active private slice; three fresh seeds; previously exposed and exploratory.',
        'prelude': 'Fresh declared identical warm/ordinary-continuation prelude per seed; pulse at fixed continuation update200 as a proposed constant, not validation-selected. No inheritance of current-study fitted state.',
        'arms': ['common_private_descent', 'common_plus_original_graph_contrast', 'common_plus_node_permuted_graph_contrast'],
        'signal': 'q_m=(E^T H_m E-I/4)r_pool, current pooled TRAIN CE cotangent, detached; h_m=J_m^T q_m on actual route closure.',
        'joint_projection': 'Center block geometry, remove centered route-own member-gradient columns and centered pooled-gradient column with a 5x5 Gram/pseudoinverse. Predeclare rank tolerance/residual checks and retain failure; no independent two-pass projection.',
        'common_direction_and_radius': 'v=-mean_m g_m; rho=0.5||mean_m g_m||; zero/degenerate cases retain unchanged pair. All candidates use the same radius and common direction.',
        'line_search': {'trials': 6, 'alpha0': '0.01*sqrt(d)/max_candidate_route_direction_norm', 'backtrack_factor': 2,
            'paired_shared_alpha': True, 'member_mean_gate': 'Mean member TRAIN CE <= baseline-1e-4*alpha*||gbar||^2',
            'pool_gate': 'Pooled TRAIN CE <= its own previous pooled baseline',
            'individual_members': 'All finite, report each CE change; aggregate gate is not an individual-improvement guarantee',
            'failed_pair': 'Retain unchanged pair and reason; no resampling, hidden common fallback search or scientific retry'},
        'finite_realization_metric': 'Signed original-q prediction change of graph candidate relative to common-only candidate at the same alpha; incremental JVP/finite geometry and all algebra defects recorded.',
        'continuation_contract': 'Shared dense parameters ordinary mean-member CE; active refresh only private slice; exact outside state frozen during pulse; named Adam state preserved across manual displacement, no reset/equivalence claim; matched saved/restored RNG; same native continuation/validation selector.',
        'proposed_practical_screen': 'Mean later pooled NLL improvement >=0.01nats over BOTH controls, mean accuracy decline <=0.5percentage points; constants require freeze before follow-up outcomes, not significance/coverage inference.',
        'falsifiers': ['No meaningful finite orientation to retain in post-closure diagnostic', 'Divergent closure/joint projection cannot be qualified cheaply', 'Frequent paired finite gate/rank failures', 'Finite extra signed graph target not realized', 'No practically favorable later pooled-quality effect over both controls', 'Whole paid operation cost defeats the small extension'],
        'cost_bound': {'refresh_output_forwards': 4, 'route_vjp_primals': 4, 'vjp_pullbacks': 16,
            'incremental_jvp_calls': 8, 'sparse_products_two_topologies': 6,
            'maximum_paired_trial_route_forwards': 72,
            'additional_charges': ['Closure/AD qualification','Installation checks','Clone/AD-tape peaks','Normalization/permutation/storage','Fresh warm/prelude','Rejected trials','All continuation/serving/member work','Optimizer states'],
            'runtime_or_peak_qualified': False},
        'attribution_limit': 'Equal parameter radius/shared alpha is not equal predictive Gram; comparator tests complete graph-conditioned pulse, not isolated output-orientation advantage.',
        'repeat_refreshes': 'Do not escalate from one pulse without its utility/cost evidence; no persistent schedule frozen or authorized'})

    retrieval = []
    for name in ['DISCOVERY_RETRIEVAL_run01.json', 'RETRIEVAL_run02.json', 'PRIMARY_RETRIEVAL_run03.json']:
        retrieval.extend(read(HERE / name))
    for r in retrieval:
        if 'sha256' in r:
            p = HERE / r['file']
            assert p.stat().st_size == r['bytes'] and sha(p) == r['sha256']
    assert all(r['success'] for r in retrieval)
    assert sha(index_path) == index_sha
    for x in inputs:
        p = PHASE / x['path']
        assert sha(p) == x['sha256'] and p.stat().st_size == x['bytes']
    write('VERIFICATION.json', {'schema': 'graph-error-specialization-static-verification-v1',
        'public_retrieval_bodies_verified': len(retrieval), 'http_failures': 0,
        'new_primary_dedup_verified_against_index_v19': 2,
        'exact_version_markers_verified': 2, 'unique_scoped_blocks_bound': 102,
        'retained_input_files_rechecked_unchanged': len(inputs),
        'local_parser_failures_recorded': 'LOCAL_EXTRACTION_FAILURES.json; no source absence/novelty inference',
        'primary_listing_scope_limit': 'SEA Algorithm1 caption/surrounding equations only; no complete algorithm/source audit',
        'models_data_labels_checkpoints_outcomes_remote_gpu_access': False,
        'scientific_or_analytic_numeric_experiments_run': False,
        'status_ledger_index_manuscript_changes': False,
        'scope': 'Static document/JSON/hash verification only, not execution/utility/theorem qualification'})
    payload = [{'path': str(p.relative_to(HERE)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(HERE.rglob('*')) if p.is_file() and p.name not in ['MANIFEST.json', 'SEAL.json']]
    write('MANIFEST.json', {'schema': 'graph-error-specialization-gap-search-manifest-v1',
        'created_UTC': datetime.now(timezone.utc).isoformat(), 'payload': payload,
        'bounded_literature_and_mathematical_analysis_only': True})
    write('SEAL.json', {'schema': 'graph-error-specialization-gap-search-seal-v1',
        'manifest_sha256': sha(HERE / 'MANIFEST.json'), 'payload_count': len(payload),
        'report_sha256': sha(HERE / 'REPORT.md'), 'refresh_analysis_sha256': sha(HERE / 'REFRESH_ANALYSIS.md'),
        'new_scoped_primary_method_reads': 2, 'new_full_primary_reads': 0,
        'retained_primary_rereads': 0, 'execution_authorized': False,
        'decision': 'Conditional one-pulse graph-error refresh falsifier survives; no persistent scheme, new learner/theory or utility admission'})
    print(json.dumps({'packet': str(HERE), 'manifest_sha256': sha(HERE / 'MANIFEST.json'),
                      'payload_count': len(payload), 'new_scoped_primary_reads': 2,
                      'new_full_reads': 0, 'retained_primary_rereads': 0,
                      'report_sha256': sha(HERE / 'REPORT.md')}))


if __name__ == '__main__':
    main()
