"""Local stdlib review metadata only. No scientific source imports or execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
R17 = PHASE / 'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v3_precision'
MODERN = PHASE / 'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3'
WRAPPER = PHASE / 'graph_init_precision_execution_root_v2'
LIT = PHASE / 'graph_initializer_cache_literature_followup_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(name, obj):
    path = HERE / name
    if path.exists():
        raise RuntimeError(f'Review metadata already exists: {path}')
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def record(path, inspection):
    return {'path': str(path.relative_to(PHASE)), 'bytes': path.stat().st_size,
            'sha256': sha(path), 'inspection': inspection}


def main():
    registry_path = WRAPPER / 'study_v2/GRAPH_INIT_ATTEMPT_REGISTRY.json'
    registry = read(registry_path)
    assert registry['study_id'] == 'graph_init_cfg0_outcome_aware_precision_v2'
    assert registry['anchor_directory'].endswith('/graph_init_precision_execution_root_v2/study_v2')
    for key, filename in [('source_bindings', 'SOURCE_BINDINGS.json'), ('protocol', 'PROTOCOL.json')]:
        assert registry[key]['sha256'] == sha(R17 / filename)
        assert registry[key]['path'].endswith('/round17_graph_init_driver_integration_v3_precision/' + filename)
    contexts = [{k: c[k] for k in ['graph', 'backbone', 'seed', 'source_split_index', 'config']}
                for c in registry['contexts']]
    assert len(contexts) == 6

    inspected = [
        (registry_path, 'Registered study, anchor, source/protocol descriptors and six context identity fields only; no attempt/cell outcome fields inspected'),
        (WRAPPER / 'SEAL.json', 'Source seal read'),
        (WRAPPER / 'MANIFEST.json', 'Manifest hash and payload descriptors only'),
        (WRAPPER / 'DEPLOYMENT_DEPENDENCIES.json', 'Registered sealed source dependencies; no deployment execution'),
        (WRAPPER / 'ROOT_HANDOFF.md', 'Wrapper/source lineage scope and unchanged method metadata read'),
        (WRAPPER / 'admitted_v1/CONTEXTS.json', 'Context identities and registered source bindings; first context inspected plus summaries of registered six rows'),
        (WRAPPER / 'graph_init_root_entry.py', 'Full wrapper entry source: binding to exact round17 driver/source'),
        (R17 / 'SEAL.json', 'Manifest binding fields'),
        (R17 / 'MANIFEST.json', 'Manifest hash/payload descriptor verification'),
        (R17 / 'SOURCE_BINDINGS.json', 'Full registered source binding read'),
        (R17 / 'PROTOCOL.json', 'Full registered narrow protocol read; source-state prose is not a current-runtime status inference'),
        (R17 / 'prototype/graph_band_route_initializer.py', 'Lines 1-405: method, graph masking, constrained direction, diagnostics and qualification source'),
        (R17 / 'prototype/graph_init_driver.py', 'Targeted source search plus lines 635-851; no compare/report execution or outputs'),
        (R17 / 'prototype/graph_init_training_adapter.py', 'Targeted source search plus lines 368-503: actual train/eval/continuation and preserved timepoints'),
        (MODERN / 'SEAL.json', 'Manifest binding fields'),
        (MODERN / 'MANIFEST.json', 'Manifest hash/payload descriptor verification'),
        (MODERN / 'prototype/backbone_boundary_adapter.py', 'Lines 1-155: shared/private algebra, whole trajectories, native warm clone'),
        (MODERN / 'prototype/native_polyformer.py', 'Targeted source search: activation/dropout/forward definitions; smoothness caveat'),
        (MODERN / 'prototype/native_polynormer.py', 'Targeted source search: activation/dropout/forward definitions; smoothness caveat'),
        (PHASE / 'graph_init_mechanism_analysis_root_v1/ANALYSIS_v2.md', 'Full saved mathematical analysis read; reused no primary read'),
        (LIT / 'REPORT.md', 'Full saved initializer/cache review read before further acquisition; no acquisition performed'),
        (LIT / 'INPUT_BINDINGS.json', 'Saved provenance inspected; no referenced datasets/checkpoints/labels opened'),
        (LIT / 'PAPER_CONCLUSIONS.json', 'Full saved two-primary conclusions read; no primary reread'),
        (LIT / 'REUSED_CONCLUSIONS.json', 'Lines 1-220: inherited BatchEnsemble conclusion scopes; broader prior boundaries reused from REPORT.md'),
        (LIT / 'MANIFEST.json', 'Hash only; packet unchanged'),
        (LIT / 'SEAL.json', 'Hash only; packet unchanged'),
    ]
    bindings = [record(p, s) for p, s in inspected]

    seal_checks = []
    for root in [WRAPPER, R17, MODERN]:
        manifest = read(root / 'MANIFEST.json')
        seal = read(root / 'SEAL.json')
        assert seal['manifest_sha256'] == sha(root / 'MANIFEST.json')
        payload = {x['path']: x for x in manifest['payload']}
        checked = []
        for path, _ in inspected:
            if path.is_relative_to(root):
                rel = str(path.relative_to(root))
                if rel in payload:
                    expected = payload[rel]
                    assert sha(path) == expected['sha256'], rel
                    if 'bytes' in expected:
                        assert path.stat().st_size == expected['bytes'], rel
                    checked.append(rel)
        seal_checks.append({'root': str(root.relative_to(PHASE)),
                            'manifest_sha256': sha(root / 'MANIFEST.json'),
                            'manifest_to_seal_matches': True,
                            'inspected_payloads_verified': checked,
                            'scope': 'Only statically inspected source/text payloads verified; not a scientific gate'})

    source_bindings = read(R17 / 'SOURCE_BINDINGS.json')
    for rel, expected in source_bindings['local_implementation_sha256'].items():
        assert sha(R17 / rel) == expected
    for filename in ['native_polyformer.py', 'native_polynormer.py', 'backbone_boundary_adapter.py']:
        assert sha(MODERN / 'prototype' / filename) == source_bindings['modern_model_sha256'][filename]
    assert sha(MODERN / 'MANIFEST.json') == source_bindings['modern_packet']['manifest_sha256']
    assert sha(MODERN / 'SEAL.json') == source_bindings['modern_packet']['seal_sha256']

    write('INPUT_BINDINGS.json', {'schema': 'graph-init-local-prediction-review-inputs-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'registry_identity': {k: registry[k] for k in ['schema', 'study_id', 'anchor_directory', 'source_bindings', 'protocol']},
        'registered_context_identity_excerpt': contexts,
        'bindings': bindings, 'seal_checks': seal_checks,
        'registered_method_source_hashes_match': True,
        'no_outcome_payload_inspected': True, 'no_label_array_checkpoint_access': True})

    method = str((R17 / 'prototype/graph_band_route_initializer.py').relative_to(PHASE))
    driver = str((R17 / 'prototype/graph_init_driver.py').relative_to(PHASE))
    adapter = str((R17 / 'prototype/graph_init_training_adapter.py').relative_to(PHASE))
    boundary = str((MODERN / 'prototype/backbone_boundary_adapter.py').relative_to(PHASE))
    sources = [
        {'key': 'S1', 'path': str(registry_path.relative_to(PHASE)), 'scope': 'Registered study/source/context identities only', 'facts': ['Exact study_v2 identity', 'Registered round17 v3 bindings', 'Six context identity rows']},
        {'key': 'S2', 'path': str((R17 / 'PROTOCOL.json').relative_to(PHASE)), 'scope': 'Full source protocol', 'facts': ['Five arms, slices and schedules', 'Mean member CE and mean raw-logit primary pool', 'Complete-operation and outcome-aware limitations']},
        {'key': 'S3', 'path': method, 'lines': [28, 111], 'facts': ['512/1024 private scale slices', 'Frozen deterministic whole-predictor closure', 'Identity factors and copied warm context']},
        {'key': 'S4', 'path': method, 'lines': [114, 176], 'facts': ['Simple symmetric normalized topology without added self loops', 'Four overlapping cubic Bernstein bands', 'Node alignment-null topology permutation']},
        {'key': 'S5', 'path': method, 'lines': [212, 253], 'facts': ['Detached CE cotangent normalized by TRAIN size', 'Full-graph propagation then TRAIN remasking before VJP', 'Band sum and gradient sum checks']},
        {'key': 'S6', 'path': method, 'lines': [254, 350], 'facts': ['Zero-gradient return', 'Repeated Euclidean g projection and route centering', 'Common radius rescaling without min(1,lambda)', 'Parameter-norm random control', 'JVP vectors discarded after six pair summaries', 'Six attempts and common/unchanged fallbacks']},
        {'key': 'S7', 'path': driver, 'lines': [635, 807], 'facts': ['Actual predictive closure selects complete K1 member output', 'Actual-warm AD/Adam qualification', 'Saved slices, initialized logits and scalar diagnostics', 'Exact post-warm RNG restoration']},
        {'key': 'S8', 'paths': [adapter, boundary], 'line_scopes': [[368, 499], [15, 152]], 'facts': ['Whole member maps linear(x*R,W)*S+B', 'Mean member CE with independent dropout', 'Per-update scalar traces', 'Saved native midpoint and selected prediction arrays only']},
        {'key': 'S9', 'path': 'graph_init_mechanism_analysis_root_v1/ANALYSIS_v2.md', 'scope': 'Full retained mathematical analysis', 'facts': ['Existing local descent and same-alpha cancellation', 'Jensen limitations; no new theorem credit']},
        {'key': 'S10', 'paths': ['graph_initializer_cache_literature_followup_v1/REPORT.md', 'graph_initializer_cache_literature_followup_v1/PAPER_CONCLUSIONS.json', 'graph_initializer_cache_literature_followup_v1/REUSED_CONCLUSIONS.json'], 'scope': 'Saved report/conclusions; REUSED_CONCLUSIONS lines 1-220', 'facts': ['Graph residual/Bernstein and factor/warm ancestry retained', 'Generic frozen-Jacobian construction and output scaling are prior', 'No new primary acquisition or reread']},
    ]
    for item in sources:
        paths = item.get('paths', [item.get('path')])
        item['document_sha256'] = {p: sha(PHASE / p) for p in paths}
    write('SOURCE_MAP.json', {'schema': 'graph-init-local-prediction-review-source-map-v1', 'sources': sources})

    records = [
        ('masked_graph_cotangent', 'The running method uses h_b=J_T^T(E_T^T H_b E_T)r_T; propagated errors are remasked to TRAIN before VJP.', ['S4', 'S5'], ['Exact masking/normalization and fixed warm CE cotangent'], 'This is not whole-node pseudo-label optimization or disjoint spectral-route prediction.'),
        ('constrained_linear_prediction_objective', 'For g nonzero and a resolved nonzero projected graph gradient, u_b=-rho*a_b/||a||_F solves min sum_b q_b^T Jbar u_b under route centering, g-orthogonality and joint parameter radius rho=0.5||g||.', ['S5', 'S6'], ['Ideal arithmetic', 'Detached q_b', 'Normalization floor inactive'], 'Ordinary projected linear optimization; not major theory, not guaranteed task utility.'),
        ('prediction_metric', 'The equivalent realizable prediction-space constraint is sum_b ||Jbar^+ Delta_b||^2<=rho^2 with Delta in range(Jbar), route sum zero and r^T Delta_b=0.', ['S3', 'S5', 'S6'], ['Euclidean private-factor metric', 'Minimum-norm representatives'], 'Parameter matching does not match output geometry; Jacobian/output-scale construction is prior.'),
        ('local_remainders', 'Identical first-order route CE descent and same-alpha pooled cancellation hold conditionally; DERIVATION equations 3-6 give Lipschitz/Hessian remainder bounds and numerical defects.', ['S6', 'S9'], ['Differentiable frozen model; stronger bounds require smooth neighborhood', 'Finite arithmetic is tolerance-bounded'], 'ReLU boundaries and finite six-trial failure preclude unconditional finite or utility guarantees.'),
        ('finite_orientation', 'At a nondegenerate accepted graph candidate, fixed warm graph contrast Psi(alpha) has first coefficient -rho||a||_F; finite signed response can be measured later from saved outputs.', ['S5', 'S6', 'S7'], ['Same fixed q and route/band order', 'No unknown-Hessian bound treated as measured'], 'Finite sign reversal refutes finite realization, not the conditional derivative formula.'),
        ('generic_prior_boundary', 'Graph-error-conditioned orientation and finite private-factor installation differ from an additive frozen random JVP, but graph filters, factors, warm copying, Jacobian ensembles and output scaling are established ingredients.', ['S10'], ['Retained scoped primary conclusions only'], 'No global novelty, posterior, uncertainty or superiority claim supported; no new primary read essential.'),
        ('saved_jvp_gram', 'Six pair-RMS values recover a route-centered 4x4 JVP Gram via -0.5 H4 D^2 H4; the original JVP tensors and orientation are not saved.', ['S6', 'S7'], ['Correct fixed pair order and RMS normalization'], 'Gram agreement cannot establish graph-error sign or held-out utility.'),
        ('postclosure_signed_diagnostic', 'After whole-cohort closure/admission, saved warm/initialized logits plus existing source TRAIN pack/topology can reconstruct q and assess Psi; no new forward/JVP or final labels required.', ['S5', 'S6', 'S7'], ['Registered artifact custody and separate root analysis admission', 'No diagnostic computed here'], 'Recorded lambda/g2 provide only an ideal first-order scalar prediction; thresholds cannot be inferred from unseen outcomes.'),
        ('continuation_retention', 'Fixed native-midpoint/selected logits permit a fixed warm-q orientation-retention diagnostic; only scalar losses are saved per update, and initial constraints are not maintained under continuation.', ['S7', 'S8'], ['No member relabeling, no missing midpoint imputation'], 'Persistence/selected results are descriptive and exploratory, not independent causal confirmation.'),
        ('utility_unresolved', 'Complete-operation utility remains an empirical full-cohort question with all branches/costs retained; the current random control does not isolate orientation at matched JVP Gram and alpha.', ['S2', 'S6', 'S10'], ['Outcome-aware previously exposed cohort', 'No cell outcomes inspected'], 'No additional arm, theory-based superiority claim or active-protocol change recommended by this review.'),
    ]
    write('CONCLUSIONS.json', {'schema': 'graph-init-local-prediction-review-conclusions-v1',
        'study_id': registry['study_id'],
        'records': [{'key': k, 'source_bound_conclusion': c, 'source_keys': s,
                     'assumptions': a, 'limits': l} for k, c, s, a, l in records],
        'read_accounting': {'new_primary_acquisitions': 0, 'new_scoped_primary_reads': 0,
                            'new_full_primary_reads': 0, 'retained_primary_rereads': 0,
                            'saved_conclusion_and_source_review_only': True},
        'no_model_data_labels_checkpoints_outcomes_ssh_gpu_access': True,
        'no_active_protocol_manuscript_ledger_status_index_mutation': True,
        'decision': 'Honest bounded local interpretation; post-closure diagnostics feasible; novelty and utility unestablished'})

    for item in bindings:
        path = PHASE / item['path']
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
    write('VERIFICATION.json', {'schema': 'graph-init-local-prediction-review-verification-v1',
        'static_only': True, 'input_bindings_rechecked_unchanged': len(bindings),
        'registered_source_descriptors_match': True, 'inspected_sealed_payloads_match': True,
        'active_source_imported_or_executed': False, 'outcomes_or_scientific_payloads_read': False,
        'new_primary_reads': 0, 'scope': 'Provenance/JSON verification only; not a scientific or mathematical empirical certificate'})
    payload = [record(p, 'Review artifact') for p in sorted(HERE.iterdir())
               if p.is_file() and p.name not in ['MANIFEST.json', 'SEAL.json']]
    write('MANIFEST.json', {'schema': 'graph-init-local-prediction-review-manifest-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_and_math_review_only': True,
        'payload': [{'path': Path(x['path']).name, 'bytes': x['bytes'], 'sha256': x['sha256']} for x in payload]})
    write('SEAL.json', {'schema': 'graph-init-local-prediction-review-seal-v1',
        'manifest_sha256': sha(HERE / 'MANIFEST.json'), 'payload_count': len(payload),
        'new_primary_reads': 0, 'execution_authorized': False,
        'not_a_scientific_gate': True, 'immutable_review_packet': True})
    print(json.dumps({'review': str(HERE), 'manifest_sha256': sha(HERE / 'MANIFEST.json'),
                      'payload_count': len(payload), 'input_binding_count': len(bindings),
                      'conclusion_count': len(records), 'new_primary_reads': 0}))


if __name__ == '__main__':
    main()
