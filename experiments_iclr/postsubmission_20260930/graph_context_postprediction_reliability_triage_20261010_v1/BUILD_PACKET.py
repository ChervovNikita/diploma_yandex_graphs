from pathlib import Path
import datetime
import hashlib
import json

D = Path(__file__).resolve().parent
P = D.parent
if (D / 'SEAL.json').exists():
    raise SystemExit('Preserve sealed packet; use a successor.')
UTC = datetime.datetime.now(datetime.timezone.utc).isoformat()

def dump(name, value):
    (D / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')

def bind(path, scope):
    data = (P / path).read_bytes()
    return {'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'scope': scope}

inputs = [
    ('AGENTS.md', 'Applicable local execution and research-role instructions.'),
    ('RESEARCH_WORKFLOW.md', 'Applicable research workflow; read completely.'),
    ('graph_aware_saved_prediction_aggregation_prior_scout_20261005_v1/REPORT.md', 'Complete saved graph stacking proposal and bounded literature synthesis.'),
    ('graph_aware_saved_prediction_aggregation_prior_scout_20261005_v1/READ_SCOPES.json', 'Saved primary locator and method-scope metadata; no primary body reopening.'),
    ('graph_aware_aggregation_development_reuse_pairwise_amendment_20261005_v1/REPORT.md', 'Complete development-reuse correction; link construction not transferred to node task.'),
    ('graph_aware_aggregation_development_reuse_pairwise_amendment_20261005_v1/AMENDMENT_CONCLUSIONS.json', 'Development-reuse correction and unchanged authority.'),
    ('direct12_result_synthesis_root_20261008_v1/REPORT.md', 'Complete closed aggregate classifier-refit outcome.'),
    ('amazon_polynormer_logits_graph_moment_result_independent_review_20261005_v1/REPORT.md', 'Complete closed aggregate Amazon99 review; fixed NO_GO retained.'),
    ('geeni_public_primary_recovery_20261007_v1/CONCLUSIONS.json', 'Unresolved primary-method access and abstract-only scope.'),
    ('genn_new_route_and_graph_specific_aggregation_alternative_20261005_v1/REPORT.md', 'Complete saved NLC/GENNN and same-head neighbor-removal synthesis.'),
    ('genn_new_route_and_graph_specific_aggregation_alternative_20261005_v1/READ_SCOPES.json', 'Selected locator/schema metadata; no primary body reopening.'),
    ('graph_context_residual_ensemble_closest_prior_20261009_v1/REPORT.md', 'Saved C&S/MoE graph ancestry read before compaction; retained conclusion, not a new primary read.'),
    ('graph_context_residual_ensemble_closest_prior_20261009_v1/CONCLUSIONS.json', 'Saved C&S residual semantics and graph-prior scope.'),
    ('graph_context_residual_ensemble_closest_prior_20261009_v1/READ_SCOPES.json', 'Selected saved C&S and MoE-NP scope metadata; output was truncated, no whole-file semantic reading claim.'),
    ('graph_conditioned_low_rank_structural_specialization_literature_scout_20261003_v1/PAPER_CONCLUSIONS.json', 'Selected MoSE conclusion; incidental other saved conclusion exposure not used.'),
    ('graph_conditioned_low_rank_structural_specialization_literature_scout_20261003_v1/READ_SCOPES.json', 'Selected MoSE source locator/scope excluding read_blocks; an earlier broad selector incidentally exposed saved text for other sources, not analyzed or credited.'),
    ('shared_wrapper_acquisition_priority_decision_20261010_v1/CONCLUSIONS.json', 'Complete prior priority and scope; supersede only its pooling/acquisition priority rationale.'),
    ('shared_wrapper_acquisition_priority_decision_20261010_v1/MANIFEST.json', 'Predecessor custody metadata.'),
    ('shared_wrapper_acquisition_priority_decision_20261010_v1/SEAL.json', 'Predecessor immutable seal.'),
]
bindings = [bind(path, scope) for path, scope in inputs]
dump('SOURCE_BINDINGS.json', {'schema': 'postprediction-reliability-source-bindings-v1', 'UTC': UTC, 'inputs': bindings, 'primary_payloads_reopened': False})

root_quote = ('Closed117 diagnosis, no new forwards: majority-common-missing signature FAILS in all18 comparisons. '
              'Most I4-correct/shared-wrong nodes already have a correct shared member. '
              'Recoverable fractions: SAGE ~65–69%, GCN ~74–76%, GAT ~57–69%. '
              'Wrong members on SAGE pooling losses have median top confidence .84/.99/.97. '
              'New coherent24 restores competence but adds few different correct predictions.')
dump('ROOT_CONTEXT.json', {
    'schema': 'postprediction-reliability-parent-context-v1', 'UTC': UTC,
    'provenance': 'Parent /root message quoted in the handoff summary supplied at resumption; not independently audited here.',
    'quoted_context': root_quote,
    'original_message_bytes_or_tool_receipt_available_here': False,
    'numerical_payload_access_or_new_diagnosis': False,
    'exact_denominators_and_seed_mapping_available_in_quote': False,
    'interpretation_scope': 'Motivates a prospective pooling question; approximate supplied fractions are not imported project results or predictive evidence for the proposed scorer.',
    'current24_and_full9_on77_changed_or_read': False,
})

method = {
    'name': 'Shared graph-context reliability scorer for frozen probability pooling',
    'status': 'ONE_PROSPECTIVE_ATTRIBUTED_UTILITY_HYPOTHESIS_NO_FIT_OR_SOURCE_ADMISSION',
    'inputs': 'Fixed aligned per-member node class probabilities p_vm from fully authenticated selected states; no intermediate-state or new expert construction.',
    'member_count': 'M>=2; M=1 is the unchanged single probability and has no member-weighting effect.',
    'class_count': 'C>=2',
    'graph_operator': {
        'definition': 'P_vu=1/|N_in(v)| for each distinct eligible incoming neighbor u != v; nonempty rows exclude self-loops; an isolate row is P_vv=1.',
        'support': 'The exact authorized graph-visible node/edge support, directedness and node-row map must be frozen by root before any new aggregation assessment. Use the graph as supplied; do not add reverse edges or infer a new support.',
        'missing_rows': 'All neighbor prediction rows on that support must be available in an authenticated permitted full-node field. Missing rows do not authorize silent deletion, TEST-role payload access, new export or replay.',
        'label_use': 'No ground-truth neighbor labels, TRAIN residuals, correctness masks or label propagation in features.',
    },
    'peer': 'b_vm=(sum_{j!=m} p_vj)/(M-1)',
    'neighbor': 'g_vm=(P p_m)_v and d_vm=(sum_{j!=m} g_vj)/(M-1)',
    'features': [
        'H(p_vm)/log C, with 0 log 0=0',
        'top1(p_vm)-top2(p_vm)',
        'dot(p_vm,b_vm)',
        'dot(p_vm,g_vm)',
        'dot(p_vm,d_vm)',
    ],
    'scorer': 's_vm=a^T tanh(W f_vm+b); W is 8x5, b is 8, a is 8; shared across all members/nodes, no output bias or member/class identity inputs.',
    'trainable_parameters': 56,
    'weights': 'alpha_vm=exp(s_vm)/sum_j exp(s_vj)',
    'served_prediction': 'q_v=sum_m alpha_vm p_vm',
    'objective': 'mean_{v in fusion-fit} -log q_v[y_v] + 0.01 mean_v sum_m alpha_vm log(M alpha_vm)',
    'supervision': 'Fusion-fit labels supervise mixture NLL; the scorer is not trained on an oracle member-correctness flag.',
    'prospective_fixed_fit': {
        'initialization': 'W uniform[-0.1,0.1] with one declared deterministic seed11709, b=0, a=0; scores initially zero and pool initially uniform.',
        'optimization': 'Full-batch Adam, learning rate0.01, default beta1=0.9/beta2=0.999/epsilon=1e-8, no weight decay, exactly500 updates, serve the final update; no fold-heldout early stopping, head-width/shrinkage or seed grid.',
        'qualification': 'This specifies a finite proposal only. Root must qualify numerical execution in an existing authorized interface, retain failures and freeze the complete roster/protocol before a fit. No fit is admitted here.',
    },
    'every_route_executes': True,
    'frontend_MoE_or_expert_acquisition': False,
    'route_compute_saving_claim': False,
}
dump('METHOD.json', {'schema': 'postprediction-reliability-method-v1', 'UTC': UTC, **method})

proof = {
    'schema': 'postprediction-reliability-algebra-v1', 'UTC': UTC,
    'claim_status': 'Symbolic inspection, not numerical testing or published theorem adoption.',
    'class_permutation': 'For a common class permutation matrix R, p->Rp, b->Rb, g->Rg, d->Rd. Entropy and top order values are unchanged, and (Rx)^T(Rz)=x^Tz. Thus all five f and scores/weights stay unchanged, and q->Rq. The scalar score is invariant; the probability output is equivariant.',
    'member_permutation': 'A member permutation permutes own p and g; total sums are unchanged and leave-one-out b/d follow the same member. Shared psi gives permuted scores and alpha, and the sum q is unchanged. Independent class permutations of different members are not allowed: class coordinates must be aligned.',
    'convexity': 'alpha>=0 and sum alpha=1, so q is a valid convex probability mixture. This is a representational statement, not a convex fitting-objective claim.',
    'strict_common_rival_limit': 'If some rival c has p_vm[c]>p_vm[y] for every member m at v, then q_v[c]-q_v[y]=sum_m alpha_vm (p_vm[c]-p_vm[y])>0 for finite softmax weights. This gate cannot repair that node.',
    'oracle_coverage_caveat': 'A top1-correct member provides a potentially selectable alternative, but does not guarantee learnable reliability. Conversely, all members being top1-wrong is not by itself an impossibility proof: different rivals can cancel under a mixture. Use the strict common-rival condition for the claimed impossibility.',
    'unchanged_base_competence': 'Freezing p means individual route predictions/accuracy remain unchanged algebraically; final pooled quality can improve or worsen.',
    'M1_limit': 'With one member, alpha=1 and q=p irrespective of graph context. The independent processed-single quality anchor therefore needs its own ordinary calibration/stacking, not a claimed gate gain.',
    'no_reliability_guarantee': 'Wrong high-confidence members and wrong agreeing neighborhoods can receive large weights; NLL supervision only tests whether the cues predict useful pooling on the available population.',
}
dump('PROOF.json', proof)

prior_scopes = []
prior_path = 'graph_aware_saved_prediction_aggregation_prior_scout_20261005_v1/READ_SCOPES.json'
prior = json.loads((P / prior_path).read_text())
for s in prior['scopes']:
    prior_scopes.append({
        'identity': 'arxiv:' + s['versioned_id'],
        'title': s['title'],
        'kind': 'REUSED_SAVED_PRIMARY_METHOD_SCOPE_NO_NEW_CREDIT',
        'scope_reference': prior_path,
        'selector': 'scopes/' + str(prior['scopes'].index(s)),
        'saved_HTML_locators': s['exact_HTML_ID_locators'],
        'saved_paragraph_indices': s['paragraph_indices_zero_based'],
        'saved_equation_indices': s['equation_indices_zero_based'],
        'primary_body_reopened_here': False,
        'results_proofs_author_code_reproduced_or_adopted_here': False,
    })
mose_dir = 'graph_conditioned_low_rank_structural_specialization_literature_scout_20261003_v1'
mose = next(x for x in json.loads((P / mose_dir / 'READ_SCOPES.json').read_text())['sources'] if x['key'] == 'mose')
prior_scopes.append({'identity': mose['canonical_id'], 'kind': 'REUSED_SAVED_PRIMARY_METHOD_SCOPE_NO_NEW_CREDIT',
                     'scope_reference': mose_dir + '/READ_SCOPES.json', 'selector': 'sources[key=mose]',
                     'saved_ranges_zero_based_inclusive': mose['ranges_zero_based_inclusive'],
                     'scope_note': mose['scope_note'], 'primary_body_reopened_here': False,
                     'proof_appendix_algorithm_code_or_results_adopted': False})
prior_scopes.extend([
    {'identity': 'arxiv:2010.13993v2', 'kind': 'REUSED_SAVED_PRIMARY_AND_PINNED_AUTHOR_SOURCE_CONCLUSION_NO_NEW_CREDIT',
     'scope_reference': 'graph_context_residual_ensemble_closest_prior_20261009_v1/READ_SCOPES.json',
     'scope': 'C&S relevant correction/smoothing paper passages and pinned author source sign from inherited saved review; no new code/body inspection.'},
    {'identity': 'arxiv:2402.08583v2', 'kind': 'REUSED_SAVED_PRIMARY_METHOD_CONCLUSION_NO_NEW_CREDIT',
     'scope_reference': 'graph_aware_aggregation_development_reuse_pairwise_amendment_20261005_v1/REPORT.md',
     'scope': 'Link-MoE Section4/Eqs1-3/Algorithm1 and saved role scope; second-stage link expert gate, not a node-method reproduction.'},
    {'identity': 'arxiv:2405.17139v2', 'kind': 'REUSED_SAVED_PRIMARY_METHOD_CONCLUSION_NO_NEW_CREDIT',
     'scope_reference': 'genn_new_route_and_graph_specific_aggregation_alternative_20261005_v1/READ_SCOPES.json',
     'scope': 'NLC complete Section3, AppendixA controller setup and Section4.2 frozen-feature/heldout-label roles from saved review; coefficient map/constraints remain unresolved.'},
    {'identity': 'doi:10.1145/3489517.3530416', 'kind': 'UNRESOLVED_METADATA_AND_ABSTRACT_ONLY_NO_PRIMARY_METHOD_CREDIT',
     'scope_reference': 'geeni_public_primary_recovery_20261007_v1/CONCLUSIONS.json',
     'scope': 'GEENI full methods and code absent after prior20 bounded public retrieval attempts; no fresh retries.'},
    {'identity': 'doi:10.1016/j.inffus.2024.102461', 'kind': 'UNRESOLVED_METADATA_ONLY_NO_PRIMARY_METHOD_CREDIT',
     'scope_reference': 'genn_new_route_and_graph_specific_aggregation_alternative_20261005_v1/REPORT.md',
     'scope': 'GENNN journal/SSRN identity only; exact fusion unresolved, no fresh retries.'},
])
dump('READ_SCOPES.json', {
    'schema': 'postprediction-reliability-read-scopes-v1', 'UTC': UTC,
    'source_binding_reference': 'SOURCE_BINDINGS.json', 'reused_literature_scopes': prior_scopes,
    'closed_project_aggregate_reports': ['direct12_result_synthesis_root_20261008_v1/REPORT.md', 'amazon_polynormer_logits_graph_moment_result_independent_review_20261005_v1/REPORT.md'],
    'latest_project_context': 'ROOT_CONTEXT.json parent-message custody only; no target report or payload audit of closed117/coherent24 in this packet.',
    'scope_note': 'Full reports explicitly marked complete were read; selected/truncated locator records do not imply complete JSON semantic reading. Saved primary excerpts exposed incidentally by a broad selector receive no new reading credit.',
    'accounting': {'new_HTTP_requests': 0, 'new_primary_identities': 0, 'new_scoped_primary_method_reads': 0, 'new_full_papers': 0, 'new_author_code_or_proof_audits': 0, 'new_scientific_executions': 0, 'target_payload_reads': 0},
})

conclusions = {
    'schema': 'postprediction-reliability-conclusions-v1', 'UTC': UTC,
    'status': method['status'], 'one_mechanism': method['name'],
    'predicted_benefit': 'Downweight some confidently wrong members when graph-local prediction agreement and consensus identify a retained correct alternative better than own confidence alone.',
    'principal_expected_harm': 'High-confidence wrong neighborhoods/majorities and development overfitting can suppress the correct outlier and worsen pooled accuracy/NLL.',
    'ancestry': ['Saved5October frozen graph stacking', 'META-DES classifier-competence learning', 'GATS and RBS graph-aware calibration', 'MoE-NP and Link-MoE context-conditioned expert fusion', 'MoSE structural-expert routing', 'NLC dense learned late weighting'],
    'C_and_S_boundary': 'C&S propagates permitted TRAIN-label residuals and smooths; this proposed scorer has no label-seeded propagation. Adding C&S would change information and cost and is outside the one mechanism.',
    'unknown_prior_limits': ['GEENI exact methods/sharing/aggregation unresolved', 'GENNN exact fusion unresolved'],
    'old_failure_dispositions': {
        'direct12': 'Exact regularized frozen TRAIN-classifier refit failed; no additional-head freedom gain, all four refits lower native mean accuracy. It did not test fixed-probability fusion supervision.',
        'Amazon99': 'Complete frozen aggregation/moment family failed every practical screen and remains NO_GO. This proposal is a known-method, different-bank retention question, not a reversal or generic rescue.',
        'saved5October': 'Same broad graph-conditioned frozen pooling idea already proposed; new reliability feature restriction/permutation property and new root diagnosis give scope, not novelty.',
    },
    'priority_successor': {
        'predecessor': 'shared_wrapper_acquisition_priority_decision_20261010_v1',
        'manifest_sha256': 'b514cbbdbfb2a9c6569ee2cae695a8bca2eb2f679db82edb1f409887cb644da1',
        'superseded_scope': 'Ranking3/general rationale that inactive pooling fails because correct alternatives are absent, and priority of another acquisition mechanism, for the newly diagnosed encountered Wiki banks only.',
        'new_reason': 'Parent reports most I4-correct/shared-wrong nodes already have a correct shared member; new coherent bank restores competence while adding few distinct correct outputs.',
        'not_superseded': 'All prior outcome numbers/failures, competence deficits where measured, strict common-rival impossibility and unchanged current24/full9 contracts.',
    },
    'essential_baseline_families': {
        'temperature': 'Fit one shared positive temperature and M positive per-member temperatures as two fixed controls; each followed by uniform probability pooling, same eligible fusion labels/folds/NLL and no temperature grid.',
        'non_graph_stacking': 'Matched shared5→8→1 scorer with P=I, plus one ordinary regularized multinomial linear probability stacker on all own-node member probabilities. Both lack neighbor prediction fields; same fusion roles and frozen finite schedules.',
    },
    'graph_attribution_falsifier': 'If replacing P by I matches/beats the graph scorer, no neighbor-prediction contribution is supported for this recipe/bank.',
    'practical_falsifier': 'No net served accuracy gain with acceptable proper-risk behavior beyond the strongest temperature/non-graph control across the complete paired roster. Better NLL alone is calibration utility, not recovered accuracy.',
    'development_limit': 'Aggregator-only cross-fitting on already-selected VALID is encountered development, not whole-pipeline cross-fitting or independent confirmation. The proposal itself uses observed development diagnostics.',
    'confirmation': 'Freeze the entire base/selector/fusion protocol, then separately authorized genuinely unused confirmation remains essential; this packet establishes no TEST custody/access.',
    'sharing_limit': 'A useful same-bank graph postprocessor would not establish an ensemble-specific or sharing-specific gain. Capable processed single and genuine independent-bank controls need equal fusion opportunity before such claims.',
    'cost_estimate': {'base_model_new_fits_for_fusion': 0, 'base_model_new_forwards_in_this_task': 0,
                      'graph_cache_work': 'One batched P on all M class-probability fields: O(|E|MC); peers obtained from sums without another sparse propagation.',
                      'head_work': 'O(NM(C+48)) plus8 tanh evaluations per node/member;56 trainable scalar parameters.',
                      'plain_stacker_reference': 'C(MC+1) coefficients including biases;O(NMC^2) per full forward. Charge this separately from the56-parameter candidate.',
                      'cache_storage': 'O(NMC) probability/neighbor storage, or equivalent streamed sparse bookkeeping; actual bytes/seconds unmeasured.',
                      'proposed_development_fits': 'For B admitted frozen banks,5 folds x5 fitted operators =25B small fits; native uniform pool requires no fit. Duplicate M=1 controls can collapse before any outcome.500 updates per nonduplicate fit; no final refit in this screen.',
                      'missing_retained_fields': 'Any newly authorized selected-state export/replay, I/O and storage must be charged separately; absent permitted full-node rows means this packet is not executable as written.'},
    'model_data_labels_logits_predictions_checkpoints_read': False,
    'new_fit_export_forward_source_framework_or_launch': False,
    'remote_scientific_or_TEST_access': False,
    'existing_packets_canonical_state_manuscript_or_sources_edited': False,
    'novelty_superiority_generalization_or_acceptance_claim': False,
}
dump('CONCLUSIONS.json', conclusions)

report = '''# One graph-context reliability scorer for frozen pooling

10 October 2026. Source review and one prospective known-method utility question. No fit, export, forward or execution admission.

## Decision and changed priority

Retain one fixed post-prediction scorer that weights already computed member probabilities using entropy, margin, consensus and label-free neighbor prediction agreement. Its purpose is to preserve useful alternatives during pooling. It is untested. Every route executes; this adds aggregation work and makes no route-compute saving claim.

Root supplied the following closed-family diagnosis through the parent message preserved in `ROOT_CONTEXT.json`: majority-common-missing fails in all18 comparisons; most I4-correct/shared-wrong nodes already have a correct shared member, with approximately65–69% recoverable fractions for SAGE,74–76% for GCN and57–69% for GAT. Wrong members on SAGE pooling losses have median top confidence .84/.99/.97. The new coherent24 bank restores competence but adds few different correct predictions. This packet has not independently audited those numbers, their denominators or seed mapping, and has not accessed current24 or full9on77 outputs.

This context supersedes the narrow pooling/acquisition priority rationale in the sealed `shared_wrapper_acquisition_priority_decision_20261010_v1`: absence of correct alternatives is no longer the leading explanation for those newly diagnosed nodes. Its failures and measured competence deficits remain intact. The mechanism targets retention where alternatives exist; it supplies no remedy for every shared error.

## The one predictor

Fix an authenticated selected-state bank of M>=2 aligned C-class probability vectors `p_vm`. For each node/member define the leave-one-out peer mean `b_vm=(sum_{j!=m}p_vj)/(M-1)`. Let P be the incoming-neighbor row mean on the exact authorized graph-visible support: distinct incoming neighbors, exclude self-loops on nonempty rows, and use `P_vv=1` for isolates. Keep graph directedness as supplied and add no reverse edges. Define `g_vm=(P p_m)_v` and neighbor peer mean `d_vm=(sum_{j!=m}g_vj)/(M-1)`.

The five scalar inputs are:

1. normalized entropy `H(p_vm)/log C`;
2. top1-minus-top2 probability margin;
3. own-node consensus `p_vm dot b_vm`;
4. own-member neighbor agreement `p_vm dot g_vm`;
5. peer-neighbor agreement `p_vm dot d_vm`.

Use exactly one shared scorer `s_vm=a^T tanh(W f_vm+b)`, with W8x5, b8 and a8:56 parameters, no output bias, member ID or class ID. Set `alpha_vm=softmax_m(s_vm)` and serve `q_v=sum_m alpha_vm p_vm`. Fit mixture NLL plus a fixed0.01 mean `KL(alpha_v || uniform_M)` penalty. This discourages unnecessary departures from uniform pooling; it is not a diversity reward. Fusion labels train q directly. No member-correctness oracle, TRAIN residual, neighbor label, true homophily, hidden state or backbone gradient enters the features.

The fixed prospective fit uses W uniform[-0.1,0.1] from declared seed11709, zero b/a, full-batch Adam learning rate0.01, default betas/epsilon, no weight decay and exactly500 updates. Serve the last update. The zero output a initializes uniform pooling without making the whole hidden layer identical. Width, shrinkage and seed have no proposed search. Root still owns finite numerical qualification and any admission in an existing interface; this memo creates no trainer/source implementation or authorized fit.

Under a common class permutation, entropy/top-order scalars and dot products are invariant, so weights stay the same and q follows the class permutation. Under a member permutation, the symmetric leave-one-out means and shared scorer permute the weights, leaving q unchanged. These statements require aligned class coordinates and do not permit a different class permutation for each member. `PROOF.json` records the symbolic argument.

This is a supervised reliability cue, not a correctness guarantee. High-confidence wrong members and wrong agreeing neighborhoods can suppress the correct outlier. A correct top1 member does not prove the cue can identify it. Conversely, every member being top1-wrong is not by itself an impossibility proof: different rivals can cancel. The exact limitation is a strict common rival: if one wrong class exceeds the true class in every member, no convex weighting can repair that node. The frozen individual predictions and their competence remain unchanged; served quality can worsen.

## Two essential baseline families

Retain native uniform probability pooling as the unfitted anchor. All fitted rules receive the same eligible fusion labels, fixed folds, base bank and assessment opportunity.

| Baseline family | Fixed controls and decisive interpretation |
| --- | --- |
| Global/per-member temperature calibration | Fit one positive shared T and, separately, M positive T_m values, followed by uniform probability pooling: `p_m(T_m)=softmax(log p_m/T_m)`. Fit NLL with the same fusion roles and fixed finite budget; no temperature grid. A positive temperature preserves each member's class ordering but can change the pooled ordering. If this suffices, confidence miscalibration explains the recoverable gain without learned neighbor weighting. |
| Non-graph learned pooling/plain stacking | First use the exact same56-parameter scorer, fit and penalty with P=I: the last two inputs become `p_vm dot p_vm` and own-node consensus. This keeps dimensions/architecture fixed and removes neighboring predictions; repeated coordinates are disclosed. Also retain one ordinary regularized multinomial linear stacker on concatenated own-node member probabilities, `softmax(B[p_v1;...;p_vM]+c)`, fitted with NLL and fixed0.01 mean-square penalty over B and c, zero initialization and the same500-update Adam schedule. It is a capable class-specific own-node decoder and may leave the convex hull. No feature, penalty or depth grid is proposed. |

For temperatures use log-T parameters initialized at0 and the same500-update Adam schedule. Ordinary positive temperature calibration has no graph cues. Candidate/scorer supervision learns mixture behavior; it does not certify per-member calibration. Baselines cannot silently receive fewer labels or a weaker selection procedure. These are two baseline families with fixed operators, not alternative candidate mechanisms.

If the P=I scorer matches or beats the graph scorer, this neighbor-prediction ingredient has no supported contribution. If the plain stacker matches, use ordinary stacking for the measured utility. Better calibration against the raw pool alone is insufficient. Before a sharing or ensemble-specific claim, capable processed single and genuine independent banks need equal fusion opportunity and all acquisition costs; a same-bank postprocessor win cannot supply that claim.

## What was already tested or proposed

**The5October graph stacker is the exact local ancestor.** It froze predictions, formed `p_m-mean_j p_j` and their neighbor summaries, and learned softmax member weights for a probability mixture. It established no predictive gain. This proposal narrows the inputs to class-invariant reliability scalars and supplies a fixed shared scorer after a new retention diagnosis. Those details do not clear methodological novelty or justify renaming the broad idea as a new family.

**Direct12 closed one regularized frozen TRAIN-classifier refit.** Extra unrestricted BE head freedom changed correct counts by0,-1,+1 and mean accuracy by zero; all four refit conditions lowered their corresponding native mean accuracy. Six finite endpoints failed the declared convergence condition. Its exact refit stays closed, with no penalty/solver or private-head extension. It altered each route classifier on TRAIN; it did not test a frozen probability mixer supervised on fusion-development labels. That distinction supports an untested question, not a predicted gain.

**Amazon99 is a genuine negative aggregation result.** All99 configurations completed. Selected shared4 accuracy52.5450487% was below processed single53.1275519% and processed independent4 53.2527628%. Relative to processed single, shared Brier worsened0.023703398 and NLL0.340900739; relative to processed independent, accuracy fell0.7077141pp and NLL worsened0.048939813. Every practical screen failed, and analytic local moments also failed their global/diagonal/stacker comparisons. Its fixed NO_GO remains intact. Calibration gains over a badly calibrated raw bank did not survive capable controls. A different Wiki bank with retained alternatives can justify a scoped known-method question; it does not reverse that result or establish generic graph stacking success.

## Closest primary ancestry and limits

The sources below reuse saved bounded primary-method scopes; no primary body, author implementation or proof was reopened here. `READ_SCOPES.json` preserves exact original locators and credits zero new reads.

| Source | Relevant collision and remaining scope |
| --- | --- |
| META-DES, arXiv1810.01270v1 §3 | Confidence, local posterior behavior/accuracy and output-profile matching estimate member competence. Reliability-conditioned combination is established; its labeled local reference features and majority-selection endpoint differ from the proposed label-free deployment cues and soft probability mixture. |
| GATS, arXiv2210.06391v1 §5/AppendixA.3; RBS, arXiv2206.01570v1 §§II-C2/IV | Already use graph/local prediction agreement to condition temperature calibration. Their positivity/label-role qualifications remain preserved. They provide close graph reliability ancestry, not proof that weighting improves accuracy. |
| MoE-NP, arXiv2412.00418v3 §3 | Graph-pattern/context features feed softmax expert weights. Its printed binary-logistic/multiclass ambiguity remains unresolved; adapting its bank/endpoint is not native reproduction. |
| Link-MoE, arXiv2402.08583v2 §4/Eqs1–3/Algorithm1 | Dense second-stage structure/feature-conditioned expert weighting. Its link-logit output and native supervision split differ; ordinary gate ancestry is direct. |
| MoSE, arXiv2509.09337v1 saved main-method blocks18–73 | Anonymous walks form structural subgraphs; topology-aware noisy top-K routing chooses trainable hidden-graph/random-walk-kernel experts. It changes representation/expert computation before prediction. It is graph expert-routing ancestry rather than this frozen dense late mixer. Printed balance reduction ambiguity remains; appendix algorithm/proofs/code/results were not qualified. |
| C&S, arXiv2010.13993v2 and saved pinned author source | Propagates permitted TRAIN-label residuals, scales correction and smooths. Saved source uses Y-p and adds correction despite the paper sign mismatch. The present features propagate predictions only and have no label seeds; adding C&S would change information/cost and is outside this one mechanism. |
| NLC, arXiv2405.17139v2 §3/AppendixA/§4.2 | Frozen feature-conditioned per-example model coefficients combine existing logits under supervised CE. Every backbone runs. Exact coefficient constraints/temperature map remain unqualified in saved prose. Dense late learned weighting is already established. |
| GEENI, DOI10.1145/3489517.3530416; GENNN, DOI10.1016/j.inffus.2024.102461 | Exact primary methods remain unresolved. GEENI abstract identifies likely-error nodes and suppresses outgoing messages;20 prior public attempts did not recover the body/code. GENNN has metadata/SSRN locator only. No retries, exclusion certificate or novelty inference follows. |

## Development scope, decisive result and costs

Use identical fixed five folds over the encountered VALID population for candidate and controls; each fold's labels are excluded from that fold's aggregator fit. The base selectors already used all VALID labels, and the current hypothesis was chosen after development diagnostics. Aggregator-only cross-fitting remains encountered development and does not cross-fit the selected whole pipeline. Transductive neighbor prediction context is allowed only on the unchanged authorized support and supplies no independence of connected observations. No claim of honest untouched assessment is available here.

Record the full paired roster before interpreting results. Compare served accuracy, NLL/Brier and repairs versus harms relative to native pooling. The diagnostic subset of pooling losses can describe whether retained correct alternatives survive, but its outcome-derived membership is never a gate input or selector. Stop this recipe if it provides no net served accuracy gain with acceptable proper-risk behavior beyond the strongest temperature/non-graph control; an NLL-only gain supports calibration utility. Stop the graph-context explanation if P=I matches or beats it. Root must freeze any numerical worthwhile-effect/uncertainty criterion before a study rather than choose one from outcomes. A promising development result still requires a fully frozen pipeline and separately authorized genuinely unused confirmation. This packet establishes no TEST custody or access.

One batched sparse P on the M probability fields costs O(|E|MC); neighbor peer means follow from sums, without another propagation. Own-node features and head add O(NM(C+48)) work plus8 tanh evaluations per node/member. There are56 trained parameters and O(NMC) probability/neighbor storage. The plain stacker has C(MC+1) coefficients including biases and O(NMC^2) forward work; charge its reference cost separately. With B frozen admitted banks, five folds and five fitted operators give25B small fits of500 updates each; native pooling has no fit and duplicate M=1 controls can be collapsed prospectively. No final refit is part of this development screen. These are operation/count estimates, not measured time or memory results. All base-route acquisition/serving remains charged.

Permitted full-node probability rows and exact graph support are necessary. VALID-only logits do not provide honest neighbor summaries. Missing retained fields make this design non-executable as written; they do not authorize a replay/export, access to sealed TEST-role predictions or silent support reduction. Any later authorized extraction, I/O/storage and replay cost must be declared. This task has read aggregate/source reports and metadata only, executed standard-library text/JSON/hash bookkeeping, and changed only this fresh packet. Current24/full9on77, canonical state, scientific sources and prior sealed packets remain untouched.
'''
(D / 'REPORT.md').write_text(report)

# Validate every declared input binding before any seal exists.
checks = []
for b in bindings:
    now = bind(b['path'], b['scope'])
    checks.append({'path': b['path'], 'match': now == b})
predecessor = P / 'shared_wrapper_acquisition_priority_decision_20261010_v1'
actual_predecessor = hashlib.sha256((predecessor / 'MANIFEST.json').read_bytes()).hexdigest()
predecessor_seal = json.loads((predecessor / 'SEAL.json').read_text())
predecessor_ok = actual_predecessor == predecessor_seal['manifest_sha256'] == conclusions['priority_successor']['manifest_sha256']
structural_ok = (len(method['features']) == 5 and method['trainable_parameters'] == 56
                 and method['every_route_executes'] and not method['frontend_MoE_or_expert_acquisition']
                 and len(conclusions['essential_baseline_families']) == 2)
if not (all(c['match'] for c in checks) and predecessor_ok and structural_ok):
    raise SystemExit('Pre-seal binding/specification verification failed.')
dump('VERIFICATION.json', {
    'schema': 'postprediction-reliability-preseal-verification-v1', 'UTC': UTC,
    'input_binding_checks': checks, 'predecessor_manifest_seal_match': predecessor_ok,
    'specification_metadata_checks': structural_ok, 'seal_existed_during_checks': False,
    'scope': 'Hash/JSON/specification bookkeeping only; not a numerical fit, data check, prediction verification or empirical result.',
    'PASS': True,
})
files = []
for path in sorted(D.iterdir()):
    if path.is_file() and path.name not in ('MANIFEST.json', 'SEAL.json'):
        data = path.read_bytes()
        files.append({'path': path.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
dump('MANIFEST.json', {
    'schema': 'postprediction-reliability-packet-v1', 'UTC': UTC, 'files': files,
    'input_bindings_reference': 'SOURCE_BINDINGS.json', 'verification_before_seal': 'VERIFICATION.json',
    'new_primary_reads_or_HTTP_requests': 0, 'new_scientific_executions': 0,
    'target_payload_reads': False, 'scientific_host_or_TEST_access': False,
    'new_fit_or_source_admission': False, 'canonical_or_prior_packet_changes': False,
    'new_method_novelty_predictive_gain_or_acceptance_claim': False,
})
manifest_hash = hashlib.sha256((D / 'MANIFEST.json').read_bytes()).hexdigest()
dump('SEAL.json', {'schema': 'postprediction-reliability-seal-v1', 'UTC': UTC, 'manifest_sha256': manifest_hash})
print(json.dumps({'packet': str(D), 'files': len(files), 'manifest_sha256': manifest_hash, 'preseal_verification': 'PASS'}))
