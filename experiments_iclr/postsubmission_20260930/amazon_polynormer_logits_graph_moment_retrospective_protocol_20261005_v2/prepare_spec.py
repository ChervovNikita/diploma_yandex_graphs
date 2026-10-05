"""Create the V2 specification from preserved V1 before any fusion outcome."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
V1 = PHASE / 'amazon_polynormer_logits_graph_moment_retrospective_protocol_20261005_v1'
REVIEW = PHASE / 'amazon_polynormer_logits_graph_moment_protocol_independent_review_20261005_v1'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

spec = json.loads((V1 / 'PROTOCOL.json').read_text())
spec['schema'] = 'amazon-polynormer-logits-graph-moment-retrospective-protocol-v2'
spec['created_utc'] = datetime.now(timezone.utc).isoformat()
spec['predecessor'] = {'directory': V1.name, 'PROTOCOL_sha256': digest(V1 / 'PROTOCOL.json'),
                       'REPORT_sha256': digest(V1 / 'REPORT.md')}
spec['independent_review'] = {'directory': REVIEW.name, 'REPORT_sha256': digest(REVIEW / 'REPORT.md')}
spec['graph'].update(mask_probability_dtype='native FP32 softmax and member mean',
    edge_orientation='T[u,v] receives source edge_index[0]=u, edge_index[1]=v; canonical graph is undirected',
    degree='retained outgoing edges including one self loop; duplicates forbidden',
    CSR='SciPy CSR, sorted column indices, no duplicates; FP64 values 1/degree[u]',
    bindings='SHA256 retained int64 edge tensor, CSR indptr/indices and FP64 values before outcome selection')
spec['optimizer'].update(selected_iterate='best B-fit penalized objective among steps 0..150; strict improvement, earliest exact tie',
    amsgrad=False, maximize=False, foreach=False, fused=False,
    diagnostics='initial, final-update and selected Brier/penalty/objective; gradient L2, selected step, trajectory and wall')
spec['heads'].update(formula='softmax(stable log(q_skip)+A*X+V*ReLU(W*X+b)+c)',
    standardization='stitched B training rows only; NumPy population std ddof=0, floor 1e-6',
    W_init='torch CPU Generator, uniform[-sqrt(6/(d+width)),+sqrt(6/(d+width))], Xavier gain 1',
    zero_skip='algebraic equality, report FP64 numerical discrepancy; no prediction contamination')
spec['numerics'] = {
    'dtype': 'FP64 CPU for processing; original bank native FP32 classes retained',
    'threads': 'one BLAS/OpenMP/PyTorch intra/inter-op thread; deterministic Torch algorithms',
    'environment': 'bind Python/NumPy/SciPy/Torch versions and BLAS configuration at source qualification before fitting',
    'simplex_solver': 'batch enumerate all 15 nonempty free faces, increasing free count then lexicographic indices; inactive weights fixed at 0.0125',
    'QP_objective': 'w^T A w-2b^T w; moment b=0; projection A=P P^T,b=P posterior',
    'QP_scale': 'divide A,b by max(maxabs(A),maxabs(b),1e-12) separately per node',
    'QP_face_solve': 'equality-constrained augmented KKT solve; np.linalg.solve for ridge moments, np.linalg.pinv rcond=1e-12 for semidefinite projection',
    'QP_checks': 'finite/symmetric/PSD; floor and sum errors <=1e-10; scaled and raw stationarity <=1e-10; retain distinct diagnostics',
    'QP_selection': 'lowest feasible scaled objective, differences <=1e-14 treated as numerical ties; lowest squared weight norm, tolerance1e-14; exact remaining ties keep first enumerated face',
    'QP_roundoff': 'clip negative excess above floor to zero and renormalize remaining mass0.95; repeat KKT checks',
    'projection_tie_scope': 'minimum-norm numerical optimizer over feasible faces; unique probability projection up to reported tolerances; no hidden ridge',
    'NLL': 'q_delta=(1-1e-12)*q+1e-12/5; -mean log q_delta[y]; original native uncorrected NLL uses stable uncontaminated log mixture',
    'log_skip': 'logsumexp of member log_softmax plus log weights, without exp/log round trip',
    'failed_attempts': 'preserve failures/costs; numerical repairs must leave scientific constants fixed and affect all arms consistently'
}
spec['single_control'] = {
    'family': 'single_author', 'alias': 'independent member0, its original selected checkpoint; no new base fit',
    'native_id': 0, 'head_id': 10, 'settings': 2, 'regularizers': [1e-4, 1e-2],
    'feature_dim': 17, 'width': 156, 'parameters': 3678,
    'features': {'single_p': 5, 'single_log_p': 5, 'H_single_p': 5, 'log1p_degree': 1, 'anchor_mass': 1},
    'graph': 'own native member0 FP32 predicted classes, same graph/mask/diffusion rule',
    'features_training': 'same whole-inner-fold exclusion for anchor mass, same stitched B standardization; no other bank information',
    'skip': 'stable member0 log_softmax; zero residual recovers member0',
    'fit': 'same 150-update/best-B-objective rule, seed recipe with bank=single_author and operator=10',
    'CS': 'identical permitted labels/folds/graph-adaptation opportunity',
    'selection': 'head regularizer by corrected OOF Brier; processed single finalist is best of native+C&S and tuned head+C&S, ties native first',
    'report': 'all raw/corrected configurations and native/member0 metrics, not only winner'
}
spec['budget'].update(MLP_fits=90, calibration_fits=36, single_bank_splits=3,
    single_configs_per_split=3, total_outer_config_predictions=297,
    logical_CS_H_applications=594, bank_seed_context_H_applications=54,
    single_mass_context_H_applications=27, label_free_H_applications=9,
    total_logical_H_applications=684, logical_sparse_steps=13680,
    maximum_batched_H_calls=144, maximum_batched_sparse_steps=2880,
    batching='C&S configurations share block-field sparse calls; logical costs retain every output field',
    final_refits='NOT admitted in this development run; later separately freeze/release/count at most6 bank learned fits plus3 single head fits, plus full-VALID fields/C&S and any cache reconstruction',
    historical_costs='separate original15 base fits, qualification, failed evaluatorV1 and successful evaluatorV2, replay and cacheI/O')
spec['selection']['strongest_cheap'] = 'per split: best corrected tuned IDs1..5 from own bank and best processed member0 single; ties own ID order then single'
spec['go_no_go']['shared_vs_processed_references'] = {
    'contrast': 'each split independently selected best processed shared versus best processed independent and best processed member0 single',
    'requirements_against_each': dict(spec['go_no_go']['bank_route_vs_best_cheap']),
    'interpretation': 'complete processed pipeline quality; bank-specific prediction masks differ, no causal sharing attribution',
    'failure': 'no shared-superiority or shared-method confirmation proposal; preserve any useful common processor as attributed follow-up'
}
spec['go_no_go']['moment_attribution_matched_rho'] = 'report all already available same-rho6-vs4 and6-vs5 contrasts in addition to each selected-route contrast; no isolated ingredient claim from mismatched rho'
spec['go_no_go']['stacker_or_posterior_match_Brier_tolerance_scope'] = 'mean three-split corrected Brier, tolerance0.001; applies to IDs8 and9'
spec['go_no_go']['competence'] = 'loss by a poorly fitted/unqualified comparator is not evidence of analytic superiority; report training objective and gradient diagnostics'
spec['execution'] = {'payload_access': False, 'fusion_fit': False, 'TEST_access': False,
    'status': 'specification only; source qualification/admission required separately',
    'development_refit': False, 'hidden_export': False, 'new_backbone_training': False}

report = '''# Amazon/Polynormer aggregation protocol V2

This specification amends the preserved V1 before any fusion fit or score. It addresses all six findings of the independent V1 protocol review. V1 remains the complete description of bank custody, VALID-only outer/inner folds, nine bank operators, fixed graph diffusion/C&S, metrics and future confirmation boundary. The V2 JSON is the operative machine-readable specification. The definitions below supersede conflicting V1 text; all unamended scientific constants and role restrictions remain fixed. This is retrospective development, not untouched confirmation.

## Purpose and references

Test whether transported local member-error moments improve combination of the existing complete four-member prediction banks. All members have already run. Both shared and independent banks receive identical processing opportunities. The member0 single receives its own capable, comparable-budget processing. No extra backbone fit, hidden export or GPU forward is part of this screen. All outputs, failures and costs are retained. This study can reject an idea cheaply; it cannot establish novelty or acceptance by itself.

## F1: exact graph

The graph mask uses each bank's native FP32 member softmax and arithmetic probability mean argmax; the single uses native member0 FP32 softmax argmax. Ties select the smallest class. FP64 discrepancies are diagnostics and cannot choose a different mask. Retained non-loop edges join equal predicted classes and every original self-loop remains. The source graph is canonical undirected int64 [2,210592], with duplicates forbidden. T[u,v] uses edge_index[0]=u and edge_index[1]=v, degree counts the self-loop, CSR indices are sorted and values are FP64 1/degree[u]. Bind retained edges and actual normalized CSR arrays before outcome selection. Different banks have different fixed masks.

## F2: processed single

The independent member0 is the fixed native single alias, with no favorable-member selection. Native ID0 and residual-head ID10 use the same VALID-only folds and C&S adaptation, with the single's own graph. The head has 17 inputs: probabilities5, stable log-probabilities5, H(probabilities)5, log1p(degree)1 and permitted anchor mass1. Its width156 gives3678 parameters, versus3650/3675 in bank heads. Its two regularizers, training/standardization rule and seed recipe match the bank heads; bank token is single_author, operator10. Native+C&S and the tuned head+C&S compete by that split's corrected OOF Brier. Report all three single configurations raw and corrected.

## F3: fitted-control competence

Within exactly150 full-batch Adam updates, evaluate steps0..150 and retain the lowest permitted B-training penalized objective, including the initial zero-residual state. Strict improvement wins, exact ties retain the earliest step. D labels never select the iterate. Apply this rule to bank heads, single heads and global calibration. Record initial, final-update and selected Brier, penalty, objective and gradient L2, trajectory, selected step and time. Failure to optimize a control cannot support analytic superiority. Adam uses betas(0.9,0.999), epsilon1e-8, no AMSGrad/maximize/foreach/fused path. Population standardization has ddof0 and std floor1e-6. Xavier gain1 uses the pinned Torch CPU generator and existing SHA256 seed recipe.

## F4: complete processed comparisons

For each split independently, select settings and bank finalists by the V1 corrected-OOF-Brier rule. Add the processed single to each bank's strongest cheap reference. For a shared-superiority confirmation proposal, best processed shared must meet all V1 practical quality thresholds against both best processed independent and best processed single: mean Brier improvement>=0.002, positive Brier difference in at least2/3 splits, mean accuracy gain>=0.25points, no split accuracy loss>0.5points and mean NLL harm<=0.01nats. The per-bank own-native/cheap gates also remain. These are fixed screening thresholds, not significance tests. If this fails, no shared superiority is promoted; an effective common processor may remain an attributed separate hypothesis. Because bank graphs differ, this is a complete-pipeline quality comparison, with no causal-sharing conclusion.

Display all already computed matched-rho local-full versus global-full/local-diagonal contrasts, alongside the tuned-route contrasts. The0.001 Brier match tolerance for capable full-information stacking or label-posterior projection applies to the mean three-split corrected difference. It limits interpretation; it does not erase useful quality evidence.

## F5: stable arithmetic

All new processing is FP64 CPU with one numerical thread and deterministic Torch algorithms. Bind package/BLAS versions at source qualification before fitting. Use stable log-softmax/log-sum-exp skip probabilities, then softmax(log skip+residual); never take log of an underflowed probability. Zero residual recovers the analytic skip algebraically, with FP64 discrepancy reported. New NLL uses q_delta=(1-1e-12)q+1e-12/5; the original native uncorrected NLL retains its stable uncontaminated mixture calculation.

Solve each four-variable convex problem by batched enumeration of all15 nonempty free faces, increasing face size then lexicographic member indices. Inactive weights are0.0125. Solve the equality-constrained KKT system for the scaled objective w'A w-2b'w: divide A,b by max(maxabsA,maxabsb,1e-12). Ridge moment problems use direct solve; singular posterior projection uses Moore-Penrose pseudoinverse rcond1e-12 without added ridge. Feasible candidates satisfy floor/sum/KKT checks; lowest scaled objective wins, differences<=1e-14 are numerical ties resolved by lowest squared weight norm, then first face. This implements a numerical minimum-norm projection tie rule. Clip/renormalize only roundoff excess above the floor and repeat checks. Assert finite/symmetric/PSD matrices and report scaled/raw stationarity separately at tolerance1e-10, plus probability and weight simplex checks. Preserve failed attempts; arithmetic repairs cannot change scientific constants or selectively spare an arm.

## F6: costs and boundaries

Development has90 MLP fits and36 calibration fits: the original72+36 plus18 single-head fits. There are297 outer configuration predictions,594 logical C&S H applications,54 bank seed-context H applications,27 single-mass H applications and9 label-free H applications:684 logical H applications/13680 twenty-step field propagations. Batching all C&S configurations into shared block fields reduces sparse call counts; with context reuse this is at most144 batched H calls/2880 sparse matrix calls, with differing widths. Report actual counters/time/RSS and field widths, not an equivalence between a3.1MB context and peak memory. Charge solves, feature/head work, cacheI/O and serving.

No final refit is admitted here. A separately released frozen confirmation recipe must charge up to6 selected bank learned refits plus3 single-head refits, full-VALID fields/C&S and any feature reconstruction. Historical fifteen-model acquisition, qualification, the failed CPU-configured evaluator and corrected CUDA evaluator, replay and forward costs remain separately retained. All bank members still execute in deployment.

Fusion fits and scores use each split's own VALID only. FIT, TRAIN-control and TEST stay excluded. Cross-fitting the aggregator does not remove base VALID checkpoint selection or OOF tuning bias. Across-split screening cannot alter a split's predictor using another split's outcomes, drop an unfavorable block or imply independence. Original native endpoints and manuscript scores remain unchanged. Source implementation and numerical/custody qualification are prerequisites before root execution admission.
'''
with (HERE / 'PROTOCOL.json').open('x') as f:
    json.dump(spec, f, indent=2); f.write('\n')
with (HERE / 'REPORT.md').open('x') as f:
    f.write(report)
manifest = {'schema': 'amazon-moment-protocol-v2-manifest',
    'files': [{'path': p.name, 'sha256': digest(p), 'bytes': p.stat().st_size}
              for p in [HERE / 'PROTOCOL.json', HERE / 'REPORT.md', Path(__file__)]]}
with (HERE / 'MANIFEST.json').open('x') as f:
    json.dump(manifest, f, indent=2); f.write('\n')
print(json.dumps(manifest))
