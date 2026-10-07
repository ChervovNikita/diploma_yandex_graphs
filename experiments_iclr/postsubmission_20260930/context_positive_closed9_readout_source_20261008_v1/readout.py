"""Thin closed9 adapter over the already-audited Wiki12/Wiki24 analysis API."""
import importlib.util
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SEEDS = (8101, 8203, 8307)
CONDITIONS = ('shared_common','shared_route','shared_route_permuted')
ARCHIVE_SHA = '80d30ed946a1408f861a773952b174713c487765c75e95b8e25456b30478eae6'
LIMITS = dict(exploratory=True, independent_TEST_evidence=False,
    novelty_or_confirmation_claimed=False, numerical_parity_claimed=False,
    replication='Three fixed optimizer seeds on one selected graph; no node/member independence.',
    selection_population_caveat='The same complete5274 development population selected the checkpoints and supplies this readout.',
    cohorts='Frozen from same-seed shared_common before route/permuted prediction collection; overlapping cohorts are not additive.',
    Qbar_control='Matches aggregate target mass but not per-route target concentration/entropy; assignment and concentration are not separately identified.',
    permutation='Preserves each scored-anchor positive count/class/self/panel; does not preserve public graph degrees.',
    graph_causality='X is one context; route versus permuted does not establish topology-specific causality.',
    stationary_case='Parallel normalized same-class embeddings can have zero positive cosine derivatives despite nontrivial target TV.',
    mechanism='Target differences do not prove acquired competent prediction diversity; report common-competitor rank acquisition and introduced errors.',
    reselection_or_calibration=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def configure(pins):
    """Reuse unchanged numerical/error contracts in a fresh module instance."""
    legacy = PHASE/'internal_BE_Wiki12_closed_attribution_analysis_source_20261007_v1'
    sys.path.insert(0,str(legacy))
    analysis = load_module(legacy/'analyse.py','_context9_reused_wiki12_analysis')
    original = analysis.contracts(pins)
    analysis.SEEDS = SEEDS
    analysis.CONDITIONS = CONDITIONS
    analysis.CONTRASTS = [
        ('route-common',{'shared_route':1,'shared_common':-1}),
        ('route-permuted',{'shared_route':1,'shared_route_permuted':-1})]
    analysis.LIMITS = dict(LIMITS,role=analysis.ROLE)
    base_summary,base_pair = analysis.summary,analysis.pair
    def summary(np,arrays,mask,contract):
        row = base_summary(np,arrays,mask,contract)
        row['pool_minus_mean_member_accuracy'] = (row['served_accuracy']-row['mean_member_accuracy']
            if row['served_accuracy'] is not None else None)
        return row
    def pair(np,before,after,cohorts,seed,label,contract):
        result = base_pair(np,before,after,cohorts,seed,label,contract)
        flags = contract.error_masks(np,before)
        common = flags['common_competitor']; wrong = flags['common_competitor_class'].clip(0)
        truth = before['truth']
        old_truth = np.take_along_axis(before['member_logits'],truth[None,:,None],axis=2)[...,0]
        old_wrong = np.take_along_axis(before['member_logits'],wrong[None,:,None],axis=2)[...,0]
        new_truth = np.take_along_axis(after['member_logits'],truth[None,:,None],axis=2)[...,0]
        new_wrong = np.take_along_axis(after['member_logits'],wrong[None,:,None],axis=2)[...,0]
        acquired = common & (new_truth>new_wrong).any(0)
        pool_correct = after['pool_prediction']==truth
        for row in result['cohorts']:
            mask = cohorts[row['cohort']]; eligible = mask & common
            row['baseline_common_competitor_rank_acquisition'] = dict(
                eligible_objects=int(eligible.sum()),
                at_least_one_member_strictly_reverses_baseline_competitor=int((mask&acquired).sum()),
                acquired_and_served_repaired=int((mask&acquired&pool_correct).sum()),
                acquired_but_served_still_wrong=int((mask&acquired&~pool_correct).sum()),
                served_repairs_on_baseline_common_competitor=int((eligible&pool_correct).sum()),
                mean_member_truth_minus_same_competitor_margin_change=contract.mean(np,
                    ((new_truth-new_wrong).astype(np.float64)-(old_truth-old_wrong).astype(np.float64))[:,eligible]),
                competitor_selection='Smallest class strictly above truth in every original baseline member; ties excluded.',
                causal_gradient_mechanism_established=False)
        return result
    analysis.summary,analysis.pair = summary,pair
    analysis.METRICS = tuple(analysis.METRICS)+('pool_minus_mean_member_accuracy',)
    return analysis,original


def stage1_gate(contrasts):
    """Apply the unchanged root-frozen whole9 gate; no survivor-only summary."""
    rows = {row['comparison']:row['metrics'] for row in contrasts}
    common,permuted = rows['route-common'],rows['route-permuted']
    required = [common[key] for key in ('served_accuracy','mean_member_accuracy',
                'worst_member_accuracy','served_nll')]+[permuted['served_accuracy']]
    available = all(row.get('available_seeds')==3 and row.get('mean') is not None
                    and all(x is not None and math.isfinite(x) for x in row['values']) for row in required)
    checks = dict(all_three_paired_accuracy_gains=False,mean_accuracy_gain_at_least_point2_pp=False,
        mean_gain_over_permuted_positive=False,mean_member_loss_at_most_point1_pp=False,
        worst_member_loss_at_most_point2_pp=False,mean_pooled_NLL_no_worse=False)
    if available:
        checks.update(all_three_paired_accuracy_gains=all(x>0 for x in common['served_accuracy']['values']),
            mean_accuracy_gain_at_least_point2_pp=common['served_accuracy']['mean']>=.2,
            mean_gain_over_permuted_positive=permuted['served_accuracy']['mean']>0,
            mean_member_loss_at_most_point1_pp=common['mean_member_accuracy']['mean']>=-.1,
            worst_member_loss_at_most_point2_pp=common['worst_member_accuracy']['mean']>=-.2,
            mean_pooled_NLL_no_worse=common['served_nll']['mean']<=0)
    return dict(whole9_complete_collection_required=True,all_fixed_seed_pairs_available=available,
        checks=checks,stage2_eligible=available and all(checks.values()),
        scientific_release_authorized=False,superiority_or_novelty_established=False,
        interpretation='A passing exploratory gate permits consideration of the unchanged12 competent references; it does not prove quality or acceptance.')


def run(output,collection,pins):
    """Root supplies closed9 custody and the existing selected-state collection."""
    require(collection.get('whole9_complete_before_opening') is True
        and collection.get('owner_and_children_terminal_before_opening') is True
        and collection.get('frozen_target_archive_sha256')==ARCHIVE_SHA,
        'Entire original9 and immutable target/terminal custody required before readout')
    cells = collection['cells']
    require(len(cells)==9 and {(r['seed'],r['condition']) for r in cells}
        == {(s,c) for s in SEEDS for c in CONDITIONS}, 'Whole fixed9 roster required')
    require(all(r['family_status']=='complete' for r in cells), 'No incomplete fit is a result')
    output = Path(output).resolve()
    require(output.is_relative_to(PHASE) and output != PHASE, 'Project research custody')
    analysis,original = configure(pins)
    analysis.run(output,collection,pins)
    import json
    contrasts = json.loads((output/'compact/CONTRASTS.json').read_text())['ordered_comparisons']
    gate = stage1_gate(contrasts)
    original.write(output/'compact/STAGE1_FIXED_GATE.json',gate)
    report = output/'compact/REPORT.md'
    text = report.read_text().replace('# Wiki12 exploratory component attribution','# Context-positive exploratory Stage1')
    text = text.replace('Assess fresh C-P before component attribution.',
        'Assess the fixed whole-population route-common and route-permuted contrasts.')
    text = text.replace('plain-frozen overlapping cohorts','COMMON-frozen overlapping cohorts')
    report.write_text(text+'\nStage2 eligibility: '+str(gate['stage2_eligible'])+'. '+gate['interpretation']+'\n')
    return gate
