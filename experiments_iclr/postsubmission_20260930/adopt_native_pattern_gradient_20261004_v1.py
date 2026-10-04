"""Authenticate completed scalar diagnostic and record its bounded interpretation."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path

P=Path(__file__).resolve().parent
E=P/'graph_count_conditioned_pattern_minimal_gradient_execution_root_20261004_v2'
O=E/'owned_monitor02'
D=P/'graph_count_conditioned_pattern_native_gradient_root_adoption_20261004_v1'
D.mkdir()
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def read(relative):return json.loads((O/relative).read_text())
def write(name,value):
    with (D/name).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')

observation=read('OBSERVATION.json');terminal=read('supervision/run01/TERMINAL.json')
custody=read('supervision/run01/SUPERVISOR_CUSTODY.json')
final=read('run01/FINAL_CUSTODY.json');result=read('run01/DIAGNOSTIC.json')
assert observation['supervisor_identity'] is None
assert terminal['status']=='COMPLETE_DIAGNOSTIC_ONLY' and terminal['physical_exit_code']==0
assert terminal['physical_session_closed'] and terminal['direct_child_reaped']
assert terminal['stop'] is None and not terminal['errors'] and not terminal['unresolved_cleanup']
assert terminal['linked']['status']=='COLLECTED_COMPLETE_DIAGNOSTIC_ONLY'
assert result['source_manifest_sha256']=='3fd8e1899132a0a0298f743e7a9179bc302aee24b5178e74ec20caeece7abb23'
assert result['core_manifest_sha256']=='92ae9f79c15cf6de53e39f06c241780b8089d652d59070cf9d23ed521e2740cf'
assert terminal['linked']['result_sha256']==sha(O/'run01/DIAGNOSTIC.json')
assert final['completed'] and terminal['linked']['final_custody_sha256']==sha(O/'run01/FINAL_CUSTODY.json')
assert terminal['linked']['file_custody_sha256']==final['file_custody_sha256']==sha(O/'run01/FILE_CUSTODY.json')
assert custody['terminal_sha256']==sha(O/'supervision/run01/TERMINAL.json')
for row in custody['child_output_files']:
    f=O/'run01'/row['path'];assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
terminal_row=next(r for r in custody['supervisor_output_files'] if r['path']=='TERMINAL.json')
assert terminal_row['sha256']==sha(O/'supervision/run01/TERMINAL.json')
assert result['optimizer_constructed'] is False and result['optimizer_updates']==result['fits']==0
assert result['VALID_TEST_reads'] is False and result['no_predictive_or_novelty_claim'] is True
assert result['workload']['positive_queries']==result['workload']['negative_queries']==65536
assert result['dispatch']['encoder_calls']==1 and result['dispatch']['reverse_evaluations']==3
blocks={}
for name,b in result['parameter_blocks'].items():
    assert all(v['finite'] for v in b['objectives'].values()) and b['shared_minus_separated']['finite']
    blocks[name]=dict(target_L2=b['objectives']['base']['L2'],joint_L2=b['objectives']['J_K']['L2'],
        separate_L2=b['objectives']['J_K_sep']['L2'],delta_L2=b['shared_minus_separated']['L2'],
        delta_max_abs=b['shared_minus_separated']['max_abs'],
        joint_to_target_L2=b['auxiliary_to_target']['J_K']['auxiliary_to_target_ratios']['L2'],
        joint_target_cosine=b['auxiliary_to_target']['J_K']['cosine'])
totals={key:math.sqrt(sum(b[key]**2 for b in blocks.values())) for key in ('target_L2','joint_L2','separate_L2','delta_L2')}
totals['joint_to_target_L2']=totals['joint_L2']/totals['target_L2']
totals['delta_to_target_L2']=totals['delta_L2']/totals['target_L2']
totals['delta_to_joint_L2']=totals['delta_L2']/totals['joint_L2']
populations={}
for name,b in result['populations'].items():
    census={k:b['census'][k] for k in ('queries','both_sides_nonconstant','both_sides_genuine_subset','either_side_genuine_subset','joint_group_counts')}
    sides={}
    for side,strata in b['slot_gradients'].items():
        sides[side]={}
        for stratum in ('both_variable','genuine_r_gt1','forced_counterpart','forced_r0'):
            s=strata[stratum]
            sides[side][stratum]=dict(slot_count=s['slot_count'],
                delta=s['shared_minus_separated'],joint_L2=s['J_K']['L2'],separate_L2=s['J_K_sep']['L2'])
        assert strata['forced_r0']['exact_zero_forced_gradients']
        assert strata['forced_counterpart']['shared_minus_separated']['max_abs']==0
    populations[name]=dict(census=census,slot_gradients=sides)
assert populations['negative']['census']['both_sides_nonconstant']==0
summary=dict(UTC=datetime.now(timezone.utc).isoformat(),status='ADOPTED_NATIVE_SAME_STATE_GRADIENT_DIAGNOSTIC_ONLY',
    diagnostic_sha256=sha(O/'run01/DIAGNOSTIC.json'),terminal_sha256=sha(O/'supervision/run01/TERMINAL.json'),
    source_manifest_sha256=result['source_manifest_sha256'],core_manifest_sha256=result['core_manifest_sha256'],
    losses=result['losses'],parameter_blocks=blocks,aggregate_disjoint_parameter_L2=totals,
    populations=populations,times_seconds=result['times_seconds'],dispatch=result['dispatch'],
    child_wall_seconds=result['inclusive_child_wall_seconds'],
    supervisor_wall_seconds=terminal['wall_seconds'],
    sampled_session_peak_RSS_bytes=terminal['peak_observed_session_RSS_bytes'],
    CUDA_peak_allocated_bytes=result['cuda_peak_allocated_bytes'],CUDA_peak_reserved_bytes=result['cuda_peak_reserved_bytes'],
    physical_session_closed=True,optimizer_updates=0,VALID_TEST_reads=False,
    current_predictive_scores_unchanged=True,new_methodological_advantage=False,manuscript_acceptance=False,
    scientific_interpretation='Native auxiliary derivatives reach encoder/member parameters and joint-versus-separated parameter gradients differ. The association difference is small; predictive transfer is untested.',
    limits=['One initialization and one fixed full native TRAIN batch.',
        'All per-slot differences agree under the unchanged absolute/relative tolerance; that tolerance is large relative to mean-reduced slot derivatives and is not an effect-size criterion.',
        'No optimizer update, full TRAIN epoch, full VALID, calibration, or inference benefit evaluated.',
        'Cosine alignment and nonzero derivatives do not predict a generalization gain.',
        'Measured three-reverse diagnostic is not a whole-fit ETA; dynamic dispatch is a practical bottleneck.'],
    next_action='Inspect an exact-law support-bucket optimization, then qualified representative joint/separate/target-only fits if feasible; no favourable-batch or coefficient substitution.')
write('ROOT_ADOPTION.json',summary)
lines=['# Actual native TRAIN gradient diagnostic','',
    'The exact corrected source completed one full native TRAIN batch (65,536 positives and65,536 negatives) on18.77 GPU1. One retained forward graph supplied target, joint J_K and separate-side J_K_sep gradients. The original supervisor exited, its child was reaped with exit0, collection/custody matched, and the physical session closed. No optimizer or VALID/TEST evaluation ran.','',
    '## Signal','',
    f"Joint loss{result['losses']['J_K']:.9f}; separate-side loss{result['losses']['J_K_sep']:.9f}; target loss{result['losses']['base']:.9f}.",'',
    f"Across disjoint model blocks, joint auxiliary gradient norm is{totals['joint_to_target_L2']*100:.3f}% of target-gradient norm. Joint-minus-separated norm is{totals['delta_to_target_L2']*100:.4f}% of target norm and{totals['delta_to_joint_L2']*100:.3f}% of joint-auxiliary norm. It reaches both encoder and member factors. These are descriptive values at one initialization, not predictive improvements.",'',
    'Positive queries:3,800 have both sides variable and886 have genuine subset choices on both sides. Negative queries have no both-variable support. Forced-side gradients are exactly zero, and forced-counterpart joint/separate gradients exactly agree. Every mean-reduced per-slot delta is within the unchanged tolerance; that absolute tolerance exceeds these small derivatives. Parameter-block deltas are independently visible in the recorded scalar norms. No rule was loosened.','',
    '## Cost and decision','',
    f"Child work took{result['inclusive_child_wall_seconds']:.2f}s; forward{result['times_seconds']['native_and_auxiliary_forward']:.2f}s; joint reverse{result['times_seconds']['reverse_J_K']:.2f}s; separate reverse{result['times_seconds']['reverse_J_K_sep']:.2f}s. Peak allocator:allocated{result['cuda_peak_allocated_bytes']/1e9:.2f}GB,reserved{result['cuda_peak_reserved_bytes']/1e9:.2f}GB. The implementation dispatched1,505 genuine ESP groups and132,447 slot loops. These are this diagnostic's costs, not a full-fit estimate.",'',
    'The mechanism is connected and produces a small parameter-gradient distinction. Predictive usefulness remains untested. Investigate the separately disabled exact-law batching optimization before paying for a broad study. Preserve normalization, coefficient, complete batch, precision and teacher semantics. Representative paired target-only/joint/separate fits and an independent competitive benchmark are still needed. No novelty, superiority or acceptance follows from this diagnostic.','',
    'The prior startup failure and all source/review versions remain preserved. Original paper and current predictive scores are unchanged.']
(D/'RESULTS_SUMMARY.md').write_text('\n'.join(lines)+'\n')
write('SEAL.json',dict(UTC=summary['UTC'],files=[dict(path=f.name,bytes=f.stat().st_size,sha256=sha(f))
    for f in sorted(D.iterdir()) if f.is_file()],numeric_execution=False,current_score_recalculation=False))
print(json.dumps(dict(directory=D.name,aggregate=totals,predictive_improvement=False)))
