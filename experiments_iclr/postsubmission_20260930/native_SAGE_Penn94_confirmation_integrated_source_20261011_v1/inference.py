"""Root-adopted paired-seed inference. Caller supplies pinned SciPy t distribution.

No data/provider imports, source execution, tests or arrays at module import.
"""
import itertools
import math
import statistics


def holm(rows, p_key='p_one_sided'):
    ordered = sorted(rows.items(), key=lambda item:(item[1][p_key],item[0]))
    previous, count = 0., len(ordered)
    for rank,(key,value) in enumerate(ordered):
        previous=max(previous,(count-rank)*value[p_key])
        value['Holm_adjusted_p']=min(1.,previous)
        value['Holm_rank']=rank+1
        value['Holm_pass']=value['Holm_adjusted_p']<=.05
    return rows


def contrast(integer_differences, denominator, t_distribution, primary=True):
    if len(integer_differences)!=10 or any(type(v) is not int for v in integer_differences):
        return {'complete_ten_blocks':False,'p_one_sided':1.,'symmetry_one_sided':1.,'failed_retained':True,'no_complete_case_subset':True}
    d=[100*v/denominator for v in integer_differences]
    mean,sd=statistics.mean(d),statistics.stdev(d)
    se=sd/math.sqrt(10)
    if sd==0:
        statistic,interval,simultaneous,p=None,None,None,1.
        degenerate=True
    else:
        statistic=mean/se;p=float(t_distribution.sf(statistic,9));degenerate=False
        critical=float(t_distribution.ppf(.975,9));family_critical=float(t_distribution.ppf(.99375,9))
        interval=[mean-critical*se,mean+critical*se]
        simultaneous=[mean-family_critical*se,mean+family_critical*se] if primary else None
        if not all(math.isfinite(v) for v in [statistic,p,critical,family_critical]+interval):
            return {'complete_ten_blocks':True,'p_one_sided':1.,'symmetry_one_sided':1.,'failed_retained':True,'failure':'Pinned t CDF/quantile nonfinite; no replacement'}
    observed=sum(integer_differences)
    signed=[sum(sign*v for sign,v in zip(signs,integer_differences)) for signs in itertools.product((-1,1),repeat=10)]
    return {'complete_ten_blocks':True,'paired_correct_count_differences':integer_differences,'accuracy_pp_by_block':d,
        'mean_pp':mean,'unbiased_SD_pp':sd,'SE_pp':se,'min_pp':min(d),'max_pp':max(d),
        'positive_blocks':sum(v>0 for v in integer_differences),'negative_blocks':sum(v<0 for v in integer_differences),
        't':statistic,'df':9,'p_one_sided':p,'paired_95pct_interval_pp':interval,
        'simultaneous_Bonferroni_four_contrast_95pct_interval_pp':simultaneous,'degenerate_undefined_t':degenerate,
        'symmetry_enumerations':1024,'symmetry_one_sided':sum(v>=observed for v in signed)/1024,
        'symmetry_two_sided':sum(abs(v)>=abs(observed) for v in signed)/1024,
        'symmetry_is_sensitivity_not_primary_replacement_or_extra_veto':True,
        'mean_at_least_practical_0_2pp':mean>=.2,
        'conditional_graph_and_optimizer_block_assumptions':'Approximately independent normal block differences for t; symmetry/exchangeability for signed sensitivity. Fixed seed IDs guarantee neither; nodes/I4 bodies are not units.'}


def adopted_report(scores, t_distribution, arms, seeds, denominator):
    candidate=arms[0]+'__probability_mean'
    primary_refs={'P_original':arms[1]+'__logit_mean','P_I4':arms[5]+'__probability_mean',
        'P_ordinary_M1':arms[3]+'__probability_mean','P_factor_M1':arms[4]+'__probability_mean'}
    def one(a,b,primary):
        if any(scores[s,a] is None or scores[s,b] is None for s in seeds):
            return contrast([],denominator,t_distribution,primary)
        differences=[int(scores[s,a]['counts']['pooled_correct']-scores[s,b]['counts']['pooled_correct']) for s in seeds]
        return contrast(differences,denominator,t_distribution,primary)
    primary=holm({key:one(candidate,ref,True) for key,ref in primary_refs.items()})
    symmetry=holm({key:{'p_one_sided':row['symmetry_one_sided']} for key,row in primary.items()})
    passed=all(row['complete_ten_blocks'] and row['Holm_pass'] and row.get('mean_at_least_practical_0_2pp',False) for row in primary.values())
    secondary_pairs={'S_matched_placement':(candidate,arms[2]+'__probability_mean'),
        'S_current_same_state_pool':(candidate,arms[0]+'__logit_mean'),
        'S_boundary_same_state_pool':(arms[2]+'__probability_mean',arms[2]+'__logit_mean')}
    secondary=holm({key:one(a,b,False) for key,(a,b) in secondary_pairs.items()})
    for row in secondary.values():
        row['confirmatory_gate_open']=passed
        row['gated_attribution_pass']=passed and row['complete_ten_blocks'] and row['Holm_pass']
        row['practical_margin_not_adopted']=True
    return {'primary':primary,'primary_symmetry_Holm_sensitivity':symmetry,'primary_confirmation_pass':passed,
        'primary_rule':'All four Holm one-sided positive-null adjustedp<=.05 and all four observed means>=.2pp; no all-ten-positive veto.',
        'secondary':secondary,'secondary_gate_open':passed,'secondary_outcomes_reported_when_gate_closed':True,
        'stronger_margin_null_not_adopted':True,'point_size_not_proven_lower_bound_above_0_2pp':True,
        'conditional_one_graph_population_only':True,'LINKX_descriptive_outside_primary_and_secondary_families':True,
        'confidence_screen_separate_from_raw_accuracy':True,'no_novelty_or_manuscript_verdict':True}
