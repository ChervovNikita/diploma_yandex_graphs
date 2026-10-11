"""Two frozen Physics primary contrasts. Caller supplies pinned SciPy t law.

Adapted from sealed Penn source, changing the prospective family to two tests.
No provider imports, array access or scientific work at module import.
"""
import itertools
import math
import statistics
from contracts import BLOCKS

def descriptive(values,t_distribution):
    if len(values)!=10 or any(v is None or not math.isfinite(v) for v in values):
        return {'complete_ten_blocks':False,'values_by_block':values,'mean':None,'SD':None,
                'descriptive_95pct_t_interval':None,'inconclusive':True,'no_complete_case_subset':True}
    mean,sd=statistics.mean(values),statistics.stdev(values);se=sd/math.sqrt(10)
    critical=float(t_distribution.ppf(.975,9))
    if not math.isfinite(critical):
        return {'complete_ten_blocks':True,'values_by_block':values,'inconclusive':True,
                'failure':'Pinned t quantile nonfinite; no replacement'}
    return {'complete_ten_blocks':True,'values_by_block':values,'mean':mean,'SD':sd,'SE':se,'df':9,
            'descriptive_95pct_t_interval':[mean-critical*se,mean+critical*se],
            'scope':'Conditional split/optimizer randomness on one full graph; no independent-node/domain inference'}

def holm(rows):
    previous=0.
    for rank,(key,value) in enumerate(sorted(rows.items(),key=lambda x:(x[1]['p_one_sided'],x[0]))):
        previous=max(previous,(len(rows)-rank)*value['p_one_sided'])
        value['Holm_adjusted_p']=min(1.,previous);value['Holm_rank']=rank+1
        value['Holm_superiority_pass']=value.get('complete_ten_blocks',False) and not value.get('inconclusive',False) and previous<=.05
    return rows

def contrast(integer_differences,denominator,t_distribution,primary=True):
    if len(integer_differences)!=10 or any(type(v) is not int for v in integer_differences):
        return {'complete_ten_blocks':False,'paired_correct_count_differences':integer_differences,
                'p_one_sided':1.,'inconclusive':True,'no_complete_case_subset':True}
    d=[100*v/denominator for v in integer_differences]
    result=descriptive(d,t_distribution);mean=result.get('mean');sd=result.get('SD')
    result.update(paired_correct_count_differences=integer_differences,accuracy_pp_by_block=d,
                  positive_blocks=sum(v>0 for v in integer_differences),negative_blocks=sum(v<0 for v in integer_differences),
                  zero_blocks=sum(v==0 for v in integer_differences),p_one_sided=1.,inconclusive=True)
    result['paired_95pct_interval_pp']=result.get('descriptive_95pct_t_interval')
    if mean is not None and sd and sd>0:
        statistic=mean/(sd/math.sqrt(10));p=float(t_distribution.sf(statistic,9))
        critical=float(t_distribution.ppf(.9875,9))  # 97.5% two-sided for two Bonferroni intervals.
        if all(math.isfinite(v) for v in (statistic,p,critical)):
            result.update(t=statistic,p_one_sided=p,inconclusive=False,
                simultaneous_Bonferroni_two_contrast_97_5pct_interval_pp=
                [mean-critical*sd/math.sqrt(10),mean+critical*sd/math.sqrt(10)] if primary else None)
    else:
        result['degenerate_undefined_t']=True  # No replacement test or invented rejection.
    observed=sum(integer_differences)
    signed=[sum(sign*v for sign,v in zip(signs,integer_differences)) for signs in itertools.product((-1,1),repeat=10)]
    result.update(symmetry_enumerations=1024,symmetry_one_sided=sum(v>=observed for v in signed)/1024,
                  symmetry_two_sided=sum(abs(v)>=abs(observed) for v in signed)/1024,
                  symmetry_assumption='Symmetric/exchangeable paired block differences; fixed seed IDs do not guarantee it',
                  symmetry_is_sensitivity_not_primary_replacement_or_extra_veto=True,
                  observed_mean_at_least_0_2pp=mean is not None and mean>=.2,
                  point_margin_not_proven_population_lower_bound=True)
    return result

def adopted_report(scores,t_distribution,denominator):
    def one(reference,primary):
        differences=[]
        for block in BLOCKS:
            a,b=scores.get((block,'E4')),scores.get((block,reference))
            differences.append(None if a is None or b is None else int(a['counts']['pooled_correct']-b['counts']['pooled_correct']))
        return contrast(differences,denominator,t_distribution,primary)
    primary=holm({ref:one(ref,True) for ref in ('BASE','equipped_I4')})
    superiority=all(r['Holm_superiority_pass'] for r in primary.values())
    practical=all(r.get('observed_mean_at_least_0_2pp',False) for r in primary.values())
    attribution=one('N4',False)
    attribution['conditional_gate_open']=superiority
    attribution['observed_mean_at_least_0_1pp']=attribution.get('mean') is not None and attribution['mean']>=.1
    attribution['conditional_superiority_pass']=superiority and not attribution['inconclusive'] and attribution['p_one_sided']<=.05
    attribution['conditional_attributed_practical_pass']=attribution['conditional_superiority_pass'] and attribution['observed_mean_at_least_0_1pp']
    return {'primary':primary,'primary_Holm_superiority_pass':superiority,'primary_observed_practical_screen_pass':practical,
            'primary_joint_evidence_pass':superiority and practical,'conditional_endpoint_attribution':attribution,
            'primary_rule':'Exactly two raw integer-accuracy E4 contrasts: one-sided t at zero gain, Holm alpha .05; both observed means >=.2pp reported separately',
            'simultaneous_interval_rule':'Two 97.5% two-sided Bonferroni intervals (family coverage at least95%); t(.9875,9)',
            'no_all_positive_block_veto':True,'missing_pairs_inconclusive_no_survivor_subset':True,
            'scope':'Ten paired class-stratified mask/optimizer blocks on one complete graph, not ten graph domains',
            'calibrated_accuracy_not_second_primary_test':True,'no_novelty_or_manuscript_verdict':True}
