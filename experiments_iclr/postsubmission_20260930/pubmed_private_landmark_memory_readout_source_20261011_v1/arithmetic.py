"""Inherited readout arithmetic; PubMed roles and population kept explicit."""
import itertools
import math
import statistics
from runtime import SEEDS,checked,require

def logprob(np,z):
    shifted=z-z.max(-1,keepdims=True);return shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))


def calibrated(np,lp,T):return np.exp(logprob(np,lp.astype(np.float64)/T)).mean(0)


def descriptive(v,signs=False):
    require(len(v)==3 and all(math.isfinite(x) for x in v),'Complete three-seed vector');mean=statistics.mean(v);sd=statistics.stdev(v)
    r=dict(seeds=list(SEEDS),values=v,mean=mean,sample_SD=sd,scope='Three optimizer seeds on one encountered PubMed graph and the same fixed DEV nodes; descriptive repeat spread only; no seed or node generalization interval')
    if signs:r['eight_sign_flip_assignments']=[dict(signs=list(s),mean=statistics.mean(x*z for x,z in zip(v,s))) for s in itertools.product((-1,1),repeat=3)]
    return r


def delta(a,b,k):
    n=a['counts']['nodes'];require(n==b['counts']['nodes'],'Paired population')
    if k in ('accuracy_pct','calibrated_accuracy_pct'):
        field='served_correct' if k=='accuracy_pct' else 'calibrated_served_correct';return 100*(a['counts'][field]-b['counts'][field])/n
    if k=='mean_member_accuracy_pct':
        ac,bc=a['counts']['member_correct'],b['counts']['member_correct'];return 100*(sum(ac)*len(bc)-sum(bc)*len(ac))/(len(ac)*len(bc)*n)
    return a['quality'][k]-b['quality'][k]


def pair(a,b,mask):
    repair=(~b['P'] & a['P']) & mask;harm=(b['P'] & ~a['P']) & mask
    c=dict(repairs=int(repair.sum()),harms=int(harm.sum()),coverage_added=int((a['V'] & ~b['V'] & mask).sum()),coverage_removed=int((b['V'] & ~a['V'] & mask).sum()),introduced_common_wrong_rivals=int((a['Q'] & ~b['Q'] & mask).sum()),reference_common_rivals_covered=int((b['Q'] & a['V'] & mask).sum()),reference_common_rivals_served_repaired=int((b['Q'] & a['P'] & mask).sum()),repair_causes=dict(new_member_alternative=int((repair & a['V'] & ~b['V']).sum()),both_covered=int((repair & a['V'] & b['V']).sum()),aggregation_only=int((repair & ~a['V']).sum())),harm_causes=dict(available_alternative_lost=int((harm & a['V']).sum()),removed_alternative=int((harm & ~a['V'] & b['V']).sum()),lost_aggregation_only=int((harm & ~a['V'] & ~b['V']).sum())))
    d=dict(served_correct=int((a['P'] & mask).sum()-(b['P'] & mask).sum()),coverage=int((a['V'] & mask).sum()-(b['V'] & mask).sum()),pooling_losses=int((a['V'] & ~a['P'] & mask).sum()-(b['V'] & ~b['P'] & mask).sum()),aggregation_only_correct=int((~a['V'] & a['P'] & mask).sum()-(~b['V'] & b['P'] & mask).sum()))
    require(d['served_correct']==c['repairs']-c['harms']==d['coverage']-d['pooling_losses']+d['aggregation_only_correct'],'Exact paired decomposition')
    require(sum(c['repair_causes'].values())==c['repairs'] and sum(c['harm_causes'].values())==c['harms'],'Gross partition completeness');c['delta_counts']=d;return c


def truth(np,custodian,role):
    item=custodian['role_files'][role]
    with np.load(checked(item),allow_pickle=False) as archive:
        require(set(archive.files)=={'ids','y'},'Exact role-specific target schema')
        ids,y=archive['ids'].copy(),archive['y'].copy()
    expected=[273,515,525] if role=='cal' else [274,517,525]
    count=1313 if role=='cal' else 1316
    require(ids.dtype==y.dtype==np.int64 and ids.shape==y.shape==(count,)
        and len(np.unique(ids))==count and ids.min()>=0 and ids.max()<19717
        and y.min()>=0 and y.max()<3 and np.bincount(y,minlength=3).tolist()==expected,
        'Complete unchanged PolyFormer CAL/DEV role and class quotas')
    require(all((y[sum(expected[:c]):sum(expected[:c+1])]==c).all() for c in range(3)),
        'Original within-class order and class-concatenated role custody')
    return ids,y
