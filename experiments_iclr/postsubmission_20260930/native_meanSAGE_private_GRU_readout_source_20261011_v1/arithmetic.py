"""Exact inherited probability/readout arithmetic; no numerical import at module load."""
import itertools
import json
import math
import resource
import statistics
import time
from runtime import BLOCKS,checked,require

def owner_closed(item):
    value=json.loads(checked(item).read_text())
    require(value['status']=='COMPLETE_EXIT_0' and value['exit_code']==0 and value['direct_child_wait'] and value['child_pid_absent'] and value['owned_cuda_pid_absent'],'Successful prior direct finite owner exit/reap required')
    return value


def guard(config,start):
    require(time.monotonic()-start<config['limits']['wall_seconds'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=config['limits']['rss_bytes'],'Finite reader CPU wall/RSS cap')


def truth(np,custodian,b,role):
    prefix='calibration_only' if role=='cal' else 'scorer_only';item=custodian['files'][f'{prefix}/block_{b}_{role}.npz']
    with np.load(checked(item),allow_pickle=False) as a:
        require(set(a.files)=={'ids','y'},'Role-specific truth schema');ids,y=a['ids'].copy(),a['y'].copy()
    require(ids.dtype==y.dtype==np.int64 and ids.shape==y.shape==((150,) if role=='cal' else (34093,)) and len(np.unique(ids))==len(ids) and (y>=0).all() and (y<5).all(),'Complete fixed truth role');return ids,y


def logprob(np,z):
    shifted=z-z.max(-1,keepdims=True);return shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))


def calibrated(np,lp,T):return np.exp(logprob(np,lp.astype(np.float64)/T)).mean(0)


def descriptive(v,signs=False):
    require(len(v)==3 and all(math.isfinite(x) for x in v),'Complete three-block vector');mean=statistics.mean(v);sd=statistics.stdev(v);half=4.302652729911275*sd/math.sqrt(3)
    r=dict(blocks=list(BLOCKS),values=v,mean=mean,sample_SD=sd,descriptive_95_CI=[mean-half,mean+half],scope='Three custom paired protocols on one encountered graph; descriptive only')
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


