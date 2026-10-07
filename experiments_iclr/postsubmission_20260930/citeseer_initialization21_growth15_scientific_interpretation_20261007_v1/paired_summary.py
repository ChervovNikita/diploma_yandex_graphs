"""Descriptive paired summaries for an already closed/audited metric packet.

Input rows are taken at each source-selected cycle from existing HISTORY and
selected VALID artifacts after closure. This script loads no model/checkpoint,
opens no dataset, does no inference, and has no job/TEST/remote interface.
"""
import argparse
import json
import math
import statistics

CONDITIONS={
    'initialization':['random_signs_all','tabm_first_normal','warm_identity','graph_covariance','feature_covariance','single','independent_warm4'],
    'growth':['graph_growth','unfiltered_growth','unfiltered_top8_partition','capable_single_rank8','independent_graph_growth4']}
CONTRASTS={
    'initialization':[('graph_covariance',c) for c in ('feature_covariance','warm_identity','tabm_first_normal','random_signs_all','single','independent_warm4')],
    'growth':[('graph_growth',c) for c in ('unfiltered_growth','unfiltered_top8_partition','capable_single_rank8','independent_graph_growth4')]}


def describe(values):
    mean=statistics.mean(values);sd=statistics.stdev(values)
    # df2 interval describes optimizer-seed dispersion under strong assumptions;
    # it is not a graph/test-population confidence interval or significance gate.
    half=4.302652729911275*sd/math.sqrt(3)
    return {'values':values,'mean':mean,'sample_SD':sd,'range':[min(values),max(values)],
        'exploratory_t95_interval_df2':[mean-half,mean+half]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--records',required=True)
    args=parser.parse_args();packet=json.loads(open(args.records).read())
    family=packet['family'];conditions=CONDITIONS[family]
    if packet.get('family_complete_and_audited') is not True or packet.get('TEST_access') is not False:
        raise ValueError('Actual full-family closure/audit before metric opening')
    rows=packet['records'];expected={(s,c) for s in (0,1,2) for c in conditions}
    keyed={(r['seed'],r['condition']):r for r in rows}
    if len(rows)!=len(expected) or set(keyed)!=expected:raise ValueError('Complete prespecified roster, no seed/arm omission')
    for row in rows:
        if row.get('completed_cycles')!=60 or row.get('selected_cycle') not in range(5,61,5):raise ValueError('Frozen original60/cadence5 endpoint')
        count=1 if row['condition'] in ('single','capable_single_rank8') else 4
        if len(row['member_MRR'])!=count or any(not math.isfinite(x) or not 0<=x<=1 for x in [row['MRR'],row['Hits10'],*row['member_MRR']]):
            raise ValueError('All member quality at the same selected bank')
    report={'family':family,'conditions':{},'paired_contrasts':{},'TEST_access':False,
        'uncertainty_scope':'Three optimizer seeds on one fixed graph/split; queries/negatives are not independent graph replications. VALID selector optimism and rounding4 remain.'}
    for condition in conditions:
        group=[keyed[(s,condition)] for s in (0,1,2)]
        report['conditions'][condition]={
            'MRR':describe([r['MRR'] for r in group]),'Hits10':describe([r['Hits10'] for r in group]),
            'mean_member_MRR':describe([statistics.mean(r['member_MRR']) for r in group]),
            'minimum_member_MRR':describe([min(r['member_MRR']) for r in group]),
            'pool_minus_mean_member_MRR':describe([r['MRR']-statistics.mean(r['member_MRR']) for r in group]),
            'selected_cycles':[r['selected_cycle'] for r in group]}
    for a,b in CONTRASTS[family]:
        values=[keyed[(s,a)]['MRR']-keyed[(s,b)]['MRR'] for s in (0,1,2)]
        report['paired_contrasts'][a+' minus '+b]={**describe(values),'positive_seeds':sum(x>0 for x in values),
            'negative_seeds':sum(x<0 for x in values),'zero_seeds':sum(x==0 for x in values)}
    print(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
