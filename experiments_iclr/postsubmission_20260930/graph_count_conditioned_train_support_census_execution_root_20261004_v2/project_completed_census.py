"""Produce a pinned compact projection; retain full census JSON on the server."""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('completed_census_driver', HERE / 'stage_admit_census.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
observed = json.loads((HERE / 'owned_monitor01/OBSERVATION.json').read_text())
pins = {row['path']: row for row in observed['large_server_retained_metadata']}
assert set(pins) == {'run01/seed%d_COUNTS.json' % s for s in (0, 1, 2)}
code = driver.context() + 'pins=' + repr(pins) + '\n'
code += '''
terminal=json.loads((root/'supervision/run01/TERMINAL.json').read_text())
assert terminal['status']=='COMPLETE_TRAIN_METADATA_ONLY'
linked=terminal['linked'];assert linked['status']=='COLLECTED_COMPLETE_TRAIN_METADATA_ONLY'
rows=[]
for seed in (0,1,2):
 relative='run01/seed%d_COUNTS.json'%seed;p=root/relative;pin=pins[relative]
 raw=p.read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256']
 assert next(r for r in linked['seed_files'] if Path(r['path'])==p)['sha256']==pin['sha256']
 stream=json.loads(raw);assert stream['seed']==seed and len(stream['rows'])==17
 projected=dict(seed=seed,source_descriptor=pin,negative_draw_sha256=stream['negative_draw_sha256'],
  permutation_sha256=stream['permutation_sha256'],dropped_tail_sha256=stream['dropped_tail_sha256'],rows=[])
 for expected_batch,batch in enumerate(stream['rows'],1):
  assert batch['batch']==expected_batch and set(batch['populations'])=={'positive','negative'}
  item=dict(batch=expected_batch,record_ids_sha256=batch['record_ids_sha256'],populations={})
  for name,pop in batch['populations'].items():
   assert pop['queries']==65536 and len(pop['sides'])==2
   joint=pop['joint_group_counts'];assert len(joint)==3 and all(len(r)==3 for r in joint)
   assert all(type(x)==int and x>=0 for r in joint for x in r) and sum(map(sum,joint))==65536
   assert pop['both_sides_nonconstant']==sum(joint[a][b] for a in (1,2) for b in (1,2))
   assert pop['both_sides_genuine_subset']==joint[2][2]
   assert pop['either_side_genuine_subset']==sum(joint[2])+sum(r[2] for r in joint)-joint[2][2]
   side_rows=[]
   for side,summary in enumerate(pop['sides']):
    table=summary['n_k_frequency'];assert len(table)==len({(n,k) for n,k,f in table})
    assert all(type(n)==type(k)==type(f)==int and 0<=k<=n and f>0 for n,k,f in table)
    assert sum(f for n,k,f in table)==65536
    exact=dict(empty=sum(f for n,k,f in table if n==0),zero_count=sum(f for n,k,f in table if k==0),
      full_count=sum(f for n,k,f in table if n>0 and k==n),
      categorical_or_complement=sum(f for n,k,f in table if min(k,n-k)==1),
      genuine_subset=sum(f for n,k,f in table if min(k,n-k)>1),
      total_slots=sum(n*f for n,k,f in table),total_observed_bits=sum(k*f for n,k,f in table),
      max_n=max(n for n,k,f in table),max_k=max(k for n,k,f in table),
      max_r=max(min(k,n-k) for n,k,f in table),n_times_r_sum=sum(n*min(k,n-k)*f for n,k,f in table))
    assert all(summary[k]==v for k,v in exact.items())
    groups={}
    for n,k,f in table:
     pair=(n,min(k,n-k));groups[pair]=groups.get(pair,0)+f
    marginal=[sum(f for (n,r),f in groups.items() if (0 if r==0 else 1 if r==1 else 2)==g) for g in range(3)]
    expected=[sum(joint[g]) if side==0 else sum(row[g] for row in joint) for g in range(3)]
    assert marginal==expected
    qualified=[(n,r,f) for (n,r),f in groups.items() if r>1]
    side_rows.append(dict(exact,distinct_n_r_groups=len(groups),genuine_groups=len(qualified),
      genuine_group_slot_loops=sum(n for n,r,f in qualified),
      genuine_transition_work=sum(n*r*f for n,r,f in qualified),
      maximum_genuine_group_queries=max((f for n,r,f in qualified),default=0),
      largest_group_rolling_count_cells=max((f*(r+1) for n,r,f in qualified),default=0),
      group_counts_verified=True))
   item['populations'][name]={k:v for k,v in pop.items() if k!='sides'}
   item['populations'][name]['sides']=side_rows
  projected['rows'].append(item)
 rows.append(projected)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status='FULL_CENSUS_PINNED_INTEGER_PROJECTION',
 streams=rows,histogram_rows_recomputed_and_checked=True,original_histograms_retained_server_side=True,
 source_terminal_sha256=hashlib.sha256((root/'supervision/run01/TERMINAL.json').read_bytes()).hexdigest(),
 optimizer_updates=0,models_or_features_or_VALID_TEST_read=False,
 interpretation='Descriptive TRAIN support and arithmetic cost only; no predictive or novelty claim.')))
'''
value = driver.transport('compact_integer_projection01', code)
with (HERE / 'FULL_CENSUS_COMPACT_PROJECTION.json').open('x') as stream:
    json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
summary = []
for name in ('positive', 'negative'):
    populations = [batch['populations'][name] for seed in value['streams'] for batch in seed['rows']]
    queries = sum(p['queries'] for p in populations)
    counts = {key: sum(p[key] for p in populations) for key in (
        'both_sides_nonconstant', 'both_sides_genuine_subset', 'either_side_genuine_subset')}
    summary.append(dict(population=name, query_occurrences=queries, **counts,
        fractions={key: n / queries for key, n in counts.items()},
        per_batch_both_nonconstant_range=[min(p['both_sides_nonconstant']/p['queries'] for p in populations),
            max(p['both_sides_nonconstant']/p['queries'] for p in populations)],
        per_side_max_genuine_groups=max(s['genuine_groups'] for p in populations for s in p['sides']),
        per_side_max_genuine_slot_loops=max(s['genuine_group_slot_loops'] for p in populations for s in p['sides']),
        per_side_max_genuine_transition_work=max(s['genuine_transition_work'] for p in populations for s in p['sides']),
        per_side_max_slots=max(s['total_slots'] for p in populations for s in p['sides'])))
output = dict(status=value['status'],populations=summary,optimizer_updates=0,
    interpretation=value['interpretation'])
with (HERE / 'FULL_CENSUS_AGGREGATE.json').open('x') as stream:
    json.dump(output, stream, indent=2, allow_nan=False); stream.write('\n')
print(json.dumps(output, indent=2))
