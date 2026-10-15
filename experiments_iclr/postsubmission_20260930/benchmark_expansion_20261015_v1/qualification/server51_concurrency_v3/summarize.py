"""Scalar timing summary; one fixed workload pair, no prediction-quality claim."""
import statistics

def summary(results):
 for phase in ('native_alone','structured_alone','concurrent'):
  expected={'native','structured'} if phase=='concurrent' else {phase.removesuffix('_alone')}
  if set(results[phase])!=expected:raise ValueError('Incomplete phase arms')
  for arm,row in results[phase].items():
   if row['status']!='PASS' or row['TEST_accessed'] or row['TRAIN_epochs_completed']!=110 or row['VALID_calls_completed']!=110 or row['optimizer_calls']!={'attempts':[110],'completed':[110]}:raise ValueError('Exact complete engineering cost required')
   timed=[r for r in row['epochs'] if r['stage']=='timed']
   if len(timed)!=100:raise ValueError('Fixed100epoch timing required')
   if any(r['seconds']<=0 or r['end_monotonic_ns']<=r['start_monotonic_ns'] for r in timed):raise ValueError('Finite increasing epoch times required')
 n=results['native_alone']['native'];s=results['structured_alone']['structured'];pn=results['concurrent']['native'];ps=results['concurrent']['structured']
 def wall(row):return (row['measurement_end_monotonic_ns']-row['measurement_start_monotonic_ns'])/1e9
 serial=wall(n)+wall(s);start=min(pn['measurement_start_monotonic_ns'],ps['measurement_start_monotonic_ns']);end=max(pn['measurement_end_monotonic_ns'],ps['measurement_end_monotonic_ns'])
 pair=(end-start)/1e9;overlap_start=max(pn['measurement_start_monotonic_ns'],ps['measurement_start_monotonic_ns']);overlap_end=min(pn['measurement_end_monotonic_ns'],ps['measurement_end_monotonic_ns'])
 def epochs(row,left=None,right=None):
  rows=[r['seconds'] for r in row['epochs'] if r['stage']=='timed' and (left is None or r['start_monotonic_ns']>=left) and (right is None or r['end_monotonic_ns']<=right)]
  return {'count':len(rows),'median_seconds':statistics.median(rows) if rows else None,'mean_seconds':statistics.mean(rows) if rows else None}
 return {'serial_native_measured_seconds':wall(n),'serial_structured_measured_seconds':wall(s),'serial_sum_seconds':serial,
  'concurrent_native_measured_seconds':wall(pn),'concurrent_structured_measured_seconds':wall(ps),'concurrent_mixed_workload_makespan_seconds':pair,
  'mixed_workload_speedup':serial/pair,'native_slowdown_factor':wall(pn)/wall(n),'structured_slowdown_factor':wall(ps)/wall(s),
  'concurrent_measurement_start_skew_seconds':abs(pn['measurement_start_monotonic_ns']-ps['measurement_start_monotonic_ns'])/1e9,
  'concurrent_overlap_seconds':max(0,(overlap_end-overlap_start)/1e9),
  'serial_epoch_times':{'native':epochs(n),'structured':epochs(s)},'concurrent_epoch_times':{'native':epochs(pn),'structured':epochs(ps)},
  'fully_overlapping_epoch_times':{'native':epochs(pn,overlap_start,overlap_end),'structured':epochs(ps,overlap_start,overlap_end)},
  'startup_seconds':{phase:{arm:r['startup_seconds'] for arm,r in arms.items()} for phase,arms in results.items()},
  'completed_optimizer_updates':440,'complete_TRAIN_epochs':440,'complete_VALID_calls':440,
  'scope':'one fixed native+structured PubMed pair on one physical GPU; startup/warmup excluded from measured windows; pair tail included',
  'automatic_generalization_to_other_profiles':False,'admission_decision':'root_review_required','prediction_quality_claim':False}
