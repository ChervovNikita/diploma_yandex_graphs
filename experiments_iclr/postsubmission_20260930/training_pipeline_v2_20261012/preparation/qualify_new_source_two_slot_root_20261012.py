"""Bounded two-slot engineering checks for the requested shared-GPU launcher."""
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD, invoke

HERE=Path(__file__).resolve().parent/'new_source_two_slot_root_20261012_v1'
REMOTE=GUARD+r'''
import signal
closed=json.loads((phase/'new_source_resource_closure_root_20261012_v1/CLOSURE.json').read_text())
assert closed['total_cases']==32 and closed['all_children_and_CUDA_PIDs_absent']
source=phase/'new_source_resource_stage_20261012_v2/source'
helper=source/'qualify_exact_sources.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='8606ab9e6de8561e70ad6b383e46e1ae2cced66853d567688bcb0df091bb0e14'
observed={r['case_id']:r for r in closed['cases']}
pairs=[['endpoint_value-wikics-sage-E4','endpoint_value-amazon-photo-sage-E4'],
 ['private_bootstrap-wikics-gat-shared4_paired_graph_bootstrap','private_bootstrap-wikics-sage-shared4_paired_graph_bootstrap']]
out=phase/'new_source_two_slot_root_20261012_v1';assert not out.exists();out.mkdir()
decision={'schema':'bounded-two-slot-engineering-decision-v1','pairs':pairs,'concurrent_children':2,
 'maximum_child_wall_seconds':120,'source_callback_cycles_per_child':4,'retry':False,'GO':False,
 'scientific_fits':0,'actual_optimizer_updates':0,'scores_emitted':0,
 'reason':'Measure representative concurrent source callbacks for the user-requested two processes per GPU; not a benchmark fit or a whole-roster admission',
 'single_receipts':'new_source_resource_closure_root_20261012_v1/CLOSURE.json',
 'per_child_allocator_fraction':0.4,'other_processes_never_signaled':True}
(out/'ROOT_ENGINEERING_DECISION.json').write_text(json.dumps(decision,indent=2)+'\n')
def identity(pid):
 f=Path('/proc')/str(pid)/'stat'
 if not f.exists():return None
 fields=f.read_text().rsplit(')',1)[1].split()
 return {'pid':pid,'start_ticks':int(fields[19]),'pgid':int(fields[2])}
def physics():
 d=phase/'private_GRU_acquisition_root_20261011_v1/owner_v1'
 start=json.loads((d/'OWNER_START.json').read_text());rows={}
 for k in ('owner','child'):
  expect=start[k];now=identity(expect['pid']);rows[k]=now
  if now is not None:assert all(now[n]==expect[n] for n in ('pid','pgid','start_ticks'))
 if any(v is None for v in rows.values()):assert (d/'OWNER_END.json').exists()
 return rows
def cuda():
 text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
 return {int(line.split(',')[0]):int(line.split(',')[1]) for line in text.splitlines() if ',' in line and line.split(',')[1].strip().isdigit()}
def receipt(p):return {'path':str(p.relative_to(phase)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
attempts=[]
for index,ids in enumerate(pairs):
 before=physics();free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())*1024**2
 need=sum(observed[k]['observations']['CUDA_peak_reserved_bytes'] for k in ids)*1.25+4*1024**3
 assert free>=need, 'Insufficient current free GPU memory; no launch permitted'
 memory={line.split(':',1)[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':',1)[0] in ('MemAvailable',)}
 rss_need=sum(observed[k]['observations']['process_peak_RSS_KiB']*1024 for k in ids)*1.25+4*1024**3
 assert memory['MemAvailable']>=rss_need
 fs=os.statvfs(repo);assert fs.f_bavail*fs.f_frsize>1024**3
 item={'pair_index':index,'case_ids':ids,'Physics_before':before,'current_free_cuda_bytes':free,
  'single_peak_plus_headroom_bytes':need,'current_available_RAM_bytes':memory['MemAvailable'],
  'single_RSS_plus_headroom_bytes':rss_need,'children':[],'sampled_compute_memory':[]}
 logs=[];processes=[]
 try:
  for k in ids:
   relative='new_source_two_slot_root_20261012_v1/pair_'+str(index)+'/'+k
   log=(out/(str(index)+'_'+k+'.log')).open('x');logs.append(log)
   cmd=[str(repo/'.venv/bin/python'),'-B',str(helper),'--project-root',str(phase),
    '--admission','new_source_resource_stage_20261012_v2/ENGINEERING_ADMISSION.json',
    '--case-id',k,'--output',relative]
   child=subprocess.Popen(cmd,cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   ident=identity(child.pid);assert ident and ident['pgid']==child.pid
   processes.append(child);item['children'].append({'case_id':k,'owner':ident,'output':relative})
  (out/('PAIR_'+str(index)+'_START.json')).write_text(json.dumps(item,indent=2)+'\n')
  start=time.monotonic()
  while any(p.poll() is None for p in processes):
   if time.monotonic()-start>120:raise TimeoutError('Two-slot engineering timeout')
   sample=cuda();item['sampled_compute_memory'].append({'elapsed_seconds':time.monotonic()-start,
    'owned_PID_memory_MiB':{str(p.pid):sample.get(p.pid,0) for p in processes},
    'external_total_MiB':sum(v for k,v in sample.items() if k not in {p.pid for p in processes})})
   time.sleep(0.5)
  item['wall_seconds']=time.monotonic()-start
 except Exception as error:
  item['parent_failure']=repr(error)
 finally:
  for p,row in zip(processes,item['children']):
   if p.poll() is None:
    assert identity(p.pid)==row['owner'];os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:
     assert identity(p.pid)==row['owner'];os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
   row.update(exit_code=p.wait(),child_absent=identity(p.pid) is None)
   result=phase/row['output']/'RESULT.json'
   if result.is_file():
    v=json.loads(result.read_text());assert v['actual_optimizer_updates']==v['original_optimizer_kernels']==v['scores_emitted']==v['heldout_target_files_opened']==0
    row['result']=receipt(result)
    row['observations']={k:v[k] for k in ('CUDA_peak_allocated_bytes','CUDA_peak_reserved_bytes','process_peak_RSS_KiB','warm_median_seconds','full_graph_serving_seconds')}
    row['single_warm_median_seconds']=observed[row['case_id']]['observations']['warm_median_seconds']
    row['concurrent_to_single_callback_ratio']=v['warm_median_seconds']/row['single_warm_median_seconds']
  for log in logs:log.close()
  item['Physics_after']=physics();now=cuda()
  item['owned_children_and_CUDA_PIDs_absent']=all(r['child_absent'] and r['owner']['pid'] not in now for r in item['children'])
  attempts.append(item);(out/('PAIR_'+str(index)+'_END.json')).write_text(json.dumps(item,indent=2)+'\n')
 if item.get('parent_failure') or any(r['exit_code']!=0 for r in item['children']):break
result={'schema':'bounded-two-slot-engineering-closure-v1','attempts':attempts,'scientific_fits':0,
 'actual_optimizer_updates':0,'scores_emitted':0,'heldout_target_files_opened':0,'GO':False,
 'whole_master_or_other_GPU_admitted':False,'not_full_training_throughput':True,
 'scope':'Concurrent random source TRAIN/serving/save/restore callbacks; optimizer kernels and scientific selectors omitted'}
(out/'CLOSURE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
'''

def main():
    assert not HERE.exists()
    HERE.mkdir()
    (HERE/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
    v=invoke(REMOTE,[],HERE,'ENGINEERING_PAIRS',360)
    assert v['exit_code']==0, 'Pair collector failed; inspect ownership before any new launch'
    x=json.loads(v['stdout']);(HERE/'CLOSURE.json').write_text(json.dumps(x,indent=2)+'\n')
    print(json.dumps({'pairs_attempted':len(x['attempts']),
        'pairs_passed':sum(not a.get('parent_failure') and all(r['exit_code']==0 for r in a['children']) for a in x['attempts']),
        'all_children_closed':all(a['owned_children_and_CUDA_PIDs_absent'] for a in x['attempts']),
        'fits':0,'GO':False}))

if __name__=='__main__':main()
