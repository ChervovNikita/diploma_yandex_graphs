"""Pure two-slot admission: worker VRAM, host-shared RAM/disk. No live queries."""
WORKERS=('allocation/gpu0','server77/gpu0','server77/gpu1')

def admissible(job,worker,active,capacities,states):
 if worker not in WORKERS:raise ValueError('Unauthorized worker')
 if states.get(job['id'])!='registered':return False,'already_claimed_or_terminal'
 if not all(states.get(p)=='complete' for p in job['dependencies']):return False,'dependencies_not_complete'
 host=worker.split('/')[0];gpu_live=[a for a in active if a['worker']==worker];host_live=[a for a in active if a['worker'].split('/')[0]==host]
 if len(gpu_live)>=2:return False,'two_slot_limit'
 required=('vram_mib','ram_mib','disk_mib')
 if any(k not in job['resources'] for k in required):return False,'source_resources_not_admitted'
 if worker not in capacities.get('gpu',{}) or host not in capacities.get('host',{}):return False,'host_not_verified'
 for resource,live,bound in [('vram_mib',gpu_live,capacities['gpu'][worker]),('ram_mib',host_live,capacities['host'][host]),('disk_mib',host_live,capacities['host'][host])]:
  if resource not in bound or any(resource not in a['resources'] for a in live):return False,'active_resource_reservation_missing'
  total=job['resources'][resource]+sum(a['resources'][resource] for a in live)
  if total>bound[resource]:return False,'combined_'+resource+'_reservation'
 return True,'admitted'

def ready_order(jobs,states):
 return sorted((j for j in jobs if states.get(j['id'])=='registered' and all(states.get(p)=='complete' for p in j['dependencies'])),key=lambda j:(j['kind']=='fit',-j.get('relative_cost',{}).get('value',0.),j['id']))

def require_active_campaign(plan,approval,retirement_policy):
 if approval['approved_plan_hash'] in set(retirement_policy['retired_plan_hashes']):raise ValueError('v29 retired; its GO cannot authorize new campaign')
 if plan.get('scientific_launch_enabled') is not True:raise ValueError('Disabled prepared campaign')
 from campaign_compiler import digest
 if approval['approved_plan_hash']!=digest(plan):raise ValueError('GO binds a different campaign')
 if approval.get('user_explicit_GO') is not True:raise ValueError('Explicit human GO required')
