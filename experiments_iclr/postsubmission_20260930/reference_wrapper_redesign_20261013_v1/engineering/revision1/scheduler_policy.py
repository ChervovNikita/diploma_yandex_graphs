"""Pure admission policy. No processes, hardware queries, datasets or models."""
WORKERS=('allocation/gpu0','server77/gpu0','server77/gpu1')

def admissible(job,worker,active,capacities,states):
 if worker not in WORKERS:raise ValueError('Unauthorized worker')
 if states.get(job['id'])!='registered':return False,'already_claimed_or_terminal'
 if not all(states.get(p)=='complete' for p in job['dependencies']):return False,'dependencies_not_complete'
 live=[a for a in active if a['worker']==worker]
 if len(live)>=2:return False,'two_slot_limit'
 required=('vram_mib','ram_mib','disk_mib')
 if any(k not in job['resources'] for k in required):return False,'source_resources_not_admitted'
 if worker not in capacities:return False,'host_not_verified'
 for resource in required:
  total=job['resources'][resource]+sum(a['resources'][resource] for a in live)
  if total>capacities[worker][resource]:return False,'combined_'+resource+'_reservation'
 return True,'admitted'

def ready_order(jobs,states):
 """Close own completed cohorts before more fits; preserve deterministic ordering."""
 return sorted((j for j in jobs if states.get(j['id'])=='registered' and all(states.get(p)=='complete' for p in j['dependencies'])),key=lambda j:(j['kind']=='fit',j['id']))

def require_active_campaign(plan,approval,retirement_policy):
 retired=set(retirement_policy['retired_plan_hashes'])
 if approval['approved_plan_hash'] in retired:raise ValueError('v29 retired; its GO cannot authorize new campaign')
 if plan.get('scientific_launch_enabled') is not True:raise ValueError('Disabled prepared campaign')
 from campaign_compiler import digest
 if approval['approved_plan_hash']!=digest(plan):raise ValueError('GO binds a different campaign')
 if approval.get('user_explicit_GO') is not True:raise ValueError('Explicit human GO required')
