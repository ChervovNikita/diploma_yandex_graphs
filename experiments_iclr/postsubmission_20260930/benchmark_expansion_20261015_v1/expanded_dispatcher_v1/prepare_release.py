"""Prepare the all-blocked finite production release; never dispatch a child."""
import socket,subprocess
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
from pathlib import Path
import json
from contracts import HERE,EXPANSION,guard,pin,verify,read,compile_release,GATES
from dispatcher import inspect_release

guard('allocation/gpu0')
folder=HERE/'generated_v1';assert not folder.exists();folder.mkdir()
skeleton=pin(EXPANSION/'expanded_launch_preparation_v1/generated_v1/PLAN.json')
assert skeleton['sha256']=='2f5c9adc876202fe2075fe1bfe0c31b02646bd975f418119c5ebdb7dc3c1d116'
source=read(skeleton['path'])
pins=[pin(HERE/name)for name in('contracts.py','engine.py','dispatcher.py','MANIFEST.json')]+list(source['legacy_supervisor_sources'].values())
bindings={'schema':'expanded-production-bindings-v1','skeleton':skeleton,'source_files':pins,'tasks':{},
          'scope':'No qualification,resource,runtime or human GO receipt is fabricated by this preparation'}
binding_path=folder/'BINDINGS.json';binding_path.write_text(json.dumps(bindings,indent=2,sort_keys=True)+'\n')
result=compile_release(skeleton,binding_path,folder/'RELEASE.json',EXPANSION/'campaign_outputs/expanded_dispatcher_REQUIRES_REVIEW')
preflight=inspect_release(folder/'RELEASE.json','allocation/gpu0')
assert preflight['qualified_tasks']==0 and preflight['blocked_tasks']==6730 and preflight['children_started']==0
(folder/'PREFLIGHT.json').write_text(json.dumps(preflight,indent=2,sort_keys=True)+'\n')
summary={'schema':'expanded-production-dispatcher-preparation-v1','status':'dispatcher_source_and_fake_checks_complete_native_campaign_blocked',
 'release':result['release'],'preflight':pin(folder/'PREFLIGHT.json'),'source_manifest':pin(HERE/'MANIFEST.json'),'fake_checks':pin(HERE/'CHECKS.json'),
 'parent_dryrun_provenance':skeleton,'physical_fit_intentions':6460,'I4_export_only_tasks':270,'qualified_tasks':0,'blocked_tasks':6730,
 'source_context_blocked_fit_intentions_preserved':90,'all_protocols_and_seed_contracts_preserved':True,'production_dispatch_implemented':True,
 'current_enabled_route':'allocation/gpu0 only','server77_enabled':False,'maximum_owned_children_per_GPU':2,
 'native_fits_started':0,'native_models_constructed':0,'native_optimizer_updates':0,'GPU_compute':False,'TEST_accessed':False,
 'production_resource_profiles_asserted':0,'user_GO_received_or_created':False,'campaign_ETA':None,
 'unresolved_attached_qualification_gates':list(GATES)+['human_GO','worker_route_environment_and_claim_STOP_resume_custody'],
 'scope_note':'Individual engineering evidence may exist elsewhere; this release has no attached exact per-task PASS/resource/runtime bindings. Fake subprocess tests do not qualify native sources or devices.'}
(folder/'SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({'release':result['release'],'summary':pin(folder/'SUMMARY.json'),'preflight':pin(folder/'PREFLIGHT.json'),'qualified_tasks':0,'blocked_tasks':6730,'native_children_started':0},sort_keys=True))
