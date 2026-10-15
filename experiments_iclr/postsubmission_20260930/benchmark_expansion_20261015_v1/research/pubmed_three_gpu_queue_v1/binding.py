"""Finite PubMed24 identity/path binding; no training code is replaced."""
import copy,hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;E=HERE.parents[1]
def require(ok,message):
 if not ok:raise ValueError(message)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def pin(path):
 p=Path(path).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def load_inventory():return json.loads((HERE/'CANONICAL.json').read_text())
def shard(inventory,worker):
 require(worker in inventory['assignments'],'Unknown fixed worker');seed=inventory['assignments'][worker]
 rows=[copy.deepcopy(r)for r in inventory['rows']if r['cohort_seed']==seed]
 require(len(rows)==8 and [r['kind']for r in rows]==['native']*4+['G0','uniform','structured','permuted'],'Exactly one original eight-fit cohort required')
 return rows
def relocated(row,host,worker,source_sha,map_sha):
 sys.path.insert(0,str(E/'portable_dispatcher_v3'))
 from descriptor_binding import rebind_job
 inv=load_inventory();original=row['job'];require(digest(original)==row['job_hash'],'Canonical fit changed')
 job,receipt=rebind_job(original,inv['canonical_root'],inv['canonical_phase_relative'],host['project_root'],host['phase_relative'])
 job['portable_execution']={'schema':'pubmed-three-GPU-job-binding-v1','canonical_job_sha256':row['job_hash'],
  'queue_source_manifest_sha256':source_sha,'target_map_sha256':map_sha,'worker':worker,
  'metadata_compatibility_source_sha256':pin(HERE/'binding.py')['sha256']}
 receipt.update(bound_runtime_job_hash=digest(job),declared_extra_provenance_field='portable_execution',
  native_operators_optimizer_objective_selector_and_seeds_changed=False)
 return job,receipt
def install_residual_descriptor_view(job,row,root):
 """Only the frozen extension's exact descriptor validator receives a host view.

 The transform/update/forward/state code objects and scientific arguments stay
 original. Its metadata factory must accept the relocated descriptor, including
 the explicit source/map binding included in every checkpoint's job hash.
 """
 if row['kind']not in ('uniform','structured','permuted'):return None
 from common import load_module
 ext=job['session_transform'];module,receipt=load_module(ext['path'],root,ext['sha256'])
 original_factory=module.descriptor;original_transform=module.transform.__code__
 def factory(request_root,seed,variant,manifest_sha):
  require(str(Path(request_root).resolve())==str(Path(root).resolve())and seed==job['seed']
   and variant==row['kind']and manifest_sha==job['method_settings']['manifest_sha256'],
   'Metadata view cannot change host, seed, variant or method source')
  return copy.deepcopy(job)
 module.descriptor=factory
 require(module.transform.__code__ is original_transform,'Native residual transformation changed')
 return {'module':module,'original_factory':original_factory,'source':receipt,'transform_code_unchanged':True}
def restore_descriptor_view(view):
 if view is not None:view['module'].descriptor=view['original_factory']
