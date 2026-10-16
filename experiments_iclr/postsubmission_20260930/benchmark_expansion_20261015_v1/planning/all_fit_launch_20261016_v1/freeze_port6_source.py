"""Finalize the explicit source-only port6 draft; no generated jobs or remote work."""
from pathlib import Path
import hashlib,json
E=Path(__file__).resolve().parents[2];N=E/'server51_campaign_port_v6';O=E/'server51_campaign_port_v5';PHASE=E.parent

def pin(p,field='path_relative_to_phase'):
 raw=p.read_bytes();return {field:str(p.relative_to(PHASE if field=='path_relative_to_phase' else E)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def freeze(folder):
 previous=O/folder.relative_to(N)/'MANIFEST.json';m=json.loads(previous.read_text());m['schema']=m['schema'].replace('-source-v5','-source-v6');m['parent_source_manifest']=pin(previous,'path_relative_to_expansion')
 names=[r['path'] for r in m['files']]
 if folder==N:
  names=sorted(str(p.relative_to(N)) for p in N.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=N/'MANIFEST.json')
 m['files']=[{'path':name,'bytes':(folder/name).stat().st_size,'sha256':hashlib.sha256((folder/name).read_bytes()).hexdigest()} for name in names];write(folder/'MANIFEST.json',m)
extra=[N/'source_rng_binding.py',N/'source_rng_audits/server51.json',N/'source_rng_audits/SOURCE_COMPARISON.json',E/'qualification/runtime_v2_source_rng_v1.json',E/'interaction_BUDDY_production_v5/objectives.py',E/'interaction_BUDDY_production_v3/objectives.py',E/'interaction_BUDDY_production_v3/MANIFEST.json',E/'interaction_BUDDY_production_v4/MANIFEST.json',E/'interaction_BUDDY_production_v5/MANIFEST.json']
external=[p for p in extra if not p.is_relative_to(N)]
for relative in ('composed/SOURCE_BINDINGS.json','scoring/SOURCE_BINDINGS.json','TRANSITIVE_SOURCE_CLOSURE_ADDITIONS.json'):
 v=json.loads((O/relative).read_text());rows={}
 for row in v['files']:
  path=row['path_relative_to_phase'].replace('server51_campaign_port_v5','server51_campaign_port_v6');p=PHASE/path
  # Scoring references composed manifest and binding bytes, finalized below.
  rows[path]=pin(p)
 for p in (extra if relative!='TRANSITIVE_SOURCE_CLOSURE_ADDITIONS.json' else external):rows[pin(p)['path_relative_to_phase']]=pin(p)
 v['files']=[rows[k] for k in sorted(rows)]
 if relative=='TRANSITIVE_SOURCE_CLOSURE_ADDITIONS.json':
  v.update(additional_files=len(rows),additional_bytes=sum(r['bytes'] for r in rows.values()),port6_additive_dependencies=[pin(p) for p in external])
 write(N/relative,v)
freeze(N/'composed')
v=json.loads((N/'scoring/SOURCE_BINDINGS.json').read_text());v['files']=[pin(PHASE/r['path_relative_to_phase']) for r in v['files']];write(N/'scoring/SOURCE_BINDINGS.json',v)
freeze(N/'scoring');freeze(N/'reporting')
(N/'README.md').write_text('''# Allocation + server51 campaign port v6\n\nThis successor preserves port5's actual LinkEpochAdapter policy binding repair.\nIt repairs two additional execution-source omissions observed on server51:\n\n- The native OGB source RNG audit retains allocation-root receipt paths. All 24\n  installed/native source files and dependency versions match server51 exactly.\n  The projector changes only the execution audit path and SHA for server51\n  native source-reference OGB jobs. The embedded audit preserves every original\n  assumption and all 36 integer-loader semantic cases. Runtime RNG code remains\n  byte-identical. Allocation and undeclared jobs retain their existing audit.\n- The unchanged BUDDY interaction session loads objectives.py, which was omitted\n  from production_v4/v5 packaging. The exact 4076-byte original pinned by the v3\n  manifest is restored additively and included explicitly in source closures.\n  Existing production manifests and files remain unchanged.\n\nNo generated descriptor is edited in place. Canonical fit IDs, science, source\nmethods, 6460 fits, 270 exports, runtime/checkpoint schema v2, normalization and\nownership laws remain fixed. The separate private-BN two-update stopping repair\nbelongs to the smoke successor. Root owns native regression and rollout.\n''')
a=json.loads((N/'SCHEMA_PATH_AUDIT.json').read_text());a['scorer_parent']=pin(O/'scoring/MANIFEST.json','path_relative_to_expansion');write(N/'SCHEMA_PATH_AUDIT.json',a)
write(N/'SOURCE_REPAIR.json',{'schema':'port6-source-rng-and-missing-objectives-repair-v1','parent_source_manifest':pin(O/'MANIFEST.json','path_relative_to_expansion'),'source_rng_audit':pin(N/'source_rng_audits/server51.json','path_relative_to_expansion'),'audit_semantic_evidence_unchanged':True,'all_24_source_contents_and_versions_verified_equal':True,'runtime_source_rng_files_unchanged':True,'additive_objectives':pin(E/'interaction_BUDDY_production_v5/objectives.py','path_relative_to_expansion'),'objectives_ancestor':pin(E/'interaction_BUDDY_production_v3/MANIFEST.json','path_relative_to_expansion'),'generated_jobs_modified':False,'native_executions':0,'TEST_accessed':False})
changes=[]
for row in json.loads((O/'MANIFEST.json').read_text())['files']:
 if row['path']=='SOURCE_LINEAGE.json':continue
 p=N/row['path'];actual=pin(p,'path_relative_to_expansion')
 if (actual['bytes'],actual['sha256'])!=(row['bytes'],row['sha256']):changes.append({'path':row['path'],'old':{k:row[k] for k in ('bytes','sha256')},'new':{k:actual[k] for k in ('bytes','sha256')}})
write(N/'SOURCE_LINEAGE.json',{'schema':'port6-source-lineage-v1','parent':pin(O/'MANIFEST.json','path_relative_to_expansion'),'predecessors_unchanged':True,'changed_members':changes,'external_added_dependency_pins':[pin(p) for p in external],'native_methods_changed':False})
freeze(N)
print(json.dumps(pin(N/'MANIFEST.json','path_relative_to_expansion')))
