"""Protect both preserved finite masters before any source publication."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROTECTION=r'''
import gzip
phase=repo/'experiments_iclr/postsubmission_20260930'
expected_plans={
 'massive_practical_master_20261012_v3/PLAN.json':'081b1358d83468c8062844288f28c9f741984176d441466df8c7daded82a57ef',
 'massive_practical_master_20261012_v4/PLAN.json':'e0a446486d4d47fbf5168d64f5823b828a8d4bdb152addddbf72094e8caa9647'}
def canonical_digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
protected={}
for relative,digest in expected_plans.items():
 plan=json.loads((phase/relative).read_bytes())
 assert canonical_digest(plan)==digest,'Preserve the frozen master identity: '+relative
 for entry in plan['studies']:
  path=phase/entry['path'];raw=path.read_bytes()
  spec=json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)
  assert canonical_digest(spec)==entry['spec_hash']
  for bound in spec['source']['files']:
   previous=protected.get(bound['path']);assert previous is None or previous==bound
   protected[bound['path']]=bound
for candidate in payload['files']:
 prefix='experiments_iclr/postsubmission_20260930/'
 relative=candidate['target'][len(prefix):] if candidate['target'].startswith(prefix) else None
 if relative in protected:
  bound=protected[relative]
  assert (candidate['bytes'],candidate['sha256'])==(bound['bytes'],bound['sha256']), 'Publication must preserve frozen byte authority: '+relative
'''


def main():
    spec=importlib.util.spec_from_file_location('_protected_v29',HERE/'publish_exact_inventory_v29.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.FROZEN_PROTECTION=PROTECTION
    module.main()


if __name__=='__main__':main()
