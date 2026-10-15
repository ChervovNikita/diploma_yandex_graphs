"""Run the exact frozen metadata projector with stat-validated hash memoization.

No frozen source is edited. The only substituted function is metadata file pin:
it returns the same path/bytes/SHA dictionary after one byte hash per immutable
snapshot. Native runtime, descriptor derivation and no-overwrite rules stay exact.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'server51_campaign_port_v1'
PARENT_SHA='5107ebd694b1333629ec306808f64eb3e63c2b9fc24e4f4e3cbab7f0649b9209'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def require(ok,message):
 if not ok:raise ValueError(message)
def raw_pin(path):
 path=Path(path).resolve();raw=path.read_bytes();return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def signature(path):
 s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
class SnapshotPins:
 def __init__(self,underlying=raw_pin):
  self.underlying=underlying;self.rows={};self.hits=0;self.misses=0;self.hashed_bytes=0;self.invalidations=0;self.bytes_avoided=0
 def __call__(self,path):
  path=Path(path).resolve();key=str(path);before=signature(path);row=self.rows.get(key)
  if row is not None and row['stat']==before:self.hits+=1;self.bytes_avoided+=row['pin']['bytes'];return copy.deepcopy(row['pin'])
  if row is not None:self.invalidations+=1
  value=self.underlying(path);after=signature(path);require(before==after,'File changed while hashing: '+key)
  self.misses+=1;self.hashed_bytes+=value['bytes'];self.rows[key]={'stat':after,'pin':copy.deepcopy(value)};return value
 def verify_unchanged(self):
  for path,row in self.rows.items():require(signature(path)==row['stat'],'Snapshot changed during metadata generation: '+path)
 def summary(self):return {'cache_hits':self.hits,'actual_byte_hashes':self.misses,'bytes_hashed':self.hashed_bytes,'repeated_bytes_not_rehashed':self.bytes_avoided,'invalidated_snapshots':self.invalidations,'unique_paths':len(self.rows),'stat_fields':['device','inode','size','mtime_ns','ctime_ns']}
def main():
 require(socket.gethostname()=='anogena-2-0','Allocation-only metadata wrapper; no project reads')
 require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[UUID],'Exact sole allocation GPU required')
 parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--wrapper-sha256',required=True);parser.add_argument('--cache-receipt',required=True)
 args,forward=parser.parse_known_args();own=raw_pin(__file__);require(own['sha256']==args.wrapper_sha256,'Exact reviewed wrapper source required')
 receipt=Path(args.cache_receipt).resolve();require(receipt.is_relative_to(HERE) and not receipt.exists(),'Fresh own cache receipt required')
 parent_manifest=raw_pin(PARENT/'MANIFEST.json');require(parent_manifest['sha256']==PARENT_SHA,'Frozen projection source differs')
 for row in json.loads((PARENT/'MANIFEST.json').read_text())['files']:
  actual=raw_pin(PARENT/row['path']);require(all(actual[k]==row[k] for k in ('bytes','sha256')),'Frozen source member differs')
 sys.path.insert(0,str(PARENT));spec=importlib.util.spec_from_file_location('_frozen_server51_projection_cached',PARENT/'prepare_projection.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 original=module.pin;cache=SnapshotPins(original);module.pin=cache
 report={'schema':'server51-frozen-projector-snapshot-cache-invocation-v1','status':'running','actual_outer_producer':own,
  'unchanged_inner_source_manifest':parent_manifest,'substitution':'metadata pin only; identical receipt shape/hash semantics',
  'native_work':0,'claims_created':0,'human_GO':False,'frozen_sources_modified':False,'resume_or_nooverwrite_rules_changed':False}
 started=time.monotonic();previous_argv=sys.argv;sys.argv=[str(PARENT/'prepare_projection.py'),*forward]
 try:
  module.main();cache.verify_unchanged();report['status']='PASS'
 except BaseException as error:
  report.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
 finally:
  sys.argv=previous_argv;module.pin=original;report.update(wall_seconds=time.monotonic()-started,cache=cache.summary())
  with receipt.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
  print(json.dumps({'cache_receipt':raw_pin(receipt),'status':report['status'],'cache':report['cache']},sort_keys=True))
if __name__=='__main__':main()
