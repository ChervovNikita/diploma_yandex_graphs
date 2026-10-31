"""Metadata/source key equality only. No model/runtime/epoch is constructed."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "training_pipeline_v2_20261012/preparation"))
from prepare_molecular_packets_allocation import GUARD, invoke

REMOTE = GUARD + r'''
import ast,collections,gzip
packet=phase/'ncnc_completion_successor_root_20261012_v3'
raw=(packet/'matrix/PLAN.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='1b9bd2332371db38ea42fd985d725751caf3f82fd663a95569e062d253ab3760'
plan=json.loads(raw);canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
assert hashlib.sha256(canonical(plan)).hexdigest()=='46463adc3e5fc9301c95f09f40a93b475fb8005335d34c50e7f685498e6526aa'
pin=plan['studies'][0];p=phase/pin['path'];source=p.read_bytes();study=json.loads(gzip.decompress(source))
assert hashlib.sha256(canonical(study)).hexdigest()==pin['spec_hash']
fits=[r for r in study['runs'] if r['kind']=='model_fit'];providers=[r for r in fits if r['adapter'].split(':',1)[0].endswith('link_prototypes.py')]
assert len(fits)==375 and len(providers)==249
keys=collections.defaultdict(list);counts=collections.Counter();verified={};laws={}
for r in providers:
 c=r['config'];assert r['budget']=={'unit':'epochs','policy':'full_budget','maximum':100}
 key={'schema':'declared_shared_replay_inputs_only-v1','inputs':c['link_inputs'],'sources':c['prototype_sources'],
      'geometry':{'nodes':c['provider_nodes'],'TRAIN_records':1179052,'ordered_raw_entries':2358104,
                  'batch_size':c['training']['batch_size'],'full_batches':c['train_minibatches_per_epoch']},
      'source_master_seed':2026100401,'epochs':100,'dtype':'little-endian-int64'}
 k=hashlib.sha256(canonical(key)).hexdigest();keys[k].append(r['run_id']);counts[r['method']]+=1
 for receipt in list(c['link_inputs'].values())+list(c['prototype_sources'].values()):
  rel=receipt['path'];target=phase/rel;assert target.resolve().is_relative_to(phase)
  if rel not in verified:
   raw=target.read_bytes();assert len(raw)==receipt['bytes'] and hashlib.sha256(raw).hexdigest()==receipt['sha256']
   verified[rel]=receipt
 adapter=r['adapter'].split(':',1)[0]
 if adapter not in laws:
  path=phase/adapter;raw=path.read_bytes();tree=ast.parse(raw)
  cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='LinkPrototypeAdapter')
  fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='replay_for_epoch')
  literal=ast.get_source_segment(raw.decode(),fn)
  assert literal.count('2026100401')==2 and 'draw_epoch' in literal and 'write_epoch' in literal
  assert 'self.arm' not in literal and 'self.run' not in literal and 'seed' not in literal
  laws[adapter]={'source_sha256':hashlib.sha256(raw).hexdigest(),'replay_AST_sha256':hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest(),'replay_for_epoch_source':literal}
assert len(keys)==1 and len({v['replay_AST_sha256'] for v in laws.values()})==1
result={'schema':'declared-shared-replay-source-key-equality-v1','status':'PASS_METADATA_ONLY_UNQUALIFIED_RUNTIME',
 'plan_hash':hashlib.sha256(canonical(plan)).hexdigest(),'study_spec_hash':pin['spec_hash'],
 'source_provider_fits':len(providers),'native_boundary_fits_excluded':len(fits)-len(providers),
 'declared_input_source_geometry_keys':len(keys),'key_hash':next(iter(keys)),
 'exact_run_roster_sha256':hashlib.sha256(canonical(sorted(next(iter(keys.values()))))).hexdigest(),
 'provider_fit_counts_by_method':dict(counts),'one_declared_key':key,'verified_file_receipts':list(verified.values()),
 'adapter_replay_laws':laws,'runtime_device_key_qualified':False,'actual_TRAIN_arrays_decoded':False,
 'SELECT_or_heldout_values_decoded':False,'source_epochs_generated':0,'actual_input_equivalence_proven':False,
 'scientific_fits':0,'optimizer_updates':0,'GO':False,'packet_modified':False}
print(json.dumps(result))
'''


if __name__ == "__main__":
    result = invoke(REMOTE, [], HERE, "SOURCE_KEY_OBSERVATION", 180)
    assert result["exit_code"] == 0, "Preserve failed source metadata observation"
    value = json.loads(result["stdout"])
    (HERE / "SOURCE_KEY_EQUALITY.json").write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    print(json.dumps({k: value[k] for k in ("status", "source_provider_fits", "native_boundary_fits_excluded", "declared_input_source_geometry_keys", "actual_input_equivalence_proven")}))
