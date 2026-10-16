import tempfile,unittest
from pathlib import Path
from prepare_cached import SnapshotPins,raw_pin,HERE
class Checks(unittest.TestCase):
 def test_receipts_identical_cache_invalidated_on_mutation(self):
  with tempfile.TemporaryDirectory(prefix='.source_test_',dir=HERE) as folder:
   p=Path(folder)/'test.bin';p.write_bytes(b'initial');cache=SnapshotPins()
   first=cache(p);self.assertEqual(first,raw_pin(p));self.assertEqual(cache(p),first);self.assertEqual(cache.hits,1)
   first['path']='caller-mutated';self.assertEqual(cache(p),raw_pin(p))
   p.write_bytes(b'changed');self.assertEqual(cache(p),raw_pin(p));self.assertEqual(cache.invalidations,1)
   cache.verify_unchanged();p.write_bytes(b'mutated again')
   with self.assertRaises(ValueError):cache.verify_unchanged()
if __name__=='__main__':unittest.main(verbosity=2)
