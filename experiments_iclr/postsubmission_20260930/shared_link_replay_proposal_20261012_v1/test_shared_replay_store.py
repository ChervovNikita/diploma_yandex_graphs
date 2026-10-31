"""Fabricated integer file lifecycle only; no graph/model/library runtime."""
import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from shared_replay_store import SharedReplayStore, stream_key

HERE = Path(__file__).resolve().parent


def temporary():
    return tempfile.TemporaryDirectory(dir=HERE, prefix="synthetic_fixture_")


def key(seed=2026100401, runtime="synthetic-fixture-only"):
    return stream_key(input_receipts={x: {"sha256": x} for x in ("graph", "train", "select")},
                      source_receipts={"provider": "fabricated-fixture"},
                      geometry={"records": 3, "batch": 2}, master_seed=seed,
                      runtime={"identity": runtime})


def produce(stage):
    stage.mkdir()
    (stage / "PERMUTATION.npy").write_bytes(b"fabricated permutation; not numpy")
    (stage / "NEGATIVE_PAIRS.npy").write_bytes(b"fabricated negatives; not numpy")
    (stage / "EPOCH.json").write_text('{"fabricated":true}\n')


def verify(stage):
    assert json.loads((stage / "EPOCH.json").read_text())["fabricated"] is True
    assert (stage / "PERMUTATION.npy").read_bytes().startswith(b"fabricated")
    assert (stage / "NEGATIVE_PAIRS.npy").read_bytes().startswith(b"fabricated")


class StoreTests(unittest.TestCase):
    def test_concurrent_consumers_generate_once_and_bind_leases(self):
        with temporary() as directory:
            store = SharedReplayStore(directory, key()); calls = []
            def producer(stage):
                calls.append(1); produce(stage)
            with ThreadPoolExecutor(max_workers=2) as pool:
                receipts = list(pool.map(lambda _: store.ensure_epoch(1, producer, verify), range(2)))
            self.assertEqual(len(calls), 1)
            self.assertEqual(receipts[0], receipts[1])
            for name in ("fit_A", "fit_B"):
                lease = store.bind_consumer(name, 1, *receipts[0])
                self.assertTrue(json.loads(lease.read_text())["paid_consumer_or_retry_open"])

    def test_corruption_is_denied_without_regeneration(self):
        with temporary() as directory:
            store = SharedReplayStore(directory, key())
            epoch, _ = store.ensure_epoch(1, produce, verify)
            (epoch / "PERMUTATION.npy").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                store.ensure_epoch(1, lambda _: self.fail("Must not silently regenerate"), verify)

    def test_failed_attempt_stays_and_next_attempt_is_fresh(self):
        with temporary() as directory:
            store = SharedReplayStore(directory, key())
            def failing(stage):
                stage.mkdir(); (stage / "partial.bin").write_bytes(b"failure"); raise RuntimeError("fixture")
            with self.assertRaises(RuntimeError):
                store.ensure_epoch(1, failing, verify)
            self.assertEqual(len(list(store.store.glob("FAILED_*.json"))), 1)
            self.assertEqual(len(list(store.store.glob("attempt_1_*"))), 1)
            store.ensure_epoch(1, produce, verify)

    def test_master_runtime_or_input_difference_separates_keys(self):
        with temporary() as directory:
            a = SharedReplayStore(directory, key())
            b = SharedReplayStore(directory, key(seed=2026100402))
            c = SharedReplayStore(directory, key(runtime="different-runtime"))
            self.assertEqual(len({a.key_hash, b.key_hash, c.key_hash}), 3)

    def test_source_epoch_range_and_runtime_are_mandatory(self):
        with temporary() as directory:
            store = SharedReplayStore(directory, key())
            with self.assertRaises(ValueError):
                store.ensure_epoch(101, produce, verify)
        with self.assertRaises(ValueError):
            stream_key(input_receipts={x: {} for x in ("graph", "train", "select")},
                       source_receipts={"provider": "x"}, geometry={"records": 3},
                       master_seed=1, runtime={})


if __name__ == "__main__":
    unittest.main()
