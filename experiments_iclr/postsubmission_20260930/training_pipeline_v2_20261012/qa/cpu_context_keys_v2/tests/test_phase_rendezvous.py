"""Metadata-only distributed receipt checks; all Git remotes are local temp dirs."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from phase_closure import write_host_receipts
from phase_rendezvous import (GitReceiptTransport, branch_ref, check_proof, decode_proof,
                              encode_proof, exchange_once, gates_for, install_immutable,
                              proof_path, proposal, verify_global)
from phase_rendezvous import transport_policy, MAX_COMPRESSED_BYTES, MAX_PROOF_BYTES
from spec import canonical, fingerprint


class RegistryFixture:
    def __init__(self, rows):
        self.rows = rows
    def states(self):
        return {k: row["state"] for k, row in self.rows.items()}
    def run(self, study, identifier):
        return self.rows[study + "/" + identifier]


class PhaseRendezvousTests(unittest.TestCase):
    def fixture(self):
        runs, placement, rows = {}, {}, {}
        for host in ("allocation", "server77"):
            for stage in ("search", "search_selection", "comparative"):
                key = "fixture/" + host + "_" + stage
                row = {"run_id": key.split("/")[1],
                       "kind": "selection" if stage == "search_selection" else "model_fit",
                       "config": {"stage": stage}}
                runs[key] = {"study": "fixture", "run": row}
                placement[key] = {"worker": host + "/gpu0"}
                rows[key] = {"state": "complete", "contract": row,
                             "result": {"fixture_source_digest": "a" * 64}}
        plan = {"output_root": "outputs", "global_gates": {
            "whole_search": {"path": "gates/search.json", "source_stages": ["search", "search_selection"],
                             "required_hosts": ["allocation", "server77"]},
            "whole_comparative": {"path": "gates/comparative.json", "source_stages": ["comparative"],
                                  "required_hosts": ["allocation", "server77"]}}}
        return plan, fingerprint(plan), runs, placement, rows

    def write(self, root, plan, digest, runs, placement, rows, host):
        write_host_receipts(root, plan, digest, runs, placement, RegistryFixture(rows), host)

    def test_only_exact_metadata_and_complete_required_selection_are_accepted(self):
        plan, digest, runs, placement, rows = self.fixture()
        rows["fixture/allocation_search"]["state"] = "failed"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, plan, digest, runs, placement, rows, "allocation")
            raw = proof_path(root, plan, "whole_search", "allocation").read_bytes()
            self.assertEqual(decode_proof(encode_proof(raw)), raw)
            self.assertEqual(encode_proof(raw), encode_proof(raw))
            check_proof(raw, plan, digest, runs, placement, "whole_search", "allocation")
            original = json.loads(raw)
            for mutate in (lambda v: v.update(extra_output="forbidden"),
                           lambda v: v["results"]["fixture/allocation_search"].update(score=0.5),
                           lambda v: v["results"]["fixture/allocation_search_selection"].update(state="failed"),
                           lambda v: v.update(plan_hash="b" * 64),
                           lambda v: v["results"]["fixture/allocation_search"].update(contract_hash="c" * 64)):
                value = copy.deepcopy(original); mutate(value)
                with self.assertRaises(ValueError):
                    check_proof((canonical(value) + "\n").encode(), plan, digest, runs,
                                placement, "whole_search", "allocation")

    def test_test_author_and_export_gates_cannot_enter_transport(self):
        plan, digest, _, _, _ = self.fixture()
        self.assertIn(digest, branch_ref(digest, "allocation"))
        self.assertEqual(proposal(plan, digest, "allocation")["scientific_workers_started"], 0)
        for stage in ("evaluation", "bank_export", "author_cli_group"):
            invalid = copy.deepcopy(plan)
            invalid["global_gates"]["whole_search"]["source_stages"] = [stage]
            with self.assertRaises(ValueError):
                gates_for(invalid)
        with self.assertRaises(ValueError):
            branch_ref("short", "allocation")

    def test_final_author_terminal_hashes_preserve_failure_without_output(self):
        row = {"run_id": "author", "kind": "author_cli_group", "config": {"stage": "author_cli"}}
        runs = {"fixture/author": {"study": "fixture", "run": row}}
        placement = {"fixture/author": {"worker": "allocation/gpu0"}}
        plan = {"global_gates": {"whole_author": {"path": "gates/author.json",
                "source_stages": ["author_cli"], "required_hosts": ["allocation"]}}}
        digest = fingerprint(plan)
        rows = {"fixture/author": {"state": "failed", "contract": row, "result": {"terminal": True}}}
        with tempfile.TemporaryDirectory() as directory:
            self.write(Path(directory), plan, digest, runs, placement, rows, "allocation")
            raw = proof_path(directory, plan, "whole_author", "allocation").read_bytes()
            value = check_proof(raw, plan, digest, runs, placement, "whole_author", "allocation")
            self.assertFalse(value["all_successful"])
            self.assertNotIn("terminal", value["results"]["fixture/author"])
            value["all_successful"] = True
            with self.assertRaises(ValueError):
                check_proof((canonical(value)+"\n").encode(), plan, digest, runs, placement,
                            "whole_author", "allocation")

    def test_received_bytes_are_atomic_immutable_and_decompression_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proof.json"
            self.assertTrue(install_immutable(path, b"complete"))
            self.assertFalse(install_immutable(path, b"complete"))
            with self.assertRaises(ValueError):
                install_immutable(path, b"changed")
            self.assertEqual(path.read_bytes(), b"complete")
            self.assertEqual(list(Path(directory).glob(".phase-proof-*")), [])
        blob = encode_proof(b"x" * 1000)
        with patch("phase_rendezvous.MAX_PROOF_BYTES", 20):
            with self.assertRaises(ValueError):
                decode_proof(blob)

    def test_runtime_requires_plan_bound_existing_destination_and_limits(self):
        with self.assertRaises(ValueError):
            transport_policy({}, "origin")
        policy = {"schema": "phase-git-rendezvous-v1", "remote_name": "origin",
                  "remote_url_sha256": "a" * 64, "max_compressed_host_bytes": MAX_COMPRESSED_BYTES,
                  "max_uncompressed_proof_bytes": MAX_PROOF_BYTES}
        self.assertEqual(transport_policy({"phase_rendezvous": policy}, "origin"), policy)
        with self.assertRaises(ValueError):
            transport_policy({"phase_rendezvous": policy}, "another_remote")

    @unittest.skipUnless(shutil.which("git"), "Git is unavailable")
    def test_two_host_local_git_exchange_closes_each_phase_without_mac(self):
        plan, digest, runs, placement, rows = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            remote, source = base / "remote.git", base / "source"
            subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
            subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(source), "remote", "add", "origin", str(remote)],
                           check=True, capture_output=True)
            roots = {h: base / h for h in ("allocation", "server77")}
            for root in roots.values(): root.mkdir()
            transports = {h: GitReceiptTransport(base / (h + ".git"), source) for h in roots}
            pending = copy.deepcopy(rows)
            for key, row in pending.items():
                if key.endswith("comparative"): row["state"] = "registered"
            self.write(roots["allocation"], plan, digest, runs, placement, pending, "allocation")
            first = exchange_once(roots["allocation"], plan, digest, runs, placement,
                                  "allocation", transports["allocation"])
            self.assertEqual(first["closed_gates"], [])
            self.write(roots["server77"], plan, digest, runs, placement, pending, "server77")
            second = exchange_once(roots["server77"], plan, digest, runs, placement,
                                   "server77", transports["server77"])
            self.assertEqual(second["closed_gates"], ["whole_search"])
            exchange_once(roots["allocation"], plan, digest, runs, placement,
                          "allocation", transports["allocation"])
            original = proof_path(roots["allocation"], plan, "whole_search", "server77").read_bytes()
            for host in roots:
                self.write(roots[host], plan, digest, runs, placement, rows, host)
                exchange_once(roots[host], plan, digest, runs, placement, host, transports[host])
            complete = exchange_once(roots["allocation"], plan, digest, runs, placement,
                                     "allocation", transports["allocation"])
            self.assertTrue(complete["all_gates_closed"])
            self.assertTrue(verify_global(roots["allocation"], plan, digest, "whole_comparative"))
            self.assertEqual(proof_path(roots["allocation"], plan, "whole_search", "server77").read_bytes(), original)
            # Only two per-plan/per-host branches, flat allowlisted gzip blobs.
            refs = subprocess.check_output(["git", "--git-dir", str(remote), "for-each-ref", "--format=%(refname)"])
            self.assertEqual(set(refs.decode().splitlines()), {branch_ref(digest, h) for h in roots})
            for host in roots:
                commit = transports[host].fetch_host(digest, host)
                self.assertEqual(set(transports[host].read_tree(commit, plan["global_gates"])),
                                 set(plan["global_gates"]))
            proof_path(roots["allocation"], plan, "whole_search", "server77").write_bytes(original + b" ")
            with self.assertRaises(ValueError):
                verify_global(roots["allocation"], plan, digest, "whole_search")

    @unittest.skipUnless(shutil.which("git"), "Git is unavailable")
    def test_non_allowlisted_git_tree_is_rejected_before_installation(self):
        plan, digest, _, _, _ = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory);remote = base / "remote.git";source = base / "source"
            subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
            subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(source), "remote", "add", "origin", str(remote)], check=True, capture_output=True)
            transport = GitReceiptTransport(base / "transport.git", source)
            blob = transport.git("hash-object", "-w", "--stdin", data=b"forbidden checkpoint").stdout.strip()
            tree = transport.git("mktree", data=b"100644 blob " + blob + b"\tcheckpoint.pt\n").stdout.strip().decode()
            commit = transport.git("commit-tree", tree, data=b"malformed fixture\n").stdout.strip().decode()
            with self.assertRaises(ValueError):
                transport.read_tree(commit, plan["global_gates"])


if __name__ == "__main__":
    unittest.main()
