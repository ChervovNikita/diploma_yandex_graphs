"""Prepare or run a host-local Git rendezvous for immutable phase proofs.

No model, checkpoint, output directory, role tensor or credential is transferred.
The default CLI only describes the proposal. Explicit --once/--watch requires
the existing plan-bound human GO and reuses an existing configured Git remote.
This process never starts scientific workers; a host startup wrapper must run
it alongside the unchanged launcher, in a session independent of the Mac.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

try:
    from .phase_closure import collect, expected_keys
    from .spec import canonical, confined, fingerprint
except ImportError:
    from phase_closure import collect, expected_keys
    from spec import canonical, confined, fingerprint


# The new Coauthor split cohorts require an 11.1 MiB conservative host-tree
# bound. This cap governs hash-only phase metadata, never model/data artifacts.
MAX_COMPRESSED_BYTES = 64 * 1024**2
MAX_PROOF_BYTES = 64 * 1024**2
ALLOWED_STAGES = {"search", "search_selection", "comparative", "bank_export", "calibration"}
HOSTS = {"allocation", "server77"}
HASH = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
PROOF_FIELDS = {"status", "host", "closed_phase", "plan_hash", "run_keys",
                "results", "closure_authority", "terminal_roster", "all_successful"}


def gates_for(plan):
    """Only pre-outcome engine closures may enter the transport."""
    gates = plan.get("global_gates", {})
    if not gates:
        raise ValueError("An explicit nonempty global gate roster is required")
    for name, gate in gates.items():
        stages = gate.get("source_stages")
        hosts = gate.get("required_hosts")
        author_terminal = name == "whole_author" and stages == ["author_cli"]
        if (not isinstance(name, str) or not NAME.fullmatch(name)
                or not isinstance(stages, list) or not stages
                or len(set(stages)) != len(stages)
                or (not set(stages) <= ALLOWED_STAGES and not author_terminal)
                or not isinstance(hosts, list) or not hosts
                or len(set(hosts)) != len(hosts) or not set(hosts) <= HOSTS):
            raise ValueError("Only declared TEST-free engine phase gates are admitted")
    return gates


def branch_ref(digest, host):
    if not HASH.fullmatch(digest) or host not in HOSTS:
        raise ValueError("A full immutable plan hash and declared host are required")
    return "refs/heads/codex/phase-receipts/" + digest + "/" + host


def proof_path(root, plan, gate, host):
    return confined(root, str(Path(plan["global_gates"][gate]["path"]).parent
                               / (host + "_" + gate + ".json")))


def check_proof(raw, plan, digest, runs, placement, gate_name, host):
    """Reject arbitrary payloads, incomplete comparisons and changed contracts."""
    gates = gates_for(plan)
    if host not in gates[gate_name]["required_hosts"] or len(raw) > MAX_PROOF_BYTES:
        raise ValueError("Unexpected host or oversized phase proof")
    value = json.loads(raw)
    keys = expected_keys(plan, runs, placement, gate_name, host)
    if (not keys or not isinstance(value, dict) or set(value) != PROOF_FIELDS
            or value["status"] != "complete" or value["host"] != host
            or value["closed_phase"] != gate_name or value["plan_hash"] != digest
            or value["run_keys"] != keys or set(value["results"]) != set(keys)
            or value["closure_authority"] != "whole_assigned_host_registry"
            or value["terminal_roster"] is not True
            or type(value["all_successful"]) is not bool):
        raise ValueError("Phase proof does not cover the exact assigned host cohort")
    for key in keys:
        result, row = value["results"][key], runs[key]["run"]
        stage = row["config"].get("stage")
        permitted = {"search": "model_fit", "search_selection": "selection",
                     "comparative": "model_fit", "bank_export": "ensemble_assembly", "calibration": "calibration"}
        if gate_name == "whole_author":
            permitted["author_cli"] = "author_cli_group"
        if row["kind"] != permitted.get(stage):
            raise ValueError("Only engine closure jobs or whole-author terminal groups are admitted")
        if (not isinstance(result, dict)
                or set(result) != {"state", "result_hash", "contract_hash"}
                or not isinstance(result["result_hash"], str)
                or not HASH.fullmatch(result["result_hash"])
                or result["contract_hash"] != fingerprint(row)):
            raise ValueError("Only exact contract/result digests may be transported")
        terminal_failure = ((row["kind"] == "model_fit" and stage == "search")
                            or (gate_name == "whole_author" and row["kind"] == "author_cli_group"))
        if result["state"] != "complete" and not (terminal_failure and result["state"] == "failed"):
            raise ValueError("An incomplete selection/comparative/CAL cohort cannot close")
    if value["all_successful"] != all(value["results"][key]["state"] == "complete" for key in keys):
        raise ValueError("Terminal closure cannot claim failed runs succeeded")
    if raw != (canonical(value) + "\n").encode():
        raise ValueError("Only canonical writer-produced proof bytes are accepted")
    return value


def encode_proof(raw):
    if len(raw) > MAX_PROOF_BYTES:
        raise ValueError("Phase proof exceeds the frozen small-metadata limit")
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise ValueError("Compressed phase proof exceeds the frozen transport limit")
    return compressed


def decode_proof(blob):
    if len(blob) > MAX_COMPRESSED_BYTES:
        raise ValueError("Oversized receipt blob")
    import io
    with gzip.GzipFile(fileobj=io.BytesIO(blob), mode="rb") as stream:
        raw = stream.read(MAX_PROOF_BYTES + 1)
        if len(raw) > MAX_PROOF_BYTES or stream.read(1):
            raise ValueError("Decompressed phase proof exceeds the frozen limit")
    return raw


def install_immutable(path, raw):
    """An existing receipt is checked, never overwritten or reconstructed."""
    path = Path(path)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("An immutable received phase proof changed")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".phase-proof-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic no-clobber link: the launcher sees complete bytes or no file.
        os.link(temporary, path)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("A concurrent received phase proof differs")
        return False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def verify_global(root, plan, digest, gate_name):
    """Match the existing launcher gate format without changing its authority."""
    gate = plan["global_gates"][gate_name]
    target = confined(root, gate["path"])
    if not target.exists():
        return False
    try:
        value = json.loads(target.read_bytes())
    except json.JSONDecodeError as error:
        raise RuntimeError("Global phase proof is not yet complete") from error
    if (set(value) != {"status", "plan_hash", "closed_phase", "host_receipts", "closure_authority",
                       "terminal_roster", "all_successful"}
            or value["status"] != "complete" or value["plan_hash"] != digest
            or value["closed_phase"] != gate_name
            or value["closure_authority"] != "all_declared_hosts_with_exact_contracts"
            or value["terminal_roster"] is not True or type(value["all_successful"]) is not bool
            or set(value["host_receipts"]) != set(gate["required_hosts"])):
        raise ValueError("An existing global proof differs from the frozen gate")
    all_successful = True
    for host in gate["required_hosts"]:
        path = proof_path(root, plan, gate_name, host)
        raw = path.read_bytes()
        receipt = {"path": str(path.relative_to(Path(root).resolve())), "bytes": len(raw),
                   "sha256": hashlib.sha256(raw).hexdigest()}
        if value["host_receipts"][host] != receipt:
            raise ValueError("A host proof changed after global closure")
        all_successful = all_successful and json.loads(raw)["all_successful"]
    if value["all_successful"] != all_successful:
        raise ValueError("Global terminal closure changed success accounting")
    return True


class GitReceiptTransport:
    """Isolated bare Git database; no checkout/index from the study is used."""
    def __init__(self, repository, source_repository, remote_name="origin", *, expected_remote_hash=None):
        self.repository = Path(repository).resolve()
        if not NAME.fullmatch(remote_name):
            raise ValueError("Use an existing simple remote name")
        self.env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="",
                        GIT_AUTHOR_NAME="Phase receipt transport",
                        GIT_AUTHOR_EMAIL="phase-receipts@invalid.local",
                        GIT_COMMITTER_NAME="Phase receipt transport",
                        GIT_COMMITTER_EMAIL="phase-receipts@invalid.local")
        # Keep the remote URL private. Existing credentials/configuration are
        # reused; no credential helper, key, token or askpass is created.
        remote = self._command(["git", "-C", str(Path(source_repository).resolve()),
                                "remote", "get-url", remote_name]).stdout.strip()
        if not remote:
            raise ValueError("The source repository has no configured remote")
        if expected_remote_hash is not None and hashlib.sha256(remote).hexdigest() != expected_remote_hash:
            raise ValueError("The configured receipt destination differs from the approved plan")
        if not self.repository.exists():
            self.repository.parent.mkdir(parents=True, exist_ok=True)
            self._command(["git", "init", "--bare", str(self.repository)])
            self.git("remote", "add", "origin", remote.decode())
        elif self.git("rev-parse", "--is-bare-repository").stdout.strip() != b"true":
            raise ValueError("Receipt transport requires an isolated bare repository")
        elif self.git("remote", "get-url", "origin").stdout.strip() != remote:
            raise ValueError("The configured receipt remote changed")

    def _command(self, argv, *, data=None, accepted=(0,)):
        result = subprocess.run(argv, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=self.env, timeout=60, check=False)
        if result.returncode not in accepted:
            # Git stderr can echo URLs. Do not expose credentials or payloads.
            raise RuntimeError("Git receipt operation failed (exit " + str(result.returncode) + ")")
        return result

    def git(self, *args, data=None, accepted=(0,)):
        return self._command(["git", "--git-dir", str(self.repository), *args],
                             data=data, accepted=accepted)

    def fetch_host(self, digest, host):
        ref = branch_ref(digest, host)
        found = self.git("ls-remote", "--exit-code", "--heads", "origin", ref, accepted=(0, 2))
        if found.returncode == 2:
            return None
        destination = "refs/remotes/receipts/" + digest + "/" + host
        # Keep the short receipt-only ancestry so a later phase is provably a
        # fast-forward. Depth1 would hide the prior phase parent on peers.
        self.git("fetch", "--no-tags", "--no-write-fetch-head", "origin",
                 ref + ":" + destination)
        return self.git("rev-parse", destination).stdout.strip().decode()

    def read_tree(self, commit, allowed_gates):
        if commit is None:
            return {}
        tree = self.git("ls-tree", "-z", commit).stdout
        entries = {}
        total = 0
        for entry in tree.split(b"\0"):
            if not entry:
                continue
            meta, path = entry.split(b"\t", 1)
            mode, kind, oid = meta.split()
            name = path.decode()
            if (mode != b"100644" or kind != b"blob" or not name.endswith(".json.gz")
                    or name[:-8] not in allowed_gates):
                raise ValueError("A receipt branch contains a non-allowlisted payload")
            size = int(self.git("cat-file", "-s", oid.decode()).stdout)
            total += size
            if size > MAX_COMPRESSED_BYTES or total > MAX_COMPRESSED_BYTES:
                raise ValueError("Receipt branch exceeds the small-metadata limit")
            entries[name[:-8]] = self.git("cat-file", "blob", oid.decode()).stdout
        return entries

    def publish(self, digest, host, allowed_gates, local_blobs):
        if not set(local_blobs) <= set(allowed_gates):
            raise ValueError("A host may publish only its declared phase proofs")
        parent = self.fetch_host(digest, host)
        existing = self.read_tree(parent, allowed_gates)
        for gate, blob in local_blobs.items():
            if gate in existing and existing[gate] != blob:
                raise ValueError("A published host phase proof is immutable")
        merged = dict(existing, **local_blobs)
        if merged == existing:
            return False
        if sum(map(len, merged.values())) > MAX_COMPRESSED_BYTES:
            raise ValueError("Receipt branch exceeds the small-metadata limit")
        lines = []
        for gate, blob in sorted(merged.items()):
            oid = self.git("hash-object", "-w", "--stdin", data=blob).stdout.strip()
            lines.append(b"100644 blob " + oid + b"\t" + gate.encode() + b".json.gz\n")
        tree = self.git("mktree", data=b"".join(lines)).stdout.strip().decode()
        args = ["commit-tree", tree]
        if parent:
            args += ["-p", parent]
        commit = self.git(*args, data=b"Immutable host phase receipts\n").stdout.strip().decode()
        # Normal fast-forward push only. No force/reset, main-branch write,
        # source staging, checkpoint transfer or application message is used.
        self.git("push", "origin", commit + ":" + branch_ref(digest, host))
        return True


def exchange_once(root, plan, digest, runs, placement, host, transport):
    gates = gates_for(plan)
    if host not in HOSTS or not HASH.fullmatch(digest):
        raise ValueError("Unexpected local host/plan binding")
    own_gates = {name for name, gate in gates.items() if host in gate["required_hosts"]}
    local = {}
    for name in own_gates:
        path = proof_path(root, plan, name, host)
        if path.exists():
            if path.is_symlink() or path.stat().st_size > MAX_PROOF_BYTES:
                raise ValueError("Unexpected local phase proof storage")
            raw = path.read_bytes()
            try:
                check_proof(raw, plan, digest, runs, placement, name, host)
            except json.JSONDecodeError as error:
                # Existing phase_closure writes through open(x). A watcher can
                # see that file mid-write; no proof is transported/released.
                raise RuntimeError("Local phase proof is not yet complete") from error
            local[name] = encode_proof(raw)
    published = transport.publish(digest, host, own_gates, local)
    received = 0
    required_hosts = {h for gate in gates.values() for h in gate["required_hosts"]}
    for peer in sorted(required_hosts - {host}):
        allowed = {name for name, gate in gates.items() if peer in gate["required_hosts"]}
        blobs = transport.read_tree(transport.fetch_host(digest, peer), allowed)
        for name, blob in blobs.items():
            raw = decode_proof(blob)
            check_proof(raw, plan, digest, runs, placement, name, peer)
            received += install_immutable(proof_path(root, plan, name, peer), raw)
    closed = []
    for name, gate in gates.items():
        paths = [proof_path(root, plan, name, peer) for peer in gate["required_hosts"]]
        if not all(path.exists() for path in paths):
            continue
        for peer, path in zip(gate["required_hosts"], paths):
            check_proof(path.read_bytes(), plan, digest, runs, placement, name, peer)
        if not verify_global(root, plan, digest, name):
            try:
                collect(root, plan, digest, runs, placement, name)
            except (FileExistsError, ValueError):
                if not verify_global(root, plan, digest, name):
                    raise
            if not verify_global(root, plan, digest, name):
                raise ValueError("Global proof was not atomically installed")
        closed.append(name)
    return {"plan_hash": digest, "host": host, "published": published,
            "received_proofs": received, "closed_gates": sorted(closed),
            "all_gates_closed": len(closed) == len(gates), "scientific_workers_started": 0}


def proposal(plan, digest, host):
    gates = gates_for(plan)
    return {"schema": "phase-git-rendezvous-proposal-v1", "plan_hash": digest,
            "host": host, "branches": {h: branch_ref(digest, h) for h in sorted(HOSTS)},
            "gates": sorted(gates), "max_compressed_host_bytes": MAX_COMPRESSED_BYTES,
            "max_uncompressed_proof_bytes": MAX_PROOF_BYTES,
            "payload": "canonical state/contract/result hash closure proofs only; deterministic gzip",
            "remote": "existing configured remote; no new credentials",
            "required_plan_binding": {"field": "phase_rendezvous", "schema": "phase-git-rendezvous-v1",
                                      "remote_url_sha256": "root binds existing private URL digest",
                                      "remote_name": "root binds existing remote name",
                                      "max_compressed_host_bytes": MAX_COMPRESSED_BYTES,
                                      "max_uncompressed_proof_bytes": MAX_PROOF_BYTES},
            "startup": "host-local detached sidecar alongside approved launcher; no MacLink dependency",
            "readiness": "PREPARATION_ONLY_REQUIRES_ROOT_FROZEN_INTEGRATION_AND_GO",
            "repository_push_performed": False, "scientific_workers_started": 0}


def capacity_report(plan, runs, placement):
    """Conservative proof-size bounds from exact keys and fixed hash widths.

    Placeholder result hashes are never scientific evidence or transported.
    The raw-size bound admits the longest terminal state and boolean. Gzip's
    default zlib bound plus wrapper margin is deliberately conservative.
    """
    gates = gates_for(plan)
    hosts = sorted({host for gate in gates.values() for host in gate['required_hosts']})
    result = {}
    for host in hosts:
        details = {}
        for name, gate in gates.items():
            if host not in gate['required_hosts']:
                continue
            keys = expected_keys(plan, runs, placement, name, host)
            value = {'status': 'complete', 'host': host, 'closed_phase': name,
                     'plan_hash': '0' * 64, 'run_keys': keys,
                     'results': {key: {'state': 'complete',
                         'result_hash': hashlib.sha256(('placeholder-result/' + key).encode()).hexdigest(),
                         'contract_hash': fingerprint(runs[key]['run'])} for key in keys},
                     'closure_authority': 'whole_assigned_host_registry',
                     'terminal_roster': True, 'all_successful': False}
            raw = (canonical(value) + '\n').encode()
            n = len(raw)
            bound = n + (n >> 12) + (n >> 14) + (n >> 25) + 64
            details[name] = {'source_run_count': len(keys), 'maximum_raw_proof_bytes': n,
                             'conservative_gzip_bound_bytes': bound,
                             'placeholder_gzip_bytes': len(gzip.compress(raw, mtime=0))}
        total = sum(row['conservative_gzip_bound_bytes'] for row in details.values())
        maximum = max((row['maximum_raw_proof_bytes'] for row in details.values()), default=0)
        result[host] = {'gates': details, 'compressed_host_tree_bound_bytes': total,
                        'maximum_raw_proof_bytes': maximum,
                        'fits_fixed_limits': total <= MAX_COMPRESSED_BYTES and maximum <= MAX_PROOF_BYTES}
    return {'schema': 'phase-receipt-size-qualification-v1', 'hosts': result,
            'all_hosts_fit_fixed_limits': all(row['fits_fixed_limits'] for row in result.values()),
            'scientific_results_read': False, 'placeholder_proofs_transported': False,
            'uncompressed_limits_and_conservative_gzip_bounds': True}


def transport_policy(plan, remote_name):
    value = plan.get("phase_rendezvous")
    if (not isinstance(value, dict)
            or set(value) != {"schema", "remote_name", "remote_url_sha256",
                              "max_compressed_host_bytes", "max_uncompressed_proof_bytes"}
            or value["schema"] != "phase-git-rendezvous-v1"
            or value["remote_name"] != remote_name
            or not isinstance(value["remote_url_sha256"], str)
            or not HASH.fullmatch(value["remote_url_sha256"])
            or value["max_compressed_host_bytes"] != MAX_COMPRESSED_BYTES
            or value["max_uncompressed_proof_bytes"] != MAX_PROOF_BYTES):
        raise ValueError("Root must freeze the exact receipt destination and limits into the GO-bound plan")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--host", choices=sorted(HOSTS), required=True)
    parser.add_argument("--source-repository", help="Existing repository with the approved Git remote")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--approval", help="Existing explicit human GO, required for exchange")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if not 15 <= args.poll_seconds <= 300:
        raise ValueError("Use a bounded 15–300 second polling interval")
    root = Path(args.project_root).resolve()
    plan_path = confined(root, args.plan)
    try:
        from .launcher import host_check, load_plan
    except ImportError:
        from launcher import host_check, load_plan
    live = args.once or args.watch
    if live and (not args.approval or not args.source_repository):
        raise ValueError("Exchange requires existing plan-bound human GO and an existing Git repository")
    if live and args.host == "allocation":
        host_check(args.host, {})  # Verify route before project metadata reads.
    approval = confined(root, args.approval) if live else None
    plan, digest, _, runs, placement, _ = load_plan(root, plan_path, approval)
    if live and args.host == "server77":
        host_check(args.host, plan)  # Only an exact frozen hostname/UUID binding.
    print(canonical(proposal(plan, digest, args.host)), flush=True)
    if not live:
        return
    policy = transport_policy(plan, args.remote)
    directory = confined(root, str(Path(plan["output_root"]) / ".phase-rendezvous" / digest))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (args.host + ".lock")).open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport = GitReceiptTransport(directory / (args.host + ".git"),
                                        args.source_repository, args.remote,
                                        expected_remote_hash=policy["remote_url_sha256"])
        last = None
        failures = 0
        while True:
            try:
                status = exchange_once(root, plan, digest, runs, placement, args.host, transport)
                failures = 0
                if status != last:
                    print(canonical(status), flush=True)
                last = status
                if args.once or status["all_gates_closed"]:
                    return
                time.sleep(args.poll_seconds)
            except RuntimeError:
                if args.once:
                    raise
                failures += 1
                # No changed gates, scientific restart or credential prompting.
                print(canonical({"host": args.host, "plan_hash": digest,
                                 "transport_retry": failures, "gates_unchanged": True}), flush=True)
                time.sleep(min(300, args.poll_seconds * 2**min(failures, 3)))


if __name__ == "__main__":
    main()
