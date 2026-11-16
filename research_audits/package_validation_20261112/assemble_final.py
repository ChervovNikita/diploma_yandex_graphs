"""Assemble the final transfer set after isolated-environment qualification."""
import json
from pathlib import Path
import subprocess

from ablation.campaign.pack import collect_files, repository_archive
from ablation.campaign.offline import REPOSITORY, record
from ablation.campaign.store import atomic_json


def main():
    package=REPOSITORY/"transfer_packages/owner_20261112"
    manifest=REPOSITORY/"ablation/campaign/offline_manifest_owner_20261112.json"
    manifest_hash=record(manifest)["sha256"]
    for path in (REPOSITORY/"research_audits/package_validation_20261112/offline_current_v2.json",
                 package/"relocated_offline_check.json"):
        report=json.loads(path.read_text())
        assert report["passed"] and not report["network_attempts"]
        assert report["manifest_sha256"]==manifest_hash
    assert json.loads((package/"relocated_gpu_smoke.json").read_text())["passed"]
    assert json.loads((package/"scope_verification.json").read_text())["passed"]
    log=(REPOSITORY/"ablation/campaign/evidence/owner_20261112/isolated_environment_tests.log").read_text()
    assert "Ran 210 tests" in log and "OK (skipped=3)" in log
    subprocess.run(["/usr/bin/git","diff","--exit-code","HEAD"],cwd=REPOSITORY,check=True)
    frozen=json.loads(manifest.read_text())
    probe=REPOSITORY/".campaign_test_tmp/owner_relocation_20261112/gnnm_rebuttal_cluster"
    for name,expected in frozen["files"].items():
        assert record(probe/name)==expected, name
    assert record(probe/"ablation/campaign/offline_manifest_owner_20261112.json")["sha256"]==manifest_hash
    includes=[REPOSITORY/"ablation/campaign/evidence/owner_20261112",
              REPOSITORY/"ablation/campaign/runs/owner_smoke_20261112",
              REPOSITORY/"research_audits/package_validation_20261112"]
    includes += [package/name for name in (
        "environment_receipt.json","manifest_correction.json","relocated_offline_check.json",
        "relocated_gpu_smoke.json","relocated_gpu_smoke.log","relocation_layout.json",
        "scope_verification.json")]
    files=collect_files(manifest,includes)
    names=subprocess.check_output(["/usr/bin/git","ls-files","-z"],cwd=REPOSITORY).decode().split("\0")
    extras=list((REPOSITORY/"research_audits/package_validation_20261112").rglob("*.py"))
    for path in [REPOSITORY/name for name in names if name]+extras:
        if not path.resolve().is_relative_to(REPOSITORY) or not path.is_file():
            raise ValueError("Unsafe package input")
        files[str(path.relative_to(REPOSITORY))]=record(path)
    files=dict(sorted(files.items()))
    atomic_json(package/"repository_inputs.json",files)
    repository=repository_archive(package/"gnnm-rebuttal-repository.tar.gz",files,REPOSITORY.name)
    atomic_json(package/"repository_receipt.json",repository)
    environment=json.loads((package/"environment_receipt.json").read_text())
    assert record(REPOSITORY/environment["path"])=={k:environment[k] for k in ("bytes","sha256")}
    receipt=dict(schema="gnnm-campaign-transfer-package-v1",executed=True,
                 manifest=str(manifest.relative_to(REPOSITORY)),manifest_sha256=manifest_hash,
                 repository=repository,environment=environment,
                 source_commit=subprocess.check_output(["/usr/bin/git","rev-parse","HEAD"],cwd=REPOSITORY,text=True).strip(),
                 source_branch="rebuttal-cluster-runs",tracked_source_clean=True,
                 repository_files=len(files),repository_input_bytes=sum(r["bytes"] for r in files.values()),
                 scientific_campaign_started=False,transfer_started=False,
                 preparation_host_relocation_verified=True,
                 relocation_scope="Extracted core source plus verified read-only runtime hardlinks; freshly packed environment relocated and GPU-tested. Temporary environment copy removed after success.",
                 isolated_tests=dict(total=210,passed=207,skipped_legacy_archive=3),
                 gpu_dag_tasks_passed=77,additional_relocated_gpu_probes=3,
                 destination_driver_qualification="pending",
                 destination_cross_host_lock_qualification="pending",
                 required_transfer_files=["gnnm-rebuttal-repository.tar.gz","gnnm-rebuttal-conda.tar.gz",
                                          "SHA256SUMS","HANDOFF.md","transfer_receipt.json"])
    atomic_json(package/"transfer_receipt.json",receipt)
    (package/"SHA256SUMS").write_text("".join(r["sha256"]+"  "+Path(r["path"]).name+"\n"
                                            for r in (repository,environment)))
    (package/"HANDOFF.md").write_text(
        "# Ready package - owner update 12 November 2026\n\n"
        "Transfer the five required files listed in transfer_receipt.json. Check SHA256SUMS. "
        "Unpack the repository and conda archives, activate the relocated prefix and run its conda-unpack. "
        "Set GNNM_ENV and source ablation/campaign/environment.sh; user-site packages must stay disabled. "
        "Read gnnm_rebuttal_cluster/CLUSTER_START_HERE.md first.\n\n"
        "Ask the owner for GPU count and maximum concurrent fits per GPU (suggest four). "
        "Create Slurm integration at the destination after inspecting the site. "
        "Active plan: owner_20261112.json; maximum 24,510 fits and 36,277 tasks. "
        "The approximately 54,370-fit proposal is rejected.\n\n"
        "Verification: 210 isolated-environment tests (207 passed, three legacy-archive skips), "
        "77 successful GPU DAG smoke tasks and three successful relocated GPU probes; no TEST scoring. "
        "The host-package metadata mismatch caught during the first relocation check was corrected "
        "without changing executable source, data or the environment archive. "
        "Full member integrity and strict offline checks passed.\n\n"
        "No production training or transfer started. Destination driver and cross-host filesystem "
        "qualification remain required. Exact BUDDY paper-protocol replication has documented limits "
        "in CLUSTER_START_HERE.md.\n")
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=="__main__":
    main()
