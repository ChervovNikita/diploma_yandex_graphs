"""Commit reviewed disabled preparation and the superseding README, no fits."""
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

REPO = Path("/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs")
PHASE = REPO / "experiments_iclr/postsubmission_20260930"
HERE = PHASE / "reference_wrapper_redesign_20261013_v1"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main():
    assert socket.gethostname() == "anogena-2-0"
    assert subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() == ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
    os.chdir(REPO)
    assert git("branch", "--show-current") == "codex/postsubmission-research-20260930"
    assert git("remote", "get-url", "origin") == "git@github.com:ChervovNikita/diploma_yandex_graphs.git"
    assert not (HERE / "COMMIT_RECEIPT.json").exists(), "Already committed; do not recommit"
    summary = json.loads((HERE / "SUMMARY.json").read_text())
    assert summary["fits"] == 2480 and summary["source_cells"] == 27
    assert summary["scientific_runtime_ready"] is False and summary["GO"] is False
    history = HERE / "history"
    history.mkdir(exist_ok=True)
    saved_readme = history / "ROOT_README_BEFORE_REDESIGN.md"
    if saved_readme.exists():
        original = saved_readme.read_text()
    else:
        original = (REPO / "README.md").read_text()
        assert "## Revised wrapper benchmark — 13 October 2026" not in original
        saved_readme.write_text(original)
    first, remainder = original.split("\n", 1)
    remainder = remainder.replace("## Final massive study preparation — 13 October 2026", "## Superseded massive-study proposal — 13 October 2026", 1)
    (REPO / "README.md").write_text(first + "\n\n" + (HERE / "ROOT_README_BLOCK.md").read_text().rstrip() + "\n\n" + remainder.lstrip("\n"))
    names = ["README.md", str((PHASE / "AGENTS.md").relative_to(REPO)),
             str((PHASE / "final_sync_master_v29_20261013_v1/run_gpu.sh").relative_to(REPO)),
             str((PHASE / "final_sync_master_v29_20261013_v1/retired_entrypoint_history_20261013/run_gpu_v29_original.sh.txt").relative_to(REPO))]
    names += [str(p.relative_to(REPO)) for p in sorted(HERE.rglob("*")) if p.is_file()
              and not ({"transport", "__pycache__"} & set(p.relative_to(HERE).parts))
              and p.name != "PLAN.json" and p.suffix != ".pyc"]
    git("add", "-f", "--", *names)
    staged = git("diff", "--cached", "--name-only").splitlines()
    allowed_roots = [str(HERE.relative_to(REPO)) + "/", str((PHASE / "final_sync_master_v29_20261013_v1/retired_entrypoint_history_20261013").relative_to(REPO)) + "/"]
    allowed_exact = set(names[:3])
    assert all(path in allowed_exact or any(path.startswith(root) for root in allowed_roots) for path in staged)
    assert not any(path.endswith("/PLAN.json") or path.endswith(".pyc") for path in staged)
    # Raw author/reference captures retain their original bytes and whitespace.
    # CRLF is valid in frozen CSV inventories; inspect authored changes only.
    prefix = str(HERE.relative_to(REPO))
    git("-c", "core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol",
        "diff", "--cached", "--check", "--", ".",
        ":(exclude)" + prefix + "/native_sources/**",
        ":(exclude)" + prefix + "/references/**",
        ":(exclude)" + prefix + "/history/**",
        ":(exclude)experiments_iclr/postsubmission_20260930/final_sync_master_v29_20261013_v1/retired_entrypoint_history_20261013/**")
    print(git("diff", "--cached", "--stat"))
    git("commit", "-m", "Redesign campaign around fixed native backbones and shared GNNM wrappers")
    receipt = {"schema": "reference-wrapper-design-commit-v1", "branch": git("branch", "--show-current"),
               "commit": git("rev-parse", "HEAD"), "plan_hash": summary["plan_hash"],
               "committed_paths": staged, "GO": False, "runtime_ready": False, "scientific_fits_started": 0}
    (HERE / "COMMIT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "committed_paths"}))


if __name__ == "__main__":
    main()
