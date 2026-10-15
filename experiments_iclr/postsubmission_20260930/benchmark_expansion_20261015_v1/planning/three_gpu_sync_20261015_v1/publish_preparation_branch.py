"""Commit reviewed files on a new branch without changing the live HEAD or index.

Invoked on the verified allocation through remote_control.call. The temporary
index and receipts stay inside the project. This does not push or run models.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess


def publish(root, selection, receipt):
    root = Path(root).resolve()
    selection = Path(selection).resolve()
    receipt = Path(receipt).resolve()
    if not selection.is_relative_to(root) or not receipt.is_relative_to(root):
        raise ValueError("Project-local selection and receipt required")
    value = json.loads(selection.read_text())
    if value.get("reviewed") is not True or value.get("full_campaign_GO") is not False:
        raise ValueError("Explicit reviewed preparation-only selection required")
    branch = value["branch"]
    if not branch.startswith("codex/"):
        raise ValueError("Use a distinct codex/ preparation branch")

    def git(*args, env=None, data=None):
        return subprocess.check_output(
            ["git", *args], cwd=root, env=env, input=data, text=True
        ).strip()

    git("check-ref-format", "refs/heads/" + branch)
    head = git("rev-parse", "HEAD")
    symbolic = git("symbolic-ref", "HEAD")
    if head != value["expected_live_head"] or symbolic == "refs/heads/" + branch:
        raise ValueError("Live HEAD changed or destination is the live branch")
    if receipt.exists():
        raise ValueError("Preserve previous publication receipt")
    existing = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", "refs/heads/" + branch], cwd=root
    )
    if existing.returncode != 1:
        raise ValueError("Require a fresh preparation branch")
    index_path = Path(git("rev-parse", "--git-path", "index"))
    if not index_path.is_absolute():
        index_path = root / index_path
    index_before = hashlib.sha256(index_path.read_bytes()).hexdigest() if index_path.exists() else None
    rows = value["files"]
    if len({row["path"] for row in rows}) != len(rows) or not rows:
        raise ValueError("Unique nonempty file selection required")
    contents = []
    for row in rows:
        path = root / row["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("No symlink or escaped path")
        if path.suffix not in {".py", ".md", ".json", ".txt", ".csv", ".patch"} or "results" in path.relative_to(root).parts:
            raise ValueError("Only selected source and compact records may be committed")
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"] or len(raw) > 2_000_000:
            raise ValueError("Reviewed bytes changed or file too large: " + row["path"])
        if b"PRIVATE " + b"KEY-----" in raw:
            raise ValueError("Private-key marker in selection")
        if path.suffix == ".py":
            ast.parse(raw.decode(), filename=str(path))
        contents.append((row["path"], path))
    temporary = receipt.with_suffix(".index")
    if temporary.exists():
        raise ValueError("Preserve an existing temporary index")
    env = dict(os.environ, GIT_INDEX_FILE=str(temporary))
    git("read-tree", head, env=env)
    for relative, path in contents:
        blob = git("hash-object", "-w", "--", str(path))
        git("update-index", "--add", "--cacheinfo", "100644," + blob + "," + relative, env=env)
    tree = git("write-tree", env=env)
    names = git("diff-tree", "--no-commit-id", "--name-only", "-r", head, tree).splitlines()
    if not set(names).issubset({row["path"] for row in rows}) or not names:
        raise ValueError("Unexpected tree changes or no changes")
    commit = git("commit-tree", tree, "-p", head, data=value["message"] + "\n")
    if git("rev-parse", "HEAD") != head or git("symbolic-ref", "HEAD") != symbolic:
        raise ValueError("Live HEAD changed during preparation")
    index_after = hashlib.sha256(index_path.read_bytes()).hexdigest() if index_path.exists() else None
    if index_before != index_after:
        raise ValueError("Live index changed during preparation")
    git("update-ref", "refs/heads/" + branch, commit, "0" * 40)
    result = {
        "commit": commit, "branch": branch, "parent": head,
        "live_head_unchanged": True, "live_index_unchanged": True,
        "changed_files": names, "full_campaign_GO": False,
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    }
    with receipt.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    temporary.unlink()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.root, args.selection, args.receipt)))
