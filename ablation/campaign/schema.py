"""Validate an immutable campaign and expand visible subprocess arguments."""

import hashlib
import json
import re
import string
from pathlib import Path

SCHEMA = "gnnm-campaign-v1"
PLACEHOLDERS = {"python", "repo", "root", "task_dir", "output_dir", "device", "plan"}
TERMINAL = {"complete", "failed", "skipped", "blocked"}
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)*$")


def canonical_hash(value):
    content = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Expected a relative path without parent traversal: {value}")
    return path


def validate_plan(plan):
    if plan.get("schema") != SCHEMA:
        raise ValueError(f"Expected schema {SCHEMA}")
    if not IDENTIFIER.fullmatch(plan.get("campaign_id", "")):
        raise ValueError("campaign_id must be a stable path-safe identifier")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("A campaign requires a nonempty task list")
    by_id = {}
    formatter = string.Formatter()
    for task in tasks:
        task_id = task.get("id", "")
        if not IDENTIFIER.fullmatch(task_id):
            raise ValueError(f"Invalid task identifier: {task_id!r}")
        if task_id in by_id:
            raise ValueError(f"Duplicate task identifier: {task_id}")
        by_id[task_id] = task
        if task.get("kind", "gpu") not in {"gpu", "cpu"}:
            raise ValueError(f"Invalid task kind: {task_id}")
        argv = task.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError(f"argv must be a nonempty string list: {task_id}")
        for argument in argv:
            for _, field, format_spec, conversion in formatter.parse(argument):
                if field is not None and (field not in PLACEHOLDERS or format_spec or conversion):
                    raise ValueError(f"Unsupported argument placeholder: {field}")
        for name in ("dependencies", "optional_dependencies"):
            dependencies = task.get(name, [])
            if not isinstance(dependencies, list) or not all(isinstance(x, str) for x in dependencies):
                raise ValueError(f"Invalid {name}: {task_id}")
            if len(set(dependencies)) != len(dependencies):
                raise ValueError(f"Duplicate {name}: {task_id}")
        if int(task.get("max_attempts", 3)) < 1:
            raise ValueError(f"max_attempts must be positive: {task_id}")
        resources = task.get("resources", {})
        if float(resources.get("memory_gb", 0)) < 0 or int(resources.get("cpus", 1)) < 1:
            raise ValueError(f"Invalid resource request: {task_id}")
        for output in task.get("required_outputs", []):
            relative_path(output)
        condition = task.get("when")
        if condition:
            if not isinstance(condition, dict) or not all(k in condition for k in ("task", "path", "field", "equals")):
                raise ValueError(f"Invalid condition: {task_id}")
            relative_path(condition["path"])
            if condition["task"] not in task.get("dependencies", []):
                raise ValueError(f"Condition producer must be a required dependency: {task_id}")
    children = {task_id: [] for task_id in by_id}
    remaining = {}
    for task_id, task in by_id.items():
        dependencies = set(task.get("dependencies", []) + task.get("optional_dependencies", []))
        remaining[task_id] = len(dependencies)
        for dependency in dependencies:
            if dependency not in by_id:
                raise ValueError(f"Unknown dependency {dependency} in {task_id}")
            children[dependency].append(task_id)
    ready = [task_id for task_id, count in remaining.items() if count == 0]
    visited = 0
    while ready:
        task_id = ready.pop()
        visited += 1
        for child in children[task_id]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    if visited != len(by_id):
        raise ValueError("Campaign dependency graph contains a cycle")
    return by_id


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    validate_plan(plan)
    return plan


def expand_argv(task, context):
    return [argument.format_map(context) for argument in task["argv"]]
