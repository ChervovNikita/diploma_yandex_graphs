"""Explicit task-specific policies; link policies never use the OGB fallback."""
from bookkeeping import Policy, Selector
from common import load_module


def controller(job, root=None):
    if job.get("family") not in ("node", "ogb", "link"):
        raise ValueError("An explicit supported task family is required")
    if job.get("family") != "link":
        return Policy.from_dict(job["policy"], job["family"]), None, None
    if root is None or not job.get("link_policy_module") or not job.get("link_policy_sha256"):
        raise ValueError("Link dispatch requires an explicitly pinned link policy module and repository root")
    module, pin = load_module(job["link_policy_module"], root, job["link_policy_sha256"])
    policy = module.create_policy(job, root=root)
    if getattr(policy, "family", None) != "link":
        raise ValueError("The explicit policy module did not return a link policy")
    return policy, module, pin


def create_selector(policy, module=None):
    return Selector(policy) if module is None else module.create_selector(policy)
