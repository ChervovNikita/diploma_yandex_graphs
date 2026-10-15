"""Versioned molecular BN-state candidate; no frozen pilot source is edited."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import types


def transform(session, job):
    path = Path(__file__).with_name("banks.py")
    expected = job.get("method_settings", {}).get("banks_sha256")
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Exact private-BN source binding required")
    if job.get("family") != "ogb" or job.get("base_procedure") != "gnnm_boundary_native":
        raise ValueError("This port qualifies only molecular base GNNM")
    spec = importlib.util.spec_from_file_location("_private_bn_" + expected[:16], path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    metadata = module.attach_molecular(session)
    metadata["banks_sha256"] = expected
    original_metadata = session.metadata

    def combined_metadata(self):
        value = original_metadata()
        value["candidate_extension"] = metadata
        value["boundary_BN_policy_override"] = "route private running state, original shared affine"
        return value

    session.metadata = types.MethodType(combined_metadata, session)
    return session
