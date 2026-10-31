"""Typed path audit from the campaign execution binding."""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'descriptor_binding.py'
spec=importlib.util.spec_from_file_location('_server51_composed_descriptor_paths',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
audit_runtime_paths=module.audit_runtime_paths
