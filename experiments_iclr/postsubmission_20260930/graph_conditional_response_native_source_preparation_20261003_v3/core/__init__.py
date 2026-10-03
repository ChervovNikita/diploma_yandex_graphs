"""Preparation only. Importing this package uses only Python's standard library.

Torch is imported inside numerical entry points. No entry point performs data access,
training, a numerical witness, protocol freeze, or launch on import.
"""
from .groups import FixedPlan, LabelRoles, ResolutionRequired, raw_group_keys
from .response import ActiveReference, ResponseFailure, build_active_reference, responses
from .guards import CostLedger, CompetenceReference, GuardDecision
from .transaction import Custody, ExternalStateHook, guarded_adamw_displacement

__all__ = [
    "FixedPlan", "LabelRoles", "ResolutionRequired", "raw_group_keys",
    "ActiveReference", "ResponseFailure", "build_active_reference", "responses",
    "CostLedger", "CompetenceReference", "GuardDecision", "Custody",
    "ExternalStateHook", "guarded_adamw_displacement",
]
