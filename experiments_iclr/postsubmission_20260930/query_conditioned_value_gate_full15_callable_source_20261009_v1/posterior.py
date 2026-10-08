"""Reuse pinned P0 posterior operations; account for the already registered gate."""
import hashlib
import importlib.util
from pathlib import Path
import sys

ORIGINAL = Path(__file__).resolve().parent.parent / 'label_only_staged_posterior_full_WikiCS_source_20261008_v1' / 'posterior.py'
ORIGINAL_SHA = '3d4d36cbd648dcda59e71d6fe17ef31e383f802130166627f3f356b934a521ab'
if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA:
    raise ValueError('Changed original posterior operations')
_spec = importlib.util.spec_from_file_location('_query_value_original_posterior', ORIGINAL)
_original = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = _original
_spec.loader.exec_module(_original)
anchor_reach = _original.anchor_reach
bounded = _original.bounded
readout = _original.readout


def nonlinear_single(bank, **kwargs):
    bank = _original.nonlinear_single(bank, **kwargs)
    original = bank.descriptor
    def descriptor():
        value = original()
        value.update(learned_parameter_formula='4*(128*F+128*C+4096)+256*256+64*(F+1)',
                     query_gate_shared_across_routes=True, query_gate_parameters=32832,
                     owned_parameter_count=sum(p.numel() for p in bank.parameters()))
        return value
    bank.descriptor = descriptor
    return bank
