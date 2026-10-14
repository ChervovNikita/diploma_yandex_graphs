"""Prepared link-task extensions. Importing this package imports no Torch."""
from .adapters import make_hidden_residual, make_late_gate, install_task_prehead_adapter
from .capture import TaskPreheadCapture
from .diagnostics import link_bank_diagnostics
from .objectives import build_objectives, native_record_losses, reduce_native, step_blocks
from .planning import contrast_plan, query_keys, raw_responsibilities

__all__ = ['build_objectives', 'step_blocks', 'reduce_native', 'native_record_losses',
           'query_keys', 'contrast_plan', 'raw_responsibilities', 'link_bank_diagnostics',
           'make_hidden_residual', 'make_late_gate', 'install_task_prehead_adapter']
__all__.append('TaskPreheadCapture')
