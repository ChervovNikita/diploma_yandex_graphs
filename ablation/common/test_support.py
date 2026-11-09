"""Lets the CPU tests import ``datasets`` on machines without CUDA and DGL."""

import ctypes
import sys
import types
from unittest import mock


def import_datasets():
    try:
        import datasets
    except (OSError, ImportError):
        # Only compute_metrics is needed from this module in the tests.
        sys.modules.setdefault('dgl', types.ModuleType('dgl'))
        with mock.patch.object(ctypes, 'CDLL'):
            import datasets
    return datasets
