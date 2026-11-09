"""Shared backbone with ordinary per-member matrices. Worker for the full matrix.

Run from the repository root; safe to start repeatedly and in parallel:

    python -m ablation.p04_shared_heads.run --root RUN_DIR --device cuda:0
"""

import sys
from pathlib import Path

from ablation.common.study import run_study

from .models import VARIANTS, build_model


if __name__ == '__main__':
    sys.exit(run_study(
        __doc__, VARIANTS, build_model,
        ['ablation/p01_placement/models.py', 'ablation/p04_shared_heads/models.py',
         'ablation/p04_shared_heads/run.py'],
        Path(__file__).parent / 'runs',
    ))
