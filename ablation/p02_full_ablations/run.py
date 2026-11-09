"""Full ensemble-size and initialization ablations. Worker for the full matrix.

Run from the repository root; safe to start repeatedly and in parallel:

    python -m ablation.p02_full_ablations.run --root RUN_DIR --device cuda:0
"""

import sys
from pathlib import Path

from ablation.common.study import run_study

from .models import DEFAULT_VARIANTS, VARIANTS, build_model


if __name__ == '__main__':
    sys.exit(run_study(
        __doc__, VARIANTS, build_model,
        ['ablation/models_ablation.py', 'ablation/p02_full_ablations/models.py',
         'ablation/p02_full_ablations/run.py'],
        Path(__file__).parent / 'runs',
        default_variants=DEFAULT_VARIANTS,
    ))
