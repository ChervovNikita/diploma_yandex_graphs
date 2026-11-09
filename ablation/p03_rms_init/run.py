"""Xavier-all at the scale of the default initialization. Worker for the full matrix.

Run from the repository root; safe to start repeatedly and in parallel:

    python -m ablation.p03_rms_init.run --root RUN_DIR --device cuda:0
"""

import sys
from pathlib import Path

from ablation.common.study import run_study

from .models import VARIANTS, build_model


if __name__ == '__main__':
    sys.exit(run_study(
        __doc__, VARIANTS, build_model,
        ['ablation/models_ablation.py', 'ablation/p02_full_ablations/models.py',
         'ablation/p03_rms_init/models.py', 'ablation/p03_rms_init/run.py'],
        Path(__file__).parent / 'runs',
    ))
