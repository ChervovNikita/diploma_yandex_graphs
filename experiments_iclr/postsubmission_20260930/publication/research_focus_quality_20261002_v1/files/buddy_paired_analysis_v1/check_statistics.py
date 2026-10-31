"""Analytic checks of summary calculations, without scientific inputs."""
import importlib.util
import json
import math
from pathlib import Path

path = Path(__file__).with_name('analyze.py')
spec = importlib.util.spec_from_file_location('buddy_paired_check', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert abs(module.t_critical_df2(.95) - math.sqrt(722 / 39)) < 1e-12
assert module.paired([1, 2, 3])['sign_reference_p'] == .25
assert module.paired([0, 1, -1])['sign_reference_p'] == 1
assert module.paired([0, 0, 0])['paired_t95_pp'] == [0, 0]
assert module.holm([.04, .01, .03]) == [.06, .03, .06]
summary = module.paired([1, 2, 3])
assert summary['bonferroni_family95_pp'][0] < summary['paired_t95_pp'][0]
print(json.dumps(dict(analytic_statistical_checks=6, real_cohort_opened=False)))
