"""Protect all completed finite masters and their controller sources before publication."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location('_protected_v30_no_shadow', HERE / 'publish_exact_inventory_v30.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protection = module.PROTECTION
    assert protection.count('for relative,digest in expected_plans.items():') == 1
    assert protection.count('canonical_digest(plan)==digest') == 1
    protection = protection.replace('for relative,digest in expected_plans.items():',
                                    'for relative,plan_digest in expected_plans.items():')
    protection = protection.replace('canonical_digest(plan)==digest',
                                    'canonical_digest(plan)==plan_digest')
    marker = " 'massive_practical_master_20261012_v4/PLAN.json':'e0a446486d4d47fbf5168d64f5823b828a8d4bdb152addddbf72094e8caa9647'}"
    assert protection.count(marker) == 1
    module.PROTECTION = protection.replace(marker, marker[:-1] + ",\n 'massive_practical_master_20261012_v5/PLAN.json':'1ab7dc9077523aa45d47ae5e67ddacc06502b7efd104b6f14a05d4bf81a89c4a'}")
    marker7 = " 'massive_practical_master_20261012_v5/PLAN.json':'1ab7dc9077523aa45d47ae5e67ddacc06502b7efd104b6f14a05d4bf81a89c4a'}"
    assert module.PROTECTION.count(marker7) == 1
    module.PROTECTION = module.PROTECTION.replace(marker7, marker7[:-1] + ",\n 'massive_practical_master_20261012_v7/PLAN.json':'c58fd5b0b590d581a2cbc5e12f7da3090c57ee781f2e1434ab835874c6874c08'}")
    marker8 = " 'massive_practical_master_20261012_v7/PLAN.json':'c58fd5b0b590d581a2cbc5e12f7da3090c57ee781f2e1434ab835874c6874c08'}"
    assert module.PROTECTION.count(marker8) == 1
    module.PROTECTION = module.PROTECTION.replace(marker8, marker8[:-1] + ",\n 'massive_practical_master_20261012_v8/PLAN.json':'f38437b3b2e2faa9e5a7d6400a3c5c8e4091646b7b71ea3f0be34e25163b28df'}")
    marker_extra_0 = " 'massive_practical_master_20261012_v8/PLAN.json':'f38437b3b2e2faa9e5a7d6400a3c5c8e4091646b7b71ea3f0be34e25163b28df'}"
    assert module.PROTECTION.count(marker_extra_0) == 1
    module.PROTECTION = module.PROTECTION.replace(marker_extra_0, marker_extra_0[:-1] + ",\n 'massive_practical_master_20261012_v9/PLAN.json':'f800e085e2300b2f6da5e9edd098b094f55444420e88c67faa3e4a19b16f8d22'}")
    marker_extra_1 = " 'massive_practical_master_20261012_v9/PLAN.json':'f800e085e2300b2f6da5e9edd098b094f55444420e88c67faa3e4a19b16f8d22'}"
    assert module.PROTECTION.count(marker_extra_1) == 1
    module.PROTECTION = module.PROTECTION.replace(marker_extra_1, marker_extra_1[:-1] + ",\n 'massive_practical_master_20261012_v10/PLAN.json':'c37a5832ed757db5636d425f6bcf7cf3f18266f91cfa98772e3eaa63ca3242ef'}")
    marker_extra_2 = " 'massive_practical_master_20261012_v10/PLAN.json':'c37a5832ed757db5636d425f6bcf7cf3f18266f91cfa98772e3eaa63ca3242ef'}"
    assert module.PROTECTION.count(marker_extra_2) == 1
    module.PROTECTION = module.PROTECTION.replace(marker_extra_2, marker_extra_2[:-1] + ",\n 'massive_practical_master_20261012_v13/PLAN.json':'a4e525b29ee05a5965b221073d28543151c75087a0c6252ef57f6e5e18b01797'}")
    extra = {
        'massive_practical_master_20261012_v14/PLAN.json':'364819e2b7f07e8bdafa052b55de3dbb504778c99d8885401b4b83b3781f57ec',
        'massive_practical_master_20261012_v15/PLAN.json':'c90eb56ff593e7e18d575f9b070f2c3d9f4d35f7ae341472abedf4f97cb9f421',
        'massive_practical_master_20261012_v17/PLAN.json':'88a995423d627baa159e24d794311db355e55d7c471eccb9b112c68f08e09aa1',
        'massive_practical_master_20261012_v18/PLAN.json':'7966f239f12c3acf7c116595f67a09518b4e56a3793acaf4483c7e6dfd7b117d'}
    marker = 'def canonical_digest(value):'
    assert module.PROTECTION.count(marker) == 1
    module.PROTECTION = module.PROTECTION.replace(marker, 'expected_plans.update('+repr(extra)+')\n'+marker)
    marker = ' for entry in plan[\'studies\']:'
    assert module.PROTECTION.count(marker) == 1
    module.PROTECTION = module.PROTECTION.replace(marker, " for bound in plan['immutable_source_bundle']['files']:\n  previous=protected.get(bound['path']);assert previous is None or previous==bound\n  protected[bound['path']]=bound\n"+marker)
    # This source inventory is larger; preserve every commit validation while
    # allowing bounded transport/index work to complete under its own timeout.
    import types
    scope = {'__name__':'_protected_explicit_v36', '__file__':str(Path(__file__))}
    source = (HERE/'publish_exact_inventory_v29.py').read_text()
    marker = '        return original([*argv[:-1],shlex.join(fields)],**kwargs)'
    assert source.count(marker) == 1
    source = source.replace(marker, "        kwargs['timeout'] = max(kwargs['timeout'], 600)\n"+marker)
    exec(compile(source,str(HERE/'publish_exact_inventory_v29.py')+':bounded_large_inventory','exec'),scope)
    scope['FROZEN_PROTECTION'] = module.PROTECTION
    scope['main']()


if __name__ == '__main__':
    main()
