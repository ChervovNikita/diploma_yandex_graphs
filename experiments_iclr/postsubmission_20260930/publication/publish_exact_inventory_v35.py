"""Preserve frozen source bytes without shadowing the publisher digest function."""
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
    module.main()


if __name__ == '__main__':
    main()
