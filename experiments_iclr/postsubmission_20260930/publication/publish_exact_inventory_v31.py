"""Preserve v3/v4/v5 source byte authority during new preparation commits."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    spec=importlib.util.spec_from_file_location('_protected_v30',HERE/'publish_exact_inventory_v30.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    marker=" 'massive_practical_master_20261012_v4/PLAN.json':'e0a446486d4d47fbf5168d64f5823b828a8d4bdb152addddbf72094e8caa9647'}"
    assert module.PROTECTION.count(marker)==1
    module.PROTECTION=module.PROTECTION.replace(marker,marker[:-1]+",\n 'massive_practical_master_20261012_v5/PLAN.json':'1ab7dc9077523aa45d47ae5e67ddacc06502b7efd104b6f14a05d4bf81a89c4a'}")
    module.main()


if __name__=='__main__':main()
