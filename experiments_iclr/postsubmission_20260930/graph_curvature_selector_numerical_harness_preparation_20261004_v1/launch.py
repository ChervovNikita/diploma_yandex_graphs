"""Verify preserved sources; run only synthetic implementation checks.

No acquisition, native warm/continuation, real dataset, server or predictive run.
The launcher itself uses only stdlib. Numerical checks require torch and numpy.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verified_inputs():
    pins = json.loads((HERE / "SOURCE_PINS.json").read_text())
    packet = PHASE / pins["source_packet"]
    manifest_bytes = (packet / "MANIFEST.json").read_bytes()
    assert sha(manifest_bytes) == pins["source_manifest_sha256"]
    assert sha((packet / "SEAL.json").read_bytes()) == pins["source_seal_sha256"]
    manifest = json.loads(manifest_bytes)
    for record in manifest["files"]:
        data = (packet / record["path"]).read_bytes()
        assert len(data) == record["bytes"] and sha(data) == record["sha256"], record["path"]
    for record in pins["runtime_sources"] + [pins["constants"]]:
        relative = Path(record["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        data = (PHASE / relative).read_bytes()
        assert len(data) == record["bytes"] and sha(data) == record["sha256"], record["path"]
    frozen = json.loads((PHASE / pins["constants"]["path"]).read_text())
    assert frozen["source_manifest_sha256"] == pins["source_manifest_sha256"]
    assert frozen["source_protocol_sha256"] == sha((packet / "PROTOCOL.json").read_bytes())
    return pins, packet, frozen


def exact_module(name, path):
    """Fresh process entry modules also compile source bytes, without pyc caches."""
    assert name not in sys.modules, "Preloaded entry module: " + name
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def cache_checks(driver, fixtures):
    """All fixture files are descendants of this project-local directory."""
    old_here, old_loader = driver.HERE, driver.load_sources
    names = ("harness_unverified_stub", "harness_verified_stub", "harness_bad_stub")
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="cache_", dir=str(fixtures)) as directory:
            phase = Path(directory)
            driver.HERE = phase / "packet"
            for name in names:
                path = phase / (name + ".py")
                data = b"VALUE = 'verified bytes'\n"
                path.write_bytes(data)
                record = dict(path=path.name, bytes=len(data), sha256=sha(data))
                if name == names[0]:
                    stale = ModuleType(name)
                    stale.__file__ = str(path)
                    stale.VALUE = "stale code"
                    sys.modules[name] = stale
                    try:
                        driver.load_bound_source(record)
                    except ValueError as error:
                        assert "executed-source custody" in str(error)
                    else:
                        raise AssertionError("Unverified same-path cache accepted")
                    assert sys.modules[name] is stale and stale.VALUE == "stale code"
                    results.append("unverified_same_path_cache_rejected")
                elif name == names[1]:
                    fresh = driver.load_bound_source(record)
                    assert fresh.__graph_curvature_executed_sha256__ == record["sha256"]
                    assert fresh.__graph_curvature_executed_path__ == str(path.resolve())
                    assert driver.load_bound_source(record) is fresh
                    fresh.__graph_curvature_executed_sha256__ = "wrong"
                    try:
                        driver.load_bound_source(record)
                    except ValueError as error:
                        assert "executed-source custody" in str(error)
                    else:
                        raise AssertionError("Wrong executed digest accepted")
                    results.append("verified_cache_reused_and_wrong_digest_rejected")
                else:
                    bad = b"raise RuntimeError('synthetic import failure')\n"
                    path.write_bytes(bad)
                    bad_record = dict(path=path.name, bytes=len(bad), sha256=sha(bad))
                    try:
                        driver.load_bound_source(bad_record)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("Failing import unexpectedly succeeded")
                    assert name not in sys.modules
                    results.append("failed_source_execution_removed_from_cache")
            verified_native = ModuleType("verified_native_identity_stub")
            driver.load_sources = lambda: {"native_adapter": verified_native}
            try:
                driver.acquire_fresh_and_prepare(object(), "Squirrel", 17,
                                                None, None, None, None, None, None, None)
            except ValueError as error:
                assert "identical verified native adapter" in str(error)
            else:
                raise AssertionError("Wrong native API identity reached acquisition")
            results.append("wrong_native_api_rejected_before_acquisition")
    finally:
        driver.HERE, driver.load_sources = old_here, old_loader
        for name in names:
            sys.modules.pop(name, None)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdlib-only", action="store_true")
    parser.add_argument("--report", default="RUN_RESULT.json",
                        help="Filename inside the harness folder")
    args = parser.parse_args()
    report_name = Path(args.report)
    assert len(report_name.parts) == 1 and report_name.suffix == ".json"
    assert not (HERE / report_name).exists(), "Preserve existing results; use a fresh report filename"
    fixtures = HERE / "fixtures"
    assert not fixtures.is_symlink()
    fixtures.mkdir(exist_ok=True)
    report = dict(schema="curvature-synthetic-harness-result-v1",
                  source_manifest_sha256=None, synthetic_only=True,
                  real_warm_trials=0, predictive_training_runs=0,
                  python_executable=sys.executable, python_version=sys.version,
                  executed_launcher_sha256=sha(Path(__file__).read_bytes()),
                  numerical_status="NOT_RUN", stdlib_checks=[], numerical_checks=[])
    code = 1
    try:
        pins, packet, frozen = verified_inputs()
        report["source_manifest_sha256"] = pins["source_manifest_sha256"]
        report["constants_sha256"] = pins["constants"]["sha256"]
        selector = exact_module("selector", packet / "selector.py")
        driver = exact_module("driver", packet / "driver.py")
        constants = selector.FrozenConstants(**frozen["constants"])
        constants.validate()
        report["stdlib_checks"] = cache_checks(driver, fixtures)
        report["stdlib_checks"].append("source_and_frozen_constant_pins_verified")
        packages = {name: bool(importlib.util.find_spec(name)) for name in ("torch", "numpy")}
        report["package_availability"] = packages
        if args.stdlib_only:
            report["numerical_status"] = "UNEXECUTED_EXPLICIT_STDLIB_ONLY"
            code = 0
        elif not all(packages.values()):
            report["numerical_status"] = "UNEXECUTED_MISSING_DEPENDENCIES"
            report["missing_requirements"] = [name for name, present in packages.items() if not present]
            code = 2
        else:
            numerical = exact_module("curvature_synthetic_numerical_checks", HERE / "numerical_checks.py")
            bindings = json.loads((packet / "SOURCE_BINDINGS.json").read_text())
            modules = {role: driver.load_bound_source(bindings["runtime_modules"][role])
                       for role in ("integration", "initializer", "boundary")}
            context = SimpleNamespace(selector=selector, driver=driver, constants=constants,
                                      fixtures=fixtures, **modules)
            report["numerical_checks"], report["runtime"] = numerical.run(context)
            report["numerical_status"] = ("PASSED_SYNTHETIC_IMPLEMENTATION_CHECKS"
                if all(item["passed"] for item in report["numerical_checks"])
                else "FAILED_SYNTHETIC_IMPLEMENTATION_CHECKS")
            code = 0 if report["numerical_status"].startswith("PASSED") else 1
        verified_inputs()
        report["source_and_constants_unchanged_after"] = True
    except BaseException as error:
        import traceback
        report["error"] = dict(type=type(error).__name__, message=str(error),
                               traceback=traceback.format_exc())
        report["numerical_status"] = "LAUNCHER_OR_IMPORT_FAILURE"
    finally:
        report["finished_unix_seconds"] = time.time()
        (HERE / report_name).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if fixtures.exists() and not any(fixtures.iterdir()):
            fixtures.rmdir()
    print(json.dumps({"report": str(HERE / report_name),
                      "numerical_status": report["numerical_status"], "exit_code": code}))
    return code


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
