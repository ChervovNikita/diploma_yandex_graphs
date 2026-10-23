"""Keep the frozen queue and explicitly finalize two stopped reporting failures."""
import argparse
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
NAMED = HERE.parent / "campaign_named_processes_20261021_v1"
NAMED_SHA = "b7da088e468530267e0fd2afbaafa927c82fb6b39b5ee5008f0a1222168e9824"
sys.path.insert(0, str(NAMED))
import named_worker as named
from repair_native import guard, source

c = named.c


class RecoveryWorker(named.NamedWorker):
    def __init__(self, args, own_source):
        c.require(args.worker == "allocation/gpu0", "This recovery authority is allocation only")
        c.require(c.pin(args.recovery_config)["sha256"] == args.recovery_config_sha256, "Recovery config changed")
        self.recovery_authority_pin = c.pin(args.recovery_config)
        self.recovery_config = c.read(args.recovery_config)
        c.require(self.recovery_config["worker"] == args.worker, "Wrong recovery worker")
        c.require(self.recovery_config["plan"] == c.pin(args.plan), "Recovery plan changed")
        super().__init__(args, own_source)

    @property
    def scheduler_source(self):
        return self._scheduler_source

    @scheduler_source.setter
    def scheduler_source(self, value):
        named.NamedWorker.scheduler_source.fset(self, value)
        self._scheduler_source.update(
            schema="campaign-explicit-reporting-recovery-producer-v1",
            unchanged_named_process_source_manifest=c.pin(NAMED / "MANIFEST.json"),
            recovery_authority=self.recovery_authority_pin,
            changes=[*self._scheduler_source["changes"],
                     "Permit one exact pinned README dependency in the reporting witness",
                     "Explicitly finalize the two audited reporting failures from their stopped checkpoints"],
            native_scientific_descriptor_changed=False,
            automatic_retry=False,
        )

    def load_states(self, tasks, phase):
        super().load_states(tasks, phase)
        if phase != "full":
            return
        for fit_id, row in self.recovery_config["repairs"].items():
            state = self.states[fit_id]
            if state["status"] == "failed":
                c.require(state["attempt_result"] == row["failed_attempt_result"]
                          and state["native_result"] == row["failed_native_result"],
                          "An unreviewed failed attempt cannot be retried")
                c.verify(row["failed_attempt_result"], self.plan["project_root"])
                c.verify(row["failed_native_result"], self.plan["project_root"])
                self.retry.add(fit_id)

    def invocation(self, task, phase, directory):
        invocation = super().invocation(task, phase, directory)
        if phase != "full":
            return invocation
        argv = list(invocation["argv"])
        c.require(Path(argv[3]).resolve() == NAMED / "launch_named.py", "Unexpected named native launcher")
        argv[3] = str(HERE / "repair_native.py")
        insert = argv.index("--")
        extra = ["--repair-source-sha256", self.named_source["sha256"], "--record-dir", str(directory)]
        if task["id"] in self.retry:
            c.require("--resume" in invocation["native_argv"], "Explicit recovery must resume its saved checkpoint")
            extra += ["--recovery-config", self.args.recovery_config,
                      "--recovery-config-sha256", self.args.recovery_config_sha256]
        argv[insert:insert] = extra
        return {**invocation, "argv": argv, "reporting_repair_source": self.named_source,
                "reporting_only_finalization": task["id"] in self.retry,
                "native_argv_preserved": True}

    def validate_result(self, task, phase, value, output):
        status, selected = super().validate_result(task, phase, value, output)
        if phase == "full" and task["id"] in self.recovery_config["repairs"] and status == "complete":
            row = self.recovery_config["repairs"][task["id"]]
            c.require(value.get("finalization_only_resume") is True
                      and value.get("training_epochs_this_invocation") == 0
                      and value["completed_epochs"] == row["completed_epochs"],
                      "Reporting recovery added training epochs or changed the trajectory")
            c.require(value["optimizer_calls"] == row["optimizer_calls"]
                      and all(x == 0 for values in value["optimizer_calls_this_invocation"].values() for x in values),
                      "Reporting recovery added optimizer transitions")
            c.require(value["selected_epoch"] == row["selected_epoch"], "Reporting recovery changed checkpoint selection")
        return status, selected


def main():
    guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("allocation/gpu0",), required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--process-title-dependency-dir", required=True)
    parser.add_argument("--recovery-config", required=True)
    parser.add_argument("--recovery-config-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--smoke-wall-seconds", type=int, default=1800)
    args = parser.parse_args()
    args.retry_failed = False
    c.require(args.poll_seconds >= 1 and args.smoke_wall_seconds == 1800, "Frozen polling policy required")
    own_source = source(args.source_sha256)
    named.own_source(NAMED_SHA)
    named.set_process_title(args.process_title_dependency_dir)
    instance = None
    try:
        instance = RecoveryWorker(args, own_source)
        return instance.run()
    except BaseException as exc:
        if instance is not None:
            instance.phase = "supervisor_error"
            instance.report(error=type(exc).__name__ + ": " + str(exc), children_retain_inherited_leases=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
