"""CPU tests for the independent, launch-time all-layer SAGE audit."""
from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import verify_all_layer_be_sage_frozen as verifier


class FrozenVerifierTests(unittest.TestCase):
    def test_historical_gat_state_uses_frozen_snapshot(self) -> None:
        # A later GAT rerun may change live status. The frozen verifier must
        # neither query nor substitute that live status.
        with mock.patch.object(
            verifier.study, "require_queue_finished_and_gat_idle",
            side_effect=AssertionError("live GAT state was queried"),
        ):
            manifest, _, _ = verifier.verify_frozen_inputs()
        self.assertEqual(
            manifest["upstream"]["gat_status"],
            "incomplete_or_failed_after_launcher_exit",
        )

    def test_false_launch_time_completion_is_rejected(self) -> None:
        original = verifier.json_file

        def altered(path: Path) -> dict:
            value = original(path)
            if path == verifier.study.MANIFEST_PATH:
                value = copy.deepcopy(value)
                value["upstream"]["gat_status"] = "complete"
            return value

        with mock.patch.object(verifier, "json_file", side_effect=altered):
            with self.assertRaisesRegex(
                AssertionError, "Launch-time GAT observation differs"
            ):
                verifier.verify_frozen_inputs()

    def test_incomplete_selected_matrix_fails_before_replay(self) -> None:
        manifest = verifier.json_file(verifier.study.MANIFEST_PATH)
        with mock.patch.object(
            verifier.study, "RESULT_CSV",
            Path("/nonexistent/all_layer_projector_controls.csv"),
        ):
            with self.assertRaisesRegex(
                AssertionError, "Incomplete all-layer run"
            ):
                verifier.verify_selected_files(manifest)

    def test_checkpoint_logit_mismatch_fails(self) -> None:
        saved = np.zeros((4, 5666, 18), dtype=np.float32)
        fresh = saved.copy()
        self.assertEqual(
            verifier.compare_logits(0, fresh, saved, 1e-4, 1e-4)[
                "max_logit_abs_error"
            ], 0.0,
        )
        fresh[0, 0, 0] = 1.0
        with self.assertRaisesRegex(
            AssertionError, "Checkpoint logits differ"
        ):
            verifier.compare_logits(0, fresh, saved, 1e-4, 1e-4)


if __name__ == "__main__":
    unittest.main()
