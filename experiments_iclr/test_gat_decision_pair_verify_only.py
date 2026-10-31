"""Synthetic file-integrity checks for the read-only GAT analysis gate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gat_decision_pair_analysis as analysis


CSV_TEXT = (
    "dataset,model,split,gnnm_minus_untied_pooled_accuracy_percent\r\n"
    "roman-empire,GAT,0,1.0\r\n"
)
AUDIT_TEXT = '{"input_sha256": {"data/roman_empire.npz": "abc"}}\n'


class VerifyOnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(
            prefix=".gat_verify_only_", dir=Path(__file__).resolve().parent
        )
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.csv = root / "gat_decision_pair_analysis.csv"
        self.audit = root / "gat_decision_pair_analysis_audit.json"
        self.csv.write_bytes(CSV_TEXT.encode("utf-8"))
        self.audit.write_bytes(AUDIT_TEXT.encode("utf-8"))

    def verify(self) -> None:
        with (
            patch.object(analysis, "OUTPUT", self.csv),
            patch.object(analysis, "AUDIT", self.audit),
            patch.object(
                analysis, "render_expected",
                return_value=(CSV_TEXT, AUDIT_TEXT, 5),
            ),
        ):
            analysis.verify_only()

    def test_exact_outputs_pass_without_a_write(self) -> None:
        before = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (self.csv, self.audit)
        }
        self.verify()
        after = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (self.csv, self.audit)
        }
        self.assertEqual(before, after)

    def test_corrupted_metric_and_extra_column_fail(self) -> None:
        for changed in (
            CSV_TEXT.replace(",1.0", ",2.0"),
            CSV_TEXT.replace("split,", "split,server_path,"),
        ):
            with self.subTest(changed=changed):
                self.csv.write_bytes(changed.encode("utf-8"))
                with self.assertRaisesRegex(
                    RuntimeError, "output differs from verified inputs"
                ):
                    self.verify()

    def test_corrupted_audit_and_private_path_fail(self) -> None:
        for changed in (
            AUDIT_TEXT.replace("abc", "def"),
            AUDIT_TEXT.replace(
                '"input_sha256"', '"server_path": "private/example", "input_sha256"'
            ),
        ):
            with self.subTest(changed=changed):
                self.audit.write_bytes(changed.encode("utf-8"))
                with self.assertRaisesRegex(
                    RuntimeError, "output differs from verified inputs"
                ):
                    self.verify()

    def test_missing_output_fails(self) -> None:
        self.audit.unlink()
        with self.assertRaisesRegex(
            FileNotFoundError, "Missing GAT analysis output"
        ):
            self.verify()

    def test_default_writer_keeps_generated_payloads(self) -> None:
        with (
            patch.object(analysis, "OUTPUT", self.csv),
            patch.object(analysis, "AUDIT", self.audit),
            patch.object(
                analysis, "render_expected",
                return_value=(CSV_TEXT, AUDIT_TEXT, 5),
            ),
        ):
            analysis.run()
        self.assertEqual(self.csv.read_bytes(), CSV_TEXT.encode("utf-8"))
        self.assertEqual(self.audit.read_bytes(), AUDIT_TEXT.encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
