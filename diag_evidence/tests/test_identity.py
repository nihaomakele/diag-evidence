"""Public identity contracts after the DiagEvidence rename."""
from __future__ import annotations

from importlib.resources import files
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from diag_evidence import DiagnosisAgent, Settings, __version__, run_agent
from diag_evidence.demo import SyntheticTransport, synthetic_signal
from diag_evidence.hatools.report import generate_report


class IdentityTest(unittest.TestCase):
    def test_new_environment_prefix_is_used(self):
        config = Settings.from_env({
            "DIAG_EVIDENCE_FD_BASE_URL": "http://localhost:9998",
            "DIAG_EVIDENCE_SIGNAL_WINDOW_SIZE": "32",
            "DIAG_EVIDENCE_REPORTS_DIR": "new-reports",
            "HARNESS_FD_BASE_URL": "http://localhost:9999",
            "HARNESS_SIGNAL_WINDOW_SIZE": "invalid",
            "HARNESS_REPORTS_DIR": "old-reports",
        })
        self.assertEqual(config.fd_base_url, "http://localhost:9998")
        self.assertEqual(config.signal_window_size, 32)
        self.assertEqual(config.reports_dir, "new-reports")

    def test_obsolete_environment_does_not_change_defaults(self):
        config = Settings.from_env({
            "HARNESS_FD_BASE_URL": "not-a-url",
            "HARNESS_MAX_RETRIES": "invalid",
            "HARNESS_REPORTS_DIR": "obsolete-output",
        })
        defaults = Settings.from_env({})
        self.assertEqual(config, defaults)

    def test_library_default_artifacts_use_new_identity(self):
        config = Settings.from_env({})
        self.assertEqual(Path(config.reports_dir), Path.cwd() / "diag-evidence-output" / "reports")
        self.assertEqual(Path(config.logs_dir), Path.cwd() / "diag-evidence-output" / "logs")

    def test_module_help_exposes_new_cli_name(self):
        process = subprocess.run(
            [sys.executable, "-B", "-m", "diag_evidence", "--help"],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("usage: diag-evidence ", process.stdout)
        self.assertNotIn("harness-agent", process.stdout)
        self.assertIn("demo", process.stdout)

    def test_public_api_and_packaged_sop_are_importable(self):
        self.assertTrue(callable(DiagnosisAgent))
        self.assertTrue(callable(run_agent))
        self.assertTrue(__version__)
        resource = files("diag_evidence").joinpath("skills/fault_diagnosis.md")
        self.assertTrue(resource.is_file())
        self.assertTrue(resource.read_text(encoding="utf-8").strip())

    def test_generated_report_uses_new_brand(self):
        result = DiagnosisAgent(
            transport=SyntheticTransport("consistent"), persist_trace=False,
        ).run(synthetic_signal(), skip_services_check=True)
        title = result["report_markdown"].splitlines()[0]
        self.assertEqual(title, "# 设备故障诊断报告（DiagEvidence）")

    def test_report_artifact_filename_uses_new_brand(self):
        with tempfile.TemporaryDirectory() as directory:
            result = generate_report(output_dir=directory, run_id="identity-check")
            path = Path(result["data"]["path"])
            self.assertTrue(path.name.startswith("diag_evidence_report_identity-check_"))
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
