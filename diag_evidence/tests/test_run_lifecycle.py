"""Regression tests for isolated runs, persistence failure and defensive logging."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from diag_evidence.agent import DiagnosisAgent
from diag_evidence.audit_log import AuditLogger, sanitize
from diag_evidence.demo import SCENARIOS, SyntheticTransport, evaluate, synthetic_signal
from diag_evidence.haconfig.settings import Settings


class LifecycleTest(unittest.TestCase):
    def agent(self, root, persist=True):
        config = replace(Settings.from_env(), reports_dir=str(Path(root) / "reports"),
                         logs_dir=str(Path(root) / "logs"))
        return DiagnosisAgent(transport=SyntheticTransport("consistent"), config=config,
                              persist_trace=persist)

    def test_disabled_persistence_creates_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.agent(tmp, False).run(synthetic_signal(), skip_services_check=True)
            self.assertEqual(result["terminal_state"], "COMPLETED")
            self.assertIsNone(result["report_path"])
            self.assertIsNone(result["trace_path"])
            self.assertIsNone(result["audit_path"])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_reusing_agent_keeps_old_results_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self.agent(tmp)
            first = agent.run(synthetic_signal(), skip_services_check=True)
            snapshot = deepcopy(first)
            second = agent.run(synthetic_signal(), skip_services_check=True)
            self.assertEqual(first, snapshot)
            self.assertNotEqual(first["run_id"], second["run_id"])
            for key in ("report_path", "audit_path", "trace_path"):
                self.assertNotEqual(first[key], second[key])
                self.assertTrue(Path(first[key]).exists())
                self.assertTrue(Path(second[key]).exists())
            self.assertEqual(first["input_digest"], second["input_digest"])

    def test_concurrent_separate_instances_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            def run(_):
                return self.agent(tmp).run(synthetic_signal(), skip_services_check=True)
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(run, range(12)))
            self.assertEqual(len({r["run_id"] for r in results}), 12)
            for key in ("report_path", "audit_path", "trace_path"):
                self.assertEqual(len({r[key] for r in results}), 12)
                self.assertTrue(all(Path(r[key]).is_file() for r in results))
            for r in results:
                for path in (r["audit_path"], r["trace_path"]):
                    for line in Path(path).read_text().splitlines():
                        self.assertEqual(json.loads(line)["run_id"], r["run_id"])

    def test_same_instance_concurrent_call_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self.agent(tmp, False)
            agent._lock.acquire()
            try:
                with self.assertRaises(RuntimeError):
                    agent.run(synthetic_signal())
            finally:
                agent._lock.release()

    def test_generator_input_is_consumed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.agent(tmp, False).run(iter(synthetic_signal()), skip_services_check=True)
            self.assertEqual(result["terminal_state"], "COMPLETED")

    def test_terminal_trace_write_failure_stays_consistent(self):
        from diag_evidence.agent import TraceRecorder
        original = TraceRecorder.record
        def record(recorder, state, *args, **kwargs):
            if state == "COMPLETED":
                with patch.object(Path, "open", side_effect=PermissionError):
                    return original(recorder, state, *args, **kwargs)
            return original(recorder, state, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(TraceRecorder, "record", record):
                result = self.agent(tmp).run(synthetic_signal(), skip_services_check=True)
            self.assertEqual(result["terminal_state"], "DEGRADED")
            self.assertEqual(result["trace"][-1]["state"], "DEGRADED")
            self.assertTrue(result["persistence_errors"])

    def test_write_failure_is_exposed_and_never_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocked = Path(tmp) / "file"
            blocked.write_text("occupied")
            config = replace(Settings.from_env(), reports_dir=str(blocked), logs_dir=str(blocked))
            result = DiagnosisAgent(config=config, transport=SyntheticTransport("consistent")).run(
                synthetic_signal(), skip_services_check=True)
            self.assertNotEqual(result["terminal_state"], "COMPLETED")
            self.assertFalse(result["ok"])
            self.assertTrue(result["persistence_errors"])
            self.assertIsNone(result["report_path"])

    def test_nonfinite_and_constant_inputs_make_zero_calls(self):
        bad = [[float("nan")] * 1024, [float("inf")] * 1024, [0.1] * 1024,
               [True] * 1024, [[1]] * 1024, ["1"] * 1024, [1e300, -1e300] * 512]
        for values in bad:
            transport = SyntheticTransport("consistent")
            result = DiagnosisAgent(transport=transport, persist_trace=False).run(values)
            self.assertEqual(result["terminal_state"], "ASKING_USER")
            self.assertEqual(transport.calls, [])


class AuditTest(unittest.TestCase):
    def test_recursive_blocking_and_string_redaction(self):
        fake = "sk-TEST_ONLY_NOT_A_REAL_KEY_12345"
        value = {"nested": [{"api_key": "FAKE-SECRET", "message": fake,
                              "signal": [1, 2, 3], "deeper": [{"password": "FAKE-PASSWORD"}]}]}
        text = json.dumps(sanitize(value))
        self.assertNotIn(fake, text)
        self.assertNotIn("FAKE-SECRET", text)
        self.assertNotIn("FAKE-PASSWORD", text)
        self.assertNotIn("[1, 2, 3]", text)
        self.assertIn("signal_omitted", text)

    def test_external_payload_not_in_audit_allowlist(self):
        logger = AuditLogger(persist=False)
        logger.log("fixture", {"data": {"raw": {"text": "PRIVATE-TEXT"},
                                        "cases": [{"text": "PRIVATE-TEXT"}],
                                        "markdown": "PRIVATE-TEXT"}})
        self.assertNotIn("PRIVATE-TEXT", json.dumps(logger.entries))
        json.dumps(sanitize({"value": float("nan")}), allow_nan=False)


class DemoCliTest(unittest.TestCase):
    def test_all_synthetic_expectations(self):
        result = evaluate(write=False)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["total"], len(SCENARIOS))

    def test_cli_no_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "absent"
            process = subprocess.run([sys.executable, "-m", "diag_evidence", "demo", "--scenario", "all",
                                      "--no-write", "--output-dir", str(target)],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertTrue(json.loads(process.stdout)["passed"])
            self.assertFalse(target.exists())

    def test_cli_rejects_nonfinite_json_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.json"
            source.write_text('{"signal": [NaN]}')
            process = subprocess.run([sys.executable, "-m", "diag_evidence", "run", str(source), "--no-write"],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 3)
            self.assertEqual(json.loads(process.stderr)["error"], "ValueError")

    def test_cli_invalid_environment_has_structured_error(self):
        env = dict(os.environ, DIAG_EVIDENCE_MAX_RETRIES="invalid")
        process = subprocess.run([sys.executable, "-m", "diag_evidence", "demo", "--no-write"],
                                 env=env, capture_output=True, text=True)
        self.assertEqual(process.returncode, 3)
        self.assertEqual(json.loads(process.stderr)["error"], "ValueError")
        self.assertNotIn("Traceback", process.stderr)
