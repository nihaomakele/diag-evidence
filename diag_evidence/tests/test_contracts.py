"""Regression tests for input, model, retrieval, configuration and HTTP contracts."""
from __future__ import annotations

import math
import json
import unittest
from unittest.mock import patch

import numpy as np
import requests

from diag_evidence.contracts import SignalValidationError, validate_signal
from diag_evidence.haconfig.settings import Settings
from diag_evidence.hatools import _common, diagnosis, health, rag, signal
from diag_evidence.tests import _fakes

SIGNAL = [math.sin(index / 7) for index in range(1024)]


def prediction(**changes):
    result = {"predicted_class": 2, "confidence": 0.8,
              "predicted_fault": {"name": "滚动体故障", "name_en": "Ball Fault"}}
    result.update(changes)
    return {"success": True, "diagnosis": result}


class ContractTest(unittest.TestCase):
    def test_signal_bad_values_never_reach_transport(self):
        def forbidden(*args, **kwargs):
            self.fail("invalid input reached the network")
        invalids = [None, "1" * 1024, [[1.0]] * 1024,
                    [False] * 1024, ["1"] * 1024, [0.0] * 1024, [0.1] * 1024,
                    [float("nan")] + SIGNAL[1:], [float("inf")] + SIGNAL[1:],
                    [10 ** 10000] + SIGNAL[1:], [1e200] + SIGNAL[1:],
                    [1e-9 * v for v in SIGNAL], SIGNAL[:-1]]
        for value in invalids:
            for tool in (rag.retrieve_cases, diagnosis.run_fault_diagnosis):
                with self.subTest(tool=tool.__name__, input_type=type(value).__name__):
                    self.assertEqual(tool(value, transport=forbidden)["status"], "invalid_input")
            self.assertEqual(signal.analyze_signal(value)["status"], "invalid_input")

    def test_signal_numeric_arrays_and_features_are_finite(self):
        self.assertEqual(len(validate_signal(np.asarray(SIGNAL))), 1024)
        result = signal.analyze_signal(np.asarray(SIGNAL))
        self.assertEqual(result["status"], "ok")
        for section in ("time_domain", "frequency_domain"):
            self.assertTrue(all(math.isfinite(value) for value in result["data"][section].values()))

    def test_sampling_rate_guard(self):
        for value in (True, 0, -1, float("nan"), float("inf"), "50000", 1e20):
            with self.subTest(value=value):
                self.assertEqual(signal.analyze_signal(SIGNAL, sampling_rate=value)["status"], "invalid_input")

    def test_classifier_invalid_classes(self):
        for value in (None, True, [], {}, 0.0, "2", 3, -2):
            with self.subTest(value=value):
                result = diagnosis.run_fault_diagnosis(SIGNAL, transport=_fakes.ok(prediction(predicted_class=value)))
                self.assertEqual(result["status"], "error")
                self.assertFalse(result["data"]["diagnosis"]["model_available"])

    def test_classifier_missing_class_or_confidence(self):
        for missing in ("predicted_class", "confidence"):
            body = prediction()
            del body["diagnosis"][missing]
            result = diagnosis.run_fault_diagnosis(SIGNAL, transport=_fakes.ok(body))
            self.assertEqual(result["status"], "error")

    def test_classifier_confidence_is_finite_and_bounded(self):
        for value in (None, True, "0.8", float("nan"), float("inf"), -0.1, 1.1, {}, []):
            with self.subTest(value=value):
                result = diagnosis.run_fault_diagnosis(SIGNAL, transport=_fakes.ok(prediction(confidence=value)))
                self.assertEqual(result["status"], "error")
                self.assertFalse(result["data"]["diagnosis"]["model_available"])

    def test_classifier_optional_probability_vector_contract(self):
        invalids = [{0: 1.0}, {0: 0.1, 1: 0.1, 2: float("nan")},
                    {0: 0.1, 1: 0.1, 2: 0.7}, {0: 0.1, 1: 0.1, 2: 1.8},
                    {0: True, 1: 0.2, 2: 0.8},
                    {0: 0.2, 1: 0.1, 2: 0.7}, [0.1, 0.1, 0.8], ""]
        for vector in invalids:
            with self.subTest(vector=vector):
                body = prediction(class_probabilities=vector)
                self.assertEqual(diagnosis.run_fault_diagnosis(SIGNAL, transport=_fakes.ok(body))["status"], "error")
        for vector in ({}, {"0": 0.1, "1": 0.1, "2": 0.8},
                       {str(i): {"probability": value} for i, value in enumerate((0.1, 0.1, 0.8))}):
            self.assertEqual(diagnosis.run_fault_diagnosis(
                SIGNAL, transport=_fakes.ok(prediction(class_probabilities=vector)))["status"], "ok")

    def test_classifier_label_must_match_class(self):
        body = prediction(predicted_fault={"name": "内圈故障"})
        self.assertEqual(diagnosis.run_fault_diagnosis(SIGNAL, transport=_fakes.ok(body))["status"], "error")

    def test_missing_model_sentinel_is_degraded(self):
        body = {"success": True, "diagnosis": {"predicted_class": -1, "note": "模型未加载"}}
        result = diagnosis.run_fault_diagnosis(SIGNAL, transport=_fakes.ok(body))
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["data"]["diagnosis"]["model_available"])

    def test_retrieval_counts_only_unique_valid_cases(self):
        case = {"text": "example", "score": -0.2, "meta": {"fault_type": "内圈故障"}}
        result = rag.retrieve_cases(SIGNAL, transport=_fakes.ok(
            {"results": [case, dict(case, score=0.9), None, {"text": ""}], "num_results": 999}))
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["data"]["num_results"], 1)
        self.assertEqual(result["data"]["independent_record_count"], 0)

    def test_retrieval_explicit_record_identity(self):
        cases = [{"id": str(i), "record_id": "same-record", "text": "case fragment " + str(i), "score": 0.4}
                 for i in range(3)]
        result = rag.retrieve_cases(SIGNAL, transport=_fakes.ok({"results": cases, "num_results": 3}))
        self.assertEqual(result["data"]["num_results"], 3)
        self.assertEqual(result["data"]["independent_record_count"], 1)

    def test_retrieval_score_semantics_are_not_invented(self):
        for requested in (True, False):
            result = rag.retrieve_cases(SIGNAL, normalize_scores=requested, transport=_fakes.ok({
                "results": [{"text": "case", "score": -0.1}], "num_results": 1}))
            case = result["data"]["cases"][0]
            self.assertIsNone(case["score_is_normalized"])
            self.assertEqual(case["score_type"], "unknown")
            self.assertEqual(case["score"], -0.1)
        result = rag.retrieve_cases(SIGNAL, transport=_fakes.ok({
            "score_type": "cosine", "score_is_normalized": False,
            "results": [{"text": "case", "score": 0.1}], "num_results": 1}))
        self.assertFalse(result["data"]["cases"][0]["score_is_normalized"])
        self.assertEqual(result["data"]["cases"][0]["score_type"], "cosine")

    def test_retrieval_invalid_scores_are_unavailable(self):
        for value in (None, True, "0.9", float("nan"), float("inf")):
            result = rag.retrieve_cases(SIGNAL, transport=_fakes.ok({
                "results": [{"text": "case", "score": value}], "num_results": 1}))
            self.assertIsNone(result["data"]["cases"][0]["score"])
            self.assertFalse(result["data"]["cases"][0]["score_valid"])

    def test_retrieval_malformed_results(self):
        for value in (None, {}, "cases"):
            result = rag.retrieve_cases(SIGNAL, transport=_fakes.ok({"results": value}))
            self.assertEqual(result["status"], "error")
        self.assertEqual(rag.retrieve_cases(SIGNAL, top_k=True)["status"], "invalid_input")

    def test_instance_configuration_isolated(self):
        config = Settings.from_env({"DIAG_EVIDENCE_FD_BASE_URL": "http://localhost:9991",
                                    "DIAG_EVIDENCE_SIGNAL_WINDOW_SIZE": "32", "DIAG_EVIDENCE_SAMPLING_RATE": "1000"})
        captured = []
        def transport(method, url, **kwargs):
            captured.append((url, kwargs))
            return "ok", prediction(), ""
        short_signal = SIGNAL[:32]
        self.assertEqual(diagnosis.run_fault_diagnosis(short_signal, transport=transport, config=config)["status"], "ok")
        self.assertEqual(captured[0][0], "http://localhost:9991/diagnose")
        self.assertEqual(signal.analyze_signal(short_signal, config=config)["data"]["quality"]["sampling_rate"], 1000)
        self.assertEqual(Settings().signal_window_size, 1024)

    def test_environment_is_read_after_import(self):
        with patch.dict("os.environ", {"DIAG_EVIDENCE_FD_BASE_URL": "http://localhost:9992"}):
            config = Settings.from_env()
        self.assertEqual(config.fd_base_url, "http://localhost:9992")

    def test_invalid_settings_rejected(self):
        for kwargs in ({"signal_window_size": True}, {"sampling_rate": float("nan")},
                       {"max_retries": -1}, {"retry_backoff": float("inf")},
                       {"fd_base_url": "http://user:secret@localhost"},
                       {"fd_base_url": "http://localhost?api_key=secret"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Settings(**kwargs)

    def test_health_does_not_trust_truthy_strings_or_bad_counts(self):
        result = health.check_system_status(services=["rag"], check_logs=False, transport=_fakes.ok(
            {"model_loaded": "true", "index_loaded": True, "num_texts": "many"}))
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["data"]["services"]["rag"]["capable"])
        self.assertEqual(health.check_system_status(services=[])["status"], "invalid_input")

    def test_http_client_errors_are_not_retried(self):
        response = unittest.mock.Mock(status_code=400)
        response.json.return_value = {"error": "api_key=EXAMPLE_SECRET_ONLY"}
        with patch.object(_common.requests, "request", return_value=response) as call:
            kind, body, error = _common.http_call("POST", "http://localhost", max_retries=3, backoff=0)
        self.assertEqual(kind, "client")
        self.assertEqual(call.call_count, 1)
        self.assertNotIn("EXAMPLE_SECRET_ONLY", error)

    def test_http_transient_errors_are_bounded(self):
        response = unittest.mock.Mock(status_code=503)
        with patch.object(_common.requests, "request", return_value=response) as call:
            result = _common.http_call("GET", "http://localhost", max_retries=2, backoff=0)
        self.assertEqual(result[0], "server")
        self.assertEqual(call.call_count, 3)
        with patch.object(_common.requests, "request", side_effect=requests.exceptions.Timeout) as call:
            result = _common.http_call("GET", "http://localhost", max_retries=1, backoff=0)
        self.assertEqual(result[0], "conn")
        self.assertEqual(call.call_count, 2)

    def test_http_retry_parameters_validate_before_network(self):
        with patch.object(_common.requests, "request") as call:
            for kwargs in ({"max_retries": -1}, {"max_retries": True}, {"max_retries": 100},
                           {"timeout": float("nan")}, {"timeout": 0}, {"backoff": -1},
                           {"backoff": float("inf")}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    _common.http_call("GET", "http://localhost", **kwargs)
            call.assert_not_called()

    def test_common_error_redaction(self):
        secret = "EXAMPLE_SECRET_ONLY"
        for text in ("Bearer " + secret, "api_key=" + secret, 'password:"' + secret + '"'):
            self.assertNotIn(secret, _common.redact_text(text))
        self.assertNotIn("with spaces", _common.redact_text('password="private with spaces"'))

    def test_remote_response_fields_are_whitelisted_and_json_safe(self):
        secret = "EXAMPLE_PRIVATE_VALUE_ONLY"
        body = prediction(class_probabilities={str(i): {"probability": p, "api_key": secret}
                                               for i, p in enumerate((0.1, 0.1, 0.8))})
        body["statistics"] = {"api_key": secret, "mean": float("nan"), "std": 1.1}
        body["api_key"] = secret
        output = diagnosis.run_fault_diagnosis(SIGNAL, transport=_fakes.ok(body))
        encoded = json.dumps(output, allow_nan=False)
        self.assertNotIn(secret, encoded)
        self.assertNotIn("mean", output["data"]["statistics"])
        self.assertEqual(output["data"]["statistics"]["std"], 1.1)
        for tool in (diagnosis.run_fault_diagnosis, rag.retrieve_cases):
            output = tool(SIGNAL, transport=_fakes.client_error({"error": secret}, secret))
            self.assertNotIn(secret, json.dumps(output, allow_nan=False))
        output = rag.retrieve_cases(SIGNAL, transport=_fakes.ok({
            "results": [{"text": "api_key=" + secret, "score": 0.2,
                         "rank": {"api_key": secret}, "meta": {"api_key": secret}}],
            "num_results": {"api_key": secret}}))
        self.assertNotIn(secret, json.dumps(output, allow_nan=False))
        output = health.check_system_status(services=["fd"], check_logs=False,
                                           transport=_fakes.ok({"api_key": secret}))
        self.assertNotIn(secret, json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    unittest.main()
