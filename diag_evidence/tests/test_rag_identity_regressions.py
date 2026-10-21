"""Source-scoped evidence identities, private URLs and bounded RAG contracts."""
from copy import deepcopy
import json
import unittest

from diag_evidence.agent import DiagnosisAgent
from diag_evidence.demo import SyntheticTransport, synthetic_signal
from diag_evidence.hatools import _common, rag, diagnosis


def row(text="synthetic case", *, case_id=None, source_id="A", record_id="r1", label="滚动体故障", **meta):
    result = {"text": text, "score": 0.8,
              "meta": {"source_id": source_id, "record_id": record_id, "fault_type": label, **meta}}
    if case_id is not None:
        result["case_id"] = case_id
    return result


def retrieve(rows, **kwargs):
    return rag.retrieve_cases(synthetic_signal(), transport=lambda *a, **k:
                              ("ok", {"results": deepcopy(rows), "num_results": len(rows)}, ""), **kwargs)


def workflow(rows):
    class Transport(SyntheticTransport):
        def __call__(self, method, url, **kwargs):
            if "/v1/search/text" in url:
                return "ok", {"results": deepcopy(rows), "num_results": len(rows)}, ""
            return super().__call__(method, url, **kwargs)
    return DiagnosisAgent(transport=Transport("consistent"), persist_trace=False).run(
        synthetic_signal(), skip_services_check=True)


class RAGIdentityRegressionTest(unittest.TestCase):
    def test_scoring_metadata_does_not_increase_candidate_count_or_support(self):
        rows = [row(score_type=kind, score_is_normalized=bool(index % 2), as_of=str(index))
                for index, kind in enumerate(("cosine", "dot", "reranker"))]
        result = retrieve(rows)
        self.assertEqual(result["data"]["num_results"], 1)
        self.assertEqual(result["data"]["independent_record_count"], 1)
        ids = [retrieve([item])["data"]["cases"][0]["case_id"] for item in rows]
        self.assertEqual(len(set(ids)), 1)
        run = workflow(rows)
        self.assertEqual(run["support_level"], "medium")
        self.assertEqual(run["max_allowed_strength"], "moderate")

    def test_local_case_ids_are_namespaced_by_source(self):
        rows = [row("A1", case_id="1", record_id="A1"),
                row("B1", case_id="1", source_id="B", record_id="B1", label="内圈故障"),
                row("A2", case_id="2", record_id="A2"), row("A3", case_id="3", record_id="A3")]
        result = retrieve(rows)
        cases = result["data"]["cases"]
        self.assertEqual(len(cases), 4)
        self.assertNotEqual(cases[0]["case_id"], cases[1]["case_id"])
        self.assertEqual(cases[1]["source_case_id"], "1")
        self.assertEqual(cases[1]["source_id"], "B")
        self.assertEqual(workflow(rows)["support_level"], "medium")

    def test_same_identity_with_changed_content_or_label_is_an_error(self):
        first = row(case_id="1")
        for second in (row("different text", case_id="1"),
                       row(case_id="1", label="内圈故障"), row(case_id="1", record_id="other")):
            with self.subTest(second=second):
                result = retrieve([first, second])
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["data"])
                self.assertIn("身份", result["errors"][0])
                self.assertNotIn("different text", json.dumps(result))

    def test_different_ids_do_not_make_same_record_text_distinct(self):
        result = retrieve([row(case_id=str(index)) for index in range(3)])
        self.assertEqual(result["data"]["num_results"], 1)

    def test_same_text_different_record_or_window_stays_distinct(self):
        result = retrieve([row(record_id="r1"), row(record_id="r2"),
                           row(record_id="r1", window_start="100", window_end="200")])
        self.assertEqual(result["data"]["num_results"], 3)

    def test_same_text_without_identity_and_different_label_fails_closed(self):
        result = retrieve([row(), row(label="内圈故障")])
        self.assertEqual(result["status"], "error")
        self.assertIn("标签", result["errors"][0])

    def test_record_count_uses_source_namespace_without_claiming_independence(self):
        result = retrieve([row("A", source_id="A", record_id="1"),
                           row("B", source_id="B", record_id="1")])
        self.assertEqual(result["data"]["independent_record_count"], 2)

    def test_structured_and_embedded_urls_drop_userinfo_query_fragment(self):
        secret = "SYNTHETIC_PRIVATE_URL_SENTINEL"
        url = f"https://user:{secret}@example.invalid/doc?X-Amz-Signature={secret}#token={secret}"
        result = retrieve([row("source " + url, source_url=url)])
        case = result["data"]["cases"][0]
        self.assertEqual(case["meta"]["source_url"], "https://example.invalid/doc")
        self.assertNotIn(secret, json.dumps(result, ensure_ascii=False))
        self.assertIn("https://example.invalid/doc", case["text"])
        self.assertEqual(_common.safe_source_url("https://[::1]:9000/doc?q=x#y"), "https://[::1]:9000/doc")
        self.assertIsNone(_common.safe_source_url("file:///private/path"))
        self.assertIsNone(_common.safe_source_url("https://[invalid"))

    def test_query_token_changes_do_not_change_stable_content_identity(self):
        first = row("https://example.invalid/doc?sig=first")
        second = row("https://example.invalid/doc?sig=second")
        result = retrieve([first, second])
        self.assertEqual(result["data"]["num_results"], 1)

    def test_raw_row_and_text_budgets_fail_the_whole_response(self):
        for rows in ([row(str(index)) for index in range(rag.MAX_RAG_RESULTS + 1)],
                     [row("x" * (rag.MAX_CASE_TEXT_CHARS + 1))],
                     [row(case_id="x" * (rag.MAX_METADATA_CHARS + 1))],
                     [row(score_type="x" * (rag.MAX_METADATA_CHARS + 1))]):
            with self.subTest(size=len(rows)):
                result = retrieve(rows, top_k=1)
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["data"])
                self.assertIn("预算", result["errors"][0])

    def test_unicode_surrogates_fail_with_structured_errors(self):
        rows = [row("\ud800"), row(case_id="\ud800"), row(source_id="\ud800"),
                row(score_type="\ud800")]
        for item in rows:
            with self.subTest(field=list(item)):
                result = retrieve([item])
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["data"])
                json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.assertIsNone(_common.safe_source_url("https://example.invalid/\ud800"))

    def test_transport_local_policy_violation_is_an_error(self):
        transport = lambda *a, **k: ("invalid_response", None, "local limit")
        for tool in (rag.retrieve_cases, diagnosis.run_fault_diagnosis):
            with self.subTest(tool=tool):
                result = tool(synthetic_signal(), transport=transport)
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["data"])


if __name__ == "__main__":
    unittest.main()
