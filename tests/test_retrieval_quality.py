from __future__ import annotations

import unittest
import sys
import types

import numpy as np

if "rank_bm25" not in sys.modules:
    rank_bm25_stub = types.ModuleType("rank_bm25")

    class BM25Okapi:  # noqa: D101
        def __init__(self, tokenized_corpus):  # noqa: ANN001
            self.tokenized_corpus = tokenized_corpus

        def get_scores(self, query_tokens):  # noqa: ANN001
            query_set = set(query_tokens)
            return [
                sum(1 for token in tokens if token in query_set)
                for tokens in self.tokenized_corpus
            ]

    rank_bm25_stub.BM25Okapi = BM25Okapi
    sys.modules["rank_bm25"] = rank_bm25_stub

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    def _missing_post(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("requests is not installed in this test environment")

    requests_stub.post = _missing_post
    sys.modules["requests"] = requests_stub

from rag_runes.index.hybrid_index import HybridTreeIndex
from rag_runes.schema import DocNode


class ConstantEmbedder:
    def encode(self, texts):  # noqa: ANN001
        return np.ones((len(texts), 3), dtype=np.float32)


class RetrievalQualityTests(unittest.TestCase):
    def test_general_definition_query_prefers_prose_over_forms_and_code_charts(self) -> None:
        book = DocNode("book", "book", None, "book", "Khazarian Rovas proposal", "", 1, 19)
        summary = DocNode(
            "summary",
            "book",
            "book",
            "chunk",
            "Summary / chunk 1",
            "The Khazarian Rovas is an extinct script used in the area of the Khazar Empire.",
            1,
            1,
        )
        code_chart = DocNode(
            "code-chart",
            "book",
            "book",
            "chunk",
            "Code chart of the KHAZARIAN ROVAS in the SMP / chunk 2",
            "LETTER ARCHED Q 1xx16; R KHAZARIAN ROVAS LETTER ARCHED R 1xx17; "
            "S KHAZARIAN ROVAS LETTER SH 1xx18; z KHAZARIAN ROVAS LETTER Z",
            9,
            9,
        )
        appendix = DocNode(
            "appendix",
            "book",
            "book",
            "chunk",
            "Appendix: Proposal Summary Form / A. Administrative / chunk 1",
            "1. Title: Proposal for encoding the Khazarian Rovas script in the SMP of the UCS. "
            "2. Requester's name:",
            17,
            17,
        )
        index = HybridTreeIndex.build(
            [book, summary, code_chart, appendix],
            embedder=ConstantEmbedder(),
            embedder_name="constant",
        )

        hits = index.hybrid_rank(
            "What is Khazarian Rovas?",
            candidate_ids=["summary", "code-chart", "appendix"],
            top_k=3,
        )

        self.assertEqual(hits[0].node_id, "summary")
        self.assertLess(
            hits[[hit.node_id for hit in hits].index("appendix")].score,
            hits[[hit.node_id for hit in hits].index("summary")].score,
        )

    def test_unicode_query_keeps_code_chart_available(self) -> None:
        book = DocNode("book", "book", None, "book", "Khazarian Rovas proposal", "", 1, 19)
        summary = DocNode(
            "summary",
            "book",
            "book",
            "chunk",
            "Summary / chunk 1",
            "The Khazarian Rovas is an extinct script used in the area of the Khazar Empire.",
            1,
            1,
        )
        code_chart = DocNode(
            "code-chart",
            "book",
            "book",
            "chunk",
            "Code chart of the KHAZARIAN ROVAS in the SMP / chunk 2",
            "LETTER ARCHED Q 1xx16; R KHAZARIAN ROVAS LETTER ARCHED R 1xx17; "
            "S KHAZARIAN ROVAS LETTER SH 1xx18; z KHAZARIAN ROVAS LETTER Z",
            9,
            9,
        )
        index = HybridTreeIndex.build(
            [book, summary, code_chart],
            embedder=ConstantEmbedder(),
            embedder_name="constant",
        )

        hits = index.hybrid_rank(
            "Unicode code chart Khazarian Rovas letters",
            candidate_ids=["summary", "code-chart"],
            top_k=2,
        )

        self.assertEqual(hits[0].node_id, "code-chart")


if __name__ == "__main__":
    unittest.main()
