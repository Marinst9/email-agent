"""Hybrid retrieval without a database: chunking, query terms, RRF, the SQL shape, citations and reranking."""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import SecretStr

from app.agents import StructuredOutputError
from app.core.config import Settings
from app.models import KnowledgeDocument
from app.schemas.agent import RetrievedDoc
from app.services.ai_agents import RetrievalAgent, parse_citations
from app.services.chunking import chunk_pages, chunk_text, count_tokens
from app.services.embeddings import (
    EMBEDDING_DIMENSION,
    LocalEmbeddingProvider,
    build_embedding_provider,
    cosine_similarity,
)
from app.services.knowledge import KnowledgeService, hybrid_search_sql, vector_literal
from app.services.retrieval import LLMReranker, reciprocal_rank_fusion, search_terms, to_tsquery_text
from tests.fakes import FakeKnowledge

# --- Chunking -----------------------------------------------------------------------------------

POLICY = """# Рекламации

Рекламации се примаат во рок од 7 дена. Потребна е фотографија од производот.

Ако грешката е наша, нудиме повторно печатење или поврат на парите. Рекламациите ги решава менаџерот за квалитет.
"""


def test_chunks_respect_the_token_budget_and_overlap() -> None:
    text = " ".join(f"Реченица број {i} е за тест." for i in range(60))
    chunks = chunk_text(text, max_tokens=40, overlap_tokens=10)

    assert len(chunks) > 3
    assert all(c.token_count <= 40 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Consecutive chunks share their boundary sentence.
    assert chunks[0].text.splitlines()[-1] == chunks[1].text.splitlines()[0]


def test_chunks_split_on_sentences_and_repeat_the_section_heading() -> None:
    chunks = chunk_text(POLICY, max_tokens=30, overlap_tokens=0)

    assert all(c.text.startswith("# Рекламации") for c in chunks)
    # No sentence is cut in the middle.
    assert all(line.endswith((".", "Рекламации")) for c in chunks for line in c.text.splitlines())


def test_a_sentence_longer_than_the_budget_is_split_by_words() -> None:
    chunks = chunk_text("збор " * 100, max_tokens=30, overlap_tokens=5)
    assert all(c.token_count <= 30 for c in chunks)
    assert sum(c.text.count("збор") for c in chunks) >= 100


def test_pdf_pages_keep_their_page_numbers() -> None:
    chunks = chunk_pages([(1, "Прва страница."), (2, ""), (3, "Трета страница.")])
    assert [(c.page, c.text) for c in chunks] == [(1, "Прва страница."), (3, "Трета страница.")]


def test_overlap_must_be_smaller_than_the_chunk() -> None:
    with pytest.raises(ValueError):
        chunk_text("x", max_tokens=10, overlap_tokens=10)


def test_count_tokens_counts_words_and_punctuation() -> None:
    assert count_tokens("Цена: 2.400 ден.") == 7


# --- Query terms and fusion ---------------------------------------------------------------------


def test_search_terms_drop_stopwords_and_prefix_long_words() -> None:
    terms = search_terms("Здраво, колку чинат визит карти и рекламации? The price of delivery")
    assert terms == ["чинат", "визит", "карти", "реклама:*", "price", "deliv:*"]
    assert to_tsquery_text(terms[:2]) == "чинат | визит"


def test_search_terms_contain_only_tsquery_safe_characters() -> None:
    terms = search_terms("x' | !(drop) & table_name:* -- ;")
    assert all(term.replace(":*", "").isalnum() for term in terms)


def test_reciprocal_rank_fusion() -> None:
    fused = reciprocal_rank_fusion({"semantic": [1, 2, 3], "lexical": [3, 1]}, k=60)
    scores = dict(fused)
    assert [item_id for item_id, _ in fused] == [1, 3, 2]
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[2] == pytest.approx(1 / 62)


# --- SQL ------------------------------------------------------------------------------------------


def test_hybrid_sql_fuses_both_rankings_in_the_database() -> None:
    sql = hybrid_search_sql(semantic=True, lexical=True)
    assert "embedding <=> CAST(:query_vector AS vector)" in sql
    assert "to_tsquery('simple', :tsquery)" in sql and "ts_rank_cd" in sql
    assert "FULL OUTER JOIN" in sql and "1.0 / (:rrf_k + s.rnk)" in sql
    assert sql.count("LIMIT :candidates") == 2 and sql.rstrip().endswith("LIMIT :limit")
    assert "semantic" not in hybrid_search_sql(semantic=False, lexical=True)
    assert "lexical" not in hybrid_search_sql(semantic=True, lexical=False)


class _Result(list[Any]):
    def all(self) -> list[Any]:
        return list(self)


class FakeSqlSession:
    def __init__(self, rows: Sequence[Any] = ()) -> None:
        self.rows = list(rows)
        self.statements: list[tuple[str, dict[str, Any]]] = []
        self.added: list[KnowledgeDocument] = []

    async def execute(self, statement: Any, params: dict[str, Any] | None = None) -> _Result:
        self.statements.append((str(statement), params or {}))
        return _Result(self.rows)

    def add_all(self, items: Any) -> None:
        self.added.extend(items)

    async def commit(self) -> None:
        return None


class FakeEmbedder:
    model_name = "fake"
    dimension = 3

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.5, 0.25, 0.0]


def _row(**values: Any) -> SimpleNamespace:
    base = {"id": 7, "content": "Визит карти", "filename": "pricing.md", "chunk_index": 0, "page": None}
    return SimpleNamespace(**(base | {"score": 0.032, "vector_similarity": 0.81, "text_rank": 0.4} | values))


async def test_search_runs_one_hybrid_query_and_maps_real_scores() -> None:
    session, embedder = FakeSqlSession([_row(), _row(id=9, vector_similarity=None)]), FakeEmbedder()

    docs = await KnowledgeService(cast(Any, session), embedder, candidates=20).search("u@x.mk", "Цена визит карти", 2)

    ((sql, params),) = session.statements  # everything happens in a single SQL statement
    assert "FULL OUTER JOIN" in sql
    assert params["query_vector"] == "[0.5,0.25,0]" and params["tsquery"] == "цена | визит | карти"
    assert (params["user_email"], params["candidates"], params["limit"]) == ("u@x.mk", 20, 2)
    assert embedder.queries == ["Цена визит карти"]
    assert docs[0] == RetrievedDoc(
        chunk_id=7, content="Визит карти", filename="pricing.md", chunk_index=0, score=0.032,
        vector_similarity=0.81, text_rank=0.4,
    )  # fmt: skip
    assert docs[1].vector_similarity is None


async def test_search_without_embedder_is_full_text_only_and_skips_empty_queries() -> None:
    session = FakeSqlSession([_row()])
    service = KnowledgeService(cast(Any, session))

    assert await service.search("u@x.mk", "и на the") == []  # only stopwords: no query at all
    await service.search("u@x.mk", "PayPal")
    ((sql, params),) = session.statements
    assert "semantic" not in sql and "query_vector" not in params


async def test_upload_stores_pending_chunks_with_pages() -> None:
    session = FakeSqlSession()
    count = await KnowledgeService(cast(Any, session)).add_document(
        "u@x.mk", "cenovnik.pdf", [(1, "Визит карти: 900 ден."), (2, "Флаери: 3.500 ден.")]
    )

    assert count == 2
    assert [(d.page, d.chunk_index, d.embedding_status, d.embedding) for d in session.added] == [
        (1, 0, "pending", None),
        (2, 1, "pending", None),
    ]
    assert "DELETE FROM knowledge_document" in session.statements[0][0]  # replaces the previous version


def test_vector_literal() -> None:
    assert vector_literal([0.1, -2.0, 3e-9]) == "[0.1,-2,3e-09]"


# --- Citations, reranking -------------------------------------------------------------------------


def _doc(chunk_id: int, filename: str = "pricing.md") -> RetrievedDoc:
    return RetrievedDoc(chunk_id=chunk_id, content=f"chunk {chunk_id}", filename=filename, page=2, score=0.03)


def test_parse_citations_keeps_only_retrieved_ids_in_order() -> None:
    docs = [_doc(3), _doc(8, "faq_en.md")]
    text = "АКЦИЈА: ОДГОВОР\nИЗВОРИ: 8, 3, 8, 42\nПОРАКА: Цената е ... (ИЗВОРИ: 99)"

    citations = parse_citations(text, docs)

    assert [c.chunk_id for c in citations] == [8, 3]
    assert citations[0].label() == "faq_en.md, стр. 2" and citations[0].snippet == "chunk 8"
    assert parse_citations("АКЦИЈА: ОДГОВОР\nИЗВОРИ: НЕМА\nПОРАКА: x", docs) == []
    assert parse_citations("**Sources:** 3\nПОРАКА: x", docs)[0].chunk_id == 3


class FakeReranker:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.seen: list[int] = []

    async def rerank(self, query: str, docs: Sequence[RetrievedDoc], top_k: int) -> list[RetrievedDoc]:
        self.seen = [d.chunk_id for d in docs]
        if self.fail:
            raise StructuredOutputError("bad output")
        return list(reversed(docs))[:top_k]


async def test_retrieval_agent_reranks_a_larger_candidate_set() -> None:
    searcher, reranker = FakeKnowledge([_doc(i) for i in range(1, 30)]), FakeReranker()

    docs = await RetrievalAgent(searcher, reranker, top_k=3, rerank_candidates=20).execute("u", "q")

    assert searcher.queries == [("q", 20)] and len(reranker.seen) == 20
    assert [d.chunk_id for d in docs] == [20, 19, 18]


async def test_retrieval_agent_falls_back_to_fused_order_when_reranking_fails() -> None:
    searcher = FakeKnowledge([_doc(i) for i in range(1, 30)])
    docs = await RetrievalAgent(searcher, FakeReranker(fail=True), top_k=3).execute("u", "q")
    assert [d.chunk_id for d in docs] == [1, 2, 3]


async def test_llm_reranker_orders_by_relevance_and_keeps_fused_order_for_ties() -> None:
    from tests.test_structured_output import FakeMessages

    output = '{"judgements": [{"chunk_id": 2, "relevance": 3}, {"chunk_id": 1, "relevance": 1}]}'
    client = cast(Any, SimpleNamespace(messages=FakeMessages([(output, "end_turn")])))

    docs = await LLMReranker(client, "claude-haiku-4-5").rerank("q", [_doc(1), _doc(2), _doc(3)], top_k=2)

    assert [(d.chunk_id, d.rerank_score) for d in docs] == [(2, 3.0), (1, 1.0)]


# --- Embeddings -------------------------------------------------------------------------------------


def test_cosine_similarity() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0


def _settings(**values: Any) -> Settings:
    return cast(Settings, SimpleNamespace(**({"local_embedding_model": "intfloat/multilingual-e5-large"} | values)))


def test_provider_is_chosen_by_config_and_voyage_needs_a_key() -> None:
    local = build_embedding_provider(_settings(embedding_provider="local"))
    assert isinstance(local, LocalEmbeddingProvider) and local.dimension == EMBEDDING_DIMENSION
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        build_embedding_provider(_settings(embedding_provider="voyage", voyage_api_key=None, voyage_model="voyage-4"))
    voyage = build_embedding_provider(
        _settings(embedding_provider="voyage", voyage_api_key=SecretStr("test-not-a-real-key"), voyage_model="voyage-4")
    )
    assert (voyage.model_name, voyage.dimension) == ("voyage-4", EMBEDDING_DIMENSION)


class _FakeSentenceModel:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def encode(self, texts: list[str], batch_size: int, normalize_embeddings: bool) -> list[list[float]]:
        self.inputs.append(texts)
        return [[0.0, 1.0] for _ in texts]


async def test_local_provider_adds_e5_prefixes() -> None:
    provider = LocalEmbeddingProvider(dimension=2)
    model = _FakeSentenceModel()
    provider._model = model  # skip loading the real model

    assert await provider.embed_documents(["Визит карти"]) == [[0.0, 1.0]]
    assert await provider.embed_query("цена") == [0.0, 1.0]
    assert model.inputs == [["passage: Визит карти"], ["query: цена"]]
    assert await provider.embed_documents([]) == []
