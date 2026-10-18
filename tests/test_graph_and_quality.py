"""Uses the committed index in data/processed (embeds queries with the ONNX model)."""
import statistics

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from filingsight import config, llm
from filingsight.evaluation import load_questions, score_retrieval
from filingsight.graph import ask
from filingsight.pipeline import run_retrieval
from filingsight.prompts import REFUSAL


@pytest.fixture
def fake_llm(monkeypatch):
    def install(responses):
        model = FakeListChatModel(responses=responses)
        monkeypatch.setattr(config, "llm_available", lambda: True)
        monkeypatch.setattr(llm, "get_chat_model", lambda *a, **k: model)
    return install


def test_out_of_scope_year_is_refused_without_retrieval():
    s = ask("What was Uber's revenue in 2023?", use_llm=False)
    assert s["route"] == "refused" and s["answer"] == REFUSAL and "sources" not in s


def test_unrelated_question_is_refused_by_confidence_gate():
    s = ask("How many vehicles does Tesla plan to deliver next year?", use_llm=False)
    assert s["route"] == "refused"


def test_llm_answer_with_valid_citation(fake_llm):
    fake_llm(["Uber's total revenue was $17,455 million in 2021 [1]. Ignore this [9]."])
    s = ask("What was Uber's total revenue in 2021?", use_llm=True)
    assert s["route"] == "answered" and s["citations"] == [1]
    assert "[9]" not in s["answer"]  # citation to a non-existent source is stripped


def test_llm_not_found_routes_to_refusal(fake_llm):
    fake_llm(["NOT_FOUND"])
    s = ask("What was Uber's total revenue in 2021?", use_llm=True)
    assert s["route"] == "refused" and s["answer"] == REFUSAL


def test_uncited_answer_is_retried_once(fake_llm):
    fake_llm(["Revenue was big.", "Revenue was $17,455 million [1]."])
    s = ask("What was Uber's total revenue in 2021?", use_llm=True)
    assert s["attempts"] == 2 and s["citations"] == [1]


def test_retrieval_quality_regression():
    """Guards the headline number in the report: semantic + hybrid + local rerank."""
    scores = []
    for q in load_questions():
        if q["evidence"]:
            res = run_retrieval(q["question"], strategy="semantic", mode="hybrid", reranker="local")
            scores.append(score_retrieval([d for d, _ in res.context], q["evidence"], 5).hit)
    assert statistics.mean(scores) >= 0.85
