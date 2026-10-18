from langchain_core.documents import Document

from filingsight.evaluation import satisfies, score_retrieval
from filingsight.retrieval import plan_query, rrf_fuse, select_context


def test_plan_single_company_uses_filter():
    p = plan_query("What was Uber's revenue in 2021?")
    assert p.companies == ["Uber"] and p.sub_queries == [("What was Uber's revenue in 2021?", "Uber")]
    assert not p.out_of_scope_years


def test_plan_comparison_decomposes_per_company():
    p = plan_query("Compare Uber and Lyft revenue growth")
    assert [c for _, c in p.sub_queries] == ["Uber", "Lyft"]
    assert p.is_comparison


def test_plan_ambiguous_question_covers_every_company():
    p = plan_query("What was the net loss in 2021?")
    assert [c for _, c in p.sub_queries] == ["Uber", "Lyft"]


def test_plan_flags_years_outside_corpus():
    assert plan_query("Uber revenue in 2023").out_of_scope_years == [2023]


def test_rrf_rewards_agreement():
    fused = rrf_fuse([[(1, 0.9), (2, 0.8)], [(2, 9.0), (3, 5.0)]], k=3)
    assert fused[0][0] == 2


def _doc(company, text="x"):
    return Document(page_content=text, metadata={"company": company})


def test_select_context_keeps_both_companies():
    ranked = [(_doc("Uber"), 0.9)] * 6 + [(_doc("Lyft"), 0.5)]
    out = select_context(ranked, plan_query("Compare Uber and Lyft"), k=4)
    assert {d.metadata["company"] for d, _ in out} == {"Uber", "Lyft"}


def test_evidence_matching_and_scores():
    req = {"company": "Uber", "any": [["17,455"], ["$17.5 billion"]]}
    good, wrong_co = _doc("Uber", "Total revenue | $11,139 | $17,455"), _doc("Lyft", "17,455")
    assert satisfies(good, req) and not satisfies(wrong_co, req)
    s = score_retrieval([wrong_co, good], [req], k=5)
    assert s.hit == 1 and s.rr == 0.5 and s.precision == 0.5
