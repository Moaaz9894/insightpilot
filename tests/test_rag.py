import pytest
from insightpilot.rag.retriever import get_retriever

CASES = [
    ("What is the maximum discount on a promotion?", "promotions_policy"),
    ("How many days of safety stock do we keep?", "supply_chain_policy"),
    ("When is a customer put on hold for overdue invoices?", "finance_returns_policy"),
    ("What is the price floor margin?", "pricing_policy"),
    ("How often are supermarkets visited?", "distribution_policy"),
]


@pytest.mark.parametrize("q,doc", CASES)
def test_top3_contains_expected_policy(q, doc):
    assert doc in [r["doc"] for r in get_retriever().search(q, 3)]
