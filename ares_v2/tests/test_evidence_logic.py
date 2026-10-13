"""Unit tests for Evidence Retrieval, Functional, and Logic Layer."""

import pytest
import numpy as np
from ares.evidence.retriever import BM25Retriever, EvidenceRecord
from ares.evidence.credibility import CredibilityScorer
from ares.evidence.functional import EvidenceFunctional
from ares.logic.quantities import QuantityExtractor
from ares.logic.logic_score import SymbolicLogicLayer

def test_bm25_retriever():
    docs = [
        {"id": "doc1", "text": "The Eiffel Tower is located in Paris, France.", "source": "wiki"},
        {"id": "doc2", "text": "Mount Everest is the highest mountain on Earth in the Himalayas.", "source": "wiki"},
        {"id": "doc3", "text": "The Colosseum is an ancient amphitheatre in Rome, Italy.", "source": "wiki"},
    ]
    retriever = BM25Retriever()
    retriever.add_documents(docs)
    
    results = retriever.retrieve("Where is Eiffel Tower located?", top_k=2)
    assert len(results) > 0
    assert results[0].doc_id == "doc1"
    assert "Paris" in results[0].text

def test_credibility_scorer():
    scorer = CredibilityScorer()
    assert scorer.score("https://www.reuters.com/article/123") == 0.95
    assert scorer.score("who.int") == 0.98
    assert scorer.score("unknown-blog.xyz") == 0.50
    
    assert scorer.is_factcheck_domain("https://www.snopes.com/fact-check/test") is True
    assert scorer.is_factcheck_domain("https://politifact.com/rulings") is True
    assert scorer.is_factcheck_domain("https://reuters.com/news") is False

def test_evidence_functional_formula():
    func = EvidenceFunctional(tau_true=0.20, tau_false=-0.20)
    
    # 2 supporting items, S=0.8 and S=0.9, W=1.0, N=+1.0
    items = [
        {"id": "e1", "text": "ev1", "source": "wiki", "similarity": 0.8, "nli": {"soft_n": 0.9, "hard_n": 1.0}, "credibility": 1.0},
        {"id": "e2", "text": "ev2", "source": "wiki", "similarity": 0.9, "nli": {"soft_n": 0.7, "hard_n": 1.0}, "credibility": 1.0},
    ]
    
    res = func.compute("claim", items, use_soft_n=True)
    # Truth' = (0.8 * 0.9 + 0.9 * 0.7) / (0.8 + 0.9) = (0.72 + 0.63) / 1.7 = 1.35 / 1.7 ≈ 0.7941
    expected_truth = (0.8 * 0.9 + 0.9 * 0.7) / 1.7
    assert abs(res.truth_score - expected_truth) < 1e-4
    assert res.verdict == "SUPPORTS"
    assert res.calibrated_confidence > 0.80

def test_quantity_extractor_and_logic():
    extractor = QuantityExtractor()
    cmp1 = extractor.compare_quantities("Claim states 8 bombs exploded.", "Report confirms 8 bombs exploded.", rel_tolerance=0.01)
    assert len(cmp1["confirmations"]) == 1
    assert cmp1["has_numeric_conflict"] is False
    
    cmp2 = extractor.compare_quantities("Claim asserts 500 casualties.", "Official reports confirm 0 casualties.", rel_tolerance=0.01)
    assert cmp2["has_numeric_conflict"] is True
    
    logic = SymbolicLogicLayer()
    res = logic.evaluate("Claim asserts 500 casualties.", ["Official reports confirm 0 casualties."])
    assert res["logic_flag"] == "CONTRADICTION"
    assert res["logic_adjustment"] < 0
