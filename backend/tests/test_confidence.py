import pytest
from backend.app.services.confidence import ConfidenceScorer

def test_calculate_sentence_citation_coverage():
    # 3 sentences, 2 cited
    text1 = "Sentence one [1]. Sentence two [2]. Sentence three without citation."
    assert ConfidenceScorer.calculate_sentence_citation_coverage(text1) == pytest.approx(2 / 3)

    # 1 sentence, 0 cited
    text2 = "A single sentence without citations."
    assert ConfidenceScorer.calculate_sentence_citation_coverage(text2) == 0.0

    # Empty text
    assert ConfidenceScorer.calculate_sentence_citation_coverage("") == 0.0


def test_calculate_confidence_high():
    answer = "Aspirin is highly recommended for acute stroke management [1]."
    citations = [
        {
            "chunk_id": "chunk-111",
            "document_id": "doc-aaa",
            "evidence_level": "Level 1a"  # High weight: 1.0
        }
    ]
    retrieved = [
        {
            "chunk_id": "chunk-111",
            "score": 0.95,  # High similarity
            "document": {
                "id": "doc-aaa",
                "evidence_level": "Level 1a"
            }
        }
    ]

    result = ConfidenceScorer.calculate_confidence(answer, citations, retrieved)
    assert result["score"] >= 0.80
    assert result["label"] == "High"


def test_calculate_confidence_medium():
    # Only 1 of 2 sentences is cited, lower similarity, lower evidence level
    answer = "Take aspirin daily [1]. Some side effects might occur."
    citations = [
        {
            "chunk_id": "chunk-111",
            "document_id": "doc-aaa",
            "evidence_level": "Level 3"  # Weight: 0.70
        }
    ]
    retrieved = [
        {
            "chunk_id": "chunk-111",
            "score": 0.75,
            "document": {
                "id": "doc-aaa",
                "evidence_level": "Level 3"
            }
        }
    ]

    result = ConfidenceScorer.calculate_confidence(answer, citations, retrieved)
    assert 0.50 <= result["score"] < 0.80
    assert result["label"] == "Medium"


def test_calculate_confidence_low():
    # No citations at all, low similarity, weak evidence level
    answer = "I could not find sufficient evidence."
    citations = []
    retrieved = [
        {
            "chunk_id": "chunk-111",
            "score": 0.40,
            "document": {
                "id": "doc-aaa",
                "evidence_level": "Level 5"  # Weight: 0.45
            }
        }
    ]

    result = ConfidenceScorer.calculate_confidence(answer, citations, retrieved)
    assert result["score"] < 0.50
    assert result["label"] == "Low"
