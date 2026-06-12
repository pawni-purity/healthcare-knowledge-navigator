import pytest
from backend.app.services.citations import CitationEngine

def test_extract_year_from_date():
    assert CitationEngine.extract_year_from_date("2023-05-15") == 2023
    assert CitationEngine.extract_year_from_date("10/24/1998") == 1998
    assert CitationEngine.extract_year_from_date("Published in 2021.") == 2021
    assert CitationEngine.extract_year_from_date(None) is None
    assert CitationEngine.extract_year_from_date("not-a-date") is None

def test_resolve_citations_success():
    sources = [
        {
            "source_index": 1,
            "chunk_id": "chunk-111",
            "document_id": "doc-aaa",
            "title": "Aspirin Guidelines",
            "publisher": "AHA",
            "source_type": "clinical_guideline",
            "evidence_level": "Level 1a",
            "page_number": 5,
            "section_header": "Dosage",
            "publication_date": "2022-01-01"
        },
        {
            "source_index": 2,
            "chunk_id": "chunk-222",
            "document_id": "doc-bbb",
            "title": "Beta Blockers Trial",
            "publisher": "ACC",
            "source_type": "biomedical_paper",
            "evidence_level": "Level 2b",
            "page_number": 12,
            "section_header": "Results",
            "publication_date": "2019-11-20"
        }
    ]

    answer_text = "Aspirin is recommended for secondary prevention [1]. Beta blockers show mixed results [2]. Both are used [1][2]."
    
    result = CitationEngine.resolve_citations(answer_text, sources)
    
    assert result["answer"] == answer_text
    citations = result["citations"]
    assert len(citations) == 2
    
    # Check resolved indexes
    c1 = citations[0]
    assert c1["citation_index"] == 1
    assert c1["document_id"] == "doc-aaa"
    assert c1["publication_year"] == 2022
    
    c2 = citations[1]
    assert c2["citation_index"] == 2
    assert c2["document_id"] == "doc-bbb"
    assert c2["publication_year"] == 2019

def test_resolve_citations_missing_source():
    sources = [
        {
            "source_index": 1,
            "chunk_id": "chunk-111",
            "document_id": "doc-aaa",
            "title": "Aspirin Guidelines",
            "source_type": "clinical_guideline",
            "publication_date": "2022-01-01"
        }
    ]
    
    answer_text = "Aspirin should be given [1], but other drugs are contraindicated [2]."
    result = CitationEngine.resolve_citations(answer_text, sources)
    
    # [2] is missing from sources, so only [1] should be resolved
    assert len(result["citations"]) == 1
    assert result["citations"][0]["citation_index"] == 1
