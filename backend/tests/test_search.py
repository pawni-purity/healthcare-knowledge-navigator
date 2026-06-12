import pytest
import uuid
from unittest.mock import MagicMock
from fastapi import status

from backend.app.services.query_expander import MedicalQueryExpander
from backend.app.services.evaluator import RetrievalEvaluator
from backend.app.db.models import Document, Chunk

def test_query_expansion():
    """
    Verify abbreviations expand to standard terms.
    """
    assert MedicalQueryExpander.expand_query("COPD treatment guidelines") == "COPD (chronic obstructive pulmonary disease) treatment guidelines"
    assert MedicalQueryExpander.expand_query("AFib medication") == "AFib (atrial fibrillation) medication"
    assert MedicalQueryExpander.expand_query("normal text") == "normal text"
    assert MedicalQueryExpander.expand_query("") == ""

def test_evaluation_metrics_math():
    """
    Verify hit rate and reciprocal rank math functions.
    """
    doc_list = ["doc1", "doc2", "doc3"]
    
    # Hit Rate tests
    assert RetrievalEvaluator.calculate_hit_rate(doc_list, "doc2") == 1.0
    assert RetrievalEvaluator.calculate_hit_rate(doc_list, "doc4") == 0.0

    # Reciprocal Rank tests
    assert RetrievalEvaluator.calculate_reciprocal_rank(doc_list, "doc1") == 1.0
    assert RetrievalEvaluator.calculate_reciprocal_rank(doc_list, "doc2") == 0.5
    assert RetrievalEvaluator.calculate_reciprocal_rank(doc_list, "doc3") == 0.3333333333333333
    assert RetrievalEvaluator.calculate_reciprocal_rank(doc_list, "doc4") == 0.0

@pytest.mark.asyncio
async def test_search_endpoint_with_postgres_joins(client, db_session, mock_qdrant):
    """
    Inserts mock guidelines in db, mocks Qdrant hits, and tests search queries.
    """
    # 1. Seed database with mock Document and Chunks
    doc_id = uuid.uuid4()
    document = Document(
        id=doc_id,
        title="ESC Heart Failure Guidelines 2026",
        source_type="clinical_guideline",
        publisher="ESC",
        evidence_level="Level 1a",
        document_hash="mock-guideline-hash-abc"
    )
    db_session.add(document)
    await db_session.flush()

    parent_chunk_id = uuid.uuid4()
    parent_chunk = Chunk(
        id=parent_chunk_id,
        document_id=doc_id,
        chunk_type="parent",
        section_header="Treatment Recommendations",
        page_number=12,
        chunk_index=0,
        content="Beta-blockers are recommended in all patients with symptomatic HF.",
        token_count=10
    )
    db_session.add(parent_chunk)
    await db_session.flush()

    child_chunk_id = uuid.uuid4()
    child_chunk = Chunk(
        id=child_chunk_id,
        document_id=doc_id,
        parent_chunk_id=parent_chunk_id,
        chunk_type="child",
        section_header="Treatment Recommendations",
        page_number=12,
        chunk_index=1,
        content="Beta-blockers are recommended in HF patients.",
        token_count=6
    )
    db_session.add(child_chunk)
    await db_session.commit()

    mock_hit = MagicMock()
    mock_hit.id = str(child_chunk_id)
    mock_hit.score = 0.94
    mock_hit.payload = {
        "chunk_id": str(child_chunk_id),
        "document_id": str(doc_id),
        "source_type": "clinical_guideline",
        "page_number": 12,
        "section_header": "Treatment Recommendations"
    }
    mock_response = MagicMock()
    mock_response.points = [mock_hit]
    mock_qdrant.query_points.return_value = mock_response

    # 3. Post query request to endpoint
    payload = {
        "query": "Is beta blocker recommended for HF?",
        "limit": 3,
        "filters": {
            "source_type": "clinical_guideline"
        },
        "expand_query": True
    }
    response = client.post("/api/v1/search/query", json=payload)
    assert response.status_code == status.HTTP_200_OK
    results = response.json()
    assert len(results) == 1
    
    match = results[0]
    assert match["chunk_id"] == str(child_chunk_id)
    assert match["score"] == 0.94
    assert match["content"] == "Beta-blockers are recommended in HF patients."
    # Assert parent context join resolved correctly
    assert match["parent_content"] == "Beta-blockers are recommended in all patients with symptomatic HF."
    assert match["document"]["title"] == "ESC Heart Failure Guidelines 2026"
    assert match["document"]["evidence_level"] == "Level 1a"

@pytest.mark.asyncio
async def test_evaluate_endpoint(client, db_session, mock_qdrant):
    """
    Asserts retrieval metrics evaluations return computed statistics.
    """
    doc_id = uuid.uuid4()
    document = Document(
        id=doc_id,
        title="ESC HF Guidelines",
        source_type="clinical_guideline",
        document_hash="eval-hash"
    )
    db_session.add(document)
    await db_session.flush()

    child_chunk_id = uuid.uuid4()
    child_chunk = Chunk(
        id=child_chunk_id,
        document_id=doc_id,
        chunk_type="child",
        chunk_index=1,
        content="Beta-blocker indications",
        token_count=3
    )
    db_session.add(child_chunk)
    await db_session.commit()

    # Mock qdrant client search hits
    mock_hit = MagicMock()
    mock_hit.id = str(child_chunk_id)
    mock_hit.score = 0.88
    mock_hit.payload = {
        "chunk_id": str(child_chunk_id),
        "document_id": str(doc_id)
    }
    mock_response = MagicMock()
    mock_response.points = [mock_hit]
    mock_qdrant.query_points.return_value = mock_response

    # POST evaluation request (1 match query, 1 mismatch query)
    payload = {
        "dataset": [
            {
                "query": "HF treatment guidelines",
                "expected_document_id": str(doc_id)
            }
        ],
        "limit": 5
    }
    response = client.post("/api/v1/search/evaluate", json=payload)
    assert response.status_code == status.HTTP_200_OK
    metrics = response.json()
    assert metrics["total_queries"] == 1
    # Since mocked Qdrant returns child_chunk mapped to doc_id, it is a hit (100% Hit Rate & 1.0 MRR)
    assert metrics["mean_hit_rate"] == 1.0
    assert metrics["mean_reciprocal_rank"] == 1.0
    assert len(metrics["queries_evaluated"]) == 1
