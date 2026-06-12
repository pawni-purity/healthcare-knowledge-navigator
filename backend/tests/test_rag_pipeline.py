import pytest
import uuid
from unittest.mock import patch, MagicMock
from sqlalchemy.future import select

from backend.app.db.models import Document, Chunk, QueryLog
from qdrant_client.http import models as qdrant_models

@pytest.mark.asyncio
async def test_rag_ask_success_pipeline(client, db_session):
    # 1. Insert mock document and child/parent chunks into SQLite test database
    doc_id = uuid.uuid4()
    parent_chunk_id = uuid.uuid4()
    child_chunk_id = uuid.uuid4()

    doc = Document(
        id=doc_id,
        title="Test Hypertension Guidelines 2026",
        source_type="clinical_guideline",
        publisher="American Medical Association",
        evidence_level="Level 1a",
        document_hash="test-hash-hypertension-12345",
        publication_date=None
    )
    
    parent_chunk = Chunk(
        id=parent_chunk_id,
        document_id=doc_id,
        parent_chunk_id=None,
        chunk_type="parent",
        section_header="Treatment",
        page_number=3,
        chunk_index=0,
        content="This is the broad parent context narrative containing Hypertension recommendations.",
        token_count=10
    )

    child_chunk = Chunk(
        id=child_chunk_id,
        document_id=doc_id,
        parent_chunk_id=parent_chunk_id,
        chunk_type="child",
        section_header="Treatment",
        page_number=3,
        chunk_index=1,
        content="Hypertension recommendation: give ACEI first line [1].",
        token_count=5
    )

    db_session.add(doc)
    db_session.add(parent_chunk)
    db_session.add(child_chunk)
    await db_session.commit()

    # 2. Mock Qdrant client to return this child chunk hit
    mock_point = qdrant_models.ScoredPoint(
        id=str(child_chunk_id),
        version=1,
        score=0.95,
        payload={
            "chunk_id": str(child_chunk_id),
            "document_id": str(doc_id),
            "source_type": "clinical_guideline",
            "page_number": 3,
            "section_header": "Treatment"
        },
        vector=None
    )
    
    mock_query_response = MagicMock()
    mock_query_response.points = [mock_point]

    # 3. Patch both Qdrant and LLM calls
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="According to guidelines, give ACEI first line [1]."))
    ]

    with patch("backend.app.services.search.QdrantClient") as mock_qdrant_cls, \
         patch("backend.app.services.llm.OpenAI") as mock_openai_cls:
        
        # Setup Qdrant Client mock
        mock_qdrant_inst = MagicMock()
        mock_qdrant_inst.query_points.return_value = mock_query_response
        mock_qdrant_cls.return_value = mock_qdrant_inst

        # Setup OpenAI client mock
        mock_openai_inst = MagicMock()
        mock_openai_inst.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_openai_inst
        
        # Reset provider to force config reload
        from backend.app.services.llm import LLMService
        LLMService._provider = None

        # 4. Trigger Ask API Call
        payload = {
            "query": "What is the first line treatment for HTN?",
            "limit": 5
        }
        
        response = client.post("/api/v1/rag/ask", json=payload)
        
        # Assertions
        assert response.status_code == 200
        json_data = response.json()
        assert "answer" in json_data
        assert "citations" in json_data
        assert "confidence" in json_data
        
        # Validate values
        assert "give ACEI first line" in json_data["answer"]
        assert len(json_data["citations"]) == 1
        assert json_data["citations"][0]["document_id"] == str(doc_id)
        assert json_data["citations"][0]["title"] == "Test Hypertension Guidelines 2026"
        assert json_data["confidence"]["label"] == "High"
        
        # Verify query log is written to the SQLite DB
        stmt = select(QueryLog).where(QueryLog.query == payload["query"])
        result = await db_session.execute(stmt)
        log_entry = result.scalars().first()
        assert log_entry is not None
        assert log_entry.answer == json_data["answer"]
        assert log_entry.confidence_label == "High"


@pytest.mark.asyncio
async def test_rag_chat_insufficient_evidence(client, db_session):
    # Tests the empty / insufficient evidence fallback when no points are retrieved
    mock_query_response = MagicMock()
    mock_query_response.points = []

    with patch("backend.app.services.search.QdrantClient") as mock_qdrant_cls:
        mock_qdrant_inst = MagicMock()
        mock_qdrant_inst.query_points.return_value = mock_query_response
        mock_qdrant_cls.return_value = mock_qdrant_inst

        payload = {
            "query": "Is there a cure for COVID-19?",
            "history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello. How can I help you today?"}
            ],
            "limit": 5
        }

        response = client.post("/api/v1/rag/chat", json=payload)
        assert response.status_code == 200
        json_data = response.json()
        assert json_data["answer"] == "I could not find sufficient evidence in the retrieved sources."
        assert len(json_data["citations"]) == 0
        assert json_data["confidence"]["label"] == "Low"

        # Verify DB log entry exists
        stmt = select(QueryLog).where(QueryLog.query == payload["query"])
        result = await db_session.execute(stmt)
        log_entry = result.scalars().first()
        assert log_entry is not None
