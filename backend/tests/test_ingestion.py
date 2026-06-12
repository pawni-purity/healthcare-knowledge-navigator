import pytest
from unittest.mock import patch
from fastapi import status

def test_health_check(client):
    """
    Assert health check returns active services configuration.
    """
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "healthy"

def test_upload_pdf_invalid_extension(client):
    """
    Assert that files other than PDF are blocked.
    """
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    data = {
        "title": "Invalid Document",
        "source_type": "clinical_guideline"
    }
    response = client.post("/api/v1/ingestion/upload", files=files, data=data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Only PDF files are supported" in response.json()["detail"]

def test_upload_pdf_invalid_source_type(client):
    """
    Assert that unrecognized source type filters fail.
    """
    files = {"file": ("test.pdf", b"%PDF-1.4 dummy contents", "application/pdf")}
    data = {
        "title": "Invalid Source Type",
        "source_type": "invalid_type"
    }
    response = client.post("/api/v1/ingestion/upload", files=files, data=data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid source_type" in response.json()["detail"]

def test_upload_pdf_success_and_duplicate(client):
    """
    Asserts successful PDF upload triggers index, and subsequent identical hash upload triggers 409 conflict.
    """
    files = {"file": ("guideline.pdf", b"%PDF-1.4 dummy guidelines content", "application/pdf")}
    data = {
        "title": "Hypertension Guideline 2026",
        "source_type": "clinical_guideline",
        "publisher": "ACC/AHA",
        "evidence_level": "Level 1a"
    }

    # Patch PDFParser page text extraction to prevent PDF binary read failures
    with patch("backend.app.services.parser.PDFParserService.extract_text_by_page") as mock_extract:
        mock_extract.return_value = [
            {"page_number": 1, "content": "CLINICAL GUIDELINE: ACC/AHA Treatment of hypertension."}
        ]

        # 1. First upload (Expected: Success)
        response = client.post("/api/v1/ingestion/upload", files=files, data=data)
        assert response.status_code == status.HTTP_201_CREATED
        json_data = response.json()
        assert json_data["status"] == "indexed"
        assert json_data["title"] == "Hypertension Guideline 2026"
        assert json_data["total_chunks"] > 0
        doc_id = json_data["document_id"]

        # Verify listing API shows document
        list_response = client.get("/api/v1/ingestion/documents")
        assert list_response.status_code == status.HTTP_200_OK
        docs = list_response.json()
        assert len(docs) == 1
        assert docs[0]["id"] == doc_id
        assert docs[0]["title"] == "Hypertension Guideline 2026"

        # 2. Second upload with same bytes (Expected: 409 Conflict)
        duplicate_response = client.post("/api/v1/ingestion/upload", files=files, data=data)
        assert duplicate_response.status_code == status.HTTP_409_CONFLICT
        assert "already exists" in duplicate_response.json()["detail"]
