import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.app.core.database import get_db
from backend.app.db.models import Document
from backend.app.services.parser import PDFParserService
from backend.app.services.indexer import IndexerService

logger = logging.getLogger(__name__)

router = APIRouter()
def get_indexer_service() -> IndexerService:
    return IndexerService()

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    file: UploadFile = File(...),
    title: str = Form(...),
    source_type: str = Form(...),  # 'clinical_guideline', 'biomedical_paper', 'treatment_protocol'
    publisher: Optional[str] = Form(None),
    evidence_level: Optional[str] = Form(None),
    indexer_service: IndexerService = Depends(get_indexer_service),
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts a PDF document, runs parsing/hierarchical chunking, generates embeddings,
    and uploads vectors to Qdrant and metadata schemas to PostgreSQL.
    """
    # Verify file format
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported format. Only PDF files are supported."
        )

    # Validate source type parameters
    valid_source_types = ["clinical_guideline", "biomedical_paper", "treatment_protocol"]
    if source_type not in valid_source_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid source_type. Must be one of: {', '.join(valid_source_types)}"
        )

    try:
        file_bytes = await file.read()
        file_hash = indexer_service.calculate_file_hash(file_bytes)

        # Check for duplicates
        if await indexer_service.document_exists(db, file_hash):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A document with this content hash already exists."
            )

        # 1. Parse PDF pages
        pages_data = PDFParserService.extract_text_by_page(file_bytes)
        if not pages_data:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No readable text was extracted from this PDF."
            )

        # 2. Chunk text hierarchically
        chunks_structure = PDFParserService.chunk_document(pages_data)

        # 3. Index to Qdrant & SQL metadata stores
        document = await indexer_service.index_document(
            db=db,
            file_hash=file_hash,
            title=title,
            source_type=source_type,
            file_bytes=file_bytes,
            chunks_structure=chunks_structure,
            publisher=publisher,
            evidence_level=evidence_level
        )

        total_chunks = sum(1 + len(p["child_chunks"]) for p in chunks_structure)

        return {
            "document_id": str(document.id),
            "title": document.title,
            "source_type": document.source_type,
            "hash": document.document_hash,
            "total_chunks": total_chunks,
            "status": "indexed"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error during ingestion parsing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion processing failure: {str(e)}"
        )


@router.get("/documents", response_model=List[dict])
async def list_documents(db: AsyncSession = Depends(get_db)):
    """
    Lists metadata for all ingested clinical files.
    """
    try:
        result = await db.execute(select(Document).order_by(Document.created_at.desc()))
        docs = result.scalars().all()
        return [
            {
                "id": str(doc.id),
                "title": doc.title,
                "publisher": doc.publisher,
                "source_type": doc.source_type,
                "evidence_level": doc.evidence_level,
                "created_at": doc.created_at.isoformat()
            }
            for doc in docs
        ]
    except Exception as e:
        logger.error(f"Database query failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query database."
        )
