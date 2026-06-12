import logging
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.app.core.config import settings
from backend.app.db.models import Document, Chunk
from backend.app.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

class IndexerService:
    def __init__(self):
        # Establish connection to Qdrant
        if settings.QDRANT_HOST == ":memory:":
            self.qdrant_client = QdrantClient(location=":memory:")
        elif settings.QDRANT_HOST.startswith("./") or settings.QDRANT_HOST.startswith("local_"):
            self.qdrant_client = QdrantClient(path=settings.QDRANT_HOST)
        else:
            self.qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self._ensure_qdrant_collection()

    def _ensure_qdrant_collection(self):
        """
        Creates the Qdrant vector collection if it doesn't already exist.
        """
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if settings.QDRANT_COLLECTION not in collection_names:
                logger.info(f"Creating Qdrant collection: {settings.QDRANT_COLLECTION}")
                self.qdrant_client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=qdrant_models.VectorParams(
                        size=settings.EMBEDDING_DIMENSION,  # 1024 for BGE-large
                        distance=qdrant_models.Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")

    @staticmethod
    def calculate_file_hash(file_bytes: bytes) -> str:
        """
        Calculates SHA-256 checksum of file to verify document uniqueness.
        """
        return hashlib.sha256(file_bytes).hexdigest()

    async def document_exists(self, db: AsyncSession, file_hash: str) -> bool:
        """
        Checks if a document with this checksum hash is already registered in Postgres.
        """
        result = await db.execute(select(Document).where(Document.document_hash == file_hash))
        return result.scalars().first() is not None

    async def index_document(
        self,
        db: AsyncSession,
        file_hash: str,
        title: str,
        source_type: str,
        file_bytes: bytes,
        chunks_structure: List[Dict[str, Any]],
        publisher: Optional[str] = None,
        evidence_level: Optional[str] = None
    ) -> Document:
        """
        Indexes chunks and document details:
        1. Saves the relational metadata and chunk hierarchies in PostgreSQL.
        2. Computes BGE embeddings for all child chunks.
        3. Index child vectors and payloads to Qdrant.
        """
        # Create Document database record
        document = Document(
            id=uuid.uuid4(),
            title=title,
            source_type=source_type,
            publisher=publisher,
            evidence_level=evidence_level,
            document_hash=file_hash
        )
        db.add(document)
        await db.flush()  # Assure document ID is generated & bound

        # Prepare lists to index
        db_chunks = []
        child_texts = []
        child_chunk_mappings = []

        # Iterate hierarchical chunk structure
        for parent_idx, parent_data in enumerate(chunks_structure):
            parent_id = uuid.uuid4()
            
            # Parent Chunk record
            parent_chunk = Chunk(
                id=parent_id,
                document_id=document.id,
                parent_chunk_id=None,
                chunk_type="parent",
                section_header=parent_data["section_header"],
                page_number=parent_data["page_number"],
                chunk_index=parent_data["chunk_index"],
                content=parent_data["content"],
                token_count=parent_data["token_count"]
            )
            db_chunks.append(parent_chunk)

            # Child Chunk records
            for child_data in parent_data["child_chunks"]:
                child_id = uuid.uuid4()
                child_chunk = Chunk(
                    id=child_id,
                    document_id=document.id,
                    parent_chunk_id=parent_id,
                    chunk_type="child",
                    section_header=child_data["section_header"],
                    page_number=child_data["page_number"],
                    chunk_index=child_data["chunk_index"],
                    content=child_data["content"],
                    token_count=child_data["token_count"]
                )
                db_chunks.append(child_chunk)
                
                # Keep tracking for batch vector generation
                child_texts.append(child_data["content"])
                child_chunk_mappings.append({
                    "id": child_id,
                    "page_number": child_data["page_number"],
                    "section_header": child_data["section_header"]
                })

        # Insert all SQL Chunk records at once
        db.add_all(db_chunks)
        await db.flush()

        # Generate embeddings in batch for child chunks
        logger.info(f"Generating vectors for {len(child_texts)} child chunks...")
        embeddings = EmbeddingService.embed_batch(child_texts)

        # Build Qdrant points
        points = []
        for idx, (mapping, vector) in enumerate(zip(child_chunk_mappings, embeddings)):
            points.append(
                qdrant_models.PointStruct(
                    id=str(mapping["id"]),
                    vector=vector,
                    payload={
                        "chunk_id": str(mapping["id"]),
                        "document_id": str(document.id),
                        "source_type": source_type,
                        "page_number": mapping["page_number"],
                        "section_header": mapping["section_header"]
                    }
                )
            )

        # Upload points to Qdrant collection
        logger.info(f"Uploading vectors to Qdrant collection: {settings.QDRANT_COLLECTION}")
        try:
            self.qdrant_client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=points
            )
        except Exception as e:
            logger.error(f"Qdrant indexing transaction failed: {e}")
            raise RuntimeError(f"Qdrant insertion failure: {e}")

        # Complete Postgres transaction
        await db.commit()
        await db.refresh(document)
        return document
