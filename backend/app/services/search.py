import logging
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from backend.app.core.config import settings
from backend.app.db.models import Document, Chunk
from backend.app.services.embeddings import EmbeddingService
from backend.app.services.query_expander import MedicalQueryExpander

logger = logging.getLogger(__name__)

class SearchCoordinator:
    def __init__(self):
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
                        size=settings.EMBEDDING_DIMENSION,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection in SearchCoordinator: {e}")

    def _build_qdrant_filters(self, filters: Dict[str, Any]) -> Optional[qdrant_models.Filter]:
        """
        Translates simple filters dict into Qdrant Field Conditions.
        Supported keys: 'source_type', 'document_id', 'page_number'
        """
        if not filters:
            return None

        must_conditions = []

        if "source_type" in filters and filters["source_type"]:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="source_type",
                    match=qdrant_models.MatchValue(value=filters["source_type"])
                )
            )

        if "document_id" in filters and filters["document_id"]:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="document_id",
                    match=qdrant_models.MatchValue(value=str(filters["document_id"]))
                )
            )

        if "page_number" in filters and filters["page_number"] is not None:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="page_number",
                    match=qdrant_models.MatchValue(value=int(filters["page_number"]))
                )
            )

        if not must_conditions:
            return None

        return qdrant_models.Filter(must=must_conditions)

    async def semantic_search(
        self,
        db: AsyncSession,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        expand_query: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieves matching chunks:
        1. Expands medical acronyms (if expand_query is True).
        2. Vectorizes the query.
        3. Executes Qdrant search with filters.
        4. Reconstructs parent context joining Postgres records.
        """
        # 1. Expand query
        refined_query = MedicalQueryExpander.expand_query(query) if expand_query else query
        logger.info(f"Query: '{query}' -> Expanded: '{refined_query}'")

        # 2. Vectorize query
        query_vector = EmbeddingService.embed_text(refined_query)

        # 3. Retrieve child points from Qdrant
        qdrant_filter = self._build_qdrant_filters(filters) if filters else None
        
        try:
            response = self.qdrant_client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=limit
            )
            hits = response.points
        except Exception as e:
            logger.error(f"Qdrant query execution failure: {e}")
            raise RuntimeError(f"Qdrant query failed: {e}")

        if not hits:
            return []

        # 4. Resolve chunks details and parent contexts from Postgres
        results = []
        child_ids = [uuid.UUID(hit.id) for hit in hits]
        
        # Query Postgres for all retrieved child chunks in a single query
        query_stmt = (
            select(Chunk)
            .where(Chunk.id.in_(child_ids))
        )
        db_result = await db.execute(query_stmt)
        child_chunks = {chunk.id: chunk for chunk in db_result.scalars().all()}

        # Load document details and parent contexts
        for hit in hits:
            hit_uuid = uuid.UUID(hit.id)
            child_chunk = child_chunks.get(hit_uuid)
            
            if not child_chunk:
                # Fallback if SQL record was deleted or not synced
                results.append({
                    "chunk_id": hit.id,
                    "score": hit.score,
                    "content": "SQL record metadata missing",
                    "source_type": hit.payload.get("source_type"),
                    "page_number": hit.payload.get("page_number")
                })
                continue

            # Load document metadata
            doc_stmt = select(Document).where(Document.id == child_chunk.document_id)
            doc_res = await db.execute(doc_stmt)
            doc = doc_res.scalars().first()

            # Retrieve Parent Chunk content
            parent_text = child_chunk.content
            if child_chunk.parent_chunk_id:
                parent_stmt = select(Chunk).where(Chunk.id == child_chunk.parent_chunk_id)
                parent_res = await db.execute(parent_stmt)
                parent_chunk = parent_res.scalars().first()
                if parent_chunk:
                    parent_text = parent_chunk.content

            results.append({
                "chunk_id": str(child_chunk.id),
                "score": hit.score,
                "content": child_chunk.content,
                "parent_content": parent_text,
                "page_number": child_chunk.page_number,
                "section_header": child_chunk.section_header,
                "document": {
                    "id": str(doc.id) if doc else None,
                    "title": doc.title if doc else "Unknown",
                    "publisher": doc.publisher if doc else None,
                    "source_type": doc.source_type if doc else "Unknown",
                    "evidence_level": doc.evidence_level if doc else None,
                    "publication_date": doc.publication_date.isoformat() if (doc and doc.publication_date) else None
                }
            })

        return results
