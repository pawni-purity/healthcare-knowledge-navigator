import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from backend.app.core.database import get_db
from backend.app.services.search import SearchCoordinator
from backend.app.services.evaluator import RetrievalEvaluator

logger = logging.getLogger(__name__)

router = APIRouter()
def get_search_coordinator() -> SearchCoordinator:
    return SearchCoordinator()

# 1. Pydantic request / response schemas
class SearchFilterSchema(BaseModel):
    source_type: Optional[str] = Field(None, description="Filter: 'clinical_guideline' | 'biomedical_paper' | 'treatment_protocol'")
    document_id: Optional[str] = Field(None, description="Filter by Document UUID")
    page_number: Optional[int] = Field(None, description="Filter by exact page number")

class SearchQuerySchema(BaseModel):
    query: str = Field(..., description="The clinical search query")
    limit: int = Field(5, description="Number of results to retrieve", ge=1, le=50)
    filters: Optional[SearchFilterSchema] = Field(None, description="Pre-query metadata filters")
    expand_query: bool = Field(True, description="Enable clinical abbreviation expansion")

class EvaluationItemSchema(BaseModel):
    query: str = Field(..., description="Test query string")
    expected_document_id: str = Field(..., description="Expected document UUID string to measure hits")
    filters: Optional[SearchFilterSchema] = Field(None, description="Pre-query metadata filters for test query")

class EvaluationRequestSchema(BaseModel):
    dataset: List[EvaluationItemSchema] = Field(..., description="List of evaluation data items")
    limit: int = Field(5, description="Recall K limit to check hit rates")

@router.post("/query", status_code=status.HTTP_200_OK)
async def semantic_query(
    payload: SearchQuerySchema,
    search_coordinator: SearchCoordinator = Depends(get_search_coordinator),
    db: AsyncSession = Depends(get_db)
):
    """
    Executes a semantic vector search query on Qdrant, resolves parent contexts,
    and returns metadata-grounded search results.
    """
    try:
        # Convert Pydantic filters to dict
        filters_dict = payload.filters.model_dump(exclude_none=True) if payload.filters else None

        results = await search_coordinator.semantic_search(
            db=db,
            query=payload.query,
            limit=payload.limit,
            filters=filters_dict,
            expand_query=payload.expand_query
        )
        return results

    except Exception as e:
        logger.error(f"Failed to execute semantic query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search execution failure: {str(e)}"
        )

@router.post("/evaluate", status_code=status.HTTP_200_OK)
async def evaluate_retrieval(
    payload: EvaluationRequestSchema,
    search_coordinator: SearchCoordinator = Depends(get_search_coordinator),
    db: AsyncSession = Depends(get_db)
):
    """
    Evaluates semantic retrieval performance against a clinical test dataset,
    returning aggregate Mean Reciprocal Rank (MRR) and Hit Rate statistics.
    """
    try:
        # Format Pydantic list to standard dict items
        dataset_items = []
        for item in payload.dataset:
            item_dict: Dict[str, Any] = {
                "query": item.query,
                "expected_document_id": item.expected_document_id
            }
            if item.filters:
                item_dict["filters"] = item.filters.model_dump(exclude_none=True)
            dataset_items.append(item_dict)

        results = await RetrievalEvaluator.evaluate_retrieval_performance(
            db=db,
            search_coordinator=search_coordinator,
            dataset=dataset_items,
            limit=payload.limit
        )
        return results

    except Exception as e:
        logger.error(f"Failed to execute retrieval evaluation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval evaluation execution failure: {str(e)}"
        )
