import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.services.search import SearchCoordinator

logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    @staticmethod
    def calculate_hit_rate(retrieved_doc_ids: List[str], expected_doc_id: str) -> float:
        """
        Calculates if the expected document ID exists in the retrieved list (returns 1.0 or 0.0).
        """
        return 1.0 if expected_doc_id in retrieved_doc_ids else 0.0

    @staticmethod
    def calculate_reciprocal_rank(retrieved_doc_ids: List[str], expected_doc_id: str) -> float:
        """
        Calculates the reciprocal rank (1 / rank) of the first matching document.
        Returns 0.0 if not found.
        """
        try:
            rank = retrieved_doc_ids.index(expected_doc_id) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0

    @classmethod
    async def evaluate_retrieval_performance(
        cls,
        db: AsyncSession,
        search_coordinator: SearchCoordinator,
        dataset: List[Dict[str, Any]],
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluates retrieval performance over a test dataset.
        Format of dataset elements:
        {
          "query": "query string",
          "expected_document_id": "document-uuid-string",
          "filters": Optional[Dict[str, Any]]
        }
        
        Returns aggregated Hit Rate and Mean Reciprocal Rank (MRR) metrics.
        """
        if not dataset:
            return {
                "total_queries": 0,
                "mean_hit_rate": 0.0,
                "mean_reciprocal_rank": 0.0,
                "queries_evaluated": []
            }

        total_queries = len(dataset)
        sum_hit_rate = 0.0
        sum_reciprocal_rank = 0.0
        evaluation_logs = []

        for idx, item in enumerate(dataset):
            query = item["query"]
            expected_doc_id = item["expected_document_id"]
            filters = item.get("filters")

            # Run retrieval
            retrieved = await search_coordinator.semantic_search(
                db=db,
                query=query,
                limit=limit,
                filters=filters,
                expand_query=True
            )

            # Extract document IDs from matches
            retrieved_doc_ids = []
            for hit in retrieved:
                doc_meta = hit.get("document", {})
                doc_id = doc_meta.get("id")
                if doc_id:
                    retrieved_doc_ids.append(doc_id)

            # Compute stats
            hit_rate = cls.calculate_hit_rate(retrieved_doc_ids, expected_doc_id)
            rr = cls.calculate_reciprocal_rank(retrieved_doc_ids, expected_doc_id)

            sum_hit_rate += hit_rate
            sum_reciprocal_rank += rr

            evaluation_logs.append({
                "query": query,
                "expected_document_id": expected_doc_id,
                "retrieved_document_ids": retrieved_doc_ids,
                "hit": hit_rate > 0,
                "reciprocal_rank": rr
            })

        mean_hit_rate = sum_hit_rate / total_queries
        mean_mrr = sum_reciprocal_rank / total_queries

        return {
            "total_queries": total_queries,
            "mean_hit_rate": round(mean_hit_rate, 4),
            "mean_reciprocal_rank": round(mean_mrr, 4),
            "queries_evaluated": evaluation_logs
        }
