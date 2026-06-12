import time
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from backend.app.core.database import get_db
from backend.app.db.models import QueryLog
from backend.app.services.search import SearchCoordinator
from backend.app.services.context_builder import ContextBuilder
from backend.app.services.llm import LLMService
from backend.app.services.citations import CitationEngine
from backend.app.services.confidence import ConfidenceScorer

logger = logging.getLogger(__name__)

router = APIRouter()

# 1. Pydantic Schemas
class AskRequestSchema(BaseModel):
    query: str = Field(..., description="The clinical query to answer")
    limit: int = Field(5, description="Number of source chunks to retrieve", ge=1, le=20)

class ChatMessageSchema(BaseModel):
    role: str = Field(..., description="Role of the sender: 'user' or 'assistant'")
    content: str = Field(..., description="Message content text")

class ChatRequestSchema(BaseModel):
    query: str = Field(..., description="The follow-up clinical query")
    history: List[ChatMessageSchema] = Field(default_factory=list, description="Preserved conversation history")
    limit: int = Field(5, description="Number of source chunks to retrieve", ge=1, le=20)

class ConfidenceSchema(BaseModel):
    score: float = Field(..., description="Confidence score from 0.0 to 1.0")
    label: str = Field(..., description="Confidence level label: High | Medium | Low")

class CitationSchema(BaseModel):
    citation_index: int = Field(..., description="Number index matching inline brackets [x]")
    document_id: Optional[str] = Field(None, description="Source document UUID")
    chunk_id: Optional[str] = Field(None, description="Source chunk UUID")
    title: Optional[str] = Field(None, description="Source document title")
    publication_year: Optional[int] = Field(None, description="Document publication year")
    source_type: Optional[str] = Field(None, description="Source type category")
    evidence_level: Optional[str] = Field(None, description="Medical evidence grade")
    page_number: Optional[int] = Field(None, description="Source page number")
    section_header: Optional[str] = Field(None, description="Source section name")

class RAGResponseSchema(BaseModel):
    answer: str = Field(..., description="Grounded clinical response text")
    citations: List[CitationSchema] = Field(..., description="List of resolved citations mapping to source files")
    confidence: ConfidenceSchema = Field(..., description="Confidence assessment score and label")


# 2. Medical RAG System Prompt
MEDICAL_SYSTEM_PROMPT_TEMPLATE = """You are a precise, evidence-based healthcare AI assistant. Your task is to answer the user's query using ONLY the provided Source Context. Adhere strictly to the following rules:

1. Answer ONLY from the retrieved context. If the context does not contain the answer, state: "I could not find sufficient evidence in the retrieved sources."
2. Do not fabricate, extrapolate, or hallucinate any medical recommendations or clinical treatments.
3. For every statement or claim you make, append inline citations referencing the source numbers in brackets, e.g., [1], [2], [1][2], etc.
4. If there is conflicting or contradictory evidence within the sources, identify and outline the conflict clearly.
5. If the evidence is insufficient or details are missing to answer fully, explicitly note the limitation.
6. Provide structured, concise, and professional summaries of the medical evidence.

Source Context:
{context}"""


# Helper dependency instantiations
def get_search_coordinator() -> SearchCoordinator:
    return SearchCoordinator()


async def execute_rag_pipeline(
    db: AsyncSession,
    search_coordinator: SearchCoordinator,
    query: str,
    limit: int,
    history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Executes the full RAG pipeline:
    Query Expansion -> Retrieval -> Context Builder -> LLM -> Citations -> Confidence
    """
    start_time = time.perf_counter()
    
    # 1 & 2. Retrieval Engine runs expanded query + parent context reconstruction
    # (Existing SearchCoordinator performs MedicalQueryExpander internally)
    retrieved_chunks = await search_coordinator.semantic_search(
        db=db,
        query=query,
        limit=limit,
        expand_query=True
    )

    if not retrieved_chunks:
        # Fast exit if no clinical sources retrieved
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        fallback_answer = "I could not find sufficient evidence in the retrieved sources."
        fallback_confidence = {"score": 0.0, "label": "Low"}
        
        # Log to db
        log_record = QueryLog(
            query=query,
            answer=fallback_answer,
            citations=[],
            confidence_score=0.0,
            confidence_label="Low",
            response_time_ms=latency_ms
        )
        db.add(log_record)
        await db.commit()
        
        # Observability structured logging
        logger.info(
            f"RAG Pipeline Run: Query='{query}' | Citations=0 | Confidence=0.0 | Latency={latency_ms}ms"
        )
        
        return {
            "answer": fallback_answer,
            "citations": [],
            "confidence": fallback_confidence
        }

    # 3. Context Reconstruction & Deduplication
    context_data = ContextBuilder.build_context(retrieved_chunks)
    context_str = context_data["context"]
    sources = context_data["sources"]

    # 4. LLM Generation
    system_prompt = MEDICAL_SYSTEM_PROMPT_TEMPLATE.format(context=context_str)
    llm_answer = await LLMService.generate_answer(
        prompt=query,
        system_prompt=system_prompt,
        history=history
    )

    # 5. Citation Resolution
    citation_results = CitationEngine.resolve_citations(llm_answer, sources)
    final_answer = citation_results["answer"]
    citations = citation_results["citations"]

    # 6. Confidence Scoring
    confidence = ConfidenceScorer.calculate_confidence(final_answer, citations, retrieved_chunks)

    # 7. Record transaction latency
    latency_ms = int((time.perf_counter() - start_time) * 1000)

    # 8. Observability Structured Logging
    logger.info(
        f"RAG Pipeline Run: Query='{query}' | "
        f"Citations={len(citations)} | "
        f"Confidence={confidence['score']} ({confidence['label']}) | "
        f"Latency={latency_ms}ms"
    )

    # 9. Database Logging Transaction
    try:
        log_record = QueryLog(
            query=query,
            answer=final_answer,
            citations=citations,
            confidence_score=confidence["score"],
            confidence_label=confidence["label"],
            response_time_ms=latency_ms
        )
        db.add(log_record)
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to record RAG query execution in query_logs: {e}")

    return {
        "answer": final_answer,
        "citations": citations,
        "confidence": confidence
    }


@router.post("/ask", response_model=RAGResponseSchema, status_code=status.HTTP_200_OK)
async def ask_rag(
    payload: AskRequestSchema,
    search_coordinator: SearchCoordinator = Depends(get_search_coordinator),
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts a user query, retrieves clinical documents, passes them to LLM,
    and returns a cited answer with confidence scoring.
    """
    try:
        results = await execute_rag_pipeline(
            db=db,
            search_coordinator=search_coordinator,
            query=payload.query,
            limit=payload.limit
        )
        return results
    except Exception as e:
        logger.error(f"Failed to execute RAG ask: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG system failure: {str(e)}"
        )


@router.post("/chat", response_model=RAGResponseSchema, status_code=status.HTTP_200_OK)
async def chat_rag(
    payload: ChatRequestSchema,
    search_coordinator: SearchCoordinator = Depends(get_search_coordinator),
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts clinical query and previous message histories to support multi-turn conversation.
    """
    try:
        # Convert Pydantic history objects to simple dict formats for LLM
        formatted_history = []
        for msg in payload.history:
            formatted_history.append({
                "role": msg.role,
                "content": msg.content
            })

        results = await execute_rag_pipeline(
            db=db,
            search_coordinator=search_coordinator,
            query=payload.query,
            limit=payload.limit,
            history=formatted_history
        )
        return results
    except Exception as e:
        logger.error(f"Failed to execute RAG chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG system chat failure: {str(e)}"
        )
