import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ConfidenceScorer:
    # Medical Evidence Level weights mapping
    EVIDENCE_WEIGHTS = {
        "level 1a": 1.0,
        "level 1b": 1.0,
        "level 1": 1.0,
        "grade a": 1.0,
        "level 2a": 0.85,
        "level 2b": 0.85,
        "level 2": 0.85,
        "grade b": 0.85,
        "level 3a": 0.70,
        "level 3b": 0.70,
        "level 3": 0.70,
        "grade c": 0.70,
        "level 4": 0.55,
        "grade d": 0.55,
        "level 5": 0.45,
    }
    DEFAULT_EVIDENCE_WEIGHT = 0.60

    @classmethod
    def get_evidence_weight(cls, level_str: Any) -> float:
        if not level_str:
            return cls.DEFAULT_EVIDENCE_WEIGHT
        level_cleaned = str(level_str).strip().lower()
        return cls.EVIDENCE_WEIGHTS.get(level_cleaned, cls.DEFAULT_EVIDENCE_WEIGHT)

    @classmethod
    def calculate_sentence_citation_coverage(cls, answer: str) -> float:
        """
        Calculates the ratio of sentences containing at least one citation tag [x].
        """
        if not answer or not answer.strip():
            return 0.0

        # Split answer by punctuation (periods, question marks, exclamation marks followed by whitespace)
        # Avoid splitting abbreviation periods like "e.g." by matching period followed by space and capital letter or end of string
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
        if not sentences:
            return 0.0

        cited_count = 0
        for s in sentences:
            # Check if sentence has citation brackets e.g. [1]
            if re.search(r'\[[0-9]+\]', s):
                cited_count += 1

        return cited_count / len(sentences)

    @classmethod
    def calculate_confidence(
        cls, 
        answer: str, 
        citations: List[Dict[str, Any]], 
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Computes a deterministic clinical confidence score and label.

        Heuristic:
        - Similarity (30%): Average cosine similarity of cited chunks (or retrieved chunks if no citations).
        - Evidence Levels (30%): Average weight of cited sources' evidence grades.
        - Source Reinforcement (10%): Distinct documents supporting the answer.
        - Citation Coverage (30%): Percentage of sentences containing at least one citation.
        """
        if not retrieved_chunks:
            return {"score": 0.0, "label": "Low"}

        # 1. Similarity score
        # Extract cited chunk ids to get their scores, or default to top retrieved scores if none cited
        cited_chunk_ids = {c["chunk_id"] for c in citations if c.get("chunk_id")}
        
        # Build map of chunk_id -> score
        chunk_scores = {}
        for chunk in retrieved_chunks:
            c_id = chunk.get("chunk_id")
            if c_id:
                chunk_scores[str(c_id)] = chunk.get("score") or 0.70

        cited_scores = [chunk_scores[cid] for cid in cited_chunk_ids if cid in chunk_scores]
        if not cited_scores:
            # Fallback to all retrieved chunk scores if no citations mapped
            cited_scores = [chunk.get("score") or 0.70 for chunk in retrieved_chunks]
            
        avg_similarity = sum(cited_scores) / len(cited_scores) if cited_scores else 0.70

        # 2. Evidence level weights
        evidence_weights = []
        for cit in citations:
            el = cit.get("evidence_level")
            evidence_weights.append(cls.get_evidence_weight(el))
        if not evidence_weights:
            # Fallback to retrieved chunks evidence levels
            for chunk in retrieved_chunks:
                doc = chunk.get("document") or {}
                evidence_weights.append(cls.get_evidence_weight(doc.get("evidence_level")))
                
        avg_evidence_weight = sum(evidence_weights) / len(evidence_weights) if evidence_weights else cls.DEFAULT_EVIDENCE_WEIGHT

        # 3. Supporting documents reinforcement
        distinct_doc_ids = {cit["document_id"] for cit in citations if cit.get("document_id")}
        if not distinct_doc_ids:
            distinct_doc_ids = {chunk.get("document", {}).get("id") for chunk in retrieved_chunks if chunk.get("document", {}).get("id")}
            
        distinct_docs_count = len(distinct_doc_ids)
        if distinct_docs_count == 0:
            doc_reinforcement = 0.0
        elif distinct_docs_count == 1:
            doc_reinforcement = 0.85
        else:
            doc_reinforcement = 1.0  # Multi-source agreement increases confidence

        # 4. Citation coverage
        citation_coverage = cls.calculate_sentence_citation_coverage(answer)

        # 5. Composite score calculation
        # Weights: Similarity (0.30) + Evidence Level (0.30) + Reinforcement (0.10) + Coverage (0.30)
        raw_score = (
            (avg_similarity * 0.30) + 
            (avg_evidence_weight * 0.30) + 
            (doc_reinforcement * 0.10) + 
            (citation_coverage * 0.30)
        )
        
        # Ensure score bounds
        final_score = float(max(0.0, min(1.0, raw_score)))

        # Assign labels
        if final_score >= 0.80:
            label = "High"
        elif final_score >= 0.50:
            label = "Medium"
        else:
            label = "Low"

        return {
            "score": round(final_score, 4),
            "label": label
        }
