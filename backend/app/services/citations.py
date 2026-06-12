import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class CitationEngine:
    @staticmethod
    def extract_year_from_date(date_str: Any) -> Any:
        """
        Helper to extract a 4-digit publication year from a date string or datetime.
        """
        if not date_str:
            return None
        # Convert to string and try regex search for 4-digit year
        date_str_s = str(date_str)
        match = re.search(r'\b(19|20)\d{2}\b', date_str_s)
        if match:
            return int(match.group(0))
        return None

    @classmethod
    def resolve_citations(
        cls, 
        answer: str, 
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parses brackets like [1], [2], [1][2], or [1, 2] from generated answer text,
        resolves them to the matching numbered items in sources,
        and constructs a structured citations list.

        Returns:
            {
                "answer": str,
                "citations": List[Dict[str, Any]]
            }
        """
        if not answer or not sources:
            return {
                "answer": answer or "",
                "citations": []
            }

        # Normalize sources dictionary for fast lookup by source_index
        sources_map = {int(src["source_index"]): src for src in sources}

        # Find all brackets. e.g. [1], [2], [12]
        # Regex captures integers within square brackets
        matches = re.findall(r'\[([0-9]+)\]', answer)
        
        # Deduplicate citation indexes
        seen_indexes = set()
        resolved_citations = []

        for m in matches:
            idx = int(m)
            if idx in seen_indexes:
                continue
                
            if idx in sources_map:
                seen_indexes.add(idx)
                source_data = sources_map[idx]
                
                # Extract year
                pub_year = cls.extract_year_from_date(source_data.get("publication_date"))

                resolved_citations.append({
                    "citation_index": idx,
                    "document_id": source_data.get("document_id"),
                    "chunk_id": source_data.get("chunk_id"),
                    "title": source_data.get("title"),
                    "publication_year": pub_year,
                    "source_type": source_data.get("source_type"),
                    "evidence_level": source_data.get("evidence_level"),
                    "page_number": source_data.get("page_number"),
                    "section_header": source_data.get("section_header")
                })
            else:
                logger.warning(f"Answer cited Source [{idx}] but it is not in the retrieved sources map.")

        # Sort citations by citation_index
        resolved_citations.sort(key=lambda x: x["citation_index"])

        return {
            "answer": answer,
            "citations": resolved_citations
        }
