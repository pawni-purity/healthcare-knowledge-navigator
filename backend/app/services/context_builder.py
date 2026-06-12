import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ContextBuilder:
    @classmethod
    def build_context(
        cls, 
        retrieved_chunks: List[Dict[str, Any]], 
        max_tokens: int = 3000
    ) -> Dict[str, Any]:
        """
        Deduplicates chunks based on document ID and reconstructed parent content,
        preserves metadata & section hierarchy, maps chunks to numbered sources,
        and constructs a dense context string within token limits.

        Returns:
            {
                "context": str,
                "sources": List[Dict[str, Any]]
            }
        """
        if not retrieved_chunks:
            return {
                "context": "No source context available.",
                "sources": []
            }

        seen_contexts = set()
        unique_blocks: List[Dict[str, Any]] = []

        # Deduplicate overlapping parent contexts while keeping the order (relevance)
        for chunk in retrieved_chunks:
            doc_meta = chunk.get("document") or {}
            doc_id = doc_meta.get("id") or "unknown_doc"
            
            # Prefer parent_content if present, fallback to child content
            content = chunk.get("parent_content") or chunk.get("content") or ""
            content_cleaned = content.strip()
            
            if not content_cleaned:
                continue

            # Create a unique key for deduplication
            dedup_key = (doc_id, content_cleaned)
            if dedup_key in seen_contexts:
                continue
                
            seen_contexts.add(dedup_key)
            unique_blocks.append({
                "chunk_id": chunk.get("chunk_id"),
                "content": content_cleaned,
                "page_number": chunk.get("page_number"),
                "section_header": chunk.get("section_header"),
                "score": chunk.get("score"),
                "document": doc_meta
            })

        # Format context blocks sequentially and apply token budgets
        context_parts = []
        sources = []
        current_token_estimate = 0
        
        for idx, block in enumerate(unique_blocks):
            source_index = idx + 1
            doc = block["document"]
            
            # Format a clear metadata header block
            header = (
                f"Source [{source_index}]:\n"
                f"- Document ID: {doc.get('id')}\n"
                f"- Title: {doc.get('title', 'Unknown')}\n"
                f"- Publisher: {doc.get('publisher') or 'N/A'}\n"
                f"- Source Type: {doc.get('source_type', 'Unknown')}\n"
                f"- Evidence Level: {doc.get('evidence_level') or 'N/A'}\n"
                f"- Page: {block.get('page_number') or 'N/A'}\n"
                f"- Section: {block.get('section_header') or 'N/A'}\n"
            )
            
            body = f"Content:\n{block['content']}\n\n"
            block_text = header + body
            
            # Approximate tokens (4 characters per token heuristic)
            block_tokens = len(block_text) // 4
            
            if current_token_estimate + block_tokens > max_tokens:
                logger.info(f"Context builder reached max token limit ({max_tokens}). Truncating further sources.")
                break
                
            current_token_estimate += block_tokens
            context_parts.append(block_text)
            
            # Append to standard sources list for citations mapping
            sources.append({
                "source_index": source_index,
                "chunk_id": block["chunk_id"],
                "document_id": doc.get("id"),
                "title": doc.get("title"),
                "publisher": doc.get("publisher"),
                "source_type": doc.get("source_type"),
                "evidence_level": doc.get("evidence_level"),
                "page_number": block.get("page_number"),
                "section_header": block.get("section_header"),
                "publication_date": doc.get("publication_date")
            })

        final_context = "".join(context_parts).strip()
        
        return {
            "context": final_context,
            "sources": sources
        }
