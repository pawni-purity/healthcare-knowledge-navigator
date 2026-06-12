import io
import re
from typing import List, Dict, Any, Optional
from pypdf import PdfReader

class PDFParserService:
    @staticmethod
    def extract_text_by_page(file_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extracts raw text page by page from the PDF bytes.
        Returns a list of dicts: [{'page_number': int, 'content': str}]
        """
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_data = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_data.append({
                    "page_number": idx + 1,
                    "content": text.strip()
                })
        return pages_data

    @classmethod
    def chunk_document(cls, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies a parent-child hierarchical chunking strategy.
        - Parent chunks are generated per-page or per major section (approx 1200-1500 chars).
        - Child chunks are smaller sub-segments of parents (approx 300-400 chars, with 100 char overlap).
        
        Returns a list of chunk definitions:
        [
            {
                "chunk_type": "parent",
                "content": str,
                "page_number": int,
                "section_header": Optional[str],
                "chunk_index": int,
                "child_chunks": [
                     {"chunk_type": "child", "content": str, "page_number": int, "section_header": Optional[str]}
                ]
            }
        ]
        """
        chunks_structure = []
        chunk_counter = 0

        for page in pages_data:
            page_num = page["page_number"]
            text = page["content"]

            # Try to identify an approximate section header on this page
            # Usually the first line if it's short, capitalized, or has outline formatting
            first_line = text.split("\n")[0].strip() if text else ""
            section_header = None
            if len(first_line) > 3 and len(first_line) < 100:
                # Basic check for headers
                if first_line.isupper() or re.match(r'^\d+(\.\d+)*\s+[A-Z]', first_line) or "guideline" in first_line.lower():
                    section_header = first_line

            # 1. Create Parent Chunk (the page or split page)
            parent_content = text
            parent_chunk = {
                "chunk_type": "parent",
                "content": parent_content,
                "page_number": page_num,
                "section_header": section_header,
                "chunk_index": chunk_counter,
                "token_count": cls.approximate_tokens(parent_content),
                "child_chunks": []
            }
            chunk_counter += 1

            # 2. Create Child Chunks (overlapping windows)
            child_size = 400
            child_overlap = 100
            
            # Simple window chunker
            start = 0
            while start < len(parent_content):
                end = start + child_size
                child_text = parent_content[start:end].strip()
                if child_text:
                    child_chunk = {
                        "chunk_type": "child",
                        "content": child_text,
                        "page_number": page_num,
                        "section_header": section_header,
                        "chunk_index": chunk_counter,
                        "token_count": cls.approximate_tokens(child_text)
                    }
                    parent_chunk["child_chunks"].append(child_chunk)
                    chunk_counter += 1
                
                start += (child_size - child_overlap)

            chunks_structure.append(parent_chunk)

        return chunks_structure

    @staticmethod
    def approximate_tokens(text: str) -> int:
        """
        Approximates token count for estimation.
        Average word count * 1.3 is a reliable standard for clinical English text.
        """
        words = text.split()
        return int(len(words) * 1.3)
