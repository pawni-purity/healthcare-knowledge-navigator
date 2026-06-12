from backend.app.services.parser import PDFParserService

def test_chunk_document_basic():
    # Setup dummy page extraction data
    mock_pages = [
        {
            "page_number": 1,
            "content": "SECTION 1. INTRODUCTION\nThis is a short sample sentence representing a medical guideline statement."
        },
        {
            "page_number": 2,
            "content": "SECTION 2. METHODOLOGY\nPatients were randomized to receive drug therapy or active control."
        }
    ]

    # Execute chunking
    chunks = PDFParserService.chunk_document(mock_pages)

    # We expect 2 parent chunks corresponding to the 2 pages
    assert len(chunks) == 2
    
    parent_1 = chunks[0]
    assert parent_1["chunk_type"] == "parent"
    assert parent_1["page_number"] == 1
    assert parent_1["section_header"] == "SECTION 1. INTRODUCTION"
    assert len(parent_1["child_chunks"]) > 0

    child_1 = parent_1["child_chunks"][0]
    assert child_1["chunk_type"] == "child"
    assert child_1["page_number"] == 1
    assert "INTRODUCTION" in child_1["content"]

    parent_2 = chunks[1]
    assert parent_2["chunk_type"] == "parent"
    assert parent_2["page_number"] == 2
    assert parent_2["section_header"] == "SECTION 2. METHODOLOGY"

def test_approximate_tokens():
    text = "Clinical indicators for heart failure guidelines show high diagnostic value."
    tokens = PDFParserService.approximate_tokens(text)
    # 10 words * 1.3 = 13 tokens
    assert tokens == 13
