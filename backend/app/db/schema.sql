-- Enable UUID generation extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Documents Metadata Table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    authors JSONB,                     -- JSONB array of strings for list of authors
    publisher VARCHAR(255),
    source_type VARCHAR(50) NOT NULL, -- 'clinical_guideline', 'biomedical_paper', 'treatment_protocol'
    source_url TEXT,
    publication_date DATE,
    version VARCHAR(50),
    evidence_level VARCHAR(50),
    document_hash VARCHAR(64) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Hierarchical Chunks Table
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE NOT NULL,
    parent_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    chunk_type VARCHAR(20) NOT NULL,                              -- 'parent', 'child'
    section_header TEXT,
    page_number INTEGER,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance Optimization Indexes
CREATE INDEX IF NOT EXISTS idx_docs_source_type ON documents(source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_chunk_id);

-- 3. Query Logs Table
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    citations JSONB,
    confidence_score REAL NOT NULL,
    confidence_label VARCHAR(20) NOT NULL,
    response_time_ms INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at);

